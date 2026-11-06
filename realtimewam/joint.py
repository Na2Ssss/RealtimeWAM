"""Complete branch scheduling with compiled phases and explicit stream joins.

The first evaluation can publish current-frame K/V for this request's later
evaluations. All future Video and Action tokens are always recomputed.
"""

from functools import partial
import torch
from realtimewam._triton_attention import masked_attention, current_frame_attention, joint_first_frame_attention


def _branch_phase(backbone, layer, last, state, attended, post, *, ffn=False):
    finish = backbone.ffn_at_layer_for_compile if ffn else backbone.post_attn_at_layer_for_compile
    finish(layer, state, attended, post)
    if layer != last:
        return backbone.pre_attn_at_layer_for_compile(layer + 1, state)


class ParallelJointLoop:
    """Overlap branch phases, joining at every joint attention."""

    def __init__(self, arch):
        self.driver = arch.mot_driver
        self.video_backbone = vb = arch.video_backbone
        ab = arch.action_backbone
        self.first_frame_tokens = 0
        self.structured_attention = torch.cuda.get_device_capability(arch.device) in ((8, 0), (8, 9))
        self.stream = torch.cuda.Stream(device=arch.device)
        kwargs = dict(fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
        if torch.cuda.get_device_capability(arch.device) == (8, 9):
            # Tune complete fused phases, including GEMM epilogues. Ada request
            # measurements favor this over tuning isolated linear operators.
            kwargs["options"] = dict(kwargs.get("options", {}), max_autotune=True)
        self.video_first = torch.compile(partial(vb.pre_attn_at_layer_for_compile, 0), **kwargs)
        self.action_first = torch.compile(partial(ab.pre_attn_at_layer_for_compile, 0), **kwargs)
        self.attention = torch.compile(self._attention, **kwargs)
        self.video_post = torch.compile(self.video_backbone.post_attn_at_layer_for_compile, **kwargs)
        last = self.driver.num_layers - 1
        self.video = [
            torch.compile(partial(_branch_phase, vb, i, last, ffn=True), **kwargs) for i in range(last + 1)
        ]
        self.action = [torch.compile(partial(_branch_phase, ab, i, last), **kwargs) for i in range(last + 1)]

    def _attention(self, qv, qa, kv, ka, vv, va, mask, prefix=None):
        video_count, action_count = qv.shape[1], qa.shape[1]
        heads = self.driver.num_heads
        if prefix is not None and self.structured_attention and mask is None:
            return current_frame_attention(qv, qa, prefix[0], kv, ka, prefix[1], vv, va, heads)
        q = torch.cat((qv, qa), dim=1)
        k = torch.cat((kv, ka) if prefix is None else (prefix[0], kv, ka), dim=1)
        v = torch.cat((vv, va) if prefix is None else (prefix[1], vv, va), dim=1)
        if (
            prefix is None
            and self.structured_attention
            and self.driver.attention_mask_mode == "mutual"
            and self.driver.vb.video_attention_mask_mode == "first_frame_causal"
        ):
            result = joint_first_frame_attention(q, k, v, video_count, self.first_frame_tokens, heads)
        else:
            result = masked_attention(q, k, v, mask, heads)
        video, action = result.split((video_count, action_count), dim=1)
        return video.contiguous(), action.contiguous()

    def __call__(self, video_state, action_state):
        limit = max(torch._dynamo.config.cache_size_limit, 16 * self.driver.num_layers + 128)
        accumulated = max(torch._dynamo.config.accumulated_cache_size_limit, limit)
        with torch._dynamo.config.patch(cache_size_limit=limit, accumulated_cache_size_limit=accumulated):
            return self._run(video_state, action_state)

    def _run(self, video_state, action_state):
        if torch.is_grad_enabled() or self.driver.ab.training:
            raise RuntimeError("Prepared branches require eval mode and disabled gradients.")
        device = video_state.hidden_states.device
        video_count = video_state.hidden_states.shape[1]
        action_count = action_state.x_action.shape[1]
        self.first_frame_tokens = self.driver._video_tokens_per_frame(video_state)
        mask = self.driver._build_attention_mask(
            s_video=video_count, s_action=action_count, video_tokens_per_frame=self.first_frame_tokens, device=device
        )
        prefix = video_state.extras.get("first_frame_kv")
        recording = video_state.extras.get("record_first_frame", False)
        captured = []
        if prefix is not None:
            banks, final_clean = prefix
            clean = final_clean.shape[1]
            original_mod, original_rope = video_state.time_mod, video_state.rope_freqs
            video_state.hidden_states = video_state.hidden_states[:, clean:].contiguous()
            video_state.time_mod = original_mod[:, clean:].contiguous()
            video_state.rope_freqs = original_rope[clean:]
            mask = mask[clean:].contiguous()
            if self.driver.attention_mask_mode == "mutual":
                mask = None
        main = torch.cuda.current_stream(device)
        side = self.stream
        retained = [video_state.hidden_states, action_state.x_action]
        side.wait_stream(main)
        with torch.cuda.stream(side):
            action_phase = self.action_first(action_state)
        video_phase = self.video_first(video_state)
        for index in range(self.driver.num_layers):
            main.wait_stream(side)
            qv, kv, vv, video_post = video_phase
            if recording:
                captured.append(
                    (kv[:, : self.first_frame_tokens].contiguous(), vv[:, : self.first_frame_tokens].contiguous())
                )
            qa, ka, va, action_post = action_phase
            video_attended, action_attended = self.attention(
                qv, qa, kv, ka, vv, va, mask, banks[index] if prefix is not None else None
            )
            # Keep all cross-stream inputs alive until the final dependency join.
            retained.extend((video_phase, action_phase, video_attended, action_attended))
            last_action = index == self.driver.num_layers - 1 and video_state.extras.get("action_only", False)
            if not last_action:
                video_ffn = self.video_post(index, video_state, video_attended, video_post)
            # Overlap Action with Video FFN, leaving the smaller projections uncontended.
            side.wait_stream(main)
            with torch.cuda.stream(side):
                action_phase = self.action[index](action_state, action_attended, action_post)
            if last_action:
                break
            video_phase = self.video[index](video_state, *video_ffn)
        main.wait_stream(side)
        action_state.x_action.record_stream(main)
        if prefix is not None and not video_state.extras.get("action_only", False):
            video_state.hidden_states = torch.cat((final_clean, video_state.hidden_states), dim=1)
            video_state.time_mod, video_state.rope_freqs = original_mod, original_rope
        if recording:
            video_state.extras["first_frame_result"] = (
                tuple(captured),
                video_state.hidden_states[:, : self.first_frame_tokens].contiguous(),
            )
        return video_state, action_state
