<div align="center">

# RealtimeWAM: How Fast Can I Run<br>My World Action Model?

**OpenWAM inference with parallel execution, token reuse and adaptive refinement**

Huanan Liu, Ye Li, Kangye Ji, Xiaoyu Chen, Hanyun Cui, Yutian Shen,<br>
Yuan Meng, Chenglei Wu, Jingyan Jiang, Bo Li, and Zhi Wang

Tsinghua University · YuanxingGuangnian Robotics · Nanjing University

[![Paper](https://img.shields.io/badge/arXiv-2610.10079-b31b1b.svg)](https://arxiv.org/abs/2610.10079)
[![Code](https://img.shields.io/badge/code-RealtimeWAM-2563eb.svg)](https://github.com/Na2Ssss/RealtimeWAM)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

[Paper](https://arxiv.org/abs/2610.10079) · [Method](#method) · [Quick start](#quick-start) · [中文源码导读](CODE_GUIDE.md)

</div>

> **RealtimeWAM** accelerates existing World Action Models through parallel execution, selective observation FFN reuse and motion-adaptive refinement, without additional training.

This repository implements all three components for **OpenWAM** as an external runtime: it shares the original model and weights and requires no upstream source edits. Parallel execution is always enabled; the two approximate methods have independent switches. FastWAM and the full closed-loop evaluation suite are outside this release.

## Method

<p align="center">
  <img src="docs/assets/method.png" alt="RealtimeWAM: parallel computation, selective observation FFN reuse and motion-adaptive Transformer refinement." width="760">
</p>

*Overview from [Figure 4 of the paper](https://arxiv.org/html/2610.10079v1#S3.F4).*

| Component | Implementation | Control |
| --- | --- | --- |
| **Parallel execution** | [Branch scheduling](realtimewam/joint.py), [input preparation](realtimewam/preparation.py) | Always enabled |
| **Cross-observation token reuse** | [Selection and history](realtimewam/approximation.py), [FFN execution](realtimewam/blocks.py) | `reuse_tokens=True` |
| **Motion-adaptive refinement** | [Budget decision](realtimewam/approximation.py), [sampling and residual reuse](realtimewam/runtime.py) | `adaptive_2f4f=True` |

**Parallel execution.** Overlap independent input preparation and Video/Action branch phases, joining at joint attention. Observation and future-video rows batch compatible operations; observation K/V from the first evaluation is reused within the request. Compiled phases, fused operators and CUDA Graph replay reduce execution overhead.

**Token reuse.** RGB changes mark mandatory observation positions. After a dense first layer, feature changes fill the chosen refresh capacity. Later layers reuse only unrefreshed observation **FFN outputs**; attention, current residual gates, future-video and Action computation remain fresh. The default capacities are **16/80/120**, with zero reserve. **80 means refreshing 80 and reusing 40 of 120 observation positions.**

**Adaptive refinement.** Both paths retain **ten Euler updates** and share full Transformer evaluations at updates **0 and 3**. The coarse prediction then selects 2F for motion above **8 cm**, or adds evaluations at **6 and 9** for 4F. Other updates reuse Transformer residuals while computing current embeddings and output heads.

Token reuse and adaptive refinement introduce approximation. Both are disabled by default.

## Quick start

### Installation

Use a working [OpenWAM environment](https://github.com/OpenWAM-Official/OpenWAM/tree/898e2f96c02c172f078c17ff85a055a6bbd419a2) and compatible checkpoint; `import openwam` must already work. Weights and datasets are supplied separately.

```bash
git clone https://github.com/Na2Ssss/RealtimeWAM.git
cd RealtimeWAM
python -m pip install --no-deps -e .
```

Validated with Python 3.10/3.11, PyTorch 2.7.1+cu128 and matching Triton on A100 and RTX 4090. Extensions require a C++ compiler, CUDA development tools, Ninja and the `nvidia.cublas` package. Supported models use CUDA BF16, `head_dim=128`, Wan22Ti2v, `joint_self_attn`, first-frame-causal attention and the native Wan VAE.

### Basic inference

Use an RGB observation and raw proprioception in the checkpoint's coordinate order. Returned actions are denormalized NumPy arrays.

```python
import numpy as np
from PIL import Image
from realtimewam import OpenWAM

prompt = "Pick up the red block."
image = Image.open("observation.png").convert("RGB")  # 320 W × 384 H
state = np.load("proprio.npy")

model = OpenWAM("/path/to/checkpoint")
result = model.generate(prompt, image, proprio=state)
actions = result["actions"]
model.close()
```

Defaults: **384×320 image, 9 video frames, 32 output actions, ten Euler updates, seed 42**. Set `decode_video=True` to return video. Keep one model alive across requests; the first call includes compilation and graph capture.

### Enable token reuse and adaptive 2F/4F

Supply a motion metric that converts predicted actions into end-effector displacement in **metres**. For a robot whose actions and state both contain **absolute end-effector XYZ in metres in columns 0:3**, the complete metric is:

```python
def displacement(actions, raw_proprio):
    current_xyz = np.asarray(raw_proprio)[..., :3]
    return float(np.linalg.norm(actions[:32, :3] - current_xyz, axis=-1).max())

model = OpenWAM(
    "/path/to/checkpoint",
    reuse_tokens=True,
    adaptive_2f4f=True,
)
result = model.generate(
    prompt, image, proprio=state,
    motion_metric=lambda actions: displacement(actions, state),
)
print(model.architecture.last_stats)
model.close()
```

For two arms, use the maximum over both arms. Joint actions require forward kinematics; delta actions require the controller's accumulation and scaling. The metric receives the denormalized coarse prediction after the second full evaluation. Use the XYZ example only for the stated action representation.

The switches are independent: token reuse alone needs no motion callback. Adaptive refinement requires ten Euler updates and a callback returning a finite, nonnegative displacement.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `reuse_tokens` | `False` | Reuse observation FFN outputs across requests |
| `adaptive_2f4f` | `False` | Choose 2F/4F from predicted motion |
| `refresh_buckets` | `(16, 80, 120)` | Prepared observation refresh capacities |
| `reserve_fraction` | `0.0` | Additional refresh allowance outside mandatory RGB positions |
| `motion_threshold` | `0.08` | Displacement threshold in metres |

The online capacities follow [Figure 6 / Appendix E.3](https://arxiv.org/html/2610.10079v1#A5.SS3) and its evaluation configuration. To use the separate fixed-80 profiling configuration in [Appendix C.4](https://arxiv.org/html/2610.10079v1#A3.SS4), set `refresh_buckets=(80, 120), reserve_fraction=0.0`. It selects 80 when at most 80 positions are mandatory, otherwise full refresh. Initialization always refreshes every observation position.

Call **`model.reset()` when starting a new episode or changing the camera/preprocessing**. Prompt, weight and input-geometry changes invalidate incompatible history automatically. Cache history is committed only after a successful request. Calls on one model are serialized; parallelism occurs within each request.

<details>
<summary>Wrap an already loaded OpenWAM architecture</summary>

```python
from realtimewam import accelerate

fast = accelerate(architecture, reuse_tokens=True)
result = fast.generate(
    schedule, prompt,
    first_frame_image=image,
    proprio=architecture.normalize_deploy_proprio(state),
    action_num_frames=33,
)
architecture = fast.close()
```

`accelerate` accepts the same switches. This lower-level interface takes the native synchronous schedule and an already normalized state. `OpenWAM` handles checkpoint loading and normalization for you.

</details>

## Benchmark and tests

```bash
python -m realtimewam.benchmark \
  --checkpoint /path/to/checkpoint \
  --input /path/to/request.npz \
  --prompt-file /path/to/prompt.txt
```

The NPZ contains `image` (HWC uint8 RGB) and `proprio` (raw state). Add `--reuse-tokens` independently, or `--adaptive-2f4f --motion-metric motion_metric:displacement`. For the latter, save your robot-specific metric in `motion_metric.py`; the CLI callback accepts `(actions, raw_proprio)`. The benchmark measures repeated-input CPU-to-CPU inference after warmup.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  -c realtimewam/pytest.ini realtimewam/tests -q -p no:cacheprovider
```

See [VALIDATION.md](VALIDATION.md) for numerical checks and timing protocols, and [CODE_GUIDE.md](CODE_GUIDE.md) for a walkthrough of the **10 core Python files**. Detailed measurements are kept outside this README.

## Migration from WAMInfer

The project is now **RealtimeWAM**, with Python imports under `realtimewam`. Update `from WAMInfer import ...` to `from realtimewam import ...`, and CLI calls to `python -m realtimewam.benchmark`. Older tags retain the original package name. Historical evidence files retain their original source hashes.

## Citation

If you use this code, please cite RealtimeWAM. See also [CITATION.cff](CITATION.cff).

```bibtex
@misc{liu2026realtimewam,
  title         = {RealtimeWAM: How Fast Can I Run My World Action Model?},
  author        = {Huanan Liu and Ye Li and Kangye Ji and Xiaoyu Chen and Hanyun Cui and Yutian Shen and Yuan Meng and Chenglei Wu and Jingyan Jiang and Bo Li and Zhi Wang},
  year          = {2026},
  eprint        = {2610.10079},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  doi           = {10.48550/arXiv.2610.10079},
  url           = {https://arxiv.org/abs/2610.10079}
}
```

## License and acknowledgements

Code is released under [Apache-2.0](LICENSE). RealtimeWAM builds on [OpenWAM](https://github.com/OpenWAM-Official/OpenWAM); its external-addon organization was informed by [BAC](https://github.com/ky-ji/BAC/tree/82029a6fb0573fd07f4b26088219b0eb5ccc5a67). See [NOTICE](NOTICE) for provenance. Upstream weights and datasets retain their respective terms.

The method figure is reproduced from [RealtimeWAM, Figure 4](https://arxiv.org/html/2610.10079v1#S3.F4) under the paper's [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).
