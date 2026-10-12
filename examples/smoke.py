"""Offline wiring check for the public OpenWAM-Alpha-Sim-RoboTwin-Full checkpoint."""

import argparse
import json

import numpy as np
import torch
from PIL import Image

from realtimewam import OpenWAM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reuse-tokens", action="store_true")
    parser.add_argument("--adaptive-2f4f", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = OpenWAM(
        args.checkpoint,
        reuse_tokens=args.reuse_tokens,
        adaptive_2f4f=args.adaptive_2f4f,
    )
    image = Image.fromarray(np.full((384, 320, 3), 128, dtype=np.uint8))
    # This checkpoint takes a raw 20-D state, mapped internally to 80 dimensions.
    state = np.zeros(20, dtype=np.float32)
    for index, motion in enumerate((0.10, 0.10, 0.01)):
        # Fixed metre values exercise both routes; this is NOT a robot motion metric.
        result = model.generate(
            "Pick up the red block.", image, proprio=state,
            motion_metric=(lambda actions: motion) if args.adaptive_2f4f else None,
        )
        actions, stats = result["actions"], model.architecture.last_stats
        expected_steps = (0, 3) if motion > 0.08 else (0, 3, 6, 9)
        if not args.adaptive_2f4f:
            expected_steps = tuple(range(10))
        expected_refresh = (120 if index == 0 else 16) if args.reuse_tokens else None
        assert actions.shape == (32, 20) and np.isfinite(actions).all()
        assert stats["full_evaluations"] == expected_steps
        assert stats["refreshed_tokens"] == expected_refresh
        print(json.dumps(dict(request=index + 1, action_shape=actions.shape, **stats)), flush=True)
    model.close()
    print("PASS: finite actions and expected execution routes (synthetic inputs only).")


if __name__ == "__main__":
    main()
