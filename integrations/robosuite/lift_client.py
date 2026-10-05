"""Run the official CrossFormer Bridge policy in robosuite Lift / Panda."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import json_numpy
import numpy as np
import requests
import robosuite as suite

# Allow direct execution without installing the integrations directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bridge_action_adapter import BridgeToOSCAdapter  # noqa: E402


INSTRUCTION = "pick up the cube"


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> requests.Response:
    response = requests.post(
        url,
        data=json_numpy.dumps(payload),
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()
    return response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--max-steps", default=500, type=int)
    # BridgeData V2 demonstrations were collected at 5 Hz (0.2 s/action).
    parser.add_argument("--control-freq", default=5, type=int)
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--timeout", default=120.0, type=float)
    parser.add_argument(
        "--action-scale",
        default=1.0,
        type=float,
        help="Safety multiplier for the six physical Bridge pose deltas",
    )
    parser.add_argument(
        "--translation-signs",
        nargs=3,
        type=float,
        default=(1.0, 1.0, 1.0),
        metavar=("X", "Y", "Z"),
        help="Diagnostic Bridge-to-robosuite translation axis signs",
    )
    args = parser.parse_args()
    if not 0 < args.action_scale <= 1:
        parser.error("--action-scale must be in (0, 1]")

    controller_config = suite.load_controller_config(default_controller="OSC_POSE")
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=controller_config,
        use_camera_obs=True,
        has_renderer=args.render,
        has_offscreen_renderer=True,
        camera_names=args.camera,
        camera_heights=224,
        camera_widths=224,
        control_freq=args.control_freq,
        horizon=args.max_steps,
        ignore_done=False,
    )

    adapter = BridgeToOSCAdapter.from_robosuite_env(env)
    print(
        json.dumps(
            {
                "osc_input_min": adapter.input_min.tolist(),
                "osc_input_max": adapter.input_max.tolist(),
                "osc_output_min": adapter.output_min.tolist(),
                "osc_output_max": adapter.output_max.tolist(),
                "gripper_open_input": adapter.gripper_open_input,
                "gripper_closed_input": adapter.gripper_closed_input,
            },
            indent=2,
        )
    )

    observation = env.reset()
    _post_json(
        f"{args.server}/reset",
        {"instruction": INSTRUCTION},
        args.timeout,
    )

    try:
        for step in range(args.max_steps):
            image_key = f"{args.camera}_image"
            bridge_action = np.array(
                json_numpy.loads(
                    _post_json(
                        f"{args.server}/query",
                        {"image_primary": observation[image_key]},
                        args.timeout,
                    ).text
                ),
                dtype=np.float32,
                copy=True,
            )
            bridge_action[:6] *= args.action_scale
            bridge_action[:3] *= np.asarray(args.translation_signs, dtype=np.float32)
            robosuite_action = adapter.convert(bridge_action)

            observation, reward, done, info = env.step(robosuite_action)
            if args.render:
                env.render()
            if step % 10 == 0:
                eef_position = observation["robot0_eef_pos"]
                cube_position = observation["cube_pos"]
                print(
                    f"step={step} reward={reward:.3f} "
                    f"eef_cube_distance={np.linalg.norm(eef_position - cube_position):.4f} "
                    f"eef={np.round(eef_position, 4)} "
                    f"cube={np.round(cube_position, 4)} "
                    f"bridge={np.round(bridge_action, 4)} "
                    f"osc={np.round(robosuite_action, 4)}"
                )
            if done or env._check_success():
                print(f"finished at step={step}, success={env._check_success()}")
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
