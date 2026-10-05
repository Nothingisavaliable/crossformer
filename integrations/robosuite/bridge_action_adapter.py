"""Convert pretrained Bridge actions to robosuite OSC_POSE actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from robosuite.utils import transform_utils as transform


def _as_six_values(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(6, array.item(), dtype=np.float64)
    else:
        array = np.broadcast_to(array, (6,)).astype(np.float64, copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values, got {array}")
    return array


@dataclass(frozen=True)
class BridgeToOSCAdapter:
    """Map physical Bridge deltas and its binary gripper to robosuite.

    CrossFormer returns unnormalized Bridge actions as physical relative pose
    changes followed by an absolute binary gripper state (1=open, 0=closed).
    A robosuite OSC controller accepts values in its configured input range and
    internally maps them to its configured physical output range.  This class
    applies the inverse of that mapping.
    """

    input_min: np.ndarray
    input_max: np.ndarray
    output_min: np.ndarray
    output_max: np.ndarray
    gripper_open_input: float
    gripper_closed_input: float

    @classmethod
    def from_robosuite_env(cls, env: Any) -> "BridgeToOSCAdapter":
        if len(env.robots) != 1:
            raise ValueError("The initial integration supports exactly one robot")

        controller = env.robots[0].controller
        if controller.__class__.__name__ != "OperationalSpaceController":
            raise ValueError(
                "Bridge actions require robosuite OSC_POSE; got "
                f"{controller.__class__.__name__}"
            )
        if not controller.use_delta or not controller.use_ori:
            raise ValueError("OSC controller must use delta position and orientation")

        action_low, action_high = env.action_spec
        action_low = np.asarray(action_low, dtype=np.float64)
        action_high = np.asarray(action_high, dtype=np.float64)
        if action_low.shape != (7,) or action_high.shape != (7,):
            raise ValueError(
                "Expected Panda OSC_POSE action bounds with shape (7,), got "
                f"{action_low.shape} and {action_high.shape}"
            )

        # PandaGripper uses negative commands to open and positive commands to
        # close, so its actual action-space endpoints define these values.
        return cls(
            input_min=_as_six_values(controller.input_min, "input_min"),
            input_max=_as_six_values(controller.input_max, "input_max"),
            output_min=_as_six_values(controller.output_min, "output_min"),
            output_max=_as_six_values(controller.output_max, "output_max"),
            gripper_open_input=float(action_low[-1]),
            gripper_closed_input=float(action_high[-1]),
        )

    def __post_init__(self) -> None:
        input_span = self.input_max - self.input_min
        output_span = self.output_max - self.output_min
        if np.any(input_span <= 0) or np.any(output_span <= 0):
            raise ValueError("OSC input/output ranges must be strictly increasing")

    def convert(self, bridge_action: Any, *, clip: bool = True) -> np.ndarray:
        """Return one native robosuite action with shape ``(7,)``.

        Bridge reached-proprio rotations are Euler deltas. They are converted
        to the axis-angle delta consumed by OSC_POSE. The resulting six pose
        values undergo the exact inverse of robosuite's controller scaling.
        """

        bridge_action = np.asarray(bridge_action, dtype=np.float64)
        if bridge_action.shape != (7,):
            raise ValueError(
                f"Expected one Bridge action with shape (7,), got {bridge_action.shape}"
            )
        if not np.all(np.isfinite(bridge_action)):
            raise ValueError(f"Bridge action contains non-finite values: {bridge_action}")

        physical_pose = bridge_action[:6].copy()
        delta_rotation = transform.euler2mat(physical_pose[3:6])
        physical_pose[3:6] = transform.quat2axisangle(
            transform.mat2quat(delta_rotation)
        )

        input_span = self.input_max - self.input_min
        output_span = self.output_max - self.output_min
        osc_pose = self.input_min + (
            (physical_pose - self.output_min) * input_span / output_span
        )
        if clip:
            osc_pose = np.clip(osc_pose, self.input_min, self.input_max)

        # Bridge's gripper value is an absolute state. Thresholding prevents a
        # regression output near the class boundary from producing weak motion.
        gripper = (
            self.gripper_open_input
            if bridge_action[6] >= 0.5
            else self.gripper_closed_input
        )
        return np.concatenate((osc_pose, np.asarray([gripper]))).astype(np.float32)
