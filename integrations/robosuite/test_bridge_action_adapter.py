import unittest

import numpy as np

from integrations.robosuite.bridge_action_adapter import BridgeToOSCAdapter


class BridgeToOSCAdapterTest(unittest.TestCase):
    def setUp(self):
        self.adapter = BridgeToOSCAdapter(
            input_min=np.full(6, -2.0),
            input_max=np.full(6, 2.0),
            output_min=np.array([-0.1, -0.2, -0.4, -1.0, -2.0, -4.0]),
            output_max=np.array([0.1, 0.2, 0.4, 1.0, 2.0, 4.0]),
            gripper_open_input=-0.75,
            gripper_closed_input=0.8,
        )

    def test_inverse_controller_scaling_and_open_gripper(self):
        result = self.adapter.convert([0.05, 0.1, 0.2, 0.5, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(result[:6], [1, 1, 1, 1, 0, 0], atol=1e-6)
        self.assertEqual(result[6], -0.75)

    def test_clips_pose_and_closes_gripper(self):
        result = self.adapter.convert([1, -1, 1, 0, 0, 0, 0.49])
        np.testing.assert_allclose(result[:6], [2, -2, 2, 0, 0, 0])
        self.assertEqual(result[6], 0.8)

    def test_converts_euler_delta_to_osc_axis_angle(self):
        result = self.adapter.convert([0, 0, 0, 0.1, -0.2, 0.3, 1])
        # Euler composition is not equivalent to passing the three components
        # straight through as an axis-angle vector.
        self.assertFalse(np.allclose(result[3:6], [0.2, -0.2, 0.15]))
        np.testing.assert_allclose(
            result[3:6],
            [0.1378492, -0.21322572, 0.14437446],
            atol=1e-6,
        )

    def test_rejects_wrong_action_shape(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            self.adapter.convert(np.zeros(6))


if __name__ == "__main__":
    unittest.main()
