import numpy as np
import unittest

from src.sensors.experiment_sensor_system import (
    ExperimentSensorSystem,
    SENSOR_PROFILES,
    sensor_profile_for_inputs,
)


class FakeEnv:
    def __init__(self):
        self.ram = np.zeros(0x0800, dtype=np.uint8)
        self.unwrapped = self


INFO = {
    "x_pos": 40,
    "y_pos": 176,
    "player_state": 0,
    "powerup_level": 0,
}


class ExperimentSensorSystemTest(unittest.TestCase):
    def test_each_historical_profile_returns_the_expected_size(self):
        for num_inputs in sorted(SENSOR_PROFILES):
            with self.subTest(num_inputs=num_inputs):
                sensors = ExperimentSensorSystem(FakeEnv(), num_inputs)
                sensors.reset()
                vector = sensors.observe(INFO)
                self.assertEqual(vector.shape, (num_inputs,))

    def test_profile_70_exposes_previous_action_on_the_next_observation(self):
        sensors = ExperimentSensorSystem(FakeEnv(), 70)
        sensors.reset()
        sensors.set_previous_action(3)
        vector = sensors.observe(INFO)
        self.assertEqual(
            vector[62:69].tolist(),
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        )

    def test_unknown_profile_is_rejected_instead_of_silently_remapping_inputs(self):
        with self.assertRaisesRegex(RuntimeError, "Perfis disponiveis"):
            sensor_profile_for_inputs(99)


if __name__ == "__main__":
    unittest.main()
