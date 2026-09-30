# tests/test_sensor_177.py

from src.environment.mario_env import create_mario_env
from src.sensors.sensor_system import SensorSystem


env = create_mario_env(
    render_mode=None
)

try:
    observation, info = env.reset(
        seed=42
    )

    sensors = SensorSystem(
        env
    )

    sensors.reset()

    vector = sensors.observe(
        info
    )

    matrix = sensors.spatial_matrix(
        info
    )

    print(
        "Sensor size:",
        sensors.size,
    )

    print(
        "Vector size:",
        len(vector),
    )

    print(
        "Matrix shape:",
        matrix.shape,
    )

    print(
        "Valores da matriz:",
        sorted(
            set(
                matrix.flatten().tolist()
            )
        ),
    )

    print()
    print(matrix)

finally:
    env.close()