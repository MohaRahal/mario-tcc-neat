from src.environment.mario_env import create_mario_env
from src.sensors.spatial_sensor import SpatialSensor


def print_matrix(matrix):

    for row in matrix:

        line = ""

        for value in row:

            if value == 1:
                line += " # "
            else:
                line += " . "

        print(line)


def main():

    print("INICIANDO TESTE...")

    env = create_mario_env(
        render_mode=None
    )

    sensor = SpatialSensor(env)

    try:

        observation, info = env.reset(
            seed=42
        )

        print("=" * 60)
        print("TESTE DO SENSOR ESPACIAL")
        print("=" * 60)

        print(
            f"Tamanho do sensor: {sensor.size}"
        )

        for step in range(300):

            observation, reward, terminated, truncated, info = (
                env.step(1)
            )

            if step % 25 == 0:

                print()
                print("=" * 60)
                print(f"STEP: {step}")
                print(
                    f"Mario: "
                    f"x={info.get('x_pos')} "
                    f"y={info.get('y_pos')}"
                )
                print()

                matrix = sensor.matrix(info)

                print_matrix(matrix)

            if info.get("death"):
                print()
                print(
                    f"Mario morreu em "
                    f"x={info.get('x_pos')}"
                )
                break

            if info.get("clear"):
                print("Fase concluída.")
                break

            if terminated or truncated:
                print(
                    f"Ambiente finalizado. "
                    f"terminated={terminated}, "
                    f"truncated={truncated}"
                )
                break

    finally:

        env.close()


if __name__ == "__main__":
    main()