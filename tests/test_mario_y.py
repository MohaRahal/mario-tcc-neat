from src.environment.mario_env import create_mario_env


def get_base_env(env):

    current = env

    while True:

        if hasattr(current, "ram"):
            return current

        if not hasattr(current, "env"):
            raise RuntimeError(
                "Não foi possível acessar a RAM."
            )

        current = current.env


def main():

    env = create_mario_env(
        render_mode=None
    )

    try:

        observation, info = env.reset(
            seed=42
        )

        base_env = get_base_env(env)

        print("=" * 70)
        print("TESTE DO EIXO Y DO MARIO")
        print("=" * 70)

        for step in range(180):

            # Corre + pula.
            observation, reward, terminated, truncated, info = (
                env.step(2)
            )

            if step % 5 == 0:

                ram = base_env.ram

                print(
                    f"step={step:03d} "
                    f"x={int(info.get('x_pos', 0)):4d} "
                    f"y={int(info.get('y_pos', 0)):3d} "
                    f"RAM_B5={int(ram[0xB5]):3d} "
                    f"RAM_CE={int(ram[0xCE]):3d} "
                    f"RAM_3B8={int(ram[0x3B8]):3d}"
                )

            if info.get("death"):
                print("Mario morreu.")
                break

            if info.get("clear"):
                print("Fase concluída.")
                break

            if terminated or truncated:
                break

    finally:
        env.close()


if __name__ == "__main__":
    main()