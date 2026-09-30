from src.environment.mario_env import create_mario_env


def main():

    env = create_mario_env(
        render_mode=None
    )

    try:

        observation, info = env.reset(
            seed=42
        )

        print("=" * 60)
        print("TESTE DE ACESSO À RAM")
        print("=" * 60)

        current = env
        level = 0

        while True:

            print(
                f"Wrapper {level}: "
                f"{type(current).__name__}"
            )

            if hasattr(current, "ram"):

                ram = current.ram

                print()
                print("RAM ENCONTRADA!")
                print(f"Tipo: {type(ram)}")
                print(f"Tamanho: {len(ram)}")
                print(
                    f"Primeiros 20 bytes: "
                    f"{ram[:20]}"
                )

                break

            if not hasattr(current, "env"):

                print()
                print("RAM não encontrada.")

                break

            current = current.env
            level += 1

    finally:

        env.close()


if __name__ == "__main__":
    main()