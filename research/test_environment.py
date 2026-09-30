from src.environment.mario_env import create_mario_env


def main():

    env = create_mario_env(
        render_mode=None
    )

    try:

        observation, info = env.reset(
            seed=42
        )

        print("=" * 70)
        print("AMBIENTE SUPER MARIO BROS")
        print("=" * 70)

        print(
            f"Observation shape: {observation.shape}"
        )

        print(
            f"Action space: {env.action_space}"
        )

        print(
            f"Actions: {env.get_action_meanings()}"
        )

        print(
            f"Initial X: {info.get('x_pos')}"
        )

        print(
            f"Initial Y: {info.get('y_pos')}"
        )

        print("\nExecutando 20 steps...")

        for step in range(20):

            # 1 = direita no SIMPLE_MOVEMENT
            action = 1

            observation, reward, terminated, truncated, info = \
                env.step(action)

            print(
                f"step={step + 1:02d} "
                f"x={info.get('x_pos')} "
                f"y={info.get('y_pos')} "
                f"reward={reward:.2f}"
            )

            if terminated or truncated:
                break

        print("\nAmbiente funcionando corretamente.")

    finally:

        env.close()


if __name__ == "__main__":
    main()