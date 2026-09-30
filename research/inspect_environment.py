from pprint import pprint

from src.environment.mario_env import create_mario_env


def main():
    env = create_mario_env(render_mode=None)

    try:
        print("=" * 70)
        print("SUPER MARIO BROS - INSPEÇÃO DO AMBIENTE")
        print("=" * 70)

        print("\nENVIRONMENT")
        print(f"ID: SuperMarioBros-1-1-v0")

        print("\nOBSERVATION SPACE")
        print(env.observation_space)

        print("\nACTION SPACE")
        print(env.action_space)

        print("\nAÇÕES")
        print(env.get_action_meanings())

        print("\nRESET")
        observation, info = env.reset(seed=42)

        print(f"Observation type: {type(observation)}")
        print(f"Observation shape: {observation.shape}")
        print(f"Observation dtype: {observation.dtype}")

        print("\nINFO INICIAL")
        pprint(info)

        print("\nEXECUTANDO 100 STEPS")
        print("-" * 70)

        for step in range(100):
            action = env.action_space.sample()

            observation, reward, terminated, truncated, info = env.step(action)

            print(
                f"step={step:03d} | "
                f"action={action} | "
                f"reward={reward:7.2f} | "
                f"x={info.get('x_pos')} | "
                f"y={info.get('y_pos')} | "
                f"enemy_types={info.get('enemy_types')} | "
                f"death={info.get('death')} | "
                f"clear={info.get('clear')}"
            )

            if terminated or truncated:
                print("\nEPISÓDIO ENCERRADO")
                break

    finally:
        env.close()


if __name__ == "__main__":
    main()