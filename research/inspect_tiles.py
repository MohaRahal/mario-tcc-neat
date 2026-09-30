from __future__ import annotations

from src.environment.mario_env import create_mario_env


TILE_START = 0x0500
TILE_END = 0x069F


# Valores conhecidos segundo o mapa de RAM do SMB1.
TILE_NAMES = {
    0x00: "EMPTY",

    0x08: "VINE",

    0x10: "PIPE_TOP_LEFT",
    0x11: "PIPE_TOP_RIGHT",
    0x12: "PIPE_TOP_LEFT",
    0x13: "PIPE_TOP_RIGHT",
    0x14: "PIPE_LEFT",
    0x15: "PIPE_RIGHT",

    0x16: "TREETOP_LEFT",
    0x17: "TREETOP_MIDDLE",
    0x18: "TREETOP_RIGHT",

    0x1C: "HORIZONTAL_PIPE_TOP_LEFT",
    0x1D: "HORIZONTAL_PIPE_TOP_MIDDLE",
    0x1E: "HORIZONTAL_PIPE_TOP_RIGHT",
    0x1F: "HORIZONTAL_PIPE_BOTTOM_LEFT",
    0x20: "HORIZONTAL_PIPE_BOTTOM_MIDDLE",
    0x21: "HORIZONTAL_PIPE_BOTTOM_RIGHT",

    0x22: "CORAL",
    0x23: "BUMPED_BLOCK",

    0x24: "FLAG_TOP",
    0x25: "FLAG_MIDDLE",
    0x26: "VINE",

    0x51: "BRICK",
    0x52: "BRICK",
    0x54: "GROUND",
    0x55: "BRICK_POWERUP",
    0x56: "BRICK_VINE",
    0x57: "BRICK_STAR",
    0x58: "BRICK_MULTI_COIN",
    0x5A: "BRICK_POWERUP",
    0x5D: "BRICK_MULTI_COIN",
    0x5E: "BRICK_1UP",
    0x5F: "HIDDEN_BLOCK_COIN",

    0x61: "CASTLE_GROUND",

    0x67: "SPRING_TOP",
    0x68: "SPRING_BOTTOM",

    0x69: "UNDERWATER_GROUND",

    0x6B: "UNDERWATER_PIPE_TOP",
    0x6C: "UNDERWATER_PIPE_BOTTOM",

    0x88: "CLOUD",
    0x89: "BRIDGE",

    0xC0: "BLOCK_COIN",
    0xC1: "BLOCK_POWERUP",
    0xC2: "COIN",
    0xC3: "UNDERWATER_COIN",
    0xC4: "EMPTY_BROWN_BLOCK",
    0xC5: "AXE",
}


def read_tile_ram(env) -> list[int]:
    ram = env.unwrapped.ram

    return [
        int(ram[address])
        for address in range(TILE_START, TILE_END + 1)
    ]


def main():
    env = create_mario_env(render_mode=None)

    try:
        observation, info = env.reset(seed=42)

        print("=" * 80)
        print("INSPEÇÃO DOS TILES - SUPER MARIO BROS 1-1")
        print("=" * 80)

        for step in range(300):

            # Faz Mario avançar.
            action = 1

            observation, reward, terminated, truncated, info = env.step(
                action
            )

            tiles = read_tile_ram(env)

            known_tiles = {}

            for index, value in enumerate(tiles):
                if value == 0:
                    continue

                known_tiles[value] = known_tiles.get(value, 0) + 1

            if step % 25 == 0:
                print(
                    f"\nSTEP={step}"
                    f" | Mario X={info.get('x_pos')}"
                    f" | Mario Y={info.get('y_pos')}"
                )

                print("-" * 80)

                if not known_tiles:
                    print("Nenhum tile não vazio encontrado.")

                else:
                    for value, count in sorted(known_tiles.items()):
                        name = TILE_NAMES.get(
                            value,
                            "UNKNOWN"
                        )

                        print(
                            f"0x{value:02X} "
                            f"| {value:3d} "
                            f"| {name:30s} "
                            f"| quantidade={count}"
                        )

            if info.get("death"):
                print("\nMario morreu.")
                break

            if info.get("clear"):
                print("\nFase concluída.")
                break

            if terminated or truncated:
                break

    finally:
        env.close()


if __name__ == "__main__":
    main()