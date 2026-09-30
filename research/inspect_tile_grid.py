from __future__ import annotations

from src.environment.mario_env import create_mario_env


TILE_START = 0x0500
TILE_END = 0x069F

# Códigos que já identificamos.
TILE_SYMBOLS = {
    0x00: ".",

    0x51: "B",  # brick
    0x54: "#",  # ground

    0x12: "P",
    0x13: "P",
    0x14: "P",
    0x15: "P",

    0xC0: "C",
    0xC1: "?",
}


def read_tiles(env):
    ram = env.unwrapped.ram

    return [
        int(ram[address])
        for address in range(TILE_START, TILE_END + 1)
    ]


def print_grid(tiles):
    """
    Imprime a região de tiles tentando preservar a
    organização sequencial da memória.
    """

    width = 16

    for row in range(0, len(tiles), width):
        line = tiles[row:row + width]

        symbols = [
            TILE_SYMBOLS.get(value, "?")
            for value in line
        ]

        print(
            f"{row:03d}: "
            + " ".join(symbols)
        )


def main():
    env = create_mario_env(render_mode=None)

    try:
        observation, info = env.reset(seed=42)

        for step in range(201):

            if step % 25 == 0:

                tiles = read_tiles(env)

                print("\n")
                print("=" * 80)
                print(
                    f"STEP {step} | "
                    f"MARIO X={info.get('x_pos')} "
                    f"Y={info.get('y_pos')}"
                )
                print("=" * 80)

                print_grid(tiles)

            observation, reward, terminated, truncated, info = \
                env.step(1)

            if info.get("death"):
                break

            if info.get("clear"):
                break

            if terminated or truncated:
                break

    finally:
        env.close()


if __name__ == "__main__":
    main()