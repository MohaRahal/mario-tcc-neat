from __future__ import annotations

from src.environment.mario_env import create_mario_env


TILE_BASE = 0x0500

PAGE_SIZE = 0xA0      # 160 bytes
ROWS = 13
COLS = 16


TILE_SYMBOLS = {
    0x00: ".",

    # Ground / blocos
    0x51: "B",
    0x52: "B",
    0x54: "#",
    0x55: "B",
    0x56: "B",
    0x57: "B",
    0x58: "B",
    0x5A: "B",
    0x5D: "B",
    0x5E: "B",
    0x5F: "B",

    # Pipes
    0x10: "P",
    0x11: "P",
    0x12: "P",
    0x13: "P",
    0x14: "P",
    0x15: "P",

    # Outros
    0x23: "?",
    0x24: "F",
    0x25: "F",
    0x26: "V",
    0x67: "S",
    0x68: "S",
    0x88: "C",
    0x89: "=",

    # Coin blocks / power-up
    0xC0: "C",
    0xC1: "?",
    0xC2: "o",
    0xC3: "o",
    0xC4: "B",
    0xC5: "X",
}


def read_page(env, page: int) -> list[list[int]]:
    """
    Lê uma das páginas do buffer de tiles.

    page=0 -> 0x0500
    page=1 -> 0x05A0
    """

    base = TILE_BASE + page * PAGE_SIZE

    ram = env.unwrapped.ram

    grid = []

    for row in range(ROWS):

        row_values = []

        for col in range(COLS):

            address = base + row * COLS + col

            row_values.append(
                int(ram[address])
            )

        grid.append(row_values)

    return grid


def print_page(grid: list[list[int]]) -> None:

    for row in grid:

        print(
            " ".join(
                TILE_SYMBOLS.get(
                    value,
                    "?"
                )
                for value in row
            )
        )


def main():

    env = create_mario_env(render_mode=None)

    try:

        observation, info = env.reset(seed=42)

        for step in range(180):

            if step % 30 == 0:

                print("\n")
                print("=" * 70)
                print(
                    f"STEP {step} | "
                    f"MARIO X={info['x_pos']} "
                    f"Y={info['y_pos']}"
                )
                print("=" * 70)

                for page in range(2):

                    print(
                        f"\nPAGE {page}"
                    )

                    print(
                        "-" * 70
                    )

                    grid = read_page(
                        env,
                        page
                    )

                    print_page(grid)

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