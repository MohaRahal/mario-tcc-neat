from __future__ import annotations

from src.environment.mario_env import create_mario_env


TILE_MEMORY_START = 0x0500

ROWS = 13
COLUMNS = 16
PAGE_SIZE = ROWS * COLUMNS

TILE_SIZE = 16


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


def get_tile(
    ram,
    page: int,
    row: int,
    column: int,
):

    address = (
        TILE_MEMORY_START
        + page * PAGE_SIZE
        + row * COLUMNS
        + column
    )

    return int(ram[address])


def print_page(
    ram,
    page: int,
    mario_column: int | None = None,
    mario_row: int | None = None,
):

    print()
    print("=" * 60)
    print(f"PÁGINA {page}")
    print("=" * 60)

    for row in range(ROWS):

        line = []

        for column in range(COLUMNS):

            if (
                mario_column == column
                and mario_row == row
            ):
                symbol = " M "

            else:

                value = get_tile(
                    ram,
                    page,
                    row,
                    column,
                )

                symbol = (
                    " # "
                    if value != 0
                    else " . "
                )

            line.append(symbol)

        print(
            "".join(line)
        )


def main():

    env = create_mario_env(
        render_mode=None
    )

    try:

        observation, info = env.reset(
            seed=42
        )

        base_env = get_base_env(env)

        print("=" * 60)
        print("TESTE DE POSIÇÃO + TILE BUFFER")
        print("=" * 60)

        for step in range(300):

            # ação 1 = right
            observation, reward, terminated, truncated, info = (
                env.step(1)
            )

            if step % 25 == 0:

                x_pos = int(
                    info.get("x_pos", 0)
                )

                y_pos = int(
                    info.get("y_pos", 0)
                )

                # Página absoluta em que Mario se encontra.
                world_page = (
                    x_pos // 256
                )

                # O buffer possui duas páginas.
                buffer_page = (
                    world_page % 2
                )

                mario_column = (
                    x_pos % 256
                ) // TILE_SIZE

                print()
                print("=" * 60)
                print(f"STEP: {step}")
                print("=" * 60)

                print(
                    f"x_pos: {x_pos}"
                )

                print(
                    f"y_pos: {y_pos}"
                )

                print(
                    f"world_page: {world_page}"
                )

                print(
                    f"buffer_page: {buffer_page}"
                )

                print(
                    f"mario_column: {mario_column}"
                )

                # Ainda NÃO vamos assumir uma fórmula para Y.
                # Mostramos a página sem marcar a linha do Mario.
                print_page(
                    base_env.ram,
                    buffer_page,
                )

            if info.get("death"):

                print()
                print(
                    f"Mario morreu em "
                    f"x={info.get('x_pos')}."
                )

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