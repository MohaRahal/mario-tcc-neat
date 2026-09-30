from __future__ import annotations

from src.environment.mario_env import create_mario_env


ENEMY_SLOTS = 5

ENEMY_TYPE_BASE = 0x0016
ENEMY_DIRECTION_BASE = 0x0046

# Posição X do inimigo
ENEMY_X_LEVEL_BASE = 0x006E
ENEMY_X_SCREEN_BASE = 0x0087

# Velocidades
ENEMY_X_SPEED_BASE = 0x0058
ENEMY_Y_SPEED_BASE = 0x00A0

# Posição Y
ENEMY_Y_BASE = 0x00CF

# Representação relativa
ENEMY_REL_X_BASE = 0x03AE
ENEMY_REL_Y_BASE = 0x03B9


def ram(env, address: int) -> int:
    return int(env.unwrapped.ram[address])


def read_enemy(env, index: int) -> dict:
    x_page = ram(env, ENEMY_X_LEVEL_BASE + index)
    x_screen = ram(env, ENEMY_X_SCREEN_BASE + index)

    x_absolute = x_page * 0x100 + x_screen

    return {
        "slot": index,
        "type": ram(env, ENEMY_TYPE_BASE + index),
        "direction": ram(env, ENEMY_DIRECTION_BASE + index),

        "x_page": x_page,
        "x_screen": x_screen,
        "x_absolute": x_absolute,

        "y": ram(env, ENEMY_Y_BASE + index),

        "vx_raw": ram(env, ENEMY_X_SPEED_BASE + index),
        "vy_raw": ram(env, ENEMY_Y_SPEED_BASE + index),

        "rel_x_raw": ram(env, ENEMY_REL_X_BASE + index),
        "rel_y_raw": ram(env, ENEMY_REL_Y_BASE + index),
    }


def signed_byte(value: int) -> int:
    """
    Converte um byte unsigned (0..255) para signed (-128..127).
    """
    return value - 256 if value >= 128 else value


def main():
    env = create_mario_env(render_mode=None)

    try:
        observation, info = env.reset(seed=42)

        for step in range(1, 1001):

            # right
            action = 1

            observation, reward, terminated, truncated, info = env.step(action)

            mario_x = int(info["x_pos"])
            mario_y = int(info["y_pos"])

            enemies = []

            for index in range(ENEMY_SLOTS):
                enemy = read_enemy(env, index)

                if enemy["type"] == 0:
                    continue

                enemy["dx"] = enemy["x_absolute"] - mario_x
                enemy["dy"] = enemy["y"] - mario_y

                enemy["rel_x"] = signed_byte(enemy["rel_x_raw"])
                enemy["rel_y"] = signed_byte(enemy["rel_y_raw"])

                enemies.append(enemy)

            if enemies:

                print(
                    f"\nSTEP {step}"
                    f" | Mario=({mario_x}, {mario_y})"
                )

                print("-" * 110)

                for enemy in enemies:

                    print(
                        f"slot={enemy['slot']} "
                        f"type={enemy['type']} "
                        f"X={enemy['x_absolute']} "
                        f"Y={enemy['y']} "
                        f"DX={enemy['dx']} "
                        f"DY={enemy['dy']} "
                        f"dir={enemy['direction']} "
                        f"vx={enemy['vx_raw']} "
                        f"vy={enemy['vy_raw']} "
                        f"relX={enemy['rel_x']} "
                        f"relY={enemy['rel_y']}"
                    )

            if info.get("death"):
                print(f"\nMario morreu no step {step}.")
                break

            if info.get("clear"):
                print(f"\nFase concluída no step {step}.")
                break

            if terminated or truncated:
                break

    finally:
        env.close()


if __name__ == "__main__":
    main()