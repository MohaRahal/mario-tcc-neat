from __future__ import annotations

from dataclasses import dataclass


MAX_ENEMIES = 5

ENEMY_ACTIVE_BASE = 0x000F
ENEMY_TYPE_BASE = 0x0016
ENEMY_X_PAGE_BASE = 0x006E
ENEMY_X_SCREEN_BASE = 0x0087
ENEMY_Y_SCREEN_BASE = 0x00CF


@dataclass(frozen=True)
class EnemyState:
    slot: int
    enemy_type: int
    x: int
    y: int


class EnemyTracker:
    """
    Responsável exclusivamente por localizar os inimigos ativos
    do Super Mario Bros diretamente na RAM.

    Nenhuma informação produzida aqui é enviada diretamente
    para a rede neural.

    As posições são utilizadas pelo SensorSystem para inserir
    os inimigos na matriz espacial 13x13.
    """

    def __init__(self, env, max_enemies: int = MAX_ENEMIES):
        self.env = env
        self.max_enemies = max_enemies
        self.base_env = self._find_base_env()

    def _find_base_env(self):
        current = self.env

        while True:
            if hasattr(current, "ram"):
                return current

            if not hasattr(current, "env"):
                raise RuntimeError(
                    "Não foi possível encontrar acesso à RAM do ambiente."
                )

            current = current.env

    @property
    def ram(self):
        return self.base_env.ram

    def _read(self, address: int) -> int:
        return int(self.ram[address])

    def get_enemies(self) -> list[EnemyState]:
        enemies: list[EnemyState] = []

        for slot in range(self.max_enemies):
            active = self._read(
                ENEMY_ACTIVE_BASE + slot
            )

            if active == 0:
                continue

            enemy_type = self._read(
                ENEMY_TYPE_BASE + slot
            )

            x_page = self._read(
                ENEMY_X_PAGE_BASE + slot
            )

            x_screen = self._read(
                ENEMY_X_SCREEN_BASE + slot
            )

            enemy_x = (
                x_page * 256
                + x_screen
            )

            enemy_y = (
                self._read(
                    ENEMY_Y_SCREEN_BASE + slot
                )
                + 24
            )

            enemies.append(
                EnemyState(
                    slot=slot,
                    enemy_type=enemy_type,
                    x=enemy_x,
                    y=enemy_y,
                )
            )

        return enemies