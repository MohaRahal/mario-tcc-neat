from __future__ import annotations

import numpy as np

from .enemy_tracker import EnemyTracker
from .spatial_sensor import SpatialSensor


EMPTY = 0.0
SOLID = 1.0
ENEMY = -1.0


class SensorSystem:
    """
    Sistema final de percepção utilizado pelo agente.

    A rede neural recebe apenas uma matriz espacial 13x13.

    Cada célula representa:

        -1 = inimigo
         0 = espaço vazio
         1 = cenário sólido

    Total:
        13 x 13 = 169 entradas
    """

    def __init__(self, env):
        self.env = env

        self.spatial_sensor = SpatialSensor(env)
        self.enemy_tracker = EnemyTracker(env)

    @property
    def size(self) -> int:
        return self.spatial_sensor.size

    def reset(self) -> None:
        """
        Mantido para preservar uma interface consistente entre
        SensorSystem e Evaluator.

        A arquitetura atual não possui estado interno no sensor.
        """
        pass

    def observe(self, info: dict) -> np.ndarray:
        matrix = self.matrix(info)

        vector = matrix.flatten().astype(
            np.float32
        )

        if vector.size != self.size:
            raise RuntimeError(
                f"Esperados {self.size} sensores, "
                f"recebidos {vector.size}."
            )

        return vector

    def matrix(self, info: dict) -> np.ndarray:
        matrix = self.spatial_sensor.matrix(
            info
        ).copy()

        mario_x, mario_y = (
            self.spatial_sensor.get_mario_position()
        )

        enemies = (
            self.enemy_tracker.get_enemies()
        )

        for enemy in enemies:
            self._insert_enemy(
                matrix=matrix,
                mario_x=mario_x,
                mario_y=mario_y,
                enemy_x=enemy.x,
                enemy_y=enemy.y,
            )

        return matrix

    def _insert_enemy(
        self,
        matrix: np.ndarray,
        mario_x: int,
        mario_y: int,
        enemy_x: int,
        enemy_y: int,
    ) -> None:
        """
        Insere um inimigo na mesma célula espacial utilizada
        pelo cenário.

        A tolerância de 8 pixels corresponde à lógica utilizada
        pelo MarI/O para associar sprites às células da grade.
        """

        tile_size = self.spatial_sensor.TILE_SIZE

        for row_index, row in enumerate(
            range(
                -self.spatial_sensor.up,
                self.spatial_sensor.down + 1,
            )
        ):
            cell_y = (
                mario_y
                + row * tile_size
            )

            for column_index, column in enumerate(
                range(
                    -self.spatial_sensor.left,
                    self.spatial_sensor.right + 1,
                )
            ):
                cell_x = (
                    mario_x
                    + column * tile_size
                )

                distance_x = abs(
                    enemy_x - cell_x
                )

                distance_y = abs(
                    enemy_y - cell_y
                )

                if (
                    distance_x <= 8
                    and distance_y <= 8
                ):
                    matrix[
                        row_index,
                        column_index,
                    ] = ENEMY

                    break