from __future__ import annotations

import numpy as np


class SpatialSensor:
    TILE_SIZE = 16

    TILE_MEMORY_START = 0x0500

    TILE_BUFFER_ROWS = 13
    TILE_BUFFER_COLUMNS = 16

    PAGE_SIZE = (
        TILE_BUFFER_ROWS
        * TILE_BUFFER_COLUMNS
    )

    RADIUS = 6

    EMPTY = 0.0
    SOLID = 1.0

    def __init__(
        self,
        env,
        left: int = RADIUS,
        right: int = RADIUS,
        up: int = RADIUS,
        down: int = RADIUS,
    ):
        self.env = env

        self.left = left
        self.right = right
        self.up = up
        self.down = down

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

    @property
    def rows(self) -> int:
        return (
            self.up
            + self.down
            + 1
        )

    @property
    def columns(self) -> int:
        return (
            self.left
            + self.right
            + 1
        )

    @property
    def size(self) -> int:
        return (
            self.rows
            * self.columns
        )

    def get_mario_position(self) -> tuple[int, int]:
        mario_x = (
            int(self.ram[0x6D]) * 256
            + int(self.ram[0x86])
        )

        mario_y = (
            int(self.ram[0x03B8])
            + 16
        )

        return mario_x, mario_y

    def _get_tile(
        self,
        mario_x: int,
        mario_y: int,
        dx: int,
        dy: int,
    ) -> float:
        x = (
            mario_x
            + dx
            + 8
        )

        y = (
            mario_y
            + dy
            - 16
        )

        page = (
            x // 256
        ) % 2

        sub_x = (
            x % 256
        ) // self.TILE_SIZE

        sub_y = (
            y - 32
        ) // self.TILE_SIZE

        if (
            sub_y < 0
            or sub_y >= self.TILE_BUFFER_ROWS
        ):
            return self.EMPTY

        address = (
            self.TILE_MEMORY_START
            + page * self.PAGE_SIZE
            + sub_y * self.TILE_BUFFER_COLUMNS
            + sub_x
        )

        value = int(
            self.ram[address]
        )

        if value != 0:
            return self.SOLID

        return self.EMPTY

    def observe(self, info: dict) -> np.ndarray:
        mario_x, mario_y = self.get_mario_position()

        values: list[float] = []

        for row in range(
            -self.up,
            self.down + 1,
        ):
            for column in range(
                -self.left,
                self.right + 1,
            ):
                dx = (
                    column
                    * self.TILE_SIZE
                )

                dy = (
                    row
                    * self.TILE_SIZE
                )

                values.append(
                    self._get_tile(
                        mario_x=mario_x,
                        mario_y=mario_y,
                        dx=dx,
                        dy=dy,
                    )
                )

        vector = np.asarray(
            values,
            dtype=np.float32,
        )

        if vector.size != self.size:
            raise RuntimeError(
                f"Esperados {self.size} valores espaciais, "
                f"recebidos {vector.size}."
            )

        return vector

    def matrix(self, info: dict) -> np.ndarray:
        return self.observe(info).reshape(
            self.rows,
            self.columns,
        )