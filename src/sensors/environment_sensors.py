from __future__ import annotations

from dataclasses import dataclass


TILE_RAM_START = 0x0500
TILE_RAM_END = 0x069F


EMPTY_TILE = 0x00


GROUND_TILES = {
    0x54,
    0x61,
    0x69,
}


PIPE_TILES = {
    0x10,
    0x11,
    0x12,
    0x13,
    0x14,
    0x15,
    0x1C,
    0x1D,
    0x1E,
    0x1F,
    0x20,
    0x21,
}


BRICK_TILES = {
    0x51,
    0x52,
    0x55,
    0x56,
    0x57,
    0x58,
    0x5A,
    0x5D,
    0x5E,
    0x5F,
}


BLOCK_TILES = {
    0xC0,
    0xC1,
    0xC4,
}


@dataclass
class EnvironmentState:
    ground_count: float
    pipe_count: float
    brick_count: float
    block_count: float
    empty_count: float

    pipe_present: float
    bricks_present: float
    blocks_present: float

    def to_vector(
        self,
    ) -> list[float]:

        return [
            self.ground_count,
            self.pipe_count,
            self.brick_count,
            self.block_count,
            self.empty_count,

            self.pipe_present,
            self.bricks_present,
            self.blocks_present,
        ]


class EnvironmentSensors:

    def __init__(
        self,
        env,
    ):
        self.env = env

    def _read_tiles(
        self,
    ) -> list[int]:

        ram = self.env.unwrapped.ram

        return [
            int(
                ram[address]
            )
            for address in range(
                TILE_RAM_START,
                TILE_RAM_END + 1,
            )
        ]

    def observe(
        self,
    ) -> EnvironmentState:

        tiles = self._read_tiles()

        ground_count = 0
        pipe_count = 0
        brick_count = 0
        block_count = 0
        empty_count = 0

        for tile in tiles:

            if tile == EMPTY_TILE:

                empty_count += 1

            elif tile in GROUND_TILES:

                ground_count += 1

            elif tile in PIPE_TILES:

                pipe_count += 1

            elif tile in BRICK_TILES:

                brick_count += 1

            elif tile in BLOCK_TILES:

                block_count += 1

        return EnvironmentState(
            ground_count=float(
                ground_count
            ),

            pipe_count=float(
                pipe_count
            ),

            brick_count=float(
                brick_count
            ),

            block_count=float(
                block_count
            ),

            empty_count=float(
                empty_count
            ),

            pipe_present=float(
                pipe_count > 0
            ),

            bricks_present=float(
                brick_count > 0
            ),

            blocks_present=float(
                block_count > 0
            ),
        )