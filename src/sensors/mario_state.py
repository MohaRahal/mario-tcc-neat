from __future__ import annotations

from dataclasses import dataclass


def signed_byte(value: int) -> int:
    """
    Converte um byte unsigned (0..255)
    para signed (-128..127).
    """

    return value - 256 if value >= 128 else value


@dataclass
class MarioState:
    x: float
    y: float

    velocity_x: float
    velocity_y: float

    direction: float

    player_state: float
    powerup_level: float

    @classmethod
    def read(
        cls,
        env,
        info: dict,
    ) -> "MarioState":

        ram = env.unwrapped.ram

        # Posição fornecida pelo próprio ambiente.
        x = float(info["x_pos"])
        y = float(info["y_pos"])

        # Velocidades armazenadas pelo próprio SMB1.
        velocity_x = float(
            signed_byte(
                int(ram[0x0057])
            )
        )

        velocity_y = float(
            signed_byte(
                int(ram[0x009F])
            )
        )

        # Direção de Mario.
        direction_raw = int(
            ram[0x0045]
        )

        if direction_raw == 1:
            direction = 1.0

        elif direction_raw == 2:
            direction = -1.0

        else:
            direction = 0.0

        return cls(
            x=x,
            y=y,

            velocity_x=velocity_x,
            velocity_y=velocity_y,

            direction=direction,

            player_state=float(
                info["player_state"]
            ),

            powerup_level=float(
                info["powerup_level"]
            ),
        )

    def to_vector(self) -> list[float]:
        return [
            self.x,
            self.y,

            self.velocity_x,
            self.velocity_y,

            self.direction,

            self.player_state,
            self.powerup_level,
        ]