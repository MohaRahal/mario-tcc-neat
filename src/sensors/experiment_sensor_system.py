from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mario_state import MarioState
from .sensor_system import SensorSystem
from .spatial_sensor import SpatialSensor


MAX_ENEMIES = 5
MAX_STAGNATION_STEPS = 180
APPROACH_HORIZON = 60.0
PREVIOUS_ACTION_COUNT = 7


@dataclass(frozen=True)
class SensorProfile:
    num_inputs: int
    label: str
    include_spatial: bool = False
    include_stagnation: bool = False
    include_approach_risk: bool = False
    include_previous_action: bool = False
    recurrent_spatial: bool = False


SENSOR_PROFILES = {
    37: SensorProfile(37, "baseline 37"),
    62: SensorProfile(62, "legado 5x5", include_spatial=True),
    63: SensorProfile(
        63,
        "legado 5x5 + estagnacao",
        include_spatial=True,
        include_stagnation=True,
    ),
    68: SensorProfile(
        68,
        "legado 5x5 + aproximacao",
        include_spatial=True,
        include_stagnation=True,
        include_approach_risk=True,
    ),
    70: SensorProfile(
        70,
        "legado 5x5 + acao anterior",
        include_spatial=True,
        include_stagnation=True,
        include_previous_action=True,
    ),
    169: SensorProfile(169, "espacial recorrente 13x13", recurrent_spatial=True),
}


def sensor_profile_for_inputs(num_inputs: int) -> SensorProfile:
    try:
        return SENSOR_PROFILES[num_inputs]
    except KeyError as exc:
        supported = ", ".join(str(value) for value in sorted(SENSOR_PROFILES))
        raise RuntimeError(
            f"Arquitetura de {num_inputs} entradas nao suportada. "
            f"Perfis disponiveis: {supported}."
        ) from exc


def _signed_byte(value: int) -> int:
    return value - 256 if value >= 128 else value


@dataclass(frozen=True)
class _LegacyEnemyState:
    active: float
    enemy_type: float
    dx: float
    velocity_x: float
    velocity_y: float
    direction: float
    approach_risk: float

    def to_vector(self) -> list[float]:
        return [
            self.active,
            self.enemy_type,
            self.dx,
            self.velocity_x,
            self.velocity_y,
            self.direction,
            self.approach_risk,
        ]


class _LegacyEnemyTracker:
    """Leitura de inimigos usada pelos experimentos de 37 a 70 entradas."""

    ENEMY_TYPE_BASE = 0x0016
    ENEMY_DIRECTION_BASE = 0x0046
    ENEMY_VX_BASE = 0x0058
    ENEMY_X_PAGE_BASE = 0x006E
    ENEMY_X_SCREEN_BASE = 0x0087
    ENEMY_VY_BASE = 0x00A0

    def __init__(self, env):
        self.env = env

    def _ram(self, address: int) -> int:
        return int(self.env.unwrapped.ram[address])

    @staticmethod
    def _approach_risk(dx: float, enemy_vx: float, mario_vx: float) -> float:
        relative_velocity = enemy_vx - mario_vx
        if dx * relative_velocity >= 0:
            return 0.0
        relative_speed = abs(relative_velocity)
        if relative_speed < 1e-6:
            return 0.0
        estimated_time = abs(dx) / relative_speed
        return float(np.clip(1.0 - estimated_time / APPROACH_HORIZON, 0.0, 1.0))

    def get_enemies(
        self, mario_x: float, mario_vx: float = 0.0
    ) -> list[_LegacyEnemyState]:
        enemies: list[_LegacyEnemyState] = []
        for slot in range(MAX_ENEMIES):
            enemy_type = self._ram(self.ENEMY_TYPE_BASE + slot)
            if enemy_type == 0:
                continue

            enemy_x = (
                self._ram(self.ENEMY_X_PAGE_BASE + slot) * 256
                + self._ram(self.ENEMY_X_SCREEN_BASE + slot)
            )
            dx = float(enemy_x - mario_x)
            velocity_x = float(_signed_byte(self._ram(self.ENEMY_VX_BASE + slot)))
            velocity_y = float(_signed_byte(self._ram(self.ENEMY_VY_BASE + slot)))
            direction_raw = self._ram(self.ENEMY_DIRECTION_BASE + slot)
            direction = 1.0 if direction_raw == 1 else -1.0 if direction_raw == 2 else 0.0
            enemies.append(
                _LegacyEnemyState(
                    active=1.0,
                    enemy_type=float(enemy_type),
                    dx=dx,
                    velocity_x=velocity_x,
                    velocity_y=velocity_y,
                    direction=direction,
                    approach_risk=self._approach_risk(dx, velocity_x, mario_vx),
                )
            )

        enemies.sort(key=lambda enemy: abs(enemy.dx))
        return enemies[:MAX_ENEMIES]


class _LegacySensorSystem:
    """Implementacao preservada das arquiteturas feed-forward historicas."""

    def __init__(self, env, profile: SensorProfile):
        self.env = env
        self.profile = profile
        self.enemy_tracker = _LegacyEnemyTracker(env)
        self.spatial_sensor = (
            SpatialSensor(env, left=1, right=3, up=2, down=2)
            if profile.include_spatial
            else None
        )
        self.max_x_seen: int | None = None
        self.steps_without_progress = 0
        self.previous_action: int | None = None
        self.last_matrix = np.empty((0, 0), dtype=np.float32)
        self.mario_cell: tuple[int, int] | None = None

    @property
    def size(self) -> int:
        return self.profile.num_inputs

    def reset(self) -> None:
        self.max_x_seen = None
        self.steps_without_progress = 0
        self.previous_action = None
        self.last_matrix = np.empty((0, 0), dtype=np.float32)

    def set_previous_action(self, action: int) -> None:
        if self.profile.include_previous_action:
            if not 0 <= action < PREVIOUS_ACTION_COUNT:
                raise ValueError(f"Acao invalida: {action}.")
            self.previous_action = action

    def observe(self, info: dict) -> np.ndarray:
        mario = MarioState.read(self.env, info)
        values = mario.to_vector()

        enemy_width = 7 if self.profile.include_approach_risk else 6
        enemies = self.enemy_tracker.get_enemies(
            mario_x=mario.x,
            mario_vx=mario.velocity_x,
        )
        for enemy in enemies:
            values.extend(enemy.to_vector()[:enemy_width])
        for _ in range(MAX_ENEMIES - len(enemies)):
            values.extend([0.0] * enemy_width)

        if self.spatial_sensor is not None:
            self.last_matrix = self.spatial_sensor.matrix(info).astype(
                np.float32, copy=True
            )
            self.mario_cell = (2, 1)
            values.extend(self.last_matrix.flatten().tolist())
        else:
            self.last_matrix = np.empty((0, 0), dtype=np.float32)
            self.mario_cell = None

        if self.profile.include_previous_action:
            previous = [0.0] * PREVIOUS_ACTION_COUNT
            if self.previous_action is not None:
                previous[self.previous_action] = 1.0
            values.extend(previous)

        if self.profile.include_stagnation:
            values.append(self._stagnation_ratio(info, int(mario.x)))

        vector = np.asarray(values, dtype=np.float32)
        if vector.size != self.size:
            raise RuntimeError(
                f"Perfil {self.profile.label}: esperados {self.size} sensores, "
                f"recebidos {vector.size}."
            )
        return self._normalize(vector, enemy_width)

    def _stagnation_ratio(self, info: dict, mario_x: int) -> float:
        current_x = int(info.get("x_pos", mario_x))
        if self.max_x_seen is None or current_x > self.max_x_seen:
            self.max_x_seen = current_x
            self.steps_without_progress = 0
        else:
            self.steps_without_progress += 1
        return float(
            np.clip(self.steps_without_progress / MAX_STAGNATION_STEPS, 0.0, 1.0)
        )

    def _normalize(self, vector: np.ndarray, enemy_width: int) -> np.ndarray:
        result = vector.copy()
        result[0] = np.clip(result[0] / 4096.0, 0.0, 1.0)
        result[1] = np.clip(result[1] / 255.0, 0.0, 1.0)
        result[2] = np.clip(result[2] / 127.0, -1.0, 1.0)
        result[3] = np.clip(result[3] / 127.0, -1.0, 1.0)
        result[5] = np.clip(result[5] / 15.0, 0.0, 1.0)
        result[6] = np.clip(result[6] / 2.0, 0.0, 1.0)

        for enemy_index in range(MAX_ENEMIES):
            offset = 7 + enemy_index * enemy_width
            result[offset + 1] = np.clip(result[offset + 1] / 255.0, 0.0, 1.0)
            result[offset + 2] = np.clip(result[offset + 2] / 256.0, -1.0, 1.0)
            result[offset + 3] = np.clip(result[offset + 3] / 127.0, -1.0, 1.0)
            result[offset + 4] = np.clip(result[offset + 4] / 127.0, -1.0, 1.0)
            if self.profile.include_approach_risk:
                result[offset + 6] = np.clip(result[offset + 6], 0.0, 1.0)

        return result


class ExperimentSensorSystem:
    """Seleciona o pipeline sensorial que corresponde ao modelo salvo."""

    def __init__(self, env, num_inputs: int):
        self.profile = sensor_profile_for_inputs(num_inputs)
        self._current = SensorSystem(env) if self.profile.recurrent_spatial else None
        self._legacy = (
            None if self.profile.recurrent_spatial else _LegacySensorSystem(env, self.profile)
        )
        self.last_matrix = np.empty((0, 0), dtype=np.float32)
        self.mario_cell: tuple[int, int] | None = None

    @property
    def size(self) -> int:
        return self.profile.num_inputs

    @property
    def label(self) -> str:
        return self.profile.label

    def reset(self) -> None:
        if self._current is not None:
            self._current.reset()
        if self._legacy is not None:
            self._legacy.reset()
        self.last_matrix = np.empty((0, 0), dtype=np.float32)
        self.mario_cell = None

    def observe(self, info: dict) -> np.ndarray:
        if self._current is not None:
            self.last_matrix = self._current.matrix(info).astype(np.float32, copy=True)
            self.mario_cell = (6, 6)
            return self.last_matrix.flatten()

        if self._legacy is None:  # pragma: no cover - protegido pelo perfil
            raise RuntimeError("Pipeline sensorial nao inicializado.")
        vector = self._legacy.observe(info)
        self.last_matrix = self._legacy.last_matrix
        self.mario_cell = self._legacy.mario_cell
        return vector

    def set_previous_action(self, action: int) -> None:
        if self._legacy is not None:
            self._legacy.set_previous_action(action)
