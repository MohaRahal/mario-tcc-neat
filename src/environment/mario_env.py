from __future__ import annotations

import gymnasium as gym
import gym_super_mario_bros

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


DEFAULT_ENV_ID = "SuperMarioBros-1-1-v0"


def create_mario_env(
    env_id: str = DEFAULT_ENV_ID,
    render_mode: str | None = None,
):
    """
    Cria o ambiente do Super Mario Bros.

    render_mode:
        None        -> treinamento headless
        "human"     -> visualização do agente
        "rgb_array" -> captura de frames
    """

    env = gym.make(
        env_id,
        render_mode=render_mode,
    )

    env = JoypadSpace(
        env,
        SIMPLE_MOVEMENT,
    )

    return env