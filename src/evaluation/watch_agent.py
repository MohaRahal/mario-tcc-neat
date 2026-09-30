from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import neat
import numpy as np

from src.environment.mario_env import create_mario_env
from src.sensors.sensor_system import SensorSystem


EXPERIMENTS_DIR = Path("data/experiments")

SEED = 42

EXPECTED_INPUTS = 169
EXPECTED_OUTPUTS = 7

ACTION_REPEAT = 5

MAX_FRAMES = 10_000
MAX_STAGNATION_FRAMES = 180


def watch(experiment_name: str) -> None:
    experiment_dir = EXPERIMENTS_DIR / experiment_name

    config_path = (
        experiment_dir
        / "config"
        / "neat-config.ini"
    )

    model_path = (
        experiment_dir
        / "models"
        / "winner.pkl"
    )

    _validate_paths(
        experiment_dir=experiment_dir,
        config_path=config_path,
        model_path=model_path,
    )

    print()
    print("=" * 70)
    print(f"AVALIAÇÃO DO EXPERIMENTO: {experiment_name}")
    print("=" * 70)
    print(f"Modelo: {model_path}")
    print(f"Configuração: {config_path}")
    print()

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )

    _validate_neat_config(config)

    with model_path.open("rb") as file:
        genome = pickle.load(file)

    network = neat.nn.RecurrentNetwork.create(
        genome,
        config,
    )

    env = create_mario_env(
        render_mode="rgb_array"
    )

    sensors = SensorSystem(env)

    try:
        observation, info = env.reset(
            seed=SEED
        )

        sensors.reset()

        if sensors.size != EXPECTED_INPUTS:
            raise RuntimeError(
                f"SensorSystem possui {sensors.size} inputs. "
                f"Esperado: {EXPECTED_INPUTS}."
            )

        initial_x = int(
            info.get("x_pos", 0)
        )

        max_x = initial_x

        total_reward = 0.0

        executed_frames = 0
        decisions = 0

        stagnation_frames = 0

        clear = False
        death = False
        stuck = False

        current_action: int | None = None

        episode_finished = False
        user_interrupted = False

        while (
            executed_frames < MAX_FRAMES
            and not episode_finished
        ):
            # -------------------------------------------------
            # OBSERVAÇÃO
            # -------------------------------------------------

            inputs = sensors.observe(info)

            if len(inputs) != EXPECTED_INPUTS:
                raise RuntimeError(
                    f"A rede espera {EXPECTED_INPUTS} inputs, "
                    f"mas o SensorSystem retornou {len(inputs)}."
                )

            # -------------------------------------------------
            # REDE RECORRENTE
            # -------------------------------------------------

            outputs = network.activate(inputs)

            if len(outputs) != EXPECTED_OUTPUTS:
                raise RuntimeError(
                    f"A rede retornou {len(outputs)} outputs. "
                    f"Esperado: {EXPECTED_OUTPUTS}."
                )

            current_action = int(
                np.argmax(outputs)
            )

            decisions += 1

            # -------------------------------------------------
            # ACTION REPEAT
            # -------------------------------------------------

            for repeat_index in range(ACTION_REPEAT):
                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(current_action)

                executed_frames += 1
                total_reward += float(reward)

                current_x = int(
                    info.get(
                        "x_pos",
                        max_x,
                    )
                )

                # ---------------------------------------------
                # PROGRESSO / ESTAGNAÇÃO
                # ---------------------------------------------

                if current_x > max_x:
                    max_x = current_x
                    stagnation_frames = 0
                else:
                    stagnation_frames += 1

                # ---------------------------------------------
                # ESTADO DO EPISÓDIO
                # ---------------------------------------------

                clear = _is_level_clear(info)
                death = _is_dead(info)

                stuck = (
                    stagnation_frames
                    >= MAX_STAGNATION_FRAMES
                )

                # ---------------------------------------------
                # RENDER
                # ---------------------------------------------

                frame = _prepare_frame(
                    observation
                )

                _draw_hud(
                    frame=frame,
                    experiment_name=experiment_name,
                    current_x=current_x,
                    max_x=max_x,
                    initial_x=initial_x,
                    action=current_action,
                    repeat_index=repeat_index,
                    executed_frames=executed_frames,
                    decisions=decisions,
                    stagnation_frames=stagnation_frames,
                    total_reward=total_reward,
                )

                cv2.imshow(
                    f"Mario NEAT - {experiment_name}",
                    frame,
                )

                key = cv2.waitKey(16) & 0xFF

                if key == 27:
                    print()
                    print(
                        "Execução interrompida pelo usuário."
                    )

                    user_interrupted = True
                    episode_finished = True
                    break

                # ---------------------------------------------
                # TERMINAÇÃO
                # ---------------------------------------------

                if clear:
                    print()
                    print("=" * 70)
                    print("FASE CONCLUÍDA!")
                    print("=" * 70)

                    episode_finished = True
                    break

                if death:
                    print()
                    print("Mario morreu.")

                    episode_finished = True
                    break

                if stuck:
                    print()
                    print(
                        "Execução encerrada por estagnação."
                    )

                    episode_finished = True
                    break

                if terminated or truncated:
                    episode_finished = True
                    break

                if executed_frames >= MAX_FRAMES:
                    episode_finished = True
                    break

        _print_result(
            experiment_name=experiment_name,
            genome=genome,
            initial_x=initial_x,
            max_x=max_x,
            executed_frames=executed_frames,
            decisions=decisions,
            total_reward=total_reward,
            clear=clear,
            death=death,
            stuck=stuck,
            user_interrupted=user_interrupted,
        )

    finally:
        env.close()
        cv2.destroyAllWindows()


def _validate_paths(
    experiment_dir: Path,
    config_path: Path,
    model_path: Path,
) -> None:
    if not experiment_dir.exists():
        raise FileNotFoundError(
            f"Experimento não encontrado: {experiment_dir}"
        )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuração não encontrada: {config_path}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {model_path}"
        )


def _validate_neat_config(config) -> None:
    num_inputs = config.genome_config.num_inputs
    num_outputs = config.genome_config.num_outputs
    feed_forward = config.genome_config.feed_forward

    if num_inputs != EXPECTED_INPUTS:
        raise RuntimeError(
            f"Configuração possui {num_inputs} inputs. "
            f"Esperado: {EXPECTED_INPUTS}."
        )

    if num_outputs != EXPECTED_OUTPUTS:
        raise RuntimeError(
            f"Configuração possui {num_outputs} outputs. "
            f"Esperado: {EXPECTED_OUTPUTS}."
        )

    if feed_forward:
        raise RuntimeError(
            "Este experimento exige rede recorrente. "
            "O neat-config.ini deve possuir "
            "feed_forward = False."
        )


def _is_level_clear(info: dict) -> bool:
    return bool(
        info.get("flag_get", False)
        or info.get("clear", False)
    )


def _is_dead(info: dict) -> bool:
    return bool(
        info.get("death", False)
        or info.get("is_dead", False)
        or info.get("is_dying", False)
    )


def _prepare_frame(
    observation,
) -> np.ndarray:
    frame = cv2.cvtColor(
        observation,
        cv2.COLOR_RGB2BGR,
    )

    frame = cv2.resize(
        frame,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_NEAREST,
    )

    return frame


def _draw_hud(
    frame: np.ndarray,
    experiment_name: str,
    current_x: int,
    max_x: int,
    initial_x: int,
    action: int,
    repeat_index: int,
    executed_frames: int,
    decisions: int,
    stagnation_frames: int,
    total_reward: float,
) -> None:
    lines = [
        f"EXP: {experiment_name}",
        f"X: {current_x} | Max X: {max_x}",
        f"Distancia: {max_x - initial_x}",
        f"Inputs: {EXPECTED_INPUTS}",
        f"Rede: Recurrent",
        f"Acao: {action}",
        (
            f"Action repeat: "
            f"{repeat_index + 1}/{ACTION_REPEAT}"
        ),
        f"Frames: {executed_frames}",
        f"Decisoes: {decisions}",
        (
            f"Stagnation: "
            f"{stagnation_frames}/"
            f"{MAX_STAGNATION_FRAMES}"
        ),
        f"Reward: {total_reward:.2f}",
    ]

    y = 25

    for line in lines:
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        y += 20


def _print_result(
    experiment_name: str,
    genome,
    initial_x: int,
    max_x: int,
    executed_frames: int,
    decisions: int,
    total_reward: float,
    clear: bool,
    death: bool,
    stuck: bool,
    user_interrupted: bool,
) -> None:
    print()
    print("=" * 70)
    print("RESULTADO DA AVALIAÇÃO")
    print("=" * 70)

    print(f"Experimento: {experiment_name}")
    print(f"Fitness do winner: {genome.fitness}")

    print(f"Inputs utilizados: {EXPECTED_INPUTS}")
    print("Tipo de rede: Recurrent")
    print(f"Action repeat: {ACTION_REPEAT}")

    print(f"Posição inicial X: {initial_x}")
    print(f"Maior posição X: {max_x}")

    print(
        f"Distância percorrida: "
        f"{max_x - initial_x}"
    )

    print(
        f"Frames executados: "
        f"{executed_frames}"
    )

    print(
        f"Decisões da rede: "
        f"{decisions}"
    )

    print(
        f"Reward acumulado: "
        f"{total_reward:.2f}"
    )

    print(
        f"Fase concluída: "
        f"{'Sim' if clear else 'Não'}"
    )

    print(
        f"Morreu: "
        f"{'Sim' if death else 'Não'}"
    )

    print(
        f"Estagnou: "
        f"{'Sim' if stuck else 'Não'}"
    )

    print(
        f"Interrompido pelo usuário: "
        f"{'Sim' if user_interrupted else 'Não'}"
    )

    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Executa visualmente o melhor agente "
            "do experimento NEAT."
        )
    )

    parser.add_argument(
        "--experiment",
        required=True,
        help="Nome do experimento.",
    )

    args = parser.parse_args()

    watch(
        args.experiment
    )


if __name__ == "__main__":
    main()