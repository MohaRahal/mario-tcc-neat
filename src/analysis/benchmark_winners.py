from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import neat
import numpy as np

from src.environment.mario_env import create_mario_env
from src.sensors.sensor_system import SensorSystem


EXPERIMENTS_DIR = Path(
    "data/experiments"
)

OUTPUT_DIR = Path(
    "data/analysis/winner_benchmark"
)

SEED = 42


def get_sensor_configuration(
    num_inputs: int,
) -> tuple[bool, bool]:

    if num_inputs == 62:
        return False, False

    if num_inputs == 63:
        return True, False

    if num_inputs == 68:
        return True, True

    raise RuntimeError(
        f"Quantidade de inputs não suportada: "
        f"{num_inputs}. "
        "Esperado: 62, 63 ou 68."
    )


def find_available_experiments() -> list[str]:
    """
    Localiza experimentos que possuem:

    - config/neat-config.ini
    - models/winner.pkl
    """

    experiments: list[str] = []

    if not EXPERIMENTS_DIR.exists():

        raise FileNotFoundError(
            f"Diretório não encontrado: "
            f"{EXPERIMENTS_DIR}"
        )

    for experiment_dir in (
        EXPERIMENTS_DIR.iterdir()
    ):

        if not experiment_dir.is_dir():
            continue

        config_path = (
            experiment_dir
            / "config"
            / "neat-config.ini"
        )

        winner_path = (
            experiment_dir
            / "models"
            / "winner.pkl"
        )

        if (
            config_path.exists()
            and winner_path.exists()
        ):

            experiments.append(
                experiment_dir.name
            )

    return sorted(
        experiments
    )


def print_termination_debug(
    experiment_name: str,
    steps: int,
    info: dict,
    terminated: bool,
    truncated: bool,
) -> None:

    print()
    print("=" * 70)

    print(
        f"DEBUG DE ENCERRAMENTO: "
        f"{experiment_name}"
    )

    print("=" * 70)

    print(
        f"Step final: "
        f"{steps}"
    )

    print(
        f"X final: "
        f"{info.get('x_pos')}"
    )

    print(
        f"Tempo restante: "
        f"{info.get('time')}"
    )

    print(
        f"Death: "
        f"{info.get('death')}"
    )

    print(
        f"Clear: "
        f"{info.get('clear')}"
    )

    print(
        f"Flag get: "
        f"{info.get('flag_get')}"
    )

    print(
        f"Terminated: "
        f"{terminated}"
    )

    print(
        f"Truncated: "
        f"{truncated}"
    )

    print("=" * 70)


def evaluate_winner(
    experiment_name: str,
) -> dict:

    experiment_dir = (
        EXPERIMENTS_DIR
        / experiment_name
    )

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

    # ---------------------------------------------------------
    # CONFIGURAÇÃO NEAT
    # ---------------------------------------------------------

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )

    num_inputs = (
        config.genome_config.num_inputs
    )

    (
        include_stagnation,
        include_approach_risk,
    ) = get_sensor_configuration(
        num_inputs
    )

    # ---------------------------------------------------------
    # CARREGA WINNER
    # ---------------------------------------------------------

    with model_path.open(
        "rb"
    ) as file:

        genome = pickle.load(
            file
        )

    network = (
        neat.nn.FeedForwardNetwork.create(
            genome,
            config,
        )
    )

    # ---------------------------------------------------------
    # AMBIENTE
    # ---------------------------------------------------------

    env = create_mario_env(
        render_mode=None
    )

    sensors = SensorSystem(
        env,
        include_stagnation=(
            include_stagnation
        ),
        include_approach_risk=(
            include_approach_risk
        ),
    )

    try:

        observation, info = env.reset(
            seed=SEED
        )

        sensors.reset()

        initial_x = int(
            info.get(
                "x_pos",
                0,
            )
        )

        max_x = initial_x

        final_x = initial_x

        steps = 0

        total_reward = 0.0

        clear = False

        death = False

        termination_reason = (
            "unknown"
        )

        final_time = info.get(
            "time"
        )

        flag_get = bool(
            info.get(
                "flag_get",
                False,
            )
        )

        # -----------------------------------------------------
        # EXECUÇÃO
        # -----------------------------------------------------

        while True:

            steps += 1

            inputs = sensors.observe(
                info
            )

            if len(inputs) != num_inputs:

                raise RuntimeError(
                    f"{experiment_name}: "
                    f"rede espera {num_inputs} "
                    f"inputs, mas foram recebidos "
                    f"{len(inputs)}."
                )

            outputs = network.activate(
                inputs
            )

            action = int(
                np.argmax(
                    outputs
                )
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            total_reward += float(
                reward
            )

            current_x = int(
                info.get(
                    "x_pos",
                    0,
                )
            )

            max_x = max(
                max_x,
                current_x,
            )

            final_x = current_x

            final_time = info.get(
                "time"
            )

            flag_get = bool(
                info.get(
                    "flag_get",
                    False,
                )
            )

            # -------------------------------------------------
            # ESTADO FINAL DO EPISÓDIO
            # -------------------------------------------------

            clear_now = bool(
                info.get(
                    "clear",
                    False,
                )
            )

            death_now = bool(
                info.get(
                    "death",
                    False,
                )
            )

            if (
                clear_now
                or death_now
                or terminated
                or truncated
            ):

                print_termination_debug(
                    experiment_name=(
                        experiment_name
                    ),
                    steps=steps,
                    info=info,
                    terminated=terminated,
                    truncated=truncated,
                )

            # -------------------------------------------------
            # CONDIÇÕES DE ENCERRAMENTO
            # -------------------------------------------------

            if clear_now:

                clear = True

                termination_reason = (
                    "clear"
                )

                break

            if death_now:

                death = True

                termination_reason = (
                    "death"
                )

                break

            if terminated:

                termination_reason = (
                    "terminated"
                )

                break

            if truncated:

                termination_reason = (
                    "truncated"
                )

                break

        distance = (
            max_x
            - initial_x
        )

        winner_fitness = (
            float(genome.fitness)
            if genome.fitness is not None
            else None
        )

        return {
            "experiment": (
                experiment_name
            ),

            "seed": (
                SEED
            ),

            "num_inputs": (
                num_inputs
            ),

            "stagnation_sensor": (
                include_stagnation
            ),

            "approach_risk_sensor": (
                include_approach_risk
            ),

            "winner_fitness": (
                winner_fitness
            ),

            "initial_x": (
                initial_x
            ),

            "max_x": (
                max_x
            ),

            "final_x": (
                final_x
            ),

            "distance": (
                distance
            ),

            "steps": (
                steps
            ),

            "total_reward": (
                total_reward
            ),

            "final_time": (
                final_time
            ),

            "flag_get": (
                flag_get
            ),

            "clear": (
                clear
            ),

            "death": (
                death
            ),

            "termination_reason": (
                termination_reason
            ),
        }

    finally:

        env.close()


def save_results(
    results: list[dict],
) -> Path:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "winner_benchmark.csv"
    )

    fieldnames = [
        "experiment",
        "seed",
        "num_inputs",
        "stagnation_sensor",
        "approach_risk_sensor",
        "winner_fitness",
        "initial_x",
        "max_x",
        "final_x",
        "distance",
        "steps",
        "total_reward",
        "final_time",
        "flag_get",
        "clear",
        "death",
        "termination_reason",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    return output_path


def print_result(
    result: dict,
) -> None:

    print()
    print("-" * 70)

    print(
        f"Experimento: "
        f"{result['experiment']}"
    )

    print(
        f"Inputs: "
        f"{result['num_inputs']}"
    )

    print(
        f"Winner fitness: "
        f"{result['winner_fitness']}"
    )

    print(
        f"Maior X: "
        f"{result['max_x']}"
    )

    print(
        f"X final: "
        f"{result['final_x']}"
    )

    print(
        f"Distância: "
        f"{result['distance']}"
    )

    print(
        f"Steps: "
        f"{result['steps']}"
    )

    print(
        f"Reward: "
        f"{result['total_reward']:.2f}"
    )

    print(
        f"Tempo restante: "
        f"{result['final_time']}"
    )

    print(
        f"Flag get: "
        f"{result['flag_get']}"
    )

    print(
        "Fase concluída: "
        f"{'Sim' if result['clear'] else 'Não'}"
    )

    print(
        "Morreu: "
        f"{'Sim' if result['death'] else 'Não'}"
    )

    print(
        "Encerramento: "
        f"{result['termination_reason']}"
    )


def print_comparison(
    results: list[dict],
) -> None:

    print()
    print("=" * 110)

    print(
        "RANKING POR DISTÂNCIA"
    )

    print("=" * 110)

    ranking = sorted(
        results,
        key=lambda item: (
            item["clear"],
            item["distance"],
        ),
        reverse=True,
    )

    print(
        f"{'Experimento':<30}"
        f"{'Distância':>12}"
        f"{'Max X':>10}"
        f"{'Steps':>10}"
        f"{'Time':>10}"
        f"{'Clear':>10}"
        f"{'Death':>10}"
    )

    print("-" * 92)

    for result in ranking:

        print(
            f"{result['experiment']:<30}"
            f"{result['distance']:>12}"
            f"{result['max_x']:>10}"
            f"{result['steps']:>10}"
            f"{str(result['final_time']):>10}"
            f"{str(result['clear']):>10}"
            f"{str(result['death']):>10}"
        )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Avalia os winners dos experimentos "
            "NEAT sob o mesmo protocolo e registra "
            "dados detalhados de encerramento."
        )
    )

    parser.add_argument(
        "--experiments",
        nargs="*",
        help=(
            "Experimentos que serão avaliados. "
            "Se omitido, todos os experimentos "
            "com winner.pkl e neat-config.ini "
            "serão utilizados."
        ),
    )

    args = parser.parse_args()

    available = (
        find_available_experiments()
    )

    if args.experiments:

        requested = set(
            args.experiments
        )

        experiments = [
            experiment
            for experiment in available
            if experiment in requested
        ]

        missing = (
            requested
            - set(experiments)
        )

        if missing:

            raise FileNotFoundError(
                "Experimentos sem winner/config "
                "ou inexistentes: "
                f"{sorted(missing)}"
            )

    else:

        experiments = available

    if not experiments:

        raise RuntimeError(
            "Nenhum experimento disponível "
            "para benchmark."
        )

    print()
    print("=" * 70)

    print(
        "BENCHMARK PADRONIZADO DOS WINNERS"
    )

    print("=" * 70)

    print(
        f"Seed utilizada: {SEED}"
    )

    print(
        f"Experimentos: {len(experiments)}"
    )

    print("=" * 70)

    results: list[dict] = []

    for index, experiment in enumerate(
        experiments,
        start=1,
    ):

        print()
        print(
            f"[{index}/{len(experiments)}] "
            f"Avaliando {experiment}..."
        )

        result = evaluate_winner(
            experiment
        )

        results.append(
            result
        )

        print_result(
            result
        )

    output_path = save_results(
        results
    )

    print_comparison(
        results
    )

    print()
    print("=" * 70)

    print(
        "BENCHMARK FINALIZADO"
    )

    print("=" * 70)

    print(
        f"CSV: {output_path}"
    )


if __name__ == "__main__":
    main()