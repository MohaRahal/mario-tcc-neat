from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd


EXPERIMENTS_DIR = Path("data/experiments")

REQUIRED_COLUMNS = [
    "generation",
    "best_fitness",
    "average_fitness",
    "fitness_stdev",
    "species_count",
    "best_nodes",
    "best_connections",
    "generation_time_seconds",
]


def find_experiments() -> list[Path]:
    """
    Localiza automaticamente todos os experimentos que possuem
    metrics/training_metrics.csv.
    """

    experiments = []

    if not EXPERIMENTS_DIR.exists():
        raise FileNotFoundError(
            f"Diretório de experimentos não encontrado: "
            f"{EXPERIMENTS_DIR}"
        )

    for experiment_dir in EXPERIMENTS_DIR.iterdir():

        if not experiment_dir.is_dir():
            continue

        metrics_path = (
            experiment_dir
            / "metrics"
            / "training_metrics.csv"
        )

        if metrics_path.exists():
            experiments.append(
                experiment_dir
            )

    experiments.sort(
        key=lambda path: path.name
    )

    return experiments


def load_metrics(
    experiment_dir: Path,
) -> pd.DataFrame:

    metrics_path = (
        experiment_dir
        / "metrics"
        / "training_metrics.csv"
    )

    data = pd.read_csv(
        metrics_path
    )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise RuntimeError(
            f"{experiment_dir.name}: "
            f"colunas ausentes: "
            f"{missing_columns}"
        )

    if data.empty:
        raise RuntimeError(
            f"{experiment_dir.name}: "
            "arquivo de métricas vazio."
        )

    return data


def summarize_experiment(
    experiment_name: str,
    data: pd.DataFrame,
) -> dict:

    # ---------------------------------------------------------
    # INFORMAÇÕES GERAIS
    # ---------------------------------------------------------

    first_generation = int(
        data["generation"].iloc[0]
    )

    last_generation = int(
        data["generation"].iloc[-1]
    )

    generations_recorded = len(
        data
    )

    # ---------------------------------------------------------
    # FITNESS
    # ---------------------------------------------------------

    best_fitness_ever = float(
        data["best_fitness"].max()
    )

    best_fitness_final = float(
        data["best_fitness"].iloc[-1]
    )

    average_fitness_final = float(
        data["average_fitness"].iloc[-1]
    )

    best_average_fitness = float(
        data["average_fitness"].max()
    )

    fitness_stdev_final = float(
        data["fitness_stdev"].iloc[-1]
    )

    # ---------------------------------------------------------
    # ESPÉCIES
    # ---------------------------------------------------------

    species_final = int(
        data["species_count"].iloc[-1]
    )

    species_max = int(
        data["species_count"].max()
    )

    # ---------------------------------------------------------
    # COMPLEXIDADE
    # ---------------------------------------------------------

    best_nodes_final = int(
        data["best_nodes"].iloc[-1]
    )

    best_connections_final = int(
        data["best_connections"].iloc[-1]
    )

    max_nodes = int(
        data["best_nodes"].max()
    )

    max_connections = int(
        data["best_connections"].max()
    )

    # ---------------------------------------------------------
    # TEMPO
    # ---------------------------------------------------------

    total_training_time = float(
        data[
            "generation_time_seconds"
        ].sum()
    )

    average_generation_time = float(
        data[
            "generation_time_seconds"
        ].mean()
    )

    # ---------------------------------------------------------
    # ÚLTIMAS GERAÇÕES
    # ---------------------------------------------------------

    window_size = min(
        10,
        len(data),
    )

    last_generations = data.tail(
        window_size
    )

    recent_best_fitness_mean = float(
        last_generations[
            "best_fitness"
        ].mean()
    )

    recent_average_fitness_mean = float(
        last_generations[
            "average_fitness"
        ].mean()
    )

    return {
        "experiment": experiment_name,

        "first_generation": (
            first_generation
        ),

        "last_generation": (
            last_generation
        ),

        "generations_recorded": (
            generations_recorded
        ),

        "best_fitness_ever": (
            best_fitness_ever
        ),

        "best_fitness_final": (
            best_fitness_final
        ),

        "average_fitness_final": (
            average_fitness_final
        ),

        "best_average_fitness": (
            best_average_fitness
        ),

        "fitness_stdev_final": (
            fitness_stdev_final
        ),

        "recent_best_fitness_mean": (
            recent_best_fitness_mean
        ),

        "recent_average_fitness_mean": (
            recent_average_fitness_mean
        ),

        "species_final": (
            species_final
        ),

        "species_max": (
            species_max
        ),

        "best_nodes_final": (
            best_nodes_final
        ),

        "best_connections_final": (
            best_connections_final
        ),

        "max_nodes": (
            max_nodes
        ),

        "max_connections": (
            max_connections
        ),

        "total_training_time_seconds": (
            total_training_time
        ),

        "average_generation_time_seconds": (
            average_generation_time
        ),
    }


def create_comparison_csv(
    experiments: dict[
        str,
        pd.DataFrame,
    ],
    output_dir: Path,
) -> pd.DataFrame:

    rows = []

    for name, data in experiments.items():

        rows.append(
            summarize_experiment(
                name,
                data,
            )
        )

    comparison = pd.DataFrame(
        rows
    )

    output_path = (
        output_dir
        / "experiment_comparison.csv"
    )

    comparison.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    return comparison


def plot_best_fitness(
    experiments: dict[
        str,
        pd.DataFrame,
    ],
    output_dir: Path,
):

    plt.figure(
        figsize=(12, 7)
    )

    for name, data in experiments.items():

        plt.plot(
            data["generation"],
            data["best_fitness"],
            label=name,
        )

    plt.xlabel(
        "Geração"
    )

    plt.ylabel(
        "Melhor fitness"
    )

    plt.title(
        "Comparação do melhor fitness "
        "por geração"
    )

    plt.legend(
        fontsize=8
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    output = (
        output_dir
        / "comparison_best_fitness.png"
    )

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def plot_average_fitness(
    experiments: dict[
        str,
        pd.DataFrame,
    ],
    output_dir: Path,
):

    plt.figure(
        figsize=(12, 7)
    )

    for name, data in experiments.items():

        plt.plot(
            data["generation"],
            data["average_fitness"],
            label=name,
        )

    plt.xlabel(
        "Geração"
    )

    plt.ylabel(
        "Fitness médio"
    )

    plt.title(
        "Comparação do fitness médio "
        "por geração"
    )

    plt.legend(
        fontsize=8
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    output = (
        output_dir
        / "comparison_average_fitness.png"
    )

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def plot_species(
    experiments: dict[
        str,
        pd.DataFrame,
    ],
    output_dir: Path,
):

    plt.figure(
        figsize=(12, 7)
    )

    for name, data in experiments.items():

        plt.plot(
            data["generation"],
            data["species_count"],
            label=name,
        )

    plt.xlabel(
        "Geração"
    )

    plt.ylabel(
        "Quantidade de espécies"
    )

    plt.title(
        "Comparação da diversidade "
        "de espécies"
    )

    plt.legend(
        fontsize=8
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    output = (
        output_dir
        / "comparison_species.png"
    )

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def plot_complexity(
    experiments: dict[
        str,
        pd.DataFrame,
    ],
    output_dir: Path,
):

    plt.figure(
        figsize=(12, 7)
    )

    for name, data in experiments.items():

        plt.plot(
            data["generation"],
            data["best_connections"],
            label=name,
        )

    plt.xlabel(
        "Geração"
    )

    plt.ylabel(
        "Conexões do melhor genoma"
    )

    plt.title(
        "Comparação da complexidade "
        "das redes"
    )

    plt.legend(
        fontsize=8
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    output = (
        output_dir
        / "comparison_connections.png"
    )

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def print_summary(
    comparison: pd.DataFrame,
):

    print()
    print("=" * 100)
    print("COMPARAÇÃO DOS EXPERIMENTOS")
    print("=" * 100)

    columns = [
        "experiment",
        "generations_recorded",
        "best_fitness_ever",
        "average_fitness_final",
        "fitness_stdev_final",
        "species_final",
        "best_nodes_final",
        "best_connections_final",
    ]

    print(
        comparison[
            columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "IMPORTANTE:"
    )
    print(
        "Fitness bruto não deve ser usado "
        "isoladamente para comparar "
        "experimentos que utilizaram "
        "funções de fitness diferentes."
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compara as métricas de "
            "treinamento dos experimentos."
        )
    )

    parser.add_argument(
        "--experiments",
        nargs="*",
        help=(
            "Experimentos específicos. "
            "Se omitido, todos os experimentos "
            "com training_metrics.csv serão "
            "analisados."
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "data/analysis/"
            "experiment_comparison"
        ),
        help=(
            "Diretório onde os resultados "
            "da comparação serão salvos."
        ),
    )

    args = parser.parse_args()

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # LOCALIZA EXPERIMENTOS
    # ---------------------------------------------------------

    available_experiments = (
        find_experiments()
    )

    if args.experiments:

        requested = set(
            args.experiments
        )

        experiment_dirs = [
            experiment
            for experiment
            in available_experiments
            if experiment.name
            in requested
        ]

        found = {
            experiment.name
            for experiment
            in experiment_dirs
        }

        missing = (
            requested
            - found
        )

        if missing:
            raise FileNotFoundError(
                "Experimentos não encontrados "
                "ou sem métricas: "
                f"{sorted(missing)}"
            )

    else:

        experiment_dirs = (
            available_experiments
        )

    if not experiment_dirs:
        raise RuntimeError(
            "Nenhum experimento com "
            "training_metrics.csv encontrado."
        )

    # ---------------------------------------------------------
    # CARREGA MÉTRICAS
    # ---------------------------------------------------------

    experiments = {}

    print()
    print("=" * 70)
    print("EXPERIMENTOS ENCONTRADOS")
    print("=" * 70)

    for experiment_dir in experiment_dirs:

        print(
            experiment_dir.name
        )

        experiments[
            experiment_dir.name
        ] = load_metrics(
            experiment_dir
        )

    # ---------------------------------------------------------
    # COMPARAÇÃO
    # ---------------------------------------------------------

    comparison = (
        create_comparison_csv(
            experiments,
            output_dir,
        )
    )

    plot_best_fitness(
        experiments,
        output_dir,
    )

    plot_average_fitness(
        experiments,
        output_dir,
    )

    plot_species(
        experiments,
        output_dir,
    )

    plot_complexity(
        experiments,
        output_dir,
    )

    print_summary(
        comparison
    )

    print()
    print("=" * 70)
    print("RESULTADOS GERADOS")
    print("=" * 70)

    print(
        output_dir.resolve()
    )

    print()
    print(
        "- experiment_comparison.csv"
    )

    print(
        "- comparison_best_fitness.png"
    )

    print(
        "- comparison_average_fitness.png"
    )

    print(
        "- comparison_species.png"
    )

    print(
        "- comparison_connections.png"
    )


if __name__ == "__main__":
    main()