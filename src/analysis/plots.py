from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd


EXPERIMENTS_DIR = Path("data/experiments")


def load_metrics(experiment_name: str):

    experiment_dir = (
        EXPERIMENTS_DIR
        / experiment_name
    )

    metrics_path = (
        experiment_dir
        / "metrics"
        / "training_metrics.csv"
    )

    plots_dir = (
        experiment_dir
        / "plots"
    )

    if not experiment_dir.exists():
        raise FileNotFoundError(
            f"Experimento não encontrado: {experiment_dir}"
        )

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Arquivo de métricas não encontrado: {metrics_path}"
        )

    plots_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = pd.read_csv(
        metrics_path
    )

    return data, plots_dir


def plot_fitness(
    data: pd.DataFrame,
    plots_dir: Path,
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        data["generation"],
        data["best_fitness"],
        label="Melhor fitness",
    )

    plt.plot(
        data["generation"],
        data["average_fitness"],
        label="Fitness médio",
    )

    plt.xlabel(
        "Geração"
    )

    plt.ylabel(
        "Fitness"
    )

    plt.title(
        "Evolução do fitness por geração"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    output = (
        plots_dir
        / "fitness_evolution.png"
    )

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

    plt.close()


def plot_network_complexity(
    data: pd.DataFrame,
    plots_dir: Path,
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        data["generation"],
        data["best_nodes"],
        label="Nós",
    )

    plt.plot(
        data["generation"],
        data["best_connections"],
        label="Conexões",
    )

    plt.xlabel(
        "Geração"
    )

    plt.ylabel(
        "Quantidade"
    )

    plt.title(
        "Evolução da complexidade da rede"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    output = (
        plots_dir
        / "network_complexity.png"
    )

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

    plt.close()


def plot_species(
    data: pd.DataFrame,
    plots_dir: Path,
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        data["generation"],
        data["species_count"],
    )

    plt.xlabel(
        "Geração"
    )

    plt.ylabel(
        "Número de espécies"
    )

    plt.title(
        "Quantidade de espécies por geração"
    )

    plt.grid(
        alpha=0.3
    )

    output = (
        plots_dir
        / "species_evolution.png"
    )

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

    plt.close()


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Gera gráficos de um experimento "
            "de treinamento NEAT."
        )
    )

    parser.add_argument(
        "--experiment",
        required=True,
        help="Nome do experimento.",
    )

    args = parser.parse_args()

    data, plots_dir = load_metrics(
        args.experiment
    )

    print()
    print("=" * 60)
    print(f"GRÁFICOS DO EXPERIMENTO: {args.experiment}")
    print("=" * 60)

    plot_fitness(
        data,
        plots_dir,
    )

    plot_network_complexity(
        data,
        plots_dir,
    )

    plot_species(
        data,
        plots_dir,
    )

    print()
    print("Gráficos gerados em:")
    print(plots_dir)


if __name__ == "__main__":
    main()