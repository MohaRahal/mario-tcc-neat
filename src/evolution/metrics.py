from __future__ import annotations

import csv
import time
from pathlib import Path
from statistics import mean, pstdev

import neat


class MetricsReporter(neat.reporting.BaseReporter):
    """
    Registra métricas de cada geração em CSV.

    Métricas:
    - geração
    - fitness médio
    - desvio padrão do fitness
    - melhor fitness
    - quantidade de espécies
    - nós do melhor genoma
    - conexões do melhor genoma
    - tempo da geração
    """

    def __init__(
        self,
        output_path: str | Path,
    ):
        self.output_path = Path(output_path)

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.generation = 0
        self.generation_start = None

        self._create_file()

    def _create_file(self):

        if self.output_path.exists():
            return

        with self.output_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                [
                    "generation",
                    "best_fitness",
                    "average_fitness",
                    "fitness_stdev",
                    "species_count",
                    "best_nodes",
                    "best_connections",
                    "generation_time_seconds",
                ]
            )

    def start_generation(
        self,
        generation,
    ):
        self.generation = generation
        self.generation_start = time.perf_counter()

    def post_evaluate(
        self,
        config,
        population,
        species,
        best_genome,
    ):

        fitness_values = [
            genome.fitness
            for genome in population.values()
            if genome.fitness is not None
        ]

        average_fitness = mean(
            fitness_values
        )

        if len(fitness_values) > 1:
            fitness_stdev = pstdev(
                fitness_values
            )
        else:
            fitness_stdev = 0.0

        generation_time = (
            time.perf_counter()
            - self.generation_start
        )

        nodes = len(
            best_genome.nodes
        )

        connections = len(
            best_genome.connections
        )

        species_count = len(
            species.species
        )

        with self.output_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                [
                    self.generation,
                    best_genome.fitness,
                    average_fitness,
                    fitness_stdev,
                    species_count,
                    nodes,
                    connections,
                    generation_time,
                ]
            )