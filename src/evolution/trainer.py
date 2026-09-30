from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import os
import pickle
import shutil
import time

import neat

from src.evolution.evaluator import evaluate_genome
from src.evolution.metrics import MetricsReporter


CONFIG_PATH = Path("configs/neat-config.ini")
EXPERIMENTS_DIR = Path("data/experiments")

RESERVED_LOGICAL_CPUS = 2

EXPECTED_INPUTS = 169
EXPECTED_OUTPUTS = 7
EXPECTED_POPULATION = 300


def validate_config(config) -> None:
    num_inputs = config.genome_config.num_inputs
    num_outputs = config.genome_config.num_outputs
    feed_forward = config.genome_config.feed_forward
    population_size = config.pop_size

    errors: list[str] = []

    if num_inputs != EXPECTED_INPUTS:
        errors.append(
            f"num_inputs = {num_inputs}; esperado {EXPECTED_INPUTS}"
        )

    if num_outputs != EXPECTED_OUTPUTS:
        errors.append(
            f"num_outputs = {num_outputs}; esperado {EXPECTED_OUTPUTS}"
        )

    if feed_forward:
        errors.append(
            "feed_forward = True; esperado False"
        )

    if population_size != EXPECTED_POPULATION:
        errors.append(
            f"pop_size = {population_size}; esperado {EXPECTED_POPULATION}"
        )

    if errors:
        raise RuntimeError(
            "Configuração incompatível com o EXP-012:\n- "
            + "\n- ".join(errors)
        )


def get_resume_config_path(
    checkpoint_path: Path,
) -> Path:
    """
    Estrutura esperada:

    data/experiments/<experimento>/
        checkpoints/neat-checkpoint-X
        config/neat-config.ini
    """

    experiment_dir = checkpoint_path.parent.parent

    config_path = (
        experiment_dir
        / "config"
        / "neat-config.ini"
    )

    if not config_path.exists():
        raise FileNotFoundError(
            "Não foi possível localizar a configuração "
            f"original do checkpoint: {config_path}"
        )

    return config_path


def create_experiment(
    experiment_name: str,
    generations: int,
    workers: int,
    available_cpus: int,
    config_source: Path,
    resume_from: Path | None = None,
) -> dict[str, Path]:
    experiment_dir = (
        EXPERIMENTS_DIR
        / experiment_name
    )

    if experiment_dir.exists():
        raise FileExistsError(
            f"O experimento '{experiment_name}' já existe em "
            f"{experiment_dir}. Use outro nome para não "
            "sobrescrever resultados anteriores."
        )

    checkpoints_dir = (
        experiment_dir
        / "checkpoints"
    )

    metrics_dir = (
        experiment_dir
        / "metrics"
    )

    models_dir = (
        experiment_dir
        / "models"
    )

    plots_dir = (
        experiment_dir
        / "plots"
    )

    config_dir = (
        experiment_dir
        / "config"
    )

    for directory in (
        checkpoints_dir,
        metrics_dir,
        models_dir,
        plots_dir,
        config_dir,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # Guarda exatamente a configuração utilizada
    # pela população deste experimento.
    shutil.copy2(
        config_source,
        config_dir / "neat-config.ini",
    )

    metadata = {
        "experiment": experiment_name,
        "created_at": datetime.now().isoformat(),
        "additional_generations": generations,
        "evaluation_mode": "parallel",
        "available_logical_cpus": available_cpus,
        "reserved_logical_cpus": RESERVED_LOGICAL_CPUS,
        "workers": workers,
        "config_source": str(config_source),
        "training_mode": (
            "resume"
            if resume_from is not None
            else "new"
        ),
        "resume_from": (
            str(resume_from)
            if resume_from is not None
            else None
        ),
        "architecture": {
            "inputs": EXPECTED_INPUTS,
            "outputs": EXPECTED_OUTPUTS,
            "network": "recurrent",
            "population": EXPECTED_POPULATION,
            "action_repeat": 5,
        },
        "status": "running",
    }

    metadata_path = (
        experiment_dir
        / "experiment.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return {
        "root": experiment_dir,
        "checkpoints": checkpoints_dir,
        "metrics": metrics_dir,
        "models": models_dir,
        "plots": plots_dir,
        "config": config_dir,
    }


def get_best_genome(
    statistics: neat.StatisticsReporter,
    population,
):
    if statistics.most_fit_genomes:
        return max(
            statistics.most_fit_genomes,
            key=lambda genome: genome.fitness,
        )

    evaluated_genomes = [
        genome
        for genome in population.population.values()
        if genome.fitness is not None
    ]

    if not evaluated_genomes:
        return None

    return max(
        evaluated_genomes,
        key=lambda genome: genome.fitness,
    )


def save_winner(genome, model_path: Path) -> None:
    temporary_path = model_path.with_name(
        f".{model_path.name}.tmp"
    )

    try:
        with temporary_path.open("wb") as file:
            pickle.dump(
                genome,
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        os.replace(temporary_path, model_path)
    finally:
        temporary_path.unlink(missing_ok=True)


class WinnerCheckpointer(neat.Checkpointer):
    def __init__(self, model_path: Path, **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.best_genome = None

    def post_evaluate(
        self,
        config,
        population,
        species_set,
        best_genome,
    ) -> None:
        if (
            self.best_genome is None
            or best_genome.fitness > self.best_genome.fitness
        ):
            self.best_genome = best_genome

    def save_checkpoint(
        self,
        config,
        population,
        species_set,
        generation,
    ) -> None:
        super().save_checkpoint(
            config,
            population,
            species_set,
            generation,
        )

        if self.best_genome is not None:
            save_winner(self.best_genome, self.model_path)
            print(f"Modelo winner atualizado: {self.model_path}")


def train(
    experiment_name: str,
    generations: int = 10,
    workers: int | None = None,
    resume_from: str | Path | None = None,
):
    if generations < 1:
        raise ValueError(
            "A quantidade de gerações deve ser "
            "maior ou igual a 1."
        )

    available_cpus = (
        os.cpu_count()
        or 1
    )

    if workers is None:
        workers = max(
            1,
            available_cpus
            - RESERVED_LOGICAL_CPUS,
        )

    if workers < 1:
        raise ValueError(
            "A quantidade de workers deve ser "
            "maior ou igual a 1."
        )

    workers = min(
        workers,
        available_cpus,
    )

    checkpoint_path: Path | None = None

    # =========================================================
    # POPULAÇÃO / CONFIGURAÇÃO
    # =========================================================

    if resume_from is not None:
        checkpoint_path = Path(
            resume_from
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint não encontrado: "
                f"{checkpoint_path}"
            )

        if not checkpoint_path.is_file():
            raise ValueError(
                "O checkpoint informado não é "
                f"um arquivo: {checkpoint_path}"
            )

        print()
        print(
            f"Restaurando população de: "
            f"{checkpoint_path}"
        )

        population = (
            neat.Checkpointer.restore_checkpoint(
                str(checkpoint_path)
            )
        )

        config = population.config

        # IMPORTANTE:
        # valida o config que veio dentro do checkpoint.
        validate_config(config)

        config_source = (
            get_resume_config_path(
                checkpoint_path
            )
        )

        print("Checkpoint restaurado.")
        print(
            f"Próxima geração: "
            f"{population.generation + 1}"
        )

    else:
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(
                f"Configuração não encontrada: "
                f"{CONFIG_PATH}"
            )

        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(CONFIG_PATH),
        )

        # O experimento nem nasce se a configuração
        # estiver diferente da arquitetura final.
        validate_config(config)

        population = neat.Population(
            config
        )

        config_source = CONFIG_PATH

    # =========================================================
    # CRIA EXPERIMENTO
    # =========================================================

    paths = create_experiment(
        experiment_name=experiment_name,
        generations=generations,
        workers=workers,
        available_cpus=available_cpus,
        config_source=config_source,
        resume_from=checkpoint_path,
    )

    print()
    print("=" * 70)
    print(f"EXPERIMENTO: {experiment_name}")
    print("=" * 70)

    if checkpoint_path is None:
        print("Modo de treinamento: NOVO")
    else:
        print(
            "Modo de treinamento: CONTINUAÇÃO"
        )
        print(
            f"Checkpoint original: "
            f"{checkpoint_path}"
        )

    print(
        f"Gerações máximas desta execução: "
        f"{generations}"
    )

    print(
        f"População: "
        f"{config.pop_size}"
    )

    print(
        f"Inputs: "
        f"{config.genome_config.num_inputs}"
    )

    print(
        f"Outputs: "
        f"{config.genome_config.num_outputs}"
    )

    print("Rede: RECORRENTE")
    print("Action repeat: 5")

    print(
        "Fitness threshold: "
        f"{config.fitness_threshold}"
    )

    print("Modo de avaliação: paralelo")

    print(
        f"CPUs lógicas disponíveis: "
        f"{available_cpus}"
    )

    print(
        f"CPUs lógicas reservadas: "
        f"{RESERVED_LOGICAL_CPUS}"
    )

    print(
        f"Workers utilizados: "
        f"{workers}"
    )

    print(
        f"Diretório: "
        f"{paths['root']}"
    )

    print("=" * 70)
    print()

    # =========================================================
    # REPORTERS
    # =========================================================

    stdout_reporter = (
        neat.StdOutReporter(
            True
        )
    )

    statistics = (
        neat.StatisticsReporter()
    )

    metrics = MetricsReporter(
        paths["metrics"]
        / "training_metrics.csv"
    )

    checkpointer = WinnerCheckpointer(
        model_path=paths["models"] / "winner.pkl",
        generation_interval=5,
        time_interval_seconds=None,
        filename_prefix=str(
            paths["checkpoints"]
            / "neat-checkpoint-"
        ),
    )

    population.add_reporter(
        stdout_reporter
    )

    population.add_reporter(
        statistics
    )

    population.add_reporter(
        metrics
    )

    population.add_reporter(
        checkpointer
    )

    # =========================================================
    # AVALIAÇÃO PARALELA
    # =========================================================

    parallel_evaluator = (
        neat.ParallelEvaluator(
            workers,
            evaluate_genome,
        )
    )

    # =========================================================
    # TREINAMENTO
    # =========================================================

    starting_generation = (
        population.generation
    )

    training_start = (
        time.perf_counter()
    )

    winner = None
    interrupted = False

    try:
        winner = population.run(
            parallel_evaluator.evaluate,
            generations,
        )

    except KeyboardInterrupt:
        interrupted = True

        print()
        print("=" * 70)
        print(
            "TREINAMENTO INTERROMPIDO "
            "PELO USUÁRIO"
        )
        print("=" * 70)

        winner = get_best_genome(
            statistics,
            population,
        )

    total_training_time = (
        time.perf_counter()
        - training_start
    )

    ending_generation = (
        population.generation
    )

    # =========================================================
    # SALVA WINNER
    # =========================================================

    model_path: Path | None = None

    if winner is not None:
        model_path = (
            paths["models"]
            / "winner.pkl"
        )

        save_winner(winner, model_path)

    threshold_reached = bool(
        winner is not None
        and winner.fitness is not None
        and winner.fitness
        >= config.fitness_threshold
    )

    # =========================================================
    # METADADOS
    # =========================================================

    metadata_path = (
        paths["root"]
        / "experiment.json"
    )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(
            file
        )

    metadata.update(
        {
            "starting_generation": (
                starting_generation
            ),
            "ending_generation": (
                ending_generation
            ),
            "completed_generations": (
                ending_generation
                - starting_generation
            ),
            "winner_fitness": (
                float(winner.fitness)
                if winner is not None
                and winner.fitness is not None
                else None
            ),
            "fitness_threshold_reached": (
                threshold_reached
            ),
            "training_time_seconds": (
                total_training_time
            ),
            "finished_at": (
                datetime.now().isoformat()
            ),
            "interrupted": interrupted,
            "status": (
                "interrupted"
                if interrupted
                else "completed"
            ),
        }
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # =========================================================
    # RESULTADO
    # =========================================================

    print()
    print("=" * 70)

    if interrupted:
        print(
            "TREINAMENTO INTERROMPIDO E SALVO"
        )

    elif threshold_reached:
        print(
            "FITNESS THRESHOLD ATINGIDO"
        )

    else:
        print(
            "LIMITE DE GERAÇÕES ATINGIDO"
        )

    print("=" * 70)

    print(
        f"Experimento: "
        f"{experiment_name}"
    )

    print(
        f"Geração inicial: "
        f"{starting_generation}"
    )

    print(
        f"Geração final: "
        f"{ending_generation}"
    )

    if winner is not None:
        print(
            f"Melhor fitness salvo: "
            f"{winner.fitness}"
        )
    else:
        print(
            "Nenhum genoma avaliado "
            "disponível para salvar."
        )

    print(
        f"Threshold atingido: "
        f"{'Sim' if threshold_reached else 'Não'}"
    )

    print(
        f"Tempo total: "
        f"{total_training_time:.2f} segundos"
    )

    print(
        f"Workers utilizados: "
        f"{workers}"
    )

    if model_path is not None:
        print(
            f"Modelo: "
            f"{model_path}"
        )

    print(
        f"Métricas: "
        f"{paths['metrics'] / 'training_metrics.csv'}"
    )

    print(
        f"Metadados: "
        f"{metadata_path}"
    )

    print(
        f"Interrompido: "
        f"{'Sim' if interrupted else 'Não'}"
    )

    print("=" * 70)

    return winner
