from src.evolution.trainer import train


def main():
    train(
        experiment_name="teste-012-continuacao-360",
        generations=2640,
        resume_from=(
            "data/experiments/"
            "teste-012-recurrent-13x13/"
            "checkpoints/"
            "neat-checkpoint-360"
        ),
    )


if __name__ == "__main__":
    main()