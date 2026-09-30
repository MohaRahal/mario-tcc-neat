from src.evolution.trainer import train


def main():
    train(
        experiment_name="teste-012-recurrent-13x13",
        generations=3000,
    )


if __name__ == "__main__":
    main()