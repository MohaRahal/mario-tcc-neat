import pickle
import tempfile
import unittest
from pathlib import Path

import neat

from src.evolution.trainer import WinnerCheckpointer


class WinnerCheckpointerTest(unittest.TestCase):
    def test_checkpoint_saves_best_genome_for_live_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            model_path = root / "winner.pkl"
            checkpoint_path = root / "neat-checkpoint-2"

            config = neat.Config(
                neat.DefaultGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                "configs/neat-config.ini",
            )
            config.pop_size = 10
            config.fitness_threshold = 1_000_000

            population = neat.Population(config)
            population.add_reporter(
                WinnerCheckpointer(
                    model_path=model_path,
                    generation_interval=2,
                    time_interval_seconds=None,
                    filename_prefix=str(root / "neat-checkpoint-"),
                )
            )

            evaluated_generations = 0

            def evaluate(genomes, _config):
                nonlocal evaluated_generations

                if evaluated_generations == 0:
                    self.assertFalse(model_path.exists())
                elif evaluated_generations == 2:
                    self.assertTrue(checkpoint_path.exists())
                    with model_path.open("rb") as file:
                        self.assertEqual(pickle.load(file).fitness, 5)

                fitness = (5, 1, 2)[evaluated_generations]
                for _, genome in genomes:
                    genome.fitness = fitness

                evaluated_generations += 1

            winner = population.run(evaluate, 3)

            self.assertEqual(evaluated_generations, 3)
            self.assertEqual(winner.fitness, 5)
            with model_path.open("rb") as file:
                self.assertEqual(pickle.load(file).fitness, 5)
            self.assertFalse((root / ".winner.pkl.tmp").exists())


if __name__ == "__main__":
    unittest.main()
