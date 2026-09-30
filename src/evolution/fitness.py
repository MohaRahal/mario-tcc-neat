from __future__ import annotations


CLEAR_BONUS = 10_000.0
DEATH_PENALTY = 100.0
STUCK_PENALTY = 100.0


def calculate_fitness(
    initial_x: int,
    max_x: int,
    clear: bool,
    death: bool,
    stuck: bool,
) -> float:
    """
    Calcula a aptidão do genoma.

    O objetivo principal é simples:
    avançar o máximo possível horizontalmente.

    A conclusão da fase recebe um bônus alto para
    tornar qualquer indivíduo que alcance a bandeira
    superior aos indivíduos que apenas avançam.

    Morte e estagnação recebem penalidades pequenas
    em relação ao progresso total.
    """

    progress = max(
        0,
        max_x - initial_x,
    )

    fitness = float(progress)

    if clear:
        fitness += CLEAR_BONUS
        return fitness

    if death:
        fitness -= DEATH_PENALTY

    if stuck:
        fitness -= STUCK_PENALTY

    return fitness