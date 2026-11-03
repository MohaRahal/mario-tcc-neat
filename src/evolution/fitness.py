from __future__ import annotations


CLEAR_BONUS = 10_000.0
DEATH_PENALTY = 100.0
STUCK_PENALTY = 100.0

# Penalização propositalmente pequena.
# Serve apenas para favorecer agentes que
# concluem a fase utilizando menos saltos.
JUMP_PENALTY = 1.0


def calculate_fitness(
    initial_x: int,
    max_x: int,
    clear: bool,
    death: bool,
    stuck: bool,
    jump_count: int = 0,
) -> float:
    """
    Calcula a aptidão do genoma.

    O objetivo principal é avançar o máximo possível
    horizontalmente.

    A conclusão da fase recebe um bônus alto para
    tornar qualquer indivíduo que alcance a bandeira
    superior aos indivíduos que apenas avançam.

    Morte e estagnação recebem penalidades pequenas
    em relação ao progresso total.

    Quando o indivíduo conclui a fase, é aplicada uma
    pequena penalização relacionada à quantidade de
    saltos realizados. Isso permite que, entre agentes
    capazes de concluir a fase, sejam favorecidos aqueles
    que executam menos saltos desnecessários.
    """

    progress = max(
        0,
        max_x - initial_x,
    )

    fitness = float(
        progress
    )

    # ---------------------------------------------------------
    # CONCLUSÃO DA FASE
    # ---------------------------------------------------------

    if clear:
        fitness += CLEAR_BONUS

        # Pequeno refinamento comportamental:
        # menos saltos = fitness ligeiramente maior.
        fitness -= (
            jump_count
            * JUMP_PENALTY
        )

        return fitness

    # ---------------------------------------------------------
    # MORTE
    # ---------------------------------------------------------

    if death:
        fitness -= (
            DEATH_PENALTY
        )

    # ---------------------------------------------------------
    # ESTAGNAÇÃO
    # ---------------------------------------------------------

    if stuck:
        fitness -= (
            STUCK_PENALTY
        )

    return fitness