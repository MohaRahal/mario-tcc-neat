from __future__ import annotations

import neat
import numpy as np

from src.environment.mario_env import create_mario_env
from src.evolution.fitness import calculate_fitness
from src.sensors.sensor_system import SensorSystem


SEED = 42

EXPECTED_INPUTS = 169

MAX_FRAMES = 10_000
MAX_STAGNATION_FRAMES = 180

ACTION_REPEAT = 5

# Ações que possuem o botão A (salto):
#
# 0 = NOOP
# 1 = right
# 2 = right A
# 3 = right B
# 4 = right A B
# 5 = A
# 6 = left
#
# Portanto, as ações de salto são:
JUMP_ACTIONS = {2, 4, 5}


def evaluate_genome(genome, config) -> float:
    """
    Avalia um único genoma no ambiente Super Mario Bros.

    Arquitetura utilizada:

        SensorSystem 13x13
            ↓
        169 inputs
            ↓
        NEAT recorrente
            ↓
        7 outputs
            ↓
        ação escolhida
            ↓
        repetida por 5 frames

    O episódio termina quando:

        - Mario conclui a fase;
        - Mario morre;
        - o ambiente encerra;
        - ocorre truncamento;
        - Mario fica estagnado;
        - o limite máximo de frames é atingido.

    Também é contabilizada a quantidade de inícios
    de salto realizados pelo agente.

    Um salto mantido durante várias decisões consecutivas
    não é contado repetidamente. Apenas a transição de uma
    ação sem salto para uma ação com salto aumenta
    jump_count.
    """

    env = create_mario_env(
        render_mode=None
    )

    try:
        observation, info = env.reset(
            seed=SEED
        )

        sensors = SensorSystem(env)
        sensors.reset()

        _validate_configuration(
            sensors=sensors,
            config=config,
        )

        network = neat.nn.RecurrentNetwork.create(
            genome,
            config,
        )

        initial_x = int(
            info.get(
                "x_pos",
                0,
            )
        )

        max_x = initial_x

        stagnation_frames = 0
        executed_frames = 0

        clear = False
        death = False
        stuck = False

        # -----------------------------------------------------
        # CONTROLE DE SALTOS
        # -----------------------------------------------------

        jump_count = 0

        # Indica se a decisão anterior já continha salto.
        # Isso impede que várias decisões consecutivas com
        # botão A sejam contadas como vários saltos.
        was_jumping = False

        episode_finished = False

        while (
            executed_frames < MAX_FRAMES
            and not episode_finished
        ):
            # -------------------------------------------------
            # SENSORIAMENTO
            # -------------------------------------------------

            inputs = sensors.observe(
                info
            )

            if len(inputs) != EXPECTED_INPUTS:
                raise RuntimeError(
                    f"SensorSystem retornou "
                    f"{len(inputs)} inputs. "
                    f"Esperado: {EXPECTED_INPUTS}."
                )

            # -------------------------------------------------
            # REDE NEURAL
            # -------------------------------------------------

            outputs = network.activate(
                inputs
            )

            if len(outputs) != 7:
                raise RuntimeError(
                    f"A rede retornou "
                    f"{len(outputs)} outputs. "
                    "Esperado: 7."
                )

            action = int(
                np.argmax(
                    outputs
                )
            )

            # -------------------------------------------------
            # CONTAGEM DE SALTOS
            # -------------------------------------------------

            # Verifica se a ação escolhida possui o botão A.
            is_jumping = (
                action in JUMP_ACTIONS
            )

            # Só conta um novo salto quando ocorre:
            #
            # ação sem salto
            #       ↓
            # ação com salto
            #
            # Se continuar escolhendo ações com salto,
            # jump_count não aumenta novamente.
            if (
                is_jumping
                and not was_jumping
            ):
                jump_count += 1

            was_jumping = is_jumping

            # -------------------------------------------------
            # EXECUÇÃO DA AÇÃO
            # -------------------------------------------------

            # A decisão da rede é mantida durante cinco frames.
            for _ in range(
                ACTION_REPEAT
            ):
                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(
                    action
                )

                executed_frames += 1

                # ---------------------------------------------
                # PROGRESSO
                # ---------------------------------------------

                current_x = int(
                    info.get(
                        "x_pos",
                        max_x,
                    )
                )

                if current_x > max_x:
                    max_x = current_x

                    stagnation_frames = 0

                else:
                    stagnation_frames += 1

                # ---------------------------------------------
                # ESTADO DO EPISÓDIO
                # ---------------------------------------------

                clear = _is_level_clear(
                    info
                )

                death = _is_dead(
                    info
                )

                stuck = (
                    stagnation_frames
                    >= MAX_STAGNATION_FRAMES
                )

                # ---------------------------------------------
                # CONDIÇÕES DE ENCERRAMENTO
                # ---------------------------------------------

                if clear:
                    episode_finished = True
                    break

                if death:
                    episode_finished = True
                    break

                if terminated or truncated:
                    episode_finished = True
                    break

                if stuck:
                    episode_finished = True
                    break

                if (
                    executed_frames
                    >= MAX_FRAMES
                ):
                    episode_finished = True
                    break

        # -----------------------------------------------------
        # FITNESS
        # -----------------------------------------------------

        return calculate_fitness(
            initial_x=initial_x,
            max_x=max_x,
            clear=clear,
            death=death,
            stuck=stuck,
            jump_count=jump_count,
        )

    finally:
        env.close()


def _validate_configuration(
    sensors: SensorSystem,
    config,
) -> None:
    """
    Impede que o treinamento seja iniciado com uma
    configuração incompatível com a arquitetura final.
    """

    sensor_inputs = (
        sensors.size
    )

    configured_inputs = (
        config.genome_config.num_inputs
    )

    configured_outputs = (
        config.genome_config.num_outputs
    )

    feed_forward = (
        config.genome_config.feed_forward
    )

    if (
        sensor_inputs
        != EXPECTED_INPUTS
    ):
        raise RuntimeError(
            f"SensorSystem possui "
            f"{sensor_inputs} inputs. "
            f"Esperado: {EXPECTED_INPUTS}."
        )

    if (
        configured_inputs
        != EXPECTED_INPUTS
    ):
        raise RuntimeError(
            f"NEAT configurado com "
            f"{configured_inputs} inputs. "
            f"Esperado: {EXPECTED_INPUTS}."
        )

    if configured_outputs != 7:
        raise RuntimeError(
            f"NEAT configurado com "
            f"{configured_outputs} outputs. "
            "Esperado: 7."
        )

    if feed_forward:
        raise RuntimeError(
            "A arquitetura final exige rede recorrente. "
            "Defina feed_forward = False "
            "no neat-config.ini."
        )


def _is_level_clear(
    info: dict,
) -> bool:
    """
    Verifica conclusão da fase.

    flag_get é o indicador principal do ambiente.
    clear é mantido como compatibilidade com a lógica
    utilizada nas versões anteriores do projeto.
    """

    return bool(
        info.get(
            "flag_get",
            False,
        )
        or info.get(
            "clear",
            False,
        )
    )


def _is_dead(
    info: dict,
) -> bool:
    """
    Verifica indicadores de morte fornecidos pelo ambiente
    ou por wrappers utilizados pelo projeto.
    """

    return bool(
        info.get(
            "death",
            False,
        )
        or info.get(
            "is_dead",
            False,
        )
        or info.get(
            "is_dying",
            False,
        )
    )