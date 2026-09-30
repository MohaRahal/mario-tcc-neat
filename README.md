# Estrutura do Projeto — Mario NEAT TCC

Este documento descreve a organização do projeto responsável pelo treinamento de um agente autônomo para jogar Super Mario Bros utilizando redes neurais artificiais e o algoritmo NEAT (NeuroEvolution of Augmenting Topologies).

---

## Estrutura de Diretórios

```text
MarioTCC/
│
├── src/
│   ├── __init__.py
│   ├── main.py
│   │
│   ├── environment/
│   │   ├── __init__.py
│   │   └── mario_env.py
│   │
│   ├── sensors/
│   │   ├── __init__.py
│   │   ├── mario_state.py
│   │   ├── enemy_tracker.py
│   │   ├── environment_sensors.py
│   │   └── sensor_system.py
│   │
│   ├── evolution/
│   │   ├── __init__.py
│   │   ├── fitness.py
│   │   ├── evaluator.py
│   │   └── trainer.py
│   │
│   └── evaluation/
│       ├── __init__.py
│       └── watch_agent.py
│
├── configs/
│   └── neat-config.ini
│
├── data/
│   ├── checkpoints/
│   ├── logs/
│   └── results/
│
├── models/
│
├── research/
│   └── outputs/
│
├── docs/
│   ├── project-structure.md
│   ├── architecture.md
│   ├── sensors.md
│   ├── fitness.md
│   ├── neat-config.md
│   └── experiments.md
│
├── requirements.txt
├── README.md
└── .gitignore