from __future__ import annotations

import argparse
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import neat
import numpy as np
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from src.environment.mario_env import DEFAULT_ENV_ID, create_mario_env
from src.sensors.experiment_sensor_system import ExperimentSensorSystem


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = PROJECT_ROOT / "data" / "experiments"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "neat-config.ini"

SEED = 42
EXPECTED_INPUTS = 169
EXPECTED_OUTPUTS = len(SIMPLE_MOVEMENT)
ACTION_REPEAT = 5
MAX_FRAMES = 10_000
MAX_STAGNATION_FRAMES = 180

WINDOW_NAME = "Mario NEAT | Visualizador do agente"
SELECTOR_WINDOW = "Mario NEAT | Escolher winner.pkl"
WINDOW_WIDTH = 1440
WINDOW_HEIGHT = 860
GAME_RECT = (22, 72, 768, 720)

BACKGROUND = (15, 18, 25)
PANEL = (26, 31, 42)
PANEL_BORDER = (53, 62, 78)
TEXT = (230, 235, 242)
MUTED = (145, 155, 170)
ACCENT = (255, 190, 65)
POSITIVE = (91, 214, 140)
NEGATIVE = (91, 105, 235)


@dataclass
class Session:
    observation: np.ndarray
    info: dict
    initial_x: int
    max_x: int
    current_x: int
    total_reward: float = 0.0
    frames: int = 0
    decisions: int = 0
    stagnation: int = 0
    action: int = 0
    repeat_left: int = 0
    inputs: np.ndarray = field(
        default_factory=lambda: np.zeros(EXPECTED_INPUTS, dtype=np.float32)
    )
    perception: np.ndarray = field(
        default_factory=lambda: np.zeros((13, 13), dtype=np.float32)
    )
    outputs: np.ndarray = field(
        default_factory=lambda: np.zeros(EXPECTED_OUTPUTS, dtype=np.float32)
    )
    status: str = "Executando"
    finished: bool = False
    clear: bool = False
    death: bool = False
    stuck: bool = False


@dataclass
class ViewerState:
    paused: bool = False
    restart_requested: bool = False
    quit_requested: bool = False
    step_requested: bool = False
    buttons: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)
    display_size: tuple[int, int] = (WINDOW_WIDTH, WINDOW_HEIGHT)


@dataclass(frozen=True)
class ModelChoice:
    experiment: str
    model_path: Path
    config_path: Path
    modified_at: float


@dataclass
class SelectorState:
    choices: list[ModelChoice] = field(default_factory=list)
    selected: int = 0
    offset: int = 0
    load_requested: bool = False
    quit_requested: bool = False
    rows: list[tuple[int, tuple[int, int, int, int]]] = field(default_factory=list)
    buttons: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)
    display_size: tuple[int, int] = (WINDOW_WIDTH, WINDOW_HEIGHT)


def _open_fullscreen_window(name: str) -> tuple[int, int]:
    width, height = _screen_size()
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    return width, height


def _screen_size() -> tuple[int, int]:
    try:
        import ctypes

        user32 = ctypes.windll.user32
        width = int(user32.GetSystemMetrics(0))
        height = int(user32.GetSystemMetrics(1))
        if width > 0 and height > 0:
            return width, height
    except (AttributeError, OSError):
        pass
    return WINDOW_WIDTH, WINDOW_HEIGHT


def _fit_to_display(
    canvas: np.ndarray, display_size: tuple[int, int]
) -> np.ndarray:
    if display_size == (WINDOW_WIDTH, WINDOW_HEIGHT):
        return canvas
    return cv2.resize(canvas, display_size, interpolation=cv2.INTER_LINEAR)


def _canvas_point(
    x: int, y: int, display_size: tuple[int, int]
) -> tuple[int, int]:
    display_width, display_height = display_size
    return (
        int(x * WINDOW_WIDTH / display_width),
        int(y * WINDOW_HEIGHT / display_height),
    )


def watch(
    experiment_name: str | None = None,
    model_path: str | Path | None = None,
    config_path: str | Path | None = None,
    env_id: str = DEFAULT_ENV_ID,
) -> None:
    """Executa um genome e mostra jogo, sensores e rede em tempo real."""
    if not experiment_name and not model_path:
        choice = _choose_model()
        if choice is None:
            return
        experiment_name = choice.experiment
        model_path = choice.model_path
        config_path = choice.config_path

    model, config_file, display_name = _resolve_paths(
        experiment_name, model_path, config_path
    )
    _validate_paths(config_file, model)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_file),
    )
    _validate_neat_config(config)

    with model.open("rb") as file:
        genome = pickle.load(file)

    network_factory: Callable = (
        neat.nn.FeedForwardNetwork.create
        if config.genome_config.feed_forward
        else neat.nn.RecurrentNetwork.create
    )
    network = network_factory(genome, config)
    env = create_mario_env(env_id=env_id, render_mode="rgb_array")
    sensors = ExperimentSensorSystem(env, config.genome_config.num_inputs)
    action_repeat = 1 if config.genome_config.feed_forward else ACTION_REPEAT
    state = ViewerState()

    print(f"Modelo: {model}")
    print(f"Configuração: {config_file}")
    print("Controles: Espaço pausa | R reinicia | N passo | Esc sai")

    state.display_size = _open_fullscreen_window(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, _on_mouse, state)

    try:
        session = _reset_session(env, sensors, network)
        while not state.quit_requested:
            may_advance = not state.paused or state.step_requested
            if may_advance and not session.finished:
                _advance_one_frame(
                    session, env, sensors, network, action_repeat=action_repeat
                )
                state.step_requested = False

            canvas = _render_dashboard(
                session, state, genome, config, network, display_name
            )
            cv2.imshow(WINDOW_NAME, _fit_to_display(canvas, state.display_size))
            key = cv2.waitKey(16 if may_advance else 30) & 0xFF
            _handle_key(key, state)

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            if state.restart_requested:
                session = _reset_session(env, sensors, network)
                state.restart_requested = False
                state.paused = False

        _print_result(display_name, genome, session)
    finally:
        env.close()
        cv2.destroyAllWindows()


def _discover_models() -> list[ModelChoice]:
    choices: list[ModelChoice] = []
    if not EXPERIMENTS_DIR.exists():
        return choices

    for model_path in EXPERIMENTS_DIR.glob("**/winner.pkl"):
        experiment_dir = model_path.parent.parent
        config_path = experiment_dir / "config" / "neat-config.ini"
        try:
            modified_at = model_path.stat().st_mtime
        except OSError:
            continue
        choices.append(
            ModelChoice(
                experiment=experiment_dir.name,
                model_path=model_path.resolve(),
                config_path=config_path.resolve(),
                modified_at=modified_at,
            )
        )

    return sorted(
        choices,
        key=lambda choice: (choice.modified_at, choice.experiment.lower()),
        reverse=True,
    )


def _choose_model() -> ModelChoice | None:
    state = SelectorState(choices=_discover_models())
    state.display_size = _open_fullscreen_window(SELECTOR_WINDOW)
    cv2.setMouseCallback(SELECTOR_WINDOW, _on_selector_mouse, state)
    last_scan = time.monotonic()

    try:
        while not state.quit_requested:
            now = time.monotonic()
            if now - last_scan >= 1.0:
                _refresh_choices(state)
                last_scan = now

            selector = _render_model_selector(state)
            cv2.imshow(SELECTOR_WINDOW, _fit_to_display(selector, state.display_size))
            key = cv2.waitKeyEx(50)
            _handle_selector_key(key, state)

            if cv2.getWindowProperty(SELECTOR_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return None
            if state.load_requested and state.choices:
                return state.choices[state.selected]
    finally:
        cv2.destroyWindow(SELECTOR_WINDOW)
    return None


def _refresh_choices(state: SelectorState) -> None:
    selected_path = (
        state.choices[state.selected].model_path
        if state.choices and state.selected < len(state.choices)
        else None
    )
    state.choices = _discover_models()
    if selected_path:
        state.selected = next(
            (
                index
                for index, choice in enumerate(state.choices)
                if choice.model_path == selected_path
            ),
            min(state.selected, max(0, len(state.choices) - 1)),
        )
    else:
        state.selected = min(state.selected, max(0, len(state.choices) - 1))
    _keep_selection_visible(state)


def _render_model_selector(state: SelectorState) -> np.ndarray:
    canvas = np.full((WINDOW_HEIGHT, WINDOW_WIDTH, 3), BACKGROUND, dtype=np.uint8)
    _text(canvas, "MARIO NEAT • ESCOLHA O AGENTE", (64, 58), 0.9, TEXT, 2)
    _text(
        canvas,
        "Monitorando data/experiments/**/winner.pkl",
        (66, 88),
        0.48,
        MUTED,
    )
    _text(canvas, f"{len(state.choices)} modelo(s)", (1190, 58), 0.5, POSITIVE)

    _panel(canvas, (62, 112, 1316, 584), "MODELOS DISPONÍVEIS")
    _text(canvas, "EXPERIMENTO", (88, 162), 0.38, MUTED)
    _text(canvas, "ARQUIVO", (526, 162), 0.38, MUTED)
    _text(canvas, "MODIFICADO", (1050, 162), 0.38, MUTED)
    _text(canvas, "CONFIG", (1240, 162), 0.38, MUTED)

    state.rows.clear()
    visible_count = 13
    visible = state.choices[state.offset : state.offset + visible_count]
    for relative_index, choice in enumerate(visible):
        index = state.offset + relative_index
        y = 180 + relative_index * 38
        selected = index == state.selected
        fill = (48, 67, 72) if selected else ((31, 37, 49) if index % 2 else PANEL)
        cv2.rectangle(canvas, (78, y), (1362, y + 32), fill, -1)
        if selected:
            cv2.rectangle(canvas, (78, y), (83, y + 32), POSITIVE, -1)
        state.rows.append((index, (78, y, 1284, 32)))

        config_ok = choice.config_path.is_file()
        modified = datetime.fromtimestamp(choice.modified_at).strftime("%d/%m/%Y %H:%M")
        relative_model = choice.model_path.relative_to(PROJECT_ROOT)
        _text(canvas, choice.experiment[:46], (94, y + 21), 0.43, TEXT)
        _text(canvas, str(relative_model)[:66], (526, y + 21), 0.37, MUTED)
        _text(canvas, modified, (1050, y + 21), 0.38, TEXT)
        _text(
            canvas,
            "OK" if config_ok else "AUSENTE",
            (1240, y + 21),
            0.4,
            POSITIVE if config_ok else NEGATIVE,
        )

    if not state.choices:
        _text(canvas, "Nenhum winner.pkl encontrado.", (510, 360), 0.65, MUTED, 1)
        _text(canvas, "A lista será atualizada automaticamente.", (470, 396), 0.48, MUTED)

    if len(state.choices) > visible_count:
        first = state.offset + 1
        last = min(len(state.choices), state.offset + visible_count)
        _text(canvas, f"Mostrando {first}-{last}", (1180, 680), 0.36, MUTED)

    _draw_selector_buttons(canvas, state)
    _text(
        canvas,
        "Setas selecionam  •  Enter carrega  •  Duplo clique carrega",
        (64, 832),
        0.43,
        MUTED,
    )
    return canvas


def _draw_selector_buttons(canvas: np.ndarray, state: SelectorState) -> None:
    specs = [
        ("load", "CARREGAR [ENTER]", 850, 728, 190, bool(state.choices)),
        ("refresh", "ATUALIZAR [R]", 1052, 728, 170, True),
        ("quit", "SAIR [ESC]", 1234, 728, 144, True),
    ]
    state.buttons.clear()
    for name, label, x, y, width, enabled in specs:
        state.buttons[name] = (x, y, width, 46)
        fill = (48, 83, 75) if name == "load" and enabled else (45, 53, 67)
        if not enabled:
            fill = (31, 36, 45)
        cv2.rectangle(canvas, (x, y), (x + width, y + 46), fill, -1)
        cv2.rectangle(canvas, (x, y), (x + width, y + 46), PANEL_BORDER, 1)
        size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        _text(
            canvas,
            label,
            (x + (width - size[0]) // 2, y + 29),
            0.4,
            TEXT if enabled else MUTED,
        )


def _on_selector_mouse(
    event: int, x: int, y: int, _flags: int, state: SelectorState
) -> None:
    if event not in (cv2.EVENT_LBUTTONUP, cv2.EVENT_LBUTTONDBLCLK):
        return
    x, y = _canvas_point(x, y, state.display_size)
    for index, (rx, ry, rw, rh) in state.rows:
        if rx <= x <= rx + rw and ry <= y <= ry + rh:
            state.selected = index
            if event == cv2.EVENT_LBUTTONDBLCLK:
                state.load_requested = True
            return
    if event == cv2.EVENT_LBUTTONUP:
        for name, (bx, by, bw, bh) in state.buttons.items():
            if bx <= x <= bx + bw and by <= y <= by + bh:
                _run_selector_control(name, state)
                return


def _handle_selector_key(key: int, state: SelectorState) -> None:
    if key in (27, ord("q")):
        state.quit_requested = True
    elif key in (13, 10):
        state.load_requested = bool(state.choices)
    elif key in (ord("r"), ord("R")):
        _refresh_choices(state)
    elif key in (82, 65362, 2490368, ord("w"), ord("W")) and state.choices:
        state.selected = max(0, state.selected - 1)
        _keep_selection_visible(state)
    elif key in (84, 65364, 2621440, ord("s"), ord("S")) and state.choices:
        state.selected = min(len(state.choices) - 1, state.selected + 1)
        _keep_selection_visible(state)


def _run_selector_control(name: str, state: SelectorState) -> None:
    if name == "load" and state.choices:
        state.load_requested = True
    elif name == "refresh":
        _refresh_choices(state)
    elif name == "quit":
        state.quit_requested = True


def _keep_selection_visible(state: SelectorState) -> None:
    visible_count = 13
    if state.selected < state.offset:
        state.offset = state.selected
    elif state.selected >= state.offset + visible_count:
        state.offset = state.selected - visible_count + 1
    max_offset = max(0, len(state.choices) - visible_count)
    state.offset = max(0, min(state.offset, max_offset))


def _resolve_paths(
    experiment_name: str | None,
    model_path: str | Path | None,
    config_path: str | Path | None,
) -> tuple[Path, Path, str]:
    if not experiment_name and not model_path:
        raise ValueError("Informe --experiment ou --model.")
    if experiment_name:
        experiment_dir = EXPERIMENTS_DIR / experiment_name
        model = Path(model_path) if model_path else experiment_dir / "models" / "winner.pkl"
        config = Path(config_path) if config_path else experiment_dir / "config" / "neat-config.ini"
        return model.resolve(), config.resolve(), experiment_name

    model = Path(model_path).expanduser().resolve()  # type: ignore[arg-type]
    if config_path:
        config = Path(config_path).expanduser().resolve()
    else:
        inferred = model.parent.parent / "config" / "neat-config.ini"
        config = inferred if inferred.exists() else DEFAULT_CONFIG
    return model, config.resolve(), model.stem


def _validate_paths(config_path: Path, model_path: Path) -> None:
    if not model_path.is_file():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuração não encontrada: {config_path}. Use --config CAMINHO."
        )


def _validate_neat_config(config) -> None:
    num_inputs = config.genome_config.num_inputs
    num_outputs = config.genome_config.num_outputs
    if num_inputs <= 0:
        raise RuntimeError(
            f"A configuração possui uma quantidade inválida de entradas: {num_inputs}."
        )
    if num_outputs != EXPECTED_OUTPUTS:
        raise RuntimeError(
            f"A configuração possui {num_outputs} saídas; esperado: {EXPECTED_OUTPUTS}."
        )


def _reset_session(env, sensors: ExperimentSensorSystem, network) -> Session:
    observation, info = env.reset(seed=SEED)
    sensors.reset()
    if hasattr(network, "reset"):
        network.reset()
    initial_x = int(info.get("x_pos", 0))
    return Session(observation, info, initial_x, initial_x, initial_x)


def _advance_one_frame(
    session: Session,
    env,
    sensors: ExperimentSensorSystem,
    network,
    action_repeat: int = ACTION_REPEAT,
) -> None:
    if session.repeat_left <= 0:
        session.inputs = sensors.observe(session.info)
        session.perception = sensors.last_matrix
        # Não reduzir para float32: diferenças muito pequenas entre saídas
        # saturadas fazem parte da decisão aprendida pelo genome.
        session.outputs = np.asarray(network.activate(session.inputs), dtype=float)
        if session.outputs.size != EXPECTED_OUTPUTS:
            raise RuntimeError(
                f"A rede retornou {session.outputs.size} saídas; esperado: {EXPECTED_OUTPUTS}."
            )
        # Mantém exatamente a mesma política usada no treinamento e no
        # visualizador original: executa a saída de maior ativação da rede.
        session.action = int(np.argmax(session.outputs))
        session.repeat_left = action_repeat
        session.decisions += 1

    observation, reward, terminated, truncated, info = env.step(session.action)
    session.observation = observation
    session.info = info
    sensors.set_previous_action(session.action)
    session.frames += 1
    session.repeat_left -= 1
    session.total_reward += float(reward)
    session.current_x = int(info.get("x_pos", session.max_x))

    if session.current_x > session.max_x:
        session.max_x = session.current_x
        session.stagnation = 0
    else:
        session.stagnation += 1

    session.clear = _is_level_clear(info)
    session.death = _is_dead(info)
    session.stuck = session.stagnation >= MAX_STAGNATION_FRAMES
    session.finished = bool(
        session.clear or session.death or session.stuck or terminated
        or truncated or session.frames >= MAX_FRAMES
    )
    if session.clear:
        session.status = "Fase concluída"
    elif session.death:
        session.status = "Mario morreu"
    elif session.stuck:
        session.status = "Encerrado por estagnação"
    elif terminated or truncated:
        session.status = "Episódio encerrado"
    elif session.frames >= MAX_FRAMES:
        session.status = "Limite de frames atingido"


def _render_dashboard(
    session: Session, state: ViewerState, genome, config, network, display_name: str
) -> np.ndarray:
    canvas = np.full((WINDOW_HEIGHT, WINDOW_WIDTH, 3), BACKGROUND, dtype=np.uint8)
    _text(canvas, "MARIO NEAT • PAINEL DO AGENTE", (22, 36), 0.72, TEXT, 2)
    _text(canvas, display_name, (420, 36), 0.55, MUTED)

    game = cv2.cvtColor(session.observation, cv2.COLOR_RGB2BGR)
    gx, gy, gw, gh = GAME_RECT
    game = cv2.resize(game, (gw, gh), interpolation=cv2.INTER_NEAREST)
    canvas[gy : gy + gh, gx : gx + gw] = game
    cv2.rectangle(canvas, (gx, gy), (gx + gw, gy + gh), PANEL_BORDER, 2)
    _draw_game_overlay(canvas, session, state)

    rows, columns = session.perception.shape
    perception_title = (
        f"PERCEPÇÃO {rows} × {columns}"
        if rows and columns
        else "PERCEPÇÃO NÃO ESPACIAL"
    )
    _panel(canvas, (812, 72, 606, 250), perception_title)
    _draw_sensor_grid(canvas, session.perception, (834, 112))
    _draw_metrics(canvas, session, (1040, 116))

    _panel(canvas, (812, 336, 606, 332), "DECISÕES DA REDE • ATIVAÇÃO AO VIVO")
    _draw_network(canvas, genome, config, network, session, (830, 376, 568, 270))

    _panel(canvas, (812, 682, 606, 110), "CONTROLES")
    _draw_controls(canvas, state)
    _text(
        canvas,
        "Espaço pausa  •  R reinicia  •  N avança  •  Esc sai",
        (824, 834), 0.42, MUTED,
    )
    return canvas


def _draw_game_overlay(canvas: np.ndarray, session: Session, state: ViewerState) -> None:
    gx, gy, gw, _ = GAME_RECT
    cv2.rectangle(canvas, (gx, gy), (gx + gw, gy + 43), (10, 13, 20), -1)
    status = "PAUSADO" if state.paused else session.status.upper()
    color = ACCENT if state.paused else POSITIVE
    _text(canvas, status, (gx + 14, gy + 28), 0.55, color, 2)
    _text(
        canvas,
        f"X {session.current_x}   RECORDE {session.max_x}   FRAME {session.frames}",
        (gx + 280, gy + 28), 0.47, TEXT,
    )
    if session.finished:
        cv2.rectangle(canvas, (gx + 145, gy + 310), (gx + 625, gy + 420), (12, 15, 22), -1)
        cv2.rectangle(canvas, (gx + 145, gy + 310), (gx + 625, gy + 420), ACCENT, 2)
        _text(canvas, session.status.upper(), (gx + 178, gy + 355), 0.75, TEXT, 2)
        _text(canvas, "Clique em REINICIAR ou pressione R", (gx + 205, gy + 393), 0.5, MUTED)


def _draw_sensor_grid(
    canvas: np.ndarray, matrix: np.ndarray, origin: tuple[int, int]
) -> None:
    ox, oy = origin
    rows, columns = matrix.shape
    if not rows or not columns:
        _text(canvas, "Este agente não usa matriz espacial.", (834, 180), 0.5, MUTED)
        return
    cell = min(28, 182 // max(rows, columns))
    for row in range(rows):
        for col in range(columns):
            value = float(matrix[row, col])
            color = (45, 51, 63)
            if value > 0.5:
                color = (100, 183, 238)
            elif value < -0.5:
                color = NEGATIVE
            x, y = ox + col * cell, oy + row * cell
            cv2.rectangle(canvas, (x, y), (x + cell - 2, y + cell - 2), color, -1)
    mario_row, mario_col = ((6, 6) if rows == 13 else (2, 1))
    cx, cy = ox + mario_col * cell, oy + mario_row * cell
    cv2.rectangle(canvas, (cx, cy), (cx + cell - 2, cy + cell - 2), POSITIVE, 2)
    _text(canvas, "bloco • inimigo • posição do Mario", (834, 311), 0.36, MUTED)


def _draw_metrics(
    canvas: np.ndarray, session: Session, origin: tuple[int, int]
) -> None:
    x, y = origin
    action = "+".join(SIMPLE_MOVEMENT[session.action])
    rows = [
        ("AÇÃO", action),
        ("DISTÂNCIA", str(session.max_x - session.initial_x)),
        ("RECOMPENSA", f"{session.total_reward:.1f}"),
        ("DECISÕES", str(session.decisions)),
        ("ESTAGNAÇÃO", f"{session.stagnation}/{MAX_STAGNATION_FRAMES}"),
    ]
    for label, value in rows:
        _text(canvas, label, (x, y), 0.34, MUTED)
        _text(canvas, value, (x, y + 21), 0.5, TEXT)
        y += 34


def _draw_network(
    canvas: np.ndarray, genome, config, network, session: Session,
    rect: tuple[int, int, int, int],
) -> None:
    x, y, w, _h = rect
    _text(
        canvas,
        f"{config.genome_config.num_inputs} entradas • "
        f"{'feed-forward' if config.genome_config.feed_forward else 'recorrente'}"
        f"{' • sensores históricos' if config.genome_config.num_inputs != EXPECTED_INPUTS else ''}",
        (x, y + 4),
        0.38,
        MUTED,
    )

    for index, movement in enumerate(SIMPLE_MOVEMENT):
        row_y = y + 25 + index * 32
        selected = index == session.action
        if selected:
            cv2.rectangle(canvas, (x, row_y - 18), (x + w, row_y + 10), (48, 83, 75), -1)
            cv2.rectangle(canvas, (x, row_y - 18), (x + 5, row_y + 10), POSITIVE, -1)

        label = "+".join(movement)
        value = float(session.outputs[index])
        _text(canvas, label, (x + 16, row_y + 1), 0.45, TEXT, 1)
        _text(canvas, f"{value:+.5f}", (x + 178, row_y + 1), 0.4, MUTED)

        bar_x = x + 292
        bar_width = w - 310
        magnitude = int(bar_width * min(1.0, abs(value)))
        cv2.rectangle(
            canvas,
            (bar_x, row_y - 9),
            (bar_x + bar_width, row_y + 3),
            (45, 51, 63),
            -1,
        )
        cv2.rectangle(
            canvas,
            (bar_x, row_y - 9),
            (bar_x + magnitude, row_y + 3),
            _activation_color(value),
            -1,
        )


def _activation_color(value: float) -> tuple[int, int, int]:
    intensity = min(1.0, abs(float(value)))
    base = POSITIVE if value >= 0 else NEGATIVE
    return tuple(int(45 * (1 - intensity) + channel * intensity) for channel in base)


def _draw_controls(canvas: np.ndarray, state: ViewerState) -> None:
    specs = [
        (
            "pause",
            "CONTINUAR [ESPACO]" if state.paused else "PAUSAR [ESPACO]",
            828,
            724,
            168,
        ),
        ("step", "PASSO [N]", 1006, 724, 116),
        ("restart", "REINICIAR [R]", 1132, 724, 142),
        ("quit", "SAIR [ESC]", 1284, 724, 112),
    ]
    state.buttons.clear()
    for name, label, x, y, width in specs:
        state.buttons[name] = (x, y, width, 42)
        fill = (45, 53, 67) if name != "restart" else (48, 83, 75)
        cv2.rectangle(canvas, (x, y), (x + width, y + 42), fill, -1)
        cv2.rectangle(canvas, (x, y), (x + width, y + 42), PANEL_BORDER, 1)
        size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
        _text(canvas, label, (x + (width - size[0]) // 2, y + 26), 0.38, TEXT)


def _on_mouse(event: int, x: int, y: int, _flags: int, state: ViewerState) -> None:
    if event != cv2.EVENT_LBUTTONUP:
        return
    x, y = _canvas_point(x, y, state.display_size)
    for name, (bx, by, bw, bh) in state.buttons.items():
        if bx <= x <= bx + bw and by <= y <= by + bh:
            _run_control(name, state)
            break


def _handle_key(key: int, state: ViewerState) -> None:
    if key in (27, ord("q")):
        state.quit_requested = True
    elif key == ord(" "):
        _run_control("pause", state)
    elif key in (ord("r"), ord("R")):
        _run_control("restart", state)
    elif key in (ord("n"), ord("N")):
        _run_control("step", state)


def _run_control(name: str, state: ViewerState) -> None:
    if name == "pause":
        state.paused = not state.paused
    elif name == "restart":
        state.restart_requested = True
    elif name == "step":
        state.paused = True
        state.step_requested = True
    elif name == "quit":
        state.quit_requested = True


def _panel(canvas: np.ndarray, rect: tuple[int, int, int, int], title: str) -> None:
    x, y, w, h = rect
    cv2.rectangle(canvas, (x, y), (x + w, y + h), PANEL, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), PANEL_BORDER, 1)
    _text(canvas, title, (x + 16, y + 27), 0.48, TEXT)


def _text(
    image: np.ndarray, value: str, origin: tuple[int, int], scale: float,
    color: tuple[int, int, int], thickness: int = 1,
) -> None:
    cv2.putText(
        image, value, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
        color, thickness, cv2.LINE_AA,
    )


def _is_level_clear(info: dict) -> bool:
    return bool(info.get("flag_get", False) or info.get("clear", False))


def _is_dead(info: dict) -> bool:
    return bool(
        info.get("death", False) or info.get("is_dead", False)
        or info.get("is_dying", False)
    )


def _print_result(name: str, genome, session: Session) -> None:
    print("\n" + "=" * 64)
    print(f"RESULTADO • {name}")
    print("=" * 64)
    print(f"Fitness salvo: {getattr(genome, 'fitness', 'não informado')}")
    print(f"Status: {session.status}")
    print(f"Distância: {session.max_x - session.initial_x}")
    print(f"Frames: {session.frames} | Decisões: {session.decisions}")
    print(f"Recompensa acumulada: {session.total_reward:.2f}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualiza um agente NEAT jogando e sua atividade neural."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--experiment", help="Nome em data/experiments.")
    source.add_argument("--model", type=Path, help="Caminho para um winner.pkl.")
    parser.add_argument(
        "--config", type=Path,
        help="neat-config.ini (inferido automaticamente ao usar a estrutura do projeto).",
    )
    parser.add_argument("--env", default=DEFAULT_ENV_ID, help="ID do ambiente Gymnasium.")
    parser.epilog = "Sem argumentos, abre o seletor visual de winner.pkl."
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    watch(args.experiment, args.model, args.config, args.env)


if __name__ == "__main__":
    main()
