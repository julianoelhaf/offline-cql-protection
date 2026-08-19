"""Build the paper's dense-versus-terminal action-sequence figure."""

from __future__ import annotations

import argparse
import ast
import hashlib
import html
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_CONFIG = {
    "mode": "combined",
    "window_W": 48,
    "alpha": 0.5,
    "correct_reward": 5.0,
    "false_positive_penalty": -100.0,
    "training_seed": 0,
    "epochs": 30,
}
EXPECTED_COUNTS = {
    "fault": {"correct_first_trip": 210, "wrong_relay_first_trip": 1,
              "premature_trip": 0, "no_trip": 3},
    "nonfault": {"false_first_trip": 8, "no_trip": 3},
}


@dataclass
class Context:
    run_dir: Path
    split_path: Path
    settings_path: Path
    config: dict
    constants: dict
    settings: pd.DataFrame
    split_ids: np.ndarray
    sim_ids: np.ndarray
    actions: np.ndarray
    t_start: int
    episodes: list[dict]
    counts: dict


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repo_path(path: Path, repo: Path) -> str:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def load_constants(path: Path) -> dict:
    """Read literal action/timing definitions without importing path-sensitive code."""
    wanted = {
        "SAMPLE_RATE", "N_STEPS", "LINE_TO_ACTION", "WAIT",
        "BUS_TO_UPSTREAM_LINE", "FAULT_TYPES",
    }
    values = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in wanted:
                try:
                    values[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    missing = wanted - values.keys()
    if missing:
        raise ValueError(f"could not read constants from {path}: {sorted(missing)}")
    values["ACTION_TO_LINE"] = {
        action: line for line, action in values["LINE_TO_ACTION"].items()
    }
    return values


def parse_onset_index(value: object, sample_rate: int) -> int:
    seconds = float(str(value).strip().replace(",", "."))
    return round(seconds * sample_rate)


def expected_fault_action(event_target: str, constants: dict) -> tuple[int, str]:
    line = event_target
    if line not in constants["LINE_TO_ACTION"]:
        line = constants["BUS_TO_UPSTREAM_LINE"].get(event_target)
    if line not in constants["LINE_TO_ACTION"]:
        raise ValueError(f"no relay mapping for fault target {event_target!r}")
    return constants["LINE_TO_ACTION"][line], line


def validate_run_config(config: dict, run_dir: Path) -> Path:
    mismatches = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in EXPECTED_CONFIG.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "run configuration is not the paper default:\n"
            + json.dumps(mismatches, indent=2, sort_keys=True)
        )
    checkpoint = run_dir / f"epoch_{config['epochs']:02d}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"final checkpoint not found: {checkpoint}")
    return checkpoint


def _action_name(action: int, constants: dict) -> str:
    if action == constants["WAIT"]:
        return "WAIT"
    line = constants["ACTION_TO_LINE"].get(action)
    if line is None:
        raise ValueError(f"unknown predicted action index {action}")
    return f"TRIP {line}"


def _classify(
    actions: np.ndarray,
    *,
    t_start: int,
    onset: int,
    is_fault: bool,
    expected_action: int,
    wait_action: int,
) -> dict:
    trips = np.flatnonzero(actions != wait_action)
    if not len(trips):
        return {"outcome": "no_trip", "first_trip_index": None,
                "first_action": None}
    offset = int(trips[0])
    index = t_start + offset
    action = int(actions[offset])
    if not is_fault:
        outcome = "false_first_trip"
    elif index < onset:
        outcome = "premature_trip"
    elif action == expected_action:
        outcome = "correct_first_trip"
    else:
        outcome = "wrong_relay_first_trip"
    return {
        "outcome": outcome,
        "first_trip_index": index,
        "first_action": action,
    }


def load_context(
    run_dir: Path,
    split_path: Path,
    settings_path: Path,
    constants_path: Path,
) -> Context:
    run_dir = run_dir.resolve()
    split_path = split_path.resolve()
    settings_path = settings_path.resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    validate_run_config(config, run_dir)
    constants = load_constants(constants_path)

    settings = pd.read_csv(settings_path, sep=";").set_index("general/sim_idx")
    with np.load(split_path) as split:
        key = "indices" if "indices" in split.files else "sim_idx"
        split_ids = split[key].astype(int)

    prediction_path = run_dir / "raw_predictions.npz"
    with np.load(prediction_path) as predictions:
        sim_ids = predictions["sim_idx"].astype(int)
        t_start = int(predictions["t_start"])
        q_values = predictions["q_values"]
        if not np.isfinite(q_values).all():
            raise ValueError(f"non-finite Q-values in {prediction_path}")
        actions = q_values.argmax(axis=2).astype(np.int16)
        saved_actions = predictions["actions"].astype(np.int16)
    if not np.array_equal(actions, saved_actions):
        raise ValueError("saved actions disagree with argmax over saved Q-values")
    if not np.array_equal(sim_ids, split_ids):
        raise ValueError("prediction episode order does not match held-out split")
    if len(np.unique(sim_ids)) != len(sim_ids):
        raise ValueError("held-out split contains duplicate simulation IDs")

    episodes = []
    for row_index, (sim_id, episode_actions) in enumerate(zip(sim_ids, actions)):
        row = settings.loc[int(sim_id)]
        event_type = str(row["events/event_type"])
        event_target = str(row["events/event_target"])
        is_fault = event_type in constants["FAULT_TYPES"]
        onset = parse_onset_index(
            row["events/event_start"], constants["SAMPLE_RATE"]
        )
        expected_action, faulted_line = (
            expected_fault_action(event_target, constants)
            if is_fault else (constants["WAIT"], None)
        )
        terminal = _classify(
            episode_actions,
            t_start=t_start,
            onset=onset,
            is_fault=is_fault,
            expected_action=expected_action,
            wait_action=constants["WAIT"],
        )
        episodes.append({
            "row_index": row_index,
            "simulation_id": int(sim_id),
            "event_type": event_type,
            "event_target": event_target,
            "faulted_line": faulted_line,
            "is_fault": is_fault,
            "onset_index": onset,
            "expected_action": expected_action,
            **terminal,
        })

    fault = [episode for episode in episodes if episode["is_fault"]]
    nonfault = [episode for episode in episodes if not episode["is_fault"]]
    counts = {
        "fault": {
            outcome: sum(e["outcome"] == outcome for e in fault)
            for outcome in EXPECTED_COUNTS["fault"]
        },
        "nonfault": {
            outcome: sum(e["outcome"] == outcome for e in nonfault)
            for outcome in EXPECTED_COUNTS["nonfault"]
        },
    }
    if len(fault) != 214 or len(nonfault) != 11 or counts != EXPECTED_COUNTS:
        raise RuntimeError(
            "aggregate terminal results do not match the paper\n"
            f"configuration={json.dumps(config, sort_keys=True)}\n"
            f"fault_episodes={len(fault)}, nonfault_episodes={len(nonfault)}\n"
            f"counts={json.dumps(counts, sort_keys=True)}"
        )

    return Context(
        run_dir, split_path, settings_path, config, constants, settings,
        split_ids, sim_ids, actions, t_start, episodes, counts,
    )


def select_examples(context: Context) -> tuple[dict, dict]:
    wait = context.constants["WAIT"]

    fault_candidates = []
    for episode in context.episodes:
        if not episode["is_fault"]:
            continue
        actions = context.actions[episode["row_index"]]
        trip_offsets = np.flatnonzero(actions != wait)
        if not len(trip_offsets):
            continue
        first_offset = int(trip_offsets[0])
        first_index = context.t_start + first_offset
        later_wrong = np.flatnonzero(
            (np.arange(len(actions)) > first_offset)
            & (actions != wait)
            & (actions != episode["expected_action"])
        )
        if (
            first_index >= episode["onset_index"]
            and int(actions[first_offset]) == episode["expected_action"]
            and len(later_wrong)
        ):
            fault_candidates.append({
                **episode,
                "later_wrong_indices": (
                    context.t_start + later_wrong.astype(int)
                ).tolist(),
            })
    if not fault_candidates:
        raise RuntimeError("no fault episode satisfies the figure criteria")
    # Simulation 764 was the originally preferred fault panel, but it does not
    # satisfy the candidate criteria above, so the deterministic fallback
    # (lowest qualifying simulation id) selects simulation 655 - the episode
    # shown in the paper. tests/test_terminal_action_examples.py pins this.
    fault_example = next(
        (episode for episode in fault_candidates
         if episode["simulation_id"] == 764),
        min(fault_candidates, key=lambda episode: episode["simulation_id"]),
    )

    nonfault_candidates = []
    for episode in context.episodes:
        if episode["is_fault"]:
            continue
        actions = context.actions[episode["row_index"]]
        n_trips = int(np.count_nonzero(actions != wait))
        if n_trips and n_trips < len(actions) - n_trips:
            nonfault_candidates.append({**episode, "n_nuisance_steps": n_trips})
    if not nonfault_candidates:
        raise RuntimeError("no non-fault episode satisfies the figure criteria")
    nonfault_example = next(
        (episode for episode in nonfault_candidates
         if episode["simulation_id"] == 4449),
        min(
            nonfault_candidates,
            key=lambda episode: (
                episode["n_nuisance_steps"], episode["simulation_id"]
            ),
        ),
    )

    split_ids = set(context.split_ids.tolist())
    development_ids = set(context.settings.index.astype(int)) - split_ids
    for example in (fault_example, nonfault_example):
        assert example["simulation_id"] in split_ids
        assert example["simulation_id"] not in development_ids
        assert example["onset_index"] == round(
            0.1 * context.constants["SAMPLE_RATE"]
        )
        actions = context.actions[example["row_index"]]
        trips = np.flatnonzero(actions != wait)
        assert example["first_trip_index"] == context.t_start + int(trips[0])
    assert fault_example["outcome"] == "correct_first_trip"
    assert fault_example["later_wrong_indices"]
    assert nonfault_example["outcome"] == "false_first_trip"
    assert nonfault_example["n_nuisance_steps"] > 0
    return fault_example, nonfault_example


def build_figure_data(
    context: Context, fault_example: dict, nonfault_example: dict
) -> pd.DataFrame:
    rows = []
    wait = context.constants["WAIT"]
    sample_rate = context.constants["SAMPLE_RATE"]
    for episode_type, example in (
        ("fault", fault_example), ("non-fault", nonfault_example)
    ):
        actions = context.actions[example["row_index"]]
        for offset, predicted in enumerate(actions):
            sample_index = context.t_start + offset
            if example["is_fault"] and sample_index >= example["onset_index"]:
                expected = example["expected_action"]
            else:
                expected = wait
            if example["is_fault"]:
                category = (
                    "WAIT" if predicted == wait
                    else "correct line"
                    if predicted == example["expected_action"]
                    else "other line"
                )
            else:
                category = "WAIT" if predicted == wait else "TRIP"
            rows.append({
                "simulation_id": example["simulation_id"],
                "episode_type": episode_type,
                "time_ms": sample_index / sample_rate * 1000,
                "predicted_action_index": int(predicted),
                "predicted_action_name": _action_name(
                    int(predicted), context.constants
                ),
                "plot_category": category,
                "expected_action_index": int(expected),
                "expected_action_name": _action_name(
                    int(expected), context.constants
                ),
                "is_event_onset": sample_index == example["onset_index"],
                "is_first_trip": sample_index == example["first_trip_index"],
                "ignored_after_first_trip":
                    sample_index > example["first_trip_index"],
            })
    frame = pd.DataFrame(rows)
    for example in (fault_example, nonfault_example):
        selected = frame[frame["simulation_id"] == example["simulation_id"]]
        first = selected.index[selected["is_first_trip"]]
        assert len(first) == 1
        first_position = selected.index.get_loc(first[0])
        assert not selected.iloc[:first_position]["ignored_after_first_trip"].any()
        assert selected.iloc[first_position + 1:]["ignored_after_first_trip"].all()
        assert len(selected) == context.actions.shape[1]
    return frame


def _step_path(
    times: np.ndarray,
    values: np.ndarray,
    *,
    x0: float,
    width: float,
    y_for: dict,
    left: float,
    right: float,
) -> str:
    mask = (times >= left) & (times <= right)
    t = times[mask]
    v = values[mask]
    if not len(t):
        raise ValueError("plot interval contains no predictions")
    points = [(t[0], v[0])]
    for time, old, new in zip(t[1:], v[:-1], v[1:]):
        if new != old:
            points.extend([(time, old), (time, new)])
    points.append((t[-1], v[-1]))
    scale = lambda time: x0 + (time - left) / (right - left) * width
    return " ".join(
        ("M" if index == 0 else "L")
        + f"{scale(time):.2f},{y_for[value]:.2f}"
        for index, (time, value) in enumerate(points)
    )


def _tex_escape(text: object) -> str:
    return str(text).replace("\\", r"\textbackslash{}").replace("_", r"\_")


def _tikz_coordinates(
    times: np.ndarray,
    values: np.ndarray,
    *,
    y_for: dict[str, int],
    left: float,
    right: float,
) -> str:
    mask = (times >= left) & (times <= right)
    t = times[mask]
    v = values[mask]
    points = [(float(t[0]), y_for[v[0]])]
    for time, old, new in zip(t[1:], v[:-1], v[1:]):
        if new != old:
            points.extend([
                (float(time), y_for[old]),
                (float(time), y_for[new]),
            ])
    points.append((float(t[-1]), y_for[v[-1]]))
    return " ".join(f"({time:.4f},{value})" for time, value in points)


def write_tikz(
    frame: pd.DataFrame,
    fault_example: dict,
    nonfault_example: dict,
    sample_rate: int,
    path: Path,
) -> None:
    fault = frame[frame["simulation_id"] == fault_example["simulation_id"]]
    nonfault = frame[
        frame["simulation_id"] == nonfault_example["simulation_id"]
    ]
    first_fault = fault_example["first_trip_index"] / sample_rate * 1000
    first_nonfault = nonfault_example["first_trip_index"] / sample_rate * 1000
    onset_fault = fault_example["onset_index"] / sample_rate * 1000
    onset_nonfault = nonfault_example["onset_index"] / sample_rate * 1000
    fault_expected = np.where(
        fault["expected_action_name"].to_numpy(str) == "WAIT",
        "WAIT", "correct line",
    )
    nonfault_expected = np.full(len(nonfault), "WAIT")
    fault_title = (
        rf"\scriptsize sim\_idx={fault_example['simulation_id']} | "
        + rf"{_tex_escape(fault_example['event_type'])} on "
        + rf"{_tex_escape(fault_example['event_target'])} | zoom [80,120] ms"
    )
    nonfault_title = (
        rf"\scriptsize sim\_idx={nonfault_example['simulation_id']} | "
        + rf"{_tex_escape(nonfault_example['event_type'])} on "
        + rf"{_tex_escape(nonfault_example['event_target'])} | zoom [80,120] ms"
    )
    fault_agent = _tikz_coordinates(
        fault["time_ms"].to_numpy(float),
        fault["plot_category"].to_numpy(str),
        y_for={"WAIT": 0, "correct line": 1, "other line": 2},
        left=80, right=120,
    )
    fault_required = _tikz_coordinates(
        fault["time_ms"].to_numpy(float),
        fault_expected,
        y_for={"WAIT": 0, "correct line": 1},
        left=80, right=120,
    )
    nonfault_agent = _tikz_coordinates(
        nonfault["time_ms"].to_numpy(float),
        nonfault["plot_category"].to_numpy(str),
        y_for={"WAIT": 0, "TRIP": 1},
        left=80, right=120,
    )
    nonfault_required = _tikz_coordinates(
        nonfault["time_ms"].to_numpy(float),
        nonfault_expected,
        y_for={"WAIT": 0},
        left=80, right=120,
    )
    tikz = rf"""% Generated by make_terminal_action_examples.py; do not edit.
\begin{{tikzpicture}}
\begin{{groupplot}}[
  group style={{group size=2 by 1, horizontal sep=1.0cm}},
  width=0.46\textwidth,
  height=4.6cm,
  xmin=80, xmax=120,
  xtick={{80,90,100,110,120}},
  xlabel={{Time from episode start (ms)}},
  tick label style={{font=\scriptsize}},
  label style={{font=\scriptsize}},
  title style={{
    font=\small,
    align=center,
    at={{(axis description cs:0.5,1.19)}},
    anchor=south
  }},
  xmajorgrids,
  ymajorgrids,
  grid style={{black!12,line width=0.2pt}},
  axis line style={{black!65}},
  axis on top,
  legend style={{
    font=\scriptsize,
    draw=black!25,
    fill=white,
    at={{(0.5,1.02)}},
    anchor=south,
    legend columns=2,
    inner sep=2pt,
    column sep=5pt
  }},
]
\nextgroupplot[
  title={{{fault_title}}},
  ymin=-0.25, ymax=2.25,
  ytick={{0,1,2}},
  yticklabels={{WAIT,Correct line,Other line}},
  ylabel={{Action category}},
]
\fill[black!8] (axis cs:{first_fault:.4f},-0.25)
  rectangle (axis cs:120,2.25);
\addlegendimage{{blue!70!black,line width=1.0pt}}
\addlegendentry{{Agent}}
\addlegendimage{{orange!85!black,dashed,line width=0.9pt}}
\addlegendentry{{Required action}}
\addplot[orange!85!black,dashed,line width=0.9pt]
  coordinates {{{fault_required}}};
\addplot[blue!70!black,line width=1.0pt]
  coordinates {{{fault_agent}}};
\draw[red!75!black,dashed,line width=0.8pt]
  (axis cs:{onset_fault:.4f},-0.25) -- (axis cs:{onset_fault:.4f},2.25);
\draw[green!45!black,dash dot,line width=0.8pt]
  (axis cs:{first_fault:.4f},-0.25) -- (axis cs:{first_fault:.4f},2.25);
\addplot[green!45!black,only marks,mark=diamond*,
  mark options={{fill=white}},mark size=2.4pt]
  coordinates {{({first_fault:.4f},1)}};
\node[font=\scriptsize,anchor=north,fill=white,fill opacity=0.72,
  text opacity=1,inner sep=1pt]
  at (axis cs:{(first_fault + 120) / 2:.4f},2.16)
  {{ignored after first trip}};

\nextgroupplot[
  title={{{nonfault_title}}},
  ymin=-0.18, ymax=1.18,
  ytick={{0,1}},
  yticklabels={{WAIT,TRIP}},
  ylabel={{Action category}},
]
\fill[black!8] (axis cs:{first_nonfault:.4f},-0.18)
  rectangle (axis cs:120,1.18);
\addlegendimage{{blue!70!black,line width=1.0pt}}
\addlegendentry{{Agent}}
\addlegendimage{{orange!85!black,dashed,line width=0.9pt}}
\addlegendentry{{Required action}}
\addplot[orange!85!black,dashed,line width=0.9pt]
  coordinates {{{nonfault_required}}};
\addplot[blue!70!black,line width=1.0pt]
  coordinates {{{nonfault_agent}}};
\draw[red!75!black,dashed,line width=0.8pt]
  (axis cs:{onset_nonfault:.4f},-0.18) -- (axis cs:{onset_nonfault:.4f},1.18);
\draw[green!45!black,dash dot,line width=0.8pt]
  (axis cs:{first_nonfault:.4f},-0.18) -- (axis cs:{first_nonfault:.4f},1.18);
\addplot[green!45!black,only marks,mark=diamond*,
  mark options={{fill=white}},mark size=2.4pt]
  coordinates {{({first_nonfault:.4f},1)}};
\node[font=\scriptsize,anchor=north,fill=white,fill opacity=0.72,
  text opacity=1,inner sep=1pt]
  at (axis cs:{(first_nonfault + 120) / 2:.4f},1.10)
  {{ignored after first trip}};
\end{{groupplot}}
\node[font=\small,anchor=north,align=center]
  at ([yshift=-0.95cm]group c1r1.south)
  {{\textbf{{(a)}} Correct first trip in a fault episode}};
\node[font=\small,anchor=north,align=center]
  at ([yshift=-0.95cm]group c2r1.south)
  {{\textbf{{(b)}} Nuisance trip in a non-fault episode}};
\end{{tikzpicture}}
"""
    path.write_text(tikz, encoding="utf-8")


def write_tikz_panels(
    frame: pd.DataFrame,
    fault_example: dict,
    nonfault_example: dict,
    sample_rate: int,
    fault_path: Path,
    nonfault_path: Path,
) -> None:
    def panel(
        rows: pd.DataFrame,
        example: dict,
        *,
        categories: dict[str, int],
        ymin: float,
        ymax: float,
        yticklabels: str,
        path: Path,
    ) -> None:
        first_trip = example["first_trip_index"] / sample_rate * 1000
        onset = example["onset_index"] / sample_rate * 1000
        expected = np.where(
            rows["expected_action_name"].to_numpy(str) == "WAIT",
            "WAIT",
            "correct line" if example["is_fault"] else "TRIP",
        )
        predicted_coordinates = _tikz_coordinates(
            rows["time_ms"].to_numpy(float),
            rows["plot_category"].to_numpy(str),
            y_for=categories,
            left=80,
            right=120,
        )
        expected_coordinates = _tikz_coordinates(
            rows["time_ms"].to_numpy(float),
            expected,
            y_for=categories,
            left=80,
            right=120,
        )
        first_category = (
            "correct line" if example["is_fault"] else "TRIP"
        )
        content = rf"""% Generated by make_terminal_action_examples.py; do not edit.
\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.92\linewidth,
  height=4.2cm,
  xmin=80, xmax=120,
  ymin={ymin}, ymax={ymax},
  xtick={{80,90,100,110,120}},
  ytick={{{",".join(str(value) for value in categories.values())}}},
  yticklabels={{{yticklabels}}},
  xlabel={{Time from episode start (ms)}},
  ylabel={{Action category}},
  tick label style={{font=\scriptsize}},
  label style={{font=\scriptsize}},
  xmajorgrids,
  ymajorgrids,
  grid style={{black!12,line width=0.2pt}},
  axis line style={{black!65}},
  axis on top,
]
\fill[black!8] (axis cs:{first_trip:.4f},{ymin})
  rectangle (axis cs:120,{ymax});
\addplot[orange!85!black,dashed,line width=0.9pt]
  coordinates {{{expected_coordinates}}};
\addplot[blue!70!black,line width=1.0pt]
  coordinates {{{predicted_coordinates}}};
\draw[red!75!black,dashed,line width=0.8pt]
  (axis cs:{onset:.4f},{ymin}) -- (axis cs:{onset:.4f},{ymax});
\draw[green!45!black,dash dot,line width=0.8pt]
  (axis cs:{first_trip:.4f},{ymin}) -- (axis cs:{first_trip:.4f},{ymax});
\addplot[green!45!black,only marks,mark=diamond*,
  mark options={{fill=white}},mark size=2.4pt]
  coordinates {{({first_trip:.4f},{categories[first_category]})}};
\end{{axis}}
\end{{tikzpicture}}
"""
        path.write_text(content, encoding="utf-8")

    panel(
        frame[frame["simulation_id"] == fault_example["simulation_id"]],
        fault_example,
        categories={"WAIT": 0, "correct line": 1, "other line": 2},
        ymin=-0.25,
        ymax=2.25,
        yticklabels="WAIT,Correct line,Other line",
        path=fault_path,
    )
    panel(
        frame[frame["simulation_id"] == nonfault_example["simulation_id"]],
        nonfault_example,
        categories={"WAIT": 0, "TRIP": 1},
        ymin=-0.18,
        ymax=1.18,
        yticklabels="WAIT,TRIP",
        path=nonfault_path,
    )


def _svg_text(x: float, y: float, text: str, **attrs: object) -> str:
    settings = {"x": f"{x:.2f}", "y": f"{y:.2f}"}
    settings.update({key.replace("_", "-"): str(value) for key, value in attrs.items()})
    joined = " ".join(f'{key}="{html.escape(value)}"' for key, value in settings.items())
    return f"<text {joined}>{html.escape(text)}</text>"


def _panel_svg(
    frame: pd.DataFrame,
    example: dict,
    *,
    x0: float,
    title: str,
    categories: list[str],
    category_labels: list[str],
    sample_rate: int,
    n_steps: int,
) -> list[str]:
    plot_y, plot_h, plot_w = 38.0, 128.0, 270.0
    plot_bottom = plot_y + plot_h
    times = frame["time_ms"].to_numpy(float)
    predicted = frame["plot_category"].to_numpy(str)
    expected_names = frame["expected_action_name"].to_numpy(str)
    expected = np.array([
        "WAIT" if name == "WAIT"
        else "correct line" if len(categories) == 3
        else "TRIP"
        for name in expected_names
    ])
    y_for = {
        category: plot_bottom - index * plot_h / (len(categories) - 1)
        for index, category in enumerate(categories)
    }
    first_ms = example["first_trip_index"] / sample_rate * 1000
    onset_ms = example["onset_index"] / sample_rate * 1000
    left = max(0.0, min(80.0, onset_ms - 20.0))
    right = min(
        (n_steps - 1) / sample_rate * 1000,
        max(120.0, first_ms + 15.0),
    )
    x = lambda value: x0 + (value - left) / (right - left) * plot_w

    out = [
        _svg_text(x0 + plot_w / 2, 12, title, text_anchor="middle",
                  font_weight="bold"),
        _svg_text(
            x0 + plot_w / 2, 26,
            f"sim {example['simulation_id']} | {example['event_type']} | "
            f"{example['event_target']}",
            text_anchor="middle", font_size="7",
        ),
        f'<rect x="{x(first_ms):.2f}" y="{plot_y:.2f}" '
        f'width="{x(right)-x(first_ms):.2f}" height="{plot_h:.2f}" '
        'fill="#e7e7e7"/>',
    ]
    for category, label in zip(categories, category_labels):
        y = y_for[category]
        out.extend([
            f'<line x1="{x0:.2f}" y1="{y:.2f}" x2="{x0+plot_w:.2f}" '
            f'y2="{y:.2f}" stroke="#c8c8c8" stroke-width="0.45"/>',
            _svg_text(x0 - 6, y + 3, label, text_anchor="end"),
        ])
    ticks = np.linspace(left, right, 5)
    for tick in ticks:
        tx = x(tick)
        out.extend([
            f'<line x1="{tx:.2f}" y1="{plot_bottom:.2f}" '
            f'x2="{tx:.2f}" y2="{plot_bottom+3:.2f}" '
            'stroke="#000" stroke-width="0.6"/>',
            _svg_text(tx, plot_bottom + 13, f"{tick:g}", text_anchor="middle"),
        ])
    out.extend([
        f'<line x1="{x0:.2f}" y1="{plot_y:.2f}" x2="{x0:.2f}" '
        f'y2="{plot_bottom:.2f}" stroke="#000" stroke-width="0.7"/>',
        f'<line x1="{x0:.2f}" y1="{plot_bottom:.2f}" '
        f'x2="{x0+plot_w:.2f}" y2="{plot_bottom:.2f}" '
        'stroke="#000" stroke-width="0.7"/>',
        f'<path d="{_step_path(times, expected, x0=x0, width=plot_w, y_for=y_for, left=left, right=right)}" '
        'fill="none" stroke="#777" stroke-width="1.0" stroke-dasharray="5,3"/>',
        f'<path d="{_step_path(times, predicted, x0=x0, width=plot_w, y_for=y_for, left=left, right=right)}" '
        'fill="none" stroke="#000" stroke-width="1.25"/>',
        f'<line x1="{x(onset_ms):.2f}" y1="{plot_y:.2f}" '
        f'x2="{x(onset_ms):.2f}" y2="{plot_bottom:.2f}" '
        'stroke="#555" stroke-width="0.8" stroke-dasharray="3,2"/>',
        f'<line x1="{x(first_ms):.2f}" y1="{plot_y:.2f}" '
        f'x2="{x(first_ms):.2f}" y2="{plot_bottom:.2f}" '
        'stroke="#000" stroke-width="0.8" stroke-dasharray="1.5,2"/>',
    ])
    first_category = frame.loc[frame["is_first_trip"], "plot_category"].iloc[0]
    fy = y_for[first_category]
    fx = x(first_ms)
    out.append(
        f'<polygon points="{fx:.2f},{fy-4:.2f} {fx+4:.2f},{fy:.2f} '
        f'{fx:.2f},{fy+4:.2f} {fx-4:.2f},{fy:.2f}" '
        'fill="#fff" stroke="#000" stroke-width="0.9"/>'
    )
    out.extend([
        _svg_text(
            (x(first_ms) + x(right)) / 2, plot_y + 10,
            "ignored by terminal evaluation", text_anchor="middle",
            font_size="7",
        ),
        _svg_text(x0 + plot_w / 2, plot_bottom + 27,
                  "Time from episode start (ms)", text_anchor="middle"),
    ])
    return out


def write_svg(
    frame: pd.DataFrame,
    fault_example: dict,
    nonfault_example: dict,
    sample_rate: int,
    n_steps: int,
    path: Path,
) -> None:
    fault = frame[frame["simulation_id"] == fault_example["simulation_id"]]
    nonfault = frame[
        frame["simulation_id"] == nonfault_example["simulation_id"]
    ]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="7.16in" height="2.30in" '
        'viewBox="0 0 716 230">',
        '<rect width="716" height="230" fill="#fff"/>',
        '<g font-family="Arial, Helvetica, sans-serif" font-size="8" '
        'fill="#000" stroke-linejoin="round" stroke-linecap="square">',
    ]
    parts.extend(_panel_svg(
        fault, fault_example, x0=64,
        title="(a) Correct first trip in a fault episode",
        categories=["WAIT", "correct line", "other line"],
        category_labels=["WAIT", "Correct line", "Other line"],
        sample_rate=sample_rate, n_steps=n_steps,
    ))
    parts.extend(_panel_svg(
        nonfault, nonfault_example, x0=418,
        title="(b) Isolated nuisance trip in a non-fault episode",
        categories=["WAIT", "TRIP"],
        category_labels=["WAIT", "TRIP"],
        sample_rate=sample_rate, n_steps=n_steps,
    ))
    legend_y = 222
    parts.extend([
        '<line x1="92" y1="219" x2="112" y2="219" stroke="#000" '
        'stroke-width="1.25"/>',
        _svg_text(117, legend_y, "Prediction"),
        '<line x1="196" y1="219" x2="216" y2="219" stroke="#777" '
        'stroke-width="1" stroke-dasharray="5,3"/>',
        _svg_text(221, legend_y, "Required action"),
        '<line x1="325" y1="211" x2="325" y2="225" stroke="#555" '
        'stroke-width="0.8" stroke-dasharray="3,2"/>',
        _svg_text(331, legend_y, "Event onset"),
        '<polygon points="432,215 436,219 432,223 428,219" fill="#fff" '
        'stroke="#000" stroke-width="0.9"/>',
        _svg_text(442, legend_y, "First non-WAIT action"),
        '<rect x="576" y="213" width="18" height="12" fill="#e7e7e7"/>',
        _svg_text(600, legend_y, "Ignored after trip"),
        "</g>",
        "</svg>",
    ])
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def convert_svg(svg_path: Path, pdf_path: Path, png_path: Path) -> None:
    inkscape = shutil.which("inkscape.com") or shutil.which("inkscape")
    if not inkscape:
        raise RuntimeError("Inkscape is required to export PDF and PNG")
    subprocess.run([
        inkscape, str(svg_path), "--export-type=pdf",
        f"--export-filename={pdf_path}",
    ], check=True)
    subprocess.run([
        inkscape, str(svg_path), "--export-type=png",
        "--export-width=2148", f"--export-filename={png_path}",
    ], check=True)
    for path in (svg_path, pdf_path, png_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"empty figure output: {path}")


def run(args: argparse.Namespace) -> dict:
    repo = Path(__file__).resolve().parents[2]
    context = load_context(
        args.run_dir, args.split, args.settings, args.constants
    )
    fault_example, nonfault_example = select_examples(context)
    frame = build_figure_data(context, fault_example, nonfault_example)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.data_dir / "terminal_action_examples.csv"
    json_path = args.data_dir / "terminal_action_examples_metadata.json"
    svg_path = args.out_dir / "terminal_action_examples.svg"
    pdf_path = args.out_dir / "terminal_action_examples.pdf"
    png_path = args.out_dir / "terminal_action_examples.png"
    tikz_path = args.out_dir / "terminal_action_examples.tex"
    fault_tikz_path = args.out_dir / "terminal_action_fault.tex"
    nonfault_tikz_path = args.out_dir / "terminal_action_nonfault.tex"
    frame.to_csv(csv_path, index=False)
    write_tikz(
        frame, fault_example, nonfault_example,
        context.constants["SAMPLE_RATE"], tikz_path,
    )
    write_tikz_panels(
        frame, fault_example, nonfault_example,
        context.constants["SAMPLE_RATE"],
        fault_tikz_path, nonfault_tikz_path,
    )
    write_svg(
        frame, fault_example, nonfault_example,
        context.constants["SAMPLE_RATE"], context.constants["N_STEPS"], svg_path,
    )
    convert_svg(svg_path, pdf_path, png_path)

    checkpoint = validate_run_config(context.config, context.run_dir)
    action_mapping = {
        str(index): _action_name(index, context.constants)
        for index in sorted(
            [*context.constants["ACTION_TO_LINE"], context.constants["WAIT"]]
        )
    }
    metadata = {
        "run_configuration": {
            **context.config,
            "effective_reward": {
                "correct_fault_trip": 5.0,
                "correct_post_event_nonfault_wait": 5.0,
                "pre_event_or_fault_wait": 0.0,
                "false_positive_or_wrong_line_trip": -100.0,
            },
        },
        "checkpoint": {
            "path": repo_path(checkpoint, repo),
            "sha256": sha256(checkpoint),
        },
        "predictions": {
            "path": repo_path(context.run_dir / "raw_predictions.npz", repo),
            "sha256": sha256(context.run_dir / "raw_predictions.npz"),
            "derived_actions": "argmax over saved Q-values",
        },
        "split": {
            "path": repo_path(context.split_path, repo),
            "sha256": sha256(context.split_path),
            "episodes": len(context.split_ids),
        },
        "settings": {
            "path": repo_path(context.settings_path, repo),
            "sha256": sha256(context.settings_path),
        },
        "action_mapping": action_mapping,
        "selection": {
            "fault": {
                "rule": (
                    "prefer simulation 764 if its first post-onset trip is "
                    "correct and a later trip targets another line; otherwise "
                    "use the lowest qualifying simulation ID"
                ),
                "preferred_simulation_id": 764,
                "preferred_used": fault_example["simulation_id"] == 764,
                "selected_simulation_id": fault_example["simulation_id"],
            },
            "nonfault": {
                "rule": (
                    "prefer simulation 4449 if it contains a nuisance trip "
                    "amid predominantly WAIT predictions; otherwise minimize "
                    "non-WAIT timesteps and then simulation ID"
                ),
                "preferred_simulation_id": 4449,
                "preferred_used": nonfault_example["simulation_id"] == 4449,
                "selected_simulation_id": nonfault_example["simulation_id"],
            },
        },
        "selected_episodes": {
            "fault": {
                key: fault_example[key] for key in (
                    "simulation_id", "event_type", "event_target",
                    "faulted_line", "onset_index", "first_trip_index",
                    "first_action", "outcome", "later_wrong_indices",
                )
            },
            "nonfault": {
                key: nonfault_example[key] for key in (
                    "simulation_id", "event_type", "event_target",
                    "onset_index", "first_trip_index", "first_action",
                    "outcome", "n_nuisance_steps",
                )
            },
        },
        "timing": {
            "sample_rate_hz": context.constants["SAMPLE_RATE"],
            "t_start_index": context.t_start,
            "plot_interval_ms": {
                label: [
                    max(
                        0.0,
                        min(
                            80.0,
                            example["onset_index"]
                            / context.constants["SAMPLE_RATE"] * 1000 - 20.0,
                        ),
                    ),
                    min(
                        (context.constants["N_STEPS"] - 1)
                        / context.constants["SAMPLE_RATE"] * 1000,
                        max(
                            120.0,
                            example["first_trip_index"]
                            / context.constants["SAMPLE_RATE"] * 1000 + 15.0,
                        ),
                    ),
                ]
                for label, example in (
                    ("fault", fault_example), ("nonfault", nonfault_example)
                )
            },
            "fault_onset_time_ms":
                fault_example["onset_index"]
                / context.constants["SAMPLE_RATE"] * 1000,
            "fault_first_trip_time_ms":
                fault_example["first_trip_index"]
                / context.constants["SAMPLE_RATE"] * 1000,
            "nonfault_onset_time_ms":
                nonfault_example["onset_index"]
                / context.constants["SAMPLE_RATE"] * 1000,
            "nonfault_first_trip_time_ms":
                nonfault_example["first_trip_index"]
                / context.constants["SAMPLE_RATE"] * 1000,
        },
        "aggregate_terminal_counts": context.counts,
        "checks": {
            "aggregate_counts_match_paper": context.counts == EXPECTED_COUNTS,
            "selected_episodes_in_held_out_split": True,
            "selected_episodes_not_in_development_pool": True,
            "fault_first_trip_correct": True,
            "fault_has_later_wrong_line_prediction": True,
            "nonfault_has_nuisance_trip": True,
            "post_trip_predictions_retained_and_marked_ignored": True,
            "pdf_nonempty": pdf_path.stat().st_size > 0,
            "svg_nonempty": svg_path.stat().st_size > 0,
            "tikz_nonempty": tikz_path.stat().st_size > 0,
            "independent_tikz_panels_nonempty":
                fault_tikz_path.stat().st_size > 0
                and nonfault_tikz_path.stat().st_size > 0,
        },
    }
    json_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        f"Fault example: sim_id={fault_example['simulation_id']}, "
        f"event={fault_example['event_type']}, "
        f"target={fault_example['event_target']}, "
        f"first_trip={metadata['timing']['fault_first_trip_time_ms']:.3f} ms, "
        "outcome=correct"
    )
    print(
        f"Non-fault example: sim_id={nonfault_example['simulation_id']}, "
        f"event={nonfault_example['event_type']}, "
        f"first_trip={metadata['timing']['nonfault_first_trip_time_ms']:.3f} ms, "
        "outcome=false_trip"
    )
    print(
        "Aggregate check: fault 210 correct / 1 wrong / 3 no-trip; "
        "non-fault 8 false-trip / 3 no-trip"
    )
    return metadata


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    paper = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument(
        "--settings", type=Path, default=repo / "labels" / "settings.csv"
    )
    parser.add_argument(
        "--constants", type=Path,
        default=repo / "rl_protection" / "constants.py",
    )
    parser.add_argument("--out-dir", type=Path, default=paper / "figures")
    parser.add_argument("--data-dir", type=Path, default=paper / "figure_data")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
