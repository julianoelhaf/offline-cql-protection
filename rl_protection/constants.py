"""
constants.py — All fixed values for the power grid RL project.

Grid: CIGRE MV 14-bus benchmark (20 kV, 50 Hz)
Simulation: 4800 steps, 0–0.5 s, dt = 1/9600 s (~104 µs per step)
Fault always starts at t = 0.1 s → step index 960.
"""

import os
import pickle

import pandas as pd

# ── Simulation timing ─────────────────────────────────────────────────────────

SAMPLE_RATE    = 9600          # samples per second (1 / 104.167 µs)
DT             = 1.0 / SAMPLE_RATE   # seconds per step
N_STEPS        = 4800          # total steps per episode (0 to 0.4999 s)
FAULT_ONSET    = 960           # step index where fault turns on (t = 0.1 s)
CYCLE_LEN      = 192           # steps per 50 Hz cycle (= SAMPLE_RATE / 50)
GRID_FREQ      = 50            # Hz

# ── Training Data ─────────────────────────────────────────────────────────────

WINDOW_SIZE = int(CYCLE_LEN * 1)

# ── Action space ──────────────────────────────────────────────────────────────

# Each action trips both circuit breakers of one transmission line segment.
# Action 15 = WAIT (do nothing).
LINE_TO_ACTION = {
    'MainLn1-2':   0,
    'MainLn2-3':   1,
    'MainLn3-4':   2,
    'MainLn3-8':   3,
    'MainLn4-5':   4,
    'MainLn4-11':  5,
    'MainLn5-6':   6,
    'MainLn6-7':   7,
    'MainLn7-8':   8,
    'MainLn8-9':   9,
    'MainLn8-14': 10,
    'MainLn9-10': 11,
    'MainLn10-11':12,
    'MainLn12-13':13,
    'MainLn13-14':14,
}
WAIT           = 15
N_ACTIONS      = 16            # 15 line trips + 1 WAIT
ACTION_TO_LINE = {v: k for k, v in LINE_TO_ACTION.items()}

# For a bus fault: map to the upstream line that feeds that bus.
# "Upstream" = closest line on the shortest path from substation to that bus
# in normal radial operation (Ln8-14 is normally open).
BUS_TO_UPSTREAM_LINE = {
    'MainBus1':  'MainLn10-11',  # Bus1 at Feeder-1 head; Ln10-11 is first outgoing line
    'MainBus2':  'MainLn10-11',  # Bus2 fed from Bus1 via Ln10-11
    'MainBus3':  'MainLn2-3',
    'MainBus4':  'MainLn3-4',
    'MainBus5':  'MainLn4-5',
    'MainBus6':  'MainLn5-6',
    'MainBus7':  'MainLn6-7',
    'MainBus8':  'MainLn3-8',    # Bus8 primary infeed from Bus3 via Ln3-8
    'MainBus9':  'MainLn8-9',
    'MainBus10': 'MainLn9-10',
    'MainBus11': 'MainLn4-11',   # Bus11 primary infeed from Bus4 via Ln4-11
    'MainBus12': 'MainLn13-14',  # Bus12 at Feeder-2 head; Ln13-14 is first outgoing line
    'MainBus13': 'MainLn13-14',  # Bus13 fed from Bus12 via Ln13-14
    'MainBus14': 'MainLn1-2',    # Bus14 fed from Bus13 via Ln1-2
}

# ── Grid scenarios ────────────────────────────────────────────────────────────

# Fault event types — require a relay trip response.
FAULT_TYPES = [
    'flt_1phg_shc',           # 1-phase-to-ground short circuit
    'flt_1phg_shc_w_arc',     # 1-phase-to-ground short circuit with arc
    'flt_1phg_hif',           # 1-phase high-impedance fault
    'flt_1phg_hif_w_arc',     # 1-phase high-impedance fault with arc
    'flt_1phg_incipient',     # 1-phase incipient fault
    'flt_1phg_incipient_w_arc',# 1-phase incipient fault with arc
    'flt_2ph_shc',            # 2-phase short circuit
    'flt_2phg_shc',           # 2-phase-to-ground short circuit
    'flt_3ph_shc',            # 3-phase short circuit
]

# Switch/disturbance event types — no fault, correct response is always WAIT.
SWITCH_TYPES = [
    'switch_cable_off',
    'switch_cable_on',
    'switch_cap_off',
    'switch_cap_on',
    'switch_inrushHV',
    'switch_inrushLV',
    'switch_load_off',
    'switch_load_on',
    'switch_motor_start',
    'switch_ohl_off',
    'switch_ohl_on',
]

ALL_EVENT_TYPES = FAULT_TYPES + SWITCH_TYPES

# ── Signal columns ─────────────────────────────────────────────────────────────

# After loading a CSV and dropping the time column and the header (units) row,
# the signal array has shape (N_STEPS, N_SIGNAL_COLS).
N_SIGNAL_COLS  = 276           # 46 devices × 6 channels (Ia, Ib, Ic, Va, Vb, Vc)
N_LINE_COLS    = 174           # 29 line cubicles × 6 channels  (Mode A input)

# Column indices for the 174 line-cubicle channels within the 276-column array,
# and a mapping from line name → channel positions within that 174-channel array.
# Both computed once at import time by parsing a reference CSV header.
# ── File paths ────────────────────────────────────────────────────────────────

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))

# DATA_DIR / VAL_DATA_DIR may live outside the repo. Set POWER_GRID_DATA_DIR to
# override the portable <repo>/data default.
DATA_DIR        = os.environ.get('POWER_GRID_DATA_DIR') or os.path.join(_ROOT, 'data')
VAL_DATA_DIR    = os.environ.get('POWER_GRID_VAL_DATA_DIR') or os.path.join(DATA_DIR, 'validation')
NPY_TRAIN_DIR   = os.path.join(_ROOT, 'data_npy', 'train')
NPY_VAL_DIR     = os.path.join(_ROOT, 'data_npy', 'val')
PHASOR_TRAIN_DIR= os.path.join(_ROOT, 'data_npy', 'phasor', 'train')
PHASOR_VAL_DIR  = os.path.join(_ROOT, 'data_npy', 'phasor', 'val')
LABELS_DIR      = os.path.join(_ROOT, 'labels')
SETTINGS_CSV    = os.path.join(_ROOT, 'labels', 'settings.csv')
TRANSITIONS_NPZ = os.path.join(_ROOT, 'labels', 'transitions.npz')
GRAPH_PICKLE    = os.path.join(_ROOT, 'graphs', 'graph_benchmark.pickle')


def _build_line_indices():
    """Build (LINE_COL_INDICES, LINE_TO_CHINDEX) by parsing the CSV header.

    Raises FileNotFoundError if no reference CSV is found. We REFUSE to fall
    back to a numeric default because preprocess Stage 2 silently corrupted
    the saved phasor files when this returned `list(range(174))` for a missing
    CSV — the caller has no way to know its indices are wrong.
    """
    import re as _re

    ref = os.path.join(DATA_DIR, 'result0.csv')
    if not os.path.exists(ref):
        raise FileNotFoundError(
            f"Cannot build line index maps: no result0.csv at {ref}. "
            f"Set POWER_GRID_DATA_DIR=<dir containing result0.csv> "
            f"or place the CSV under <repo>/data/."
        )

    header = pd.read_csv(ref, nrows=0, low_memory=False).columns.tolist()
    col_indices  = []   # absolute indices into the 276-col signal array (time excluded)
    line_to_ch   = {}   # line name → list of positions within the 174-channel array
    ch_pos       = 0    # running position in the filtered 174-channel array

    for i, col in enumerate(header):
        if i == 0 or 'MainLn' not in col:
            continue
        col_indices.append(i - 1)
        m = _re.search(r'(MainLn[\d-]+)', _re.sub(r'\.\d+$', '', col))
        assert m is not None, f"unreachable: column {col!r} matched 'MainLn' but regex failed"
        line_to_ch.setdefault(m.group(1), []).append(ch_pos)
        ch_pos += 1

    return col_indices, line_to_ch

LINE_COL_INDICES, LINE_TO_CHINDEX = _build_line_indices()
# LINE_COL_INDICES : length 174 — absolute signal-column positions (time excluded)
# LINE_TO_CHINDEX  : e.g. {'MainLn1-2': [0,1,2,3,4,5,6,7,8,9,10,11], ...}
#                    6 channels per cubicle × 2 cubicles per line = 12 indices per line
#                    channel order: Ia, Ib, Ic, Ua, Ub, Uc  (repeated for each cubicle)

# Number of phasor features per episode (computed by preprocess Stage 2).
# Computed on line cubicles only (29 cubicles), all three phases per cubicle.
# Layout per cubicle: [|Va|, |Ia|, Ra, Xa, |Vb|, |Ib|, Rb, Xb, |Vc|, |Ic|, Rc, Xc]
#   →  29 × 12 = 348
N_PHASOR_COLS  = 348
