import hashlib
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_preprocess():
    constants = types.ModuleType("rl_protection.constants")
    constants.DATA_DIR = constants.VAL_DATA_DIR = ""
    constants.NPY_TRAIN_DIR = constants.NPY_VAL_DIR = ""
    constants.PHASOR_TRAIN_DIR = constants.PHASOR_VAL_DIR = ""
    constants.SETTINGS_CSV = constants.TRANSITIONS_NPZ = ""
    constants.N_STEPS = 4800
    constants.N_SIGNAL_COLS = constants.N_LINE_COLS = 174
    constants.N_PHASOR_COLS = 348
    constants.LINE_COL_INDICES = list(range(174))
    constants.GRID_FREQ = 50
    constants.SAMPLE_RATE = 9600
    constants.CYCLE_LEN = 192

    expert = types.ModuleType("rl_protection.expert")
    expert.get_expert_action = expert.build_transitions = lambda *args, **kwargs: None

    scipy = types.ModuleType("scipy")
    scipy.signal = types.ModuleType("scipy.signal")

    with mock.patch.dict(sys.modules, {
        "rl_protection.constants": constants,
        "rl_protection.expert": expert,
        "scipy": scipy,
        "scipy.signal": scipy.signal,
    }):
        sys.modules.pop("rl_protection.preprocess", None)
        return importlib.import_module("rl_protection.preprocess")


preprocess = load_preprocess()


class CausalImpedanceTests(unittest.TestCase):
    def test_future_current_cannot_change_earlier_features(self):
        base = np.empty((preprocess.N_STEPS, preprocess.N_LINE_COLS))
        for cubicle in range(preprocess.N_LINE_COLS // 6):
            start = cubicle * 6
            base[:, start:start + 3] = 1
            base[:, start + 3:start + 6] = 10

        changed = base.copy()
        cutoff = 3000
        changed[cutoff:, 0] *= 1000

        with mock.patch.object(
            preprocess,
            "sliding_phasor",
            side_effect=lambda x: x.astype(np.complex64),
        ):
            expected = preprocess.compute_phasors(base)
            actual = preprocess.compute_phasors(changed)
        np.testing.assert_allclose(actual[:cutoff], expected[:cutoff])

    def test_prefix_maximum_and_separate_rx_layout(self):
        currents = np.array([
            [0.001, 1.0, 1.0],
            [0.010, 2.0, 1.0],
            [0.005, 4.0, 2.0],
            [0.030, 1.0, 1.0],
        ])
        voltages = 10 * currents
        arr = np.column_stack([currents, voltages])

        with (
            mock.patch.object(preprocess, "N_STEPS", len(arr)),
            mock.patch.object(preprocess, "N_LINE_COLS", 6),
            mock.patch.object(preprocess, "LINE_COL_INDICES", list(range(6))),
            mock.patch.object(
                preprocess,
                "sliding_phasor",
                side_effect=lambda x: x.astype(np.complex64),
            ),
        ):
            features = preprocess.compute_phasors(arr)

        self.assertEqual(features.shape, (4, 12))
        prefix_peak = np.maximum.accumulate(np.max(currents, axis=1))
        for phase in range(3):
            base = phase * 4
            expected_r = np.where(
                currents[:, phase] < 0.005 * prefix_peak, 0.0, 10.0
            )
            np.testing.assert_allclose(features[:, base + 2], expected_r)
            np.testing.assert_allclose(features[:, base + 3], 0.0)

    def test_after_episode_maximum_matches_original(self):
        current = np.array([1.0, 2.0, 10.0, 4.0, 3.0], dtype=np.complex64)
        voltage = 5 * current
        running_peak = np.maximum.accumulate(np.abs(current))

        causal = preprocess.impedance_phasor(voltage, current, i_peak=running_peak)
        original = preprocess.impedance_phasor(
            voltage, current, i_peak=float(np.max(np.abs(current)))
        )
        np.testing.assert_allclose(causal[2:], original[2:])

    def test_split_data_and_experiment_configuration_are_unchanged(self):
        expected = {
            ROOT / "labels" / "train_labels.pt":
                "093312fca1b755c7eee0c4a921ea5776ecda37544e7c3ca465b82c904fb2c45b",
            ROOT / "labels" / "transitions.npz":
                "f44850f10c9848a71d07ebe8b27ffee5353a5d5a37c4d1971d89415064395494",
            ROOT / "labels" / "validation" / "val_labels.pt":
                "30bcae0283b5f5bfac5afab2713719941baf675f17312d1f6fc4208ab7a169b5",
            ROOT / "rl_protection" / "dataset.py":
                "63e91df5d0709d9cd8a73e6f8676a24c2af29bbdc14f5456055e1db17902fb33",
            # Updated for the public release: plot_event_type_breakdown()'s
            # F1 denominator was corrected (FN instead of TN). The training,
            # split, and _aggregate_metrics code paths are unchanged.
            ROOT / "run_ablation.py":
                "21cc8751f0c44fd0346eafc0af2ecf8e006fb79be49b688b15e5866843fe6d06",
        }
        missing_external = []
        for path, digest in expected.items():
            with self.subTest(path=path):
                if not path.is_file():
                    missing_external.append(path)
                    continue
                content = path.read_bytes()
                if path.suffix == ".py":
                    content = content.replace(b"\r\n", b"\n")
                self.assertEqual(hashlib.sha256(content).hexdigest(), digest)
        if missing_external:
            self.skipTest(
                "external label artifacts are not included in the code release"
            )


if __name__ == "__main__":
    unittest.main()
