import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def load_train_cql():
    constants = types.ModuleType("rl_protection.constants")
    constants.TRANSITIONS_NPZ = ""
    constants.N_ACTIONS = 2
    dataset = types.ModuleType("rl_protection.dataset")
    dataset.build_dataloader = lambda *args, **kwargs: None
    models = types.ModuleType("rl_protection.models")
    models.build_model = lambda *args, **kwargs: None
    with mock.patch.dict(sys.modules, {
        "rl_protection.constants": constants,
        "rl_protection.dataset": dataset,
        "rl_protection.models": models,
    }):
        sys.modules.pop("rl_protection.train_cql", None)
        return importlib.import_module("rl_protection.train_cql")


train_module = load_train_cql()


def make_loaders(generator):
    obs = torch.tensor([
        [[0.0, 1.0]], [[1.0, 0.0]], [[1.0, 1.0]], [[0.5, -0.5]],
        [[-1.0, 0.0]], [[0.0, -1.0]], [[-1.0, -1.0]], [[0.25, 0.75]],
    ])
    next_obs = obs.roll(-1, dims=0)
    actions = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    rewards = torch.tensor([1.0, -1.0, 0.5, -0.5, 1.0, -1.0, 0.5, -0.5])
    dones = torch.zeros(8)
    extra = torch.arange(8)
    data = TensorDataset(obs, next_obs, actions, rewards, dones, extra, extra)
    train = DataLoader(data, batch_size=2, shuffle=True, generator=generator)
    test = DataLoader(data, batch_size=4, shuffle=False)
    return train, test


def make_model():
    return nn.Sequential(nn.Flatten(), nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 2))


class ReproducibilityTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_training_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            full_generator = train_module.seed_everything(7, deterministic=True)
            full_model = make_model()
            full_train, full_test = make_loaders(full_generator)
            full_history = train_module.train_cql(
                full_model, full_train, full_test, n_epochs=2,
                save_dir=str(root / "full"), seed=7, deterministic=True,
            )

            first_generator = train_module.seed_everything(7, deterministic=True)
            first_model = make_model()
            first_train, first_test = make_loaders(first_generator)
            train_module.train_cql(
                first_model, first_train, first_test, n_epochs=1,
                save_dir=str(root / "resumed"), seed=7, deterministic=True,
            )

            resumed_generator = train_module.seed_everything(7, deterministic=True)
            resumed_model = make_model()
            resumed_train, resumed_test = make_loaders(resumed_generator)
            resumed_history = train_module.train_cql(
                resumed_model, resumed_train, resumed_test, n_epochs=2,
                save_dir=str(root / "resumed"),
                resume=str(root / "resumed" / "epoch_01.pt"),
                seed=7, deterministic=True,
            )

            self.assertEqual(full_history, resumed_history)
            for expected, actual in zip(
                full_model.state_dict().values(), resumed_model.state_dict().values()
            ):
                self.assertTrue(torch.equal(expected, actual))

            checkpoint = torch.load(
                root / "resumed" / "epoch_02.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.assertIn("target_model_state_dict", checkpoint)
            self.assertIn("rng_state", checkpoint)
            self.assertEqual(checkpoint["seed"], 7)
            self.assertTrue(checkpoint["deterministic"])


if __name__ == "__main__":
    unittest.main()
