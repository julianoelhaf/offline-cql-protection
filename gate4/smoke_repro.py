"""Minimal seeded CUDA repeatability smoke test for the Gate 4 model stack."""

from __future__ import annotations

import json

import torch

from rl_protection.constants import N_ACTIONS, N_LINE_COLS, N_PHASOR_COLS
from rl_protection.models import build_model
from rl_protection.train_cql import cql_loss, seed_everything


def one_step() -> tuple[float, dict[str, torch.Tensor]]:
    seed_everything(0, deterministic=False)
    device = torch.device("cuda")
    model = build_model("combined", 96).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    obs = torch.randn(8, 96, N_LINE_COLS + N_PHASOR_COLS, device=device)
    actions = torch.randint(N_ACTIONS, (8,), device=device)
    targets = torch.randn(8, device=device)

    q_all = model(obs)
    q_data = q_all.gather(1, actions[:, None]).squeeze(1)
    loss = torch.nn.functional.mse_loss(q_data, targets) + 0.5 * cql_loss(q_all, actions)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()

    return loss.item(), {name: value.detach().cpu() for name, value in model.state_dict().items()}


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Gate 4 smoke test")
    loss_a, state_a = one_step()
    loss_b, state_b = one_step()
    mismatches = [name for name in state_a if not torch.equal(state_a[name], state_b[name])]
    if loss_a != loss_b or mismatches:
        raise RuntimeError(
            f"seeded CUDA repeatability failed: losses=({loss_a}, {loss_b}), "
            f"mismatched tensors={mismatches[:5]}"
        )
    print(json.dumps({
        "status": "ok",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "loss": loss_a,
        "state_tensors": len(state_a),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
