"""
reward.py — Reward function for the relay selection RL task.

compute_reward(action, expert_action, t, is_fault_episode) -> float
    Per-step reward given the agent's action at timestep t.

episode_terminal_reward(tripped, is_fault_episode) -> float
    Bonus/penalty applied at episode end.
"""

from .constants import WAIT, DT


def compute_reward(
    action: int,
    expert_action: int,
    t_rel: int,
    is_fault_episode: bool,
    *,
    false_positive_penalty: float = -100.0,
    correct_reward: float = 5.0,
    wrong_relay_penalty: float = -100.0,
    wait_penalty: float = 0.0,
) -> float:
    """
    Return the reward for taking `action` at fault-relative timestep `t_rel`.

    t_rel = t - FAULT_ONSET  (negative = pre-fault, 0 = fault onset, positive = post-fault)

    Pre-fault period (t_rel < 0):
      WAIT        →  0.0                    (neutral)
      Any trip    →  false_positive_penalty (tripped a healthy network)

    Post-fault, switch/non-fault episode (expert = WAIT):
      WAIT        →  correct_reward         (correctly identified non-fault)
      Any trip    →  false_positive_penalty

    Post-fault, fault episode:
      Correct relay →  correct_reward
      WAIT          →  wait_penalty
      Wrong relay   →  wrong_relay_penalty
    """
    post_fault = t_rel >= 0

    if not post_fault:
        return 0.0 if action == WAIT else false_positive_penalty

    if not is_fault_episode:
        return correct_reward if action == WAIT else false_positive_penalty

    if action == expert_action:
        return correct_reward

    if action == WAIT:
        return wait_penalty

    return wrong_relay_penalty
