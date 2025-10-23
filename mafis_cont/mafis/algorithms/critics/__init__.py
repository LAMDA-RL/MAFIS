"""Critic registry."""
from mafis.algorithms.critics.soft_twin_continuous_q_critic import (
    SoftTwinContinuousQCritic,
)

CRITIC_REGISTRY = {
    "hasac": SoftTwinContinuousQCritic,
}
