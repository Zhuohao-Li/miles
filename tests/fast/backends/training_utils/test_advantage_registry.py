from argparse import Namespace

import pytest
import torch

from miles.backends.training_utils.loss_hub.advantages import (
    AdvantageEstimatorInput,
    compute_advantages,
    get_advantage_estimator,
    register_advantage_estimator,
)


def test_builtin_estimators_are_registered():
    for name in ("grpo", "gspo", "ppo", "reinforce_plus_plus", "reinforce_plus_plus_baseline"):
        assert callable(get_advantage_estimator(name))


def test_unknown_estimator_is_rejected():
    with pytest.raises(NotImplementedError, match="advantage_estimator unknown is not supported"):
        get_advantage_estimator("unknown")


def test_registered_estimator_is_dispatched():
    name = "test_estimator"

    @register_advantage_estimator(name)
    def estimator(inputs: AdvantageEstimatorInput):
        returns = [torch.full_like(inputs.kl[0], inputs.rewards[0])]
        return returns, returns

    args = Namespace(advantage_estimator=name)
    advantages, returns = compute_advantages(
        args=args,
        kl=[torch.zeros(2)],
        rewards=[3.0],
        log_probs=None,
        loss_masks=[torch.ones(2)],
        total_lengths=[2],
        response_lengths=[2],
    )

    torch.testing.assert_close(advantages[0], torch.tensor([3.0, 3.0]))
    torch.testing.assert_close(returns[0], torch.tensor([3.0, 3.0]))


def test_duplicate_estimator_is_rejected():
    with pytest.raises(ValueError, match="Advantage estimator 'grpo' is already registered"):
        register_advantage_estimator("grpo")(get_advantage_estimator("grpo"))
