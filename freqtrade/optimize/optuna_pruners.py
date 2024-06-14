import typing
from optuna.pruners import BasePruner
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState
import numpy as np
import logging

logger = logging.getLogger(__name__)


# https://github.com/optuna/optuna/issues/2042
class MultiplePruners(BasePruner):
    def __init__(
        self,
        pruners: typing.Iterable[BasePruner],
        pruning_condition: str = "any",
    ) -> None:
        self._pruners = tuple(pruners)

        self._pruning_condition_check_fn = None
        if pruning_condition == "any":
            self._pruning_condition_check_fn = any
        elif pruning_condition == "all":
            self._pruning_condition_check_fn = all
        else:
            raise ValueError(f"Invalid pruning ({pruning_condition}) condition passed!")
        assert self._pruning_condition_check_fn is not None

    def prune(
        self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial"
    ) -> bool:
        return self._pruning_condition_check_fn(
            pruner.prune(study, trial) for pruner in self._pruners
        )


# https://stackoverflow.com/questions/58820574/how-to-sample-parameters-without-duplicates-in-optuna
class RepeatPruner(BasePruner):
    def prune(
        self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial"
    ) -> bool:
        trials = study.get_trials(deepcopy=False)

        numbers = np.array([t.number for t in trials])
        bool_params = np.array([trial.params == t.params for t in trials]).astype(bool)
        # Don´t evaluate function if another with same params has been/is being evaluated before this one
        if np.sum(bool_params) > 1:
            if trial.number > np.min(numbers[bool_params]):
                logger.info(f"params already tested. pruning of trial {trial.number}")
                return True

        return False
