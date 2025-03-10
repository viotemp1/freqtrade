# pragma pylint: disable=too-many-instance-attributes, pointless-string-statement

### TO DO - dynamic resource allocation ?https://docs.ray.io/en/latest/tune/examples/includes/xgboost_dynamic_resources_example.html
"""
This module contains the hyperopt logic
"""

import logging
import random
import warnings
from datetime import datetime, timezone
import time
from math import ceil, nan
from multiprocessing import Manager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from copy import deepcopy

import rapidjson
import json
from joblib import cpu_count, dump, load
from functools import partial

from freqtrade.constants import FTHYPT_FILEVERSION, LAST_BT_RESULT_FN, Config
from freqtrade.enums import HyperoptState
from freqtrade.exceptions import OperationalException
from freqtrade.misc import file_dump_json, plural, deep_merge_dicts
from freqtrade.optimize.hyperopt.hyperopt_logger import (
    logging_mp_handle,
    logging_mp_setup,
)
from freqtrade.optimize.hyperopt.hyperopt_optimizer import HyperOptimizer
from freqtrade.optimize.hyperopt.hyperopt_output import HyperoptOutput
from freqtrade.optimize.hyperopt_tools import (
    HyperoptStateContainer,
    HyperoptTools,
    hyperopt_serializer,
)
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.util import get_progress_tracker


from tabulate import tabulate
import numpy as np
from collections import deque
from pandas import DataFrame
import pandas as pd
import os
import sys
import psutil

from rich.live import Live
from rich.table import Table
from rich.console import Console, Group
from rich.bar import Bar
from rich.text import Text
from rich.style import Style
from rich.ansi import AnsiDecoder

# import asciichartpy as acp
import plotext as plt
from progressbar import ProgressBar
from optuna.exceptions import ExperimentalWarning

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=ExperimentalWarning)
    warnings.filterwarnings("ignore", module="ray.tune.logger.tensorboardx")
    warnings.filterwarnings("ignore", module="ray.tune.callback")
    warnings.filterwarnings("ignore", module="ray.tune.execution.tune_controller")
    logging.getLogger("ray.tune.schedulers.resource_changing_scheduler").setLevel(
        logging.WARNING
    )

    from skopt import Optimizer
    from skopt.space import Dimension
    import ray
    from ray import tune, train
    from ray.train import RunConfig
    from ray.util.state import summarize_tasks
    from ray.tune.experiment import Trial
    from ray.tune.logger import LoggerCallback, CSVLoggerCallback, JsonLoggerCallback
    from ray.tune.stopper.stopper import Stopper
    from ray.tune.execution.placement_groups import PlacementGroupFactory
    from ray.tune.schedulers import ResourceChangingScheduler, ASHAScheduler, FIFOScheduler
    from ray.tune.schedulers.resource_changing_scheduler import DistributeResources


ray_results_table_max_rows = 10  # -1 - half screen
ray_reuse_actors = False

# max_used_memory = 80  # 0 or negative to deactivate, otherwise pause worker

MAX_LOSS = 100000  # just a big enough number to be bad result in loss optimization

plot_metric_list = [
    "trial_id",
    "Trades",
    "Win_Draw_Loss_Win_perc",
    "Avg_profit",
    "Profit",
    "Winrate",
    "Avg_duration",
    "loss",
    "Max_Drawdown_Acct",
    "time_total_s",
]

optunahub_samplers = [
    "auto_sampler",
    "differential_evolution",
    "hebo",
    "mocma",
    "nelder_mead",
    "nsgaii_with_tpe_warmup",
    "whale_optimization",
    "simulated_annealing",
    # "grey_wolf_optimization",
    # "implicit_natural_gradient",
    # "moead",
]

logger = logging.getLogger(__name__)
# log_queue: Any


logger = HyperOptimizer.ray_setup_func()


class Hyperopt:
    """
    Hyperopt class, this class contains all the logic to run a hyperopt simulation

    To start a hyperopt run:
    hyperopt = Hyperopt(config)
    hyperopt.start()
    """

    def __init__(self, config: Config) -> None:
        self._hyper_out: HyperoptOutput = HyperoptOutput(streaming=True)
        self.config = config

        self.random_state = self._set_random_state(
            self.config.get("hyperopt_random_state")
        )

        np.random.seed(self.random_state)
        random.seed(self.random_state)

        self.analyze_per_epoch = self.config.get("analyze_per_epoch", False)
        HyperoptStateContainer.set_state(HyperoptState.STARTUP)

        if self.config.get("hyperopt"):
            raise OperationalException(
                "Using separate Hyperopt files has been removed in 2021.9. Please convert "
                "your existing Hyperopt file to the new Hyperoptable strategy interface"
            )

        time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        strategy = str(self.config["strategy"])
        self.strategy_name = strategy
        self.results_file: Path = (
            self.config["user_data_dir"]
            / "hyperopt_results"
            / f"strategy_{strategy}_{time_now}.fthypt"
        )

        self.data_pickle_file = f'{self.config["user_data_dir"]}/hyperopt_results/hyperopt_tickerdata.pkl'
        self.detail_data_pickle_file = f'{self.config["user_data_dir"]}/hyperopt_results/hyperopt_detail_tickerdata.pkl'

        self.hyperopt_results_file: Path = (
            Path(self.config["user_data_dir"]).parent / "csv" / "hyperopt_results.csv"
        )
        self.ray_log_dir = os.path.join("/tmp", "ray", strategy)

        self.total_epochs = config.get("epochs", 0)
        self.current_best_loss = 100

        self.clean_hyperopt()

        self.num_epochs_saved = 0
        self.current_best_epoch: dict[str, Any] | None = None

        if HyperoptTools.has_space(self.config, "sell"):
            # Make sure use_exit_signal is enabled
            self.config["use_exit_signal"] = True

        self.print_all = self.config.get("print_all", False)
        self.hyperopt_table_header = 0
        self.print_json = self.config.get("print_json", False)

        self.print_all = self.config.get("print_all", False)
        # self.print_hyperopt_results = self.config.get("print_hyperopt_results", True)
        self.plot_chart = os.environ.get("RAY_PLOT_CHART", None)
        if self.plot_chart and self.plot_chart.lower() == "false":
            self.plot_chart = False
        elif self.plot_chart and self.plot_chart.lower() == "true":
            self.plot_chart = True
        else:
            self.plot_chart = self.config.get("plot_chart", True)
        self.print_progressbar = os.environ.get("RAY_PROGRESSBAR", None)
        if self.print_progressbar and self.print_progressbar.lower() == "false":
            self.print_progressbar = False
        elif self.print_progressbar and self.print_progressbar.lower() == "true":
            self.print_progressbar = True
        else:
            self.print_progressbar = self.config.get("print_progressbar", False)
        logger.info(
            f"plot_chart: {self.plot_chart} / print_progressbar: {self.print_progressbar} / isatty: {sys.stdout.isatty()}"
        )
        self.print_json = self.config.get("print_json", True)
        self.save_results_to_csv = self.config.get("save_results_to_csv", True)
        self.ray_early_stop_enable = self.config.get("ray_early_stop_enable", True)
        self.ray_early_stop_perc = self.config.get(
            "ray_early_stop_perc", 0.005
        )  # 0.001
        self.ray_early_stop_std = self.config.get("ray_early_stop_std", 0.005)  # 0.001
        self.ray_early_stop_top = self.config.get("ray_early_stop_top", 10)
        self.ray_early_stop_patience = self.config.get("ray_early_stop_patience", 0.25)
        self.ray_dashboard = self.config.get("ray_dashboard", True)
        self.ray_dashboard_port = self.config.get("ray_dashboard_port", 8265)
        self.ray_max_memory_perc = min(
            float(config.get("ray_max_memory_perc", 0.9)),
            float(os.environ.get("RAY_MAX_MEMORY_PERC", 0.9)),
        )
        if self.ray_max_memory_perc is not None:
            try:
                self.ray_max_memory = psutil.virtual_memory().total * float(
                    self.ray_max_memory_perc
                )
            except:
                self.ray_max_memory = None
        else:
            self.ray_max_memory = None

        if self.ray_max_memory and self.ray_max_memory_perc:
            logger.debug(
                f"ray_max_memory: {(self.ray_max_memory):,.2f} / ray_max_memory_perc: {(100.*self.ray_max_memory_perc):,.2f}"
            )

        self.config_jobs = self.config.get("hyperopt_jobs", -1)

        HyperOptimizer.ray_setup_func()
        self.hyperopter = HyperOptimizer(self.config)

        if hasattr(self.hyperopter.backtesting.strategy, "plot_metric"):
            self.plot_metric = getattr(
                self.hyperopter.backtesting.strategy, "plot_metric"
            )
        else:
            self.plot_metric = self.config.get(
                "plot_metric", "Profit"
            )  # Profit Winrate

    def ray_worker_logging_setup_func(self):
        logging.getLogger("ray").setLevel(logging.INFO)
        warnings.simplefilter("always")
        np.random.seed(self.random_state)
        random.seed(self.random_state)

    @staticmethod
    def get_lock_filename(config: Config) -> str:
        return str(config["user_data_dir"] / "hyperopt.lock")

    def clean_hyperopt(self) -> None:
        """
        Remove hyperopt pickle files to restart hyperopt.
        """
        for f in [self.results_file, self.data_pickle_file, self.detail_data_pickle_file]:
            p = Path(f)
            if p.is_file():
                logger.info(f"Removing `{p}`.")
                p.unlink()

    def _save_result(self, epoch: dict) -> None:
        """
        Save hyperopt results to file
        Store one line per epoch.
        While not a valid json object - this allows appending easily.
        :param epoch: result dictionary for this epoch.
        """
        epoch[FTHYPT_FILEVERSION] = 2
        with self.results_file.open("a") as f:
            rapidjson.dump(
                epoch,
                f,
                default=hyperopt_serializer,
                number_mode=rapidjson.NM_NATIVE | rapidjson.NM_NAN,
            )
            f.write("\n")

        self.num_epochs_saved += 1
        logger.debug(
            f"{self.num_epochs_saved} {plural(self.num_epochs_saved, 'epoch')} "
            f"saved to '{self.results_file}'."
        )
        # Store hyperopt filename
        latest_filename = Path.joinpath(self.results_file.parent, LAST_BT_RESULT_FN)
        file_dump_json(
            latest_filename, {"latest_hyperopt": str(self.results_file.name)}, log=False
        )

    def print_results(self, results: dict[str, Any]) -> None:
        """
        Log results if it is better than any previous evaluation
        TODO: this should be moved to HyperoptTools too
        """
        is_best = results["is_best"]

        if self.print_all or is_best:
            self._hyper_out.add_data(
                self.config,
                [results],
                self.total_epochs,
                self.print_all,
            )

    def _set_random_state(self, random_state: int | None) -> int:
        return random_state or random.randint(1, 2**16 - 1)  # noqa: S311

    # searchers: ['variant_generator', 'random', 'hyperopt', 'bohb', 'nevergrad', 'optuna', 'zoopt', 'hebo']
    # 'bayesopt' - not suported - does not suport Integer
    # 'ax' - not working
    # schedulers: ['fifo', 'async_hyperband', 'asynchyperband', 'median_stopping_rule', 'medianstopping', 'hyperband', 'hb_bohb', 'pbt', 'pbt_replay', 'pb2', 'resource_changing']
    def get_search_algo_scheduler(self, config_jobs: Dict, random_state: int):
        searcher_orig = self.hyperopter.custom_hyperopt.generate_estimator(
            dimensions=self.hyperopter.dimensions
        )
        searcher_param1 = None
        if isinstance(searcher_orig, tuple) and len(searcher_orig) == 2:
            searcher = searcher_orig[0]
            searcher_param1 = searcher_orig[1]
        elif isinstance(searcher_orig, str):
            searcher = searcher_orig
        else:
            raise Exception(
                f"generate_estimator should return either str or tuple. Got instead {searcher_orig} - {type(searcher_orig)}"
            )

        if isinstance(searcher_orig, str):
            searchers_list = [
                "variant_generator",
                "random",
                # "ax", # not working
                "hyperopt",
                # "bayesopt", # no Int parameters
                "bohb",
                "nevergrad",
                "optuna",
                "zoopt",
                "hebo",
            ]
            if searcher not in searchers_list:
                raise OperationalException(
                    f"Ray searcher {searcher} not supported. Please use one of {searchers_list}"
                )
        if searcher == "optuna" and searcher_param1 is None:
            searcher_param1 = "NSGAIIISampler"  # NSGAIIISampler auto_sampler
        self.searcher = searcher
        self.searcher_param1 = searcher_param1
        logger.info(f"Using searcher {searcher} - {searcher_param1}")
        try:
            if searcher == "nevergrad":
                from ray.tune.search.nevergrad import NevergradSearch
                import nevergrad as ng

                # "pymoo_de" 'pymoo_ga' "pymoo_brkga" "pymoo_nelder-mead" "pymoo_pattern-search" "pymoo_cmaes" "pymoo_unsga3"
                if self.searcher_param1:
                    if self.searcher_param1.startswith("pymoo_"):
                        search_algo = NevergradSearch(
                            optimizer=ng.optimizers.Pymoo(
                                algorithm=self.searcher_param1.replace("pymoo_", "")
                            ),
                        )
                    else:
                        search_algo = NevergradSearch(
                            optimizer=ng.optimizers.registry[self.searcher_param1],
                        )
                else:
                    searcher_algo = tune.create_searcher(
                        searcher,
                        random_state_seed=random_state,
                        optimizer=ng.optimizers.OnePlusOne,
                    )
            elif searcher == "zoopt":
                zoopt_search_config = {
                    "parallel_num": self.config.get(
                        "hyperopt_jobs", cpu_count()
                    ),  # how many workers to parallel
                }
                searcher_algo = tune.create_searcher(
                    searcher,
                    random_state_seed=random_state,
                    budget=self.total_epochs,
                    **zoopt_search_config,
                )
            elif searcher == "bohb":
                searcher_algo = tune.create_searcher(
                    searcher,
                    seed=random_state,
                )
            elif searcher == "bayesopt":
                searcher_algo = tune.create_searcher(
                    searcher,
                    random_state=random_state,
                )
            elif searcher == "optuna":
                import optuna

                # TPESampler NSGAIIISampler CmaEsSampler GPSampler NSGAIISampler QMCSampler
                if self.searcher_param1:
                    if self.searcher_param1 in optunahub_samplers:
                        import optunahub, inspect

                        ohsmodule = optunahub.load_module(
                            f"samplers/{self.searcher_param1}"
                        )
                        if self.searcher_param1 == "auto_sampler":
                            sampler_m = ohsmodule.AutoSampler
                        elif self.searcher_param1 == "differential_evolution":
                            sampler_m = ohsmodule.DESampler
                        elif self.searcher_param1 == "hebo":
                            sampler_m = ohsmodule.HEBOSampler
                        # elif self.searcher_param1 == "implicit_natural_gradient":
                        #     sampler_m = ohsmodule.ImplicitNaturalGradientSampler
                        # elif self.searcher_param1 == "moead":
                        #     sampler_m = ohsmodule.MOEADSampler
                        elif self.searcher_param1 == "mocma":
                            sampler_m = ohsmodule.MoCmaSampler
                        elif self.searcher_param1 == "nelder_mead":
                            sampler_m = ohsmodule.NelderMeadSampler
                        elif self.searcher_param1 == "nsgaii_with_tpe_warmup":
                            sampler_m = ohsmodule.NSGAIIWithTPEWarmupSampler
                        elif self.searcher_param1 == "whale_optimization":
                            sampler_m = ohsmodule.WhaleOptimizationSampler
                        elif self.searcher_param1 == "simulated_annealing":
                            sampler_m = ohsmodule.SimulatedAnnealingSample
                        else:
                            logger.warning(
                                f"searcher_param1 {self.searcher_param1} not supported - {optunahub_samplers}"
                            )

                        # if self.searcher_param1 in ["nsgaii_with_tpe_warmup"]:
                        #     sampler_m = inspect.getmembers(ohsmodule)[2][1]
                        # else:
                        #     sampler_m = inspect.getmembers(ohsmodule)[0][1]
                        try:
                            optuna__sampler = sampler_m(seed=random_state)
                        except:
                            logger.warning(
                                f"Cannot set random_state_seed {random_state} for {self.searcher}"
                            )
                            optuna__sampler = sampler_m()
                            pass
                    elif self.searcher_param1 == "NSGAIIISampler":
                        optuna__sampler = optuna.samplers.NSGAIIISampler(
                            seed=random_state
                        )
                    elif self.searcher_param1 == "AutoSampler":
                        import optunahub

                        optuna__sampler = optunahub.load_module(
                            "samplers/auto_sampler"
                        ).AutoSampler(seed=random_state)
                    elif self.searcher_param1 == "CmaEsSampler":
                        optuna__sampler = optuna.samplers.CmaEsSampler(
                            seed=random_state
                        )
                    elif self.searcher_param1 == "GPSampler":
                        optuna__sampler = optuna.samplers.GPSampler(seed=random_state)
                    elif self.searcher_param1 == "NSGAIISampler":
                        optuna__sampler = optuna.samplers.NSGAIISampler(
                            seed=random_state
                        )
                    elif self.searcher_param1 == "TPESampler":
                        optuna__sampler = optuna.samplers.TPESampler(seed=random_state)
                    elif self.searcher_param1 == "QMCSampler":
                        optuna__sampler = optuna.samplers.QMCSampler(
                            seed=random_state,
                            warn_independent_sampling=False,
                        )
                    elif self.searcher_param1 == "BoTorchSampler":
                        optuna__sampler = optuna.integration.BoTorchSampler(
                            seed=random_state
                        )
                    else:  # default
                        optuna__sampler = optuna.samplers.TPESampler(seed=random_state)
                    searcher_algo = tune.create_searcher(
                        searcher,
                        sampler=optuna__sampler,
                    )
                else:
                    logger.warning(
                        f"searcher_param1 for optuna not set -  {self.searcher_param1}"
                    )
            elif (
                searcher == "hebo"
            ):  # gp gpy gpy_mlp psgld svidkl deep_ensemble rf catboost svgp mcbn masked_deep_ensemble fe_deep_ensemble gumbel
                import hebo
                import torch  # hebo has torch as a dependency

                # gp gpy gpy_mlp psgld svidkl deep_ensemble rf catboost svgp mcbn masked_deep_ensemble fe_deep_ensemble gumbel
                if self.searcher_param1:
                    searcher_algo = tune.create_searcher(
                        searcher,
                        random_state_seed=random_state,
                        model_name=self.searcher_param1,
                        scramble_seed=srandom_state,
                    )
                else:  # default
                    searcher_algo = tune.create_searcher(
                        searcher,
                        random_state_seed=random_state,
                        model_name="gp",
                        scramble_seed=random_state,
                    )
            else:
                searcher_algo = tune.create_searcher(
                    searcher, random_state_seed=random_state
                )
        except Exception as e:
            logger.warning(f"Set searcher error: {repr(e)}")
            searcher_algo = tune.create_searcher(searcher)
            pass

        if isinstance(self.searcher, str) and self.searcher == "bohb":
            scheduler = tune.create_scheduler("hb_bohb")
        else:
            scheduler = tune.create_scheduler("fifo")
            # scheduler = ASHAScheduler(max_t=16)
        # self.scheduler = scheduler
        return searcher_algo, scheduler

    # # not working - training_iteration = 1 after backtest
    # @staticmethod
    # def resources_allocation_fn(
    #     config_jobs: int,
    #     max_memory_perc: float,
    #     tune_controller: "TuneController",
    #     trial: Trial,
    #     result: Dict[str, Any],
    #     scheduler: "ResourceChangingScheduler",
    # ) -> Optional[PlacementGroupFactory]:
    #     """This is a basic example of a resource allocating function.

    #     The function naively balances available CPUs over live trials.

    #     This function returns a new ``PlacementGroupFactory`` with updated
    #     resource requirements, or None. If the returned
    #     ``PlacementGroupFactory`` is equal by value to the one the
    #     trial has currently, the scheduler will skip the update process
    #     internally (same with None).

    #     See :class:`DistributeResources` for a more complex,
    #     robust approach.

    #     Args:
    #         tune_controller: Trial runner for this Tune run.
    #             Can be used to obtain information about other trials.
    #         trial: The trial to allocate new resources to.
    #         result: The latest results of trial.
    #         scheduler: The scheduler calling the function.
    #     """

    #     # print(f"tune_controller: {tune_controller} / trial: {trial} / result: {result} / scheduler: {scheduler} / config_jobs: {config_jobs}")
    #     # Get base trial resources as defined in
    #     # ``tune.with_resources``
    #     base_trial_resource = scheduler._base_trial_resources

    #     # Don't bother if this is just the first iteration
    #     print(f'training_iteration: {result["training_iteration"]} / config_jobs: {config_jobs} / base_trial_resource: {base_trial_resource.required_resources}')
    #     if result["training_iteration"] < 1 or base_trial_resource is None:
    #         return None

    #     # Assume that the number of CPUs cannot go below what was
    #     # specified in ``Tuner.fit()``.
    #     existing_required_cpus = base_trial_resource.required_resources.get("CPU", 0)
    #     if existing_required_cpus == 0:
    #         return None

    #     # Get the number of CPUs available in total (not just free)
    #     total_available_cpus = cpu_count() # tune_controller._resource_updater.get_num_cpus()
    #     # count_live_trials = len(tune_controller.get_live_trials()) # wrong calculation
    #     memory_usage_perc = psutil.virtual_memory().percent
    #     if memory_usage_perc < max_memory_perc:
    #         cpus_to_use = min(total_available_cpus//config_jobs, total_available_cpus)
    #         # tune_controller.update_pending_trial_resources(resources)
    #         # for trial in tune_controller._trials:
    #         #     if trial.status in [Trial.PENDING] and cpu_to_use != existing_required_cpus:
    #         #         print(trial, trial.status, existing_required_cpus, cpu_to_use )
    #         #         trial.update_resources(resources=PlacementGroupFactory([{"CPU": cpu_to_use}]))
    #         # logger.info(
    #         #     f"resources_allocation_fn - existing_required_cpus: {existing_required_cpus} / cpu_to_use: {cpu_to_use} / total_available_cpus: {total_available_cpus} / memory_usage_perc: {memory_usage_perc}"
    #         # )
    #     else:
    #         cpus_to_use = existing_required_cpus


    #     # Assign new CPUs to the trial in a PlacementGroupFactory
    #     return PlacementGroupFactory([{"CPU": cpus_to_use, "GPU": 0}])

    def start(self) -> None:
        results = None

        logger.info(f"Using optimizer random state: {self.random_state}")
        self.hyperopt_table_header = -1
        self.hyperopter.prepare_hyperopt(self.data_pickle_file, self.detail_data_pickle_file)

        cpus = cpu_count()
        logger.info(f"Found {cpus} CPU cores. Let's make them scream!")
        logger.info(f"Number of parallel jobs set as: {self.config_jobs}")

        not_optimized = self.hyperopter.backtesting.strategy.get_no_optimize_params()
        not_optimized = deep_merge_dicts(
            not_optimized, self.hyperopter._get_no_optimize_details()
        )

        # Searcher
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", category=ExperimentalWarning)
            self.opt, self.scheduler = self.get_search_algo_scheduler(
                self.config_jobs, self.random_state
            )

        # self.scheduler = ResourceChangingScheduler(
        #     base_scheduler=self.scheduler, # self.scheduler, FIFOScheduler()
        #     resources_allocation_function=partial(self.resources_allocation_fn, self.config_jobs, self.ray_max_memory_perc),
        # )
        # self.scheduler = ResourceChangingScheduler(
        #     base_scheduler=self.scheduler, # self.scheduler, FIFOScheduler()
        #     resources_allocation_function=DistributeResources(add_bundles=True, reserve_resources={"CPU": 4}),
        # )

        HyperOptimizer.ray_setup_func()

        try:
            # print(f"ray.init - {os.getcwd()}")
            # print(self.hyperopter.backtesting.strategy.custom_trade_info)

            # Path("./logs").mkdir(parents=True, exist_ok=True)

            with warnings.catch_warnings():
                warnings.filterwarnings(action="ignore", category=ExperimentalWarning)

                trainable_with_parameters = tune.with_parameters(
                    HyperOptimizer.objective,
                    config_ft=self.config,
                    backtesting=self.hyperopter.backtesting,
                    custom_trade_info=(
                        self.hyperopter.backtesting.strategy.custom_trade_info
                        if hasattr(
                            self.hyperopter.backtesting.strategy, "custom_trade_info"
                        )
                        else None
                    ),
                    dimensions_ft=self.hyperopter.dimensions,
                    data_pickle_file_ft=self.data_pickle_file,
                    detail_data_pickle_file_ft=self.detail_data_pickle_file,
                    min_date_ft=self.hyperopter.min_date,
                    max_date_ft=self.hyperopter.max_date,
                    total_epochs_ft=self.total_epochs,
                    custom_hyperopt_ft=self.hyperopter.custom_hyperopt,
                    _get_results_dict_ft=self.hyperopter._get_results_dict,
                    _advise_and_trim_ft=self.hyperopter.advise_and_trim,
                    # _save_result_ft=self._save_result,
                    results_file_ft=self.results_file,
                    ray_max_memory_perc=self.ray_max_memory_perc,
                )
                if self.ray_max_memory is None:
                    trainable_with_resources = tune.with_resources(
                        trainable_with_parameters, {"CPU": cpus // self.config_jobs}
                    )
                    logger.info(
                        f"ray resources per worker: CPU: {cpus // self.config_jobs}/{cpus}"
                    )
                else:
                    trainable_with_resources = tune.with_resources(
                        trainable_with_parameters,
                        PlacementGroupFactory(
                            [
                                {
                                    "CPU": 0.95 * cpus // self.config_jobs,
                                    "memory": 0.95 * self.ray_max_memory / self.config_jobs,
                                }
                            ]
                        ),
                    )
                    logger.info(
                        f"ray resources per worker: CPU: {0.95 * cpus // self.config_jobs}/{cpus} - MEM: {( 0.95 * self.ray_max_memory / self.config_jobs):,.2f}/{(self.ray_max_memory):,.2f}"
                    )
                ray.init(
                    ignore_reinit_error=True,
                    include_dashboard=self.ray_dashboard,
                    dashboard_port=find_first_free_port(
                        self.ray_dashboard_port
                    ),  # None
                    _node_ip_address="127.0.0.1",  # 127.0.0.1 0.0.0.0 socket.gethostbyname(socket.gethostname())
                    _memory=self.ray_max_memory,
                    object_store_memory=min(
                        5 * 10**9, 0.05 * psutil.virtual_memory().total
                    ),  # 10**9
                    _redis_max_memory=min(10**9, 0.01 * psutil.virtual_memory().total),
                    runtime_env={
                        "worker_process_setup_hook": self.ray_worker_logging_setup_func,
                        "env_vars": {
                            "PYTHONPATH": os.path.join(
                                self.config["user_data_dir"], "strategies"
                            )
                        },
                        "worker_process_setup_hook": HyperOptimizer.ray_setup_func,
                    },
                    _system_config={
                        "prestart_worker_first_driver": True,
                        "enable_worker_prestart": True,
                        "num_workers_soft_limit": max(self.config_jobs // 2, 2),
                    },
                    configure_logging=True,
                    logging_level="info",
                    log_to_driver=True,
                    logging_config=ray.LoggingConfig(encoding="TEXT", log_level="INFO"),
                    _temp_dir=self.ray_log_dir,
                )
                mem_available_bytes = ray.available_resources().get("memory", 0)
                mem_available_perc = (
                    100.0 * mem_available_bytes / psutil.virtual_memory().total
                )
                # logging.getLogger(__name__).setLevel(logging.INFO)
                logger.info(
                    f"ray available memory (before tune): {(mem_available_perc):,.2f}% - {(mem_available_bytes/10**9):,.2f}GB/{(psutil.virtual_memory().total/10**9):,.2f}GB"
                )

                logging.getLogger("ray.tune.schedulers.resource_changing_scheduler").setLevel(
                    logging.WARNING
                )

                if (
                    self.print_all or self.plot_chart
                ):  # self.print_hyperopt_results or  and sys.stdout.isatty()
                    r_callbacks = [
                        CSVLoggerCallback(),
                        JsonLoggerCallback(),
                        myLoggerCallback(
                            strategy=self.strategy_name,
                            print_all=self.print_all,
                            total_epochs=self.total_epochs,
                            table_max_rows=ray_results_table_max_rows,
                            plot_metric=self.plot_metric,
                        ),
                    ]
                elif self.print_progressbar or sys.stdout.isatty():
                    r_callbacks = [
                        CSVLoggerCallback(),
                        JsonLoggerCallback(),
                        myPBarCallback(
                            strategy=self.strategy_name,
                            total_epochs=self.total_epochs,
                        ),
                    ]
                else:
                    r_callbacks = [CSVLoggerCallback(), JsonLoggerCallback()]

                if self.ray_early_stop_enable:
                    stop_cb = ExperimentPlateauStopper(
                        "loss",
                        perc=self.ray_early_stop_perc,
                        std=self.ray_early_stop_std,
                        top=self.ray_early_stop_top,
                        mode="min",
                        patience=(
                            int(self.ray_early_stop_patience * self.total_epochs)
                        ),
                    )
                else:
                    stop_cb = None

                tuner = tune.Tuner(
                    trainable_with_resources, # trainable_with_parameters trainable_with_resources
                    tune_config=tune.TuneConfig(
                        metric="loss",
                        mode="min",
                        search_alg=self.opt,
                        scheduler=self.scheduler,
                        # max_concurrent_trials=0,  # self.config_jobs,
                        reuse_actors=ray_reuse_actors,
                        num_samples=self.total_epochs,
                        # trial_name_creator=lambda trial: f"{self.strategy_name}_{trial.trainable_name}_{trial.trial_id}",
                    ),
                    param_space=self.hyperopter.dimensions,
                    run_config=RunConfig(
                        # name=self.strategy_name,
                        verbose=0,
                        storage_path=self.ray_log_dir,
                        stop=stop_cb,
                        callbacks=r_callbacks,
                        log_to_file=False,
                    ),
                )

                try:
                    results = tuner.fit()
                except Exception as e:
                    logger.info(f"Tuner fit failed {e}")
                    if ray.is_initialized():
                        ray.shutdown()
                    pass
                    # os.kill(os.getpid(), signal.SIGTERM)

        except KeyboardInterrupt:
            logger.info("User interrupted..")
            if ray.is_initialized():
                ray.shutdown()
            pass

        # ['Trades', 'Win_Draw_Loss_Win_perc', 'Avg_profit', 'Profit',
        #        'Avg_duration', 'Objective', 'is_profit', 'Max_Drawdown_Acct', 'loss',
        #        'timestamp', 'checkpoint_dir_name', 'done', 'training_iteration',
        #        'trial_id', 'date', 'time_this_iter_s', 'time_total_s', 'pid',
        #        'hostname', 'node_ip', 'time_since_restore', 'iterations_since_restore',
        #        'config/buy_fastk_rsi_patterns', 'config/buy_max_slippage',
        #        'config/buy_prev_cbuys_count', 'config/buy_prev_cbuys_rwindow',
        #        'config/buy_prev_min_close_age', 'config/buy_prev_min_close_perc',
        #        'config/buy_prev_min_close_rwindow', 'config/buy_proposed_stake_limit',
        #        'config/buy_proposed_stake_limit_margin', 'config/csl_5_step1_SL',
        #        'config/csl_5_step1_time', 'config/csl_5_step2_SL',
        #        'config/csl_5_step2_time', 'config/csl_5_step3_SL',
        #        'config/csl_5_step3_time', 'config/csl_5_step4_SL',
        #        'config/sell_order_max_age', 'config/sell_order_min_profit',
        #        'config/stoploss', 'logdir']

        # print(results)
        if results:
            self.total_epochs = results.num_terminated
            logger.info(
                f"Hyperopt finished - OK: {results.num_terminated} / Failed: {results.num_errors}"
            )
            if self.current_best_epoch is None:
                self.current_best_epoch = {}
            self.current_best_epoch["tune_best_result"] = results.get_best_result(
                metric="loss", mode="min"
            )
            self.current_best_epoch[FTHYPT_FILEVERSION] = 2

            # df_results = self.current_best_epoch["tune_best_result"].metrics_dataframe
            df_results = results.get_dataframe(filter_metric="loss", filter_mode="min")
            # print(df_results.columns)
            df_results = df_results.sort_values(by="loss", ascending=True).head(1)
            df_results["training_iteration"] = df_results.index
            df_results["strategy_name"] = self.strategy_name
            df_results["hyperoptloss_name"] = self.config.get("hyperopt_loss")
            first_columns = [
                "strategy_name",
                "hyperoptloss_name",
                "profit_perc",
                "Winrate",
                "Trades",
            ]
            cols = first_columns + [
                c for c in df_results.columns if c not in first_columns
            ]
            df_results = df_results[cols]

            if self.save_results_to_csv and len(df_results) > 0:
                if not Path(self.hyperopt_results_file).is_file():
                    df_results.to_csv(
                        self.hyperopt_results_file,
                        encoding="utf-8",
                        index=False,
                    )
                else:
                    pd.concat(
                        [pd.read_csv(self.hyperopt_results_file), df_results],
                        axis=0,
                        ignore_index=True,
                    ).to_csv(
                        self.hyperopt_results_file,
                        header=True,
                        index=False,
                        encoding="utf-8",
                    )
                    # df_results.to_csv(
                    #     self.hyperopt_results_file,
                    #     encoding="utf-8",
                    #     index=False,
                    #     mode="a",
                    #     header=False,
                    # )

            df_results = df_results[
                [
                    "training_iteration",
                    "Trades",
                    "Win_Draw_Loss_Win_perc",
                    "Avg_profit",
                    "Profit",
                    "profit_perc",
                    "Winrate",
                    "Avg_duration",
                    "Objective",
                    "loss",
                    "Max_Drawdown_Acct",
                    # "trial_id",
                    # "done",
                    # "date",
                    "time_total_s",
                ]
            ]
            df_results = df_results.rename(
                columns={
                    "training_iteration": "Epoch",
                    "Win_Draw_Loss_Win_perc": "Win  Draw  Loss  Win%",
                    "Max_Drawdown_Acct": "Max Drawdown (Acct)",
                    "time_total_s": "Time to run",
                }
            )

            if len(results) > 0:
                logger.info(
                    f"Best results:\n"
                    f'{tabulate(df_results, headers="keys", tablefmt="psql", showindex=False)}'  #
                )
                # self.current_best_epoch.config
                # logger.info(
                #     f"Best params:\n"
                #     f"{json.dumps(self._get_params_details(self.current_best_epoch['tune_best_result'].config), sort_keys=False, indent=4)}"
                # )

                self.current_best_epoch["params_details"] = deepcopy(
                    self.hyperopter._get_params_details(
                        self.current_best_epoch["tune_best_result"].config
                    )
                )
                self.current_best_epoch["params_not_optimized"] = deepcopy(
                    not_optimized
                )

                HyperoptTools.try_export_params(
                    self.config,
                    self.hyperopter.backtesting.strategy.get_strategy_name(),
                    self.current_best_epoch,
                )

                # HyperoptTools.show_epoch_details(
                #     self.current_best_epoch, self.total_epochs, self.print_json
                # )

                json_results = df_results.to_json(orient="records")
                json_results = json.loads(json_results)[0]
                logger.info(
                    f"Best results json:\n {json.dumps(json_results, indent=4)}"
                )

        else:
            logger.error(f"Hyperopt error - no results - {results}")
            logger.info(f"Hyperopt finished - No epochs evaluated yet, no best result.")

        # print(self.current_best_epoch.metrics)
        # {'Trades': '4681', 'Win_Draw_Loss_Win_perc': '3517     0  1164  75.1', 'Avg_profit': '  3.12%', 'Profit': '195340381.096 USDT (19,534,038.11%)', 'Avg_duration': '0 days 21:49:00', 'Objective': '-38,157,864.48667', 'is_profit': True, 'Max_Drawdown_Acct': '  5274021.894 USDT    (5.18%)', 'loss': -38157864.486667246, 'timestamp': 1718690291, 'checkpoint_dir_name': None, 'done': True, 'training_iteration': 1, 'trial_id': '06452780', 'date': '2024-06-18_08-58-11', 'time_this_iter_s': 61.3154194355011, 'time_total_s': 61.3154194355011, 'pid': 1931179, 'hostname': 'vioUbuntu2', 'node_ip': '10.0.0.251', 'config': {'buy_fastk_rsi_patterns': 95, 'buy_max_slippage': 1.075, 'buy_prev_cbuys_count': 3, 'buy_prev_cbuys_rwindow': 5, 'buy_prev_min_close_age': 8, 'buy_prev_min_close_perc': 37.4, 'buy_prev_min_close_rwindow': 5, 'buy_proposed_stake_limit': 3731, 'buy_proposed_stake_limit_margin': 0.208, 'csl_5_step1_SL': 0.052, 'csl_5_step1_time': 591.366, 'csl_5_step2_SL': 0.035, 'csl_5_step2_time': 1625.187, 'csl_5_step3_SL': 0.075, 'csl_5_step3_time': 3717.144, 'csl_5_step4_SL': 0.248, 'sell_order_max_age': 2.8, 'sell_order_min_profit': 0.06, 'stoploss': -0.097}, 'time_since_restore': 61.3154194355011, 'iterations_since_restore': 1, 'experiment_tag': '139_buy_fastk_rsi_patterns=95,buy_max_slippage=1.0750,buy_prev_cbuys_count=3,buy_prev_cbuys_rwindow=5,buy_prev_min_close_age=8,buy_prev_min_close_perc=37.4000,buy_prev_min_close_rwindow=5,buy_proposed_stake_limit=3731,buy_proposed_stake_limit_margin=0.2080,csl_5_step1_SL=0.0520,csl_5_step1_time=591.3660,csl_5_step2_SL=0.0350,csl_5_step2_time=1625.1870,csl_5_step3_SL=0.0750,csl_5_step3_time=3717.1440,csl_5_step4_SL=0.2480,sell_order_max_age=2.8000,sell_order_min_profit=0.0600,stoploss=-0.0970'}


# https://github.com/Textualize/rich/discussions/482
class myLoggerCallback(LoggerCallback):
    def __init__(
        self,
        strategy="",
        print_all=False,
        total_epochs=-1,
        table_max_rows=-1,
        plot_metric="",
        min_refresh_time=3,  # seconds
    ) -> None:

        self.console_width = Console().width
        self.console_width_plot = self.console_width - 4
        self.console_height = Console().height
        self.decoder = AnsiDecoder()
        self.refresh_enabled = True

        if table_max_rows <= 0:
            table_max_rows = self.console_height // 3

        self.min_refresh_time = min_refresh_time
        self.trial_results = deque(maxlen=table_max_rows)  # []
        self.plot_trial_results = []
        # deque(maxlen=min(int(0.9*self.console_width), self.console_width-14)) #
        self.plot_trial_results_len = min(
            int(0.9 * self.console_width), self.console_width - 14
        )
        self.best_loss = MAX_LOSS
        self.print_all = print_all
        self.plot_metric = plot_metric
        if total_epochs <= 0:
            logger.warning(
                f"Please set total_epochs for myLoggerCallback - {total_epochs}"
            )
        self.total_epochs = total_epochs
        self.strategy = strategy
        self.count_trials = 0
        self.best_epoch = "N/A"
        self._trial_ids = set()

        self.live = None
        self.table = Table(expand=True)
        self.table_columns = [
            "Trial",
            "Trades",
            "Win  Draw  Loss  Win%",
            "Avg profit%",
            "Profit Total",
            "Winrate%",
            "Avg duration",
            "Objective",
            "Max Drawdown",
            # "Epoch",
            "TTR",
        ]
        for col in self.table_columns:
            self.table.add_column(col)
        self.table_master = self.generate_empty_table()
        self.refresh_chart = 100
        self.last_refresh_time = time.time()

    def generate_empty_table(self) -> Table:
        return Table(
            title=f"{self.strategy} {self.plot_metric} - Epoch: {self.count_trials}/{self.total_epochs} - Best: {self.best_epoch}",
            title_style=Style(color="white", bgcolor="black", bold=True),
            show_header=False,
            padding=(0, 0),
            expand=True,
        )

    # def resize_list(self, list_in: [], max_len: int):
    #     if len(list_in) > max_len:
    #         list_out = []
    #         n_averaged_elements = (len(list_in) // max_len) + 1
    #         for i in range(0, len(list_in), n_averaged_elements):
    #             slice_from_index = i
    #             slice_to_index = slice_from_index + n_averaged_elements
    #             if self.plot_metric in ["Profit", "Winrate"]:
    #                 list_out.append(np.max(list_in[slice_from_index:slice_to_index]))
    #             elif self.plot_metric == "loss":
    #                 list_out.append(np.min(list_in[slice_from_index:slice_to_index]))
    #             else:
    #                 list_out.append(np.mean(list_in[slice_from_index:slice_to_index]))
    #         list_out = list_out[-max_len:]
    #         return list_out
    #     else:
    #         return list_in

    def plot_chart_fn(self, width: int, height: int, plot_list: list, title: str = ""):
        plt.clf()
        len_plot_list = len(plot_list)
        x = range(1, len_plot_list + 1)
        plt.plot(x, plot_list, marker="hd")  # dot fhd hd
        if len_plot_list > 10:
            xticks = [i for i in range(1, len_plot_list + 1, len_plot_list // 10)]
            xlabels = [
                f"{(i):,.0f}" for i in range(1, len_plot_list + 1, len_plot_list // 10)
            ]
        else:
            xticks = [i for i in range(1, len_plot_list + 1, 1)]
            xlabels = [f"{(i):,.0f}" for i in range(1, len_plot_list + 1, 1)]
        plt.xticks(xticks, xlabels)
        plt.plotsize(width, height)
        if len(title) > 0:
            plt.title(title)
        plt.theme("dark")
        return plt.build()

    def generate_table(self) -> Table:
        """Make a new table."""
        self.table_master = self.generate_empty_table()

        self.table = Table(
            # title=f"{self.strategy} - Epoch {self.count_trials}/{self.total_epochs}",
            expand=True,
        )
        for col in self.table_columns:
            self.table.add_column(col)

        for result in self.trial_results:
            self.table.add_row(*result)

        self.table_master.add_row(self.table)

        plot_list = None
        if self.plot_metric and len(self.plot_metric) > 0:
            if self.plot_metric in plot_metric_list:
                plot_list = []
            else:
                logger.error(
                    f"plot_metric {self.plot_metric} not in {plot_metric_list}"
                )
                self.plot_metric = ""

        if plot_list is not None:
            for result in self.plot_trial_results:
                # if self.plot_metric == "Profit":
                #     profit = result  # [plot_metric_list.index(self.plot_metric)]
                #     try:
                #         profit = (
                #             profit.split("(")[1]
                #             .replace(")", "")
                #             .replace("%", "")
                #             .replace(",", "")
                #         )
                #         profit = float(profit)
                #         plot_list.append(profit)
                #     except:
                #         # print(result)
                #         # profit = math.nan
                #         plot_list.append(nan)
                #         pass
                # else:
                #     result = float(result)
                #     plot_list.append(
                #         result
                #     )  # [plot_metric_list.index(self.plot_metric)]
                plot_list.append(result)

        # print("plot_metric", self.plot_metric, "len trial_results", len(self.trial_results),  "plot_list", plot_list)
        if plot_list and len(plot_list) > 1:
            try:
                ## plot_list = plot_list[-int(0.9*self.console_width):]
                # rich_plot = acp.plot(
                #     self.resize_list(plot_list, self.plot_trial_results_len),
                #     {"height": self.console_height // 4, "format": "{:.4e}"},
                # )
                plot_list_arr = np.array(plot_list)
                if len(plot_list_arr[np.isnan(plot_list_arr) == False]) > 1:
                    plot_list_interp = np.interp(
                        np.arange(len(plot_list_arr)),
                        np.arange(len(plot_list_arr))[np.isnan(plot_list_arr) == False],
                        plot_list_arr[np.isnan(plot_list_arr) == False],
                    ).tolist()
                else:
                    plot_list_interp = plot_list
                plot = self.plot_chart_fn(
                    width=self.console_width_plot,
                    height=self.console_height // 4,
                    plot_list=plot_list_interp,
                )
                rich_plot = Group(*self.decoder.decode(plot))
                self.table_master.add_row(rich_plot)
                # print(len(plot_list), len(self.plot_trial_results))
            except Exception as e:
                print(f"myLoggerCallback - generate_table Error: {repr(e)}")
                logger.error(f"myLoggerCallback - generate_table Error: {repr(e)}")
                pass

        progress = int(self.count_trials * self.live.console.width / self.total_epochs)
        table_progress = Table(
            show_header=False,
            expand=True,
            pad_edge=False,
            show_lines=False,
            box=None,
        )
        table_progress.add_row(
            Text("Progress"),
            Bar(
                self.live.console.width,
                0,
                progress,
                color="yellow",
                bgcolor="black",
            ),
        )
        self.table_master.add_row(table_progress)

        table_memory = Table(
            show_header=False,
            expand=True,
            pad_edge=False,
            show_lines=False,
            box=None,
        )
        table_memory.add_row(
            Text("Memory  "),
            Bar(
                self.live.console.width,
                0,
                int(self.live.console.width * psutil.virtual_memory().percent / 100.0),
                color="green" if psutil.virtual_memory().percent < 90 else "red",
                bgcolor="black",
            ),
        )
        self.table_master.add_row(table_memory)

        table_cpu = Table(
            show_header=False,
            expand=True,
            pad_edge=False,
            show_lines=False,
            box=None,
        )
        table_cpu.add_row(
            Text("CPU     "),
            Bar(
                self.live.console.width,
                0,
                int(self.live.console.width * psutil.cpu_percent() / 100.0),
                color="blue",
                bgcolor="black",
            ),
        )
        self.table_master.add_row(table_cpu)

    # def on_step_begin(self, iteration, trials, **info):  ## too often
    #     # if self.live is None:
    #     #     self.live = Live(
    #     #         self.table_master,
    #     #         vertical_overflow="ellipsis",
    #     #         auto_refresh=False,
    #     #     )  # , screen=True : crop', 'ellipsis', 'visible', , refresh_per_second=0.2, transient=True,
    #     #     self.live.start(refresh=True)

    #     if self.refresh_enabled:
    #         start_date = time.time()
    #         if iteration % self.refresh_chart == 0 and self.live:
    #             # self.logger.warning(f"myLoggerCallback - on_step_begin - iteration: {iteration}")
    #             self.generate_table()
    #             self.live.update(self.table_master, refresh=True)
    #         if time.time() - start_date > 0.1:
    #             self.refresh_chart = int(2 * self.refresh_chart)
    #         self.last_refresh_time = time.time()

    def on_trial_start(self, iteration, trials, trial, **info):
        self._trial_ids.add(trial.trial_id)
        if self.live is None:
            self.live = Live(
                self.table_master,
                vertical_overflow="ellipsis",
                auto_refresh=False,
            )  # , screen=True : crop', 'ellipsis', 'visible', , refresh_per_second=0.2, transient=True,
            self.live.start(refresh=True)
            self.last_refresh_time = time.time()

        if self.refresh_enabled:
            if time.time() - self.last_refresh_time > self.min_refresh_time:
                self.generate_table()
                self.live.update(self.table_master, refresh=True)
                self.last_refresh_time = time.time()

    def append_trial_results(self, trial_id, result):
        # logger.info(f"append_trial_results result: {result}")
        loss = result["loss"]
        if abs(loss) > 100 or abs(loss) < 0.001:
            loss = f"{result['loss']:,.6e}"
        else:
            loss = f"{result['loss']:,.6f}"

        self.trial_results.append(
            (
                f"{trial_id}",
                f"{result['Trades']}",
                f"{result['Win_Draw_Loss_Win_perc']}",
                f"{(100*result['Avg_profit']):,.4f}",
                f"{(result['Profit']):,.2f}",
                f"{(result['Winrate']):,.2f}",
                f"{result['Avg_duration']}",
                loss,
                f"{(result['Max_Drawdown_Acct']):,.2f}",
                # f"{self.count_trials}",
                f"{(result['time_total_s']):,.2f}",
            )
        )

    def on_trial_result(self, iteration, trials, trial, result, **info):
        self.count_trials += 1  # len(trials)
        # print(
        #     f"Results for trial {trial} / iteration {iteration} / count trials = {self.count_trials}"
        # )
        # print(f"result: {result}")

        if self.print_all:
            self.append_trial_results(self.count_trials, result)
        elif result["loss"] < self.best_loss:
            self.best_loss = result["loss"]
            self.best_epoch = self.count_trials
            self.append_trial_results(self.count_trials, result)

        if self.plot_metric and len(self.plot_metric) > 0:
            self.plot_trial_results.append(result[self.plot_metric])

        if self.refresh_enabled:
            if time.time() - self.last_refresh_time > self.min_refresh_time:
                self.generate_table()
                self.live.update(self.table_master, refresh=True)
                self.last_refresh_time = time.time()

    def on_experiment_end(self, trials, **info):
        self.refresh_enabled = False
        if self.live and self.live.is_started:
            self.live.stop()

    def on_experiment_start(self, trials, **info):
        self.refresh_enabled = True

    def get_state(self) -> Optional[Dict]:
        return {"trial_ids": self._trial_ids.copy()}

    def set_state(self, state: Dict) -> Optional[Dict]:
        self._trial_ids = state["trial_ids"]


class myPBarCallback(LoggerCallback):
    def __init__(
        self,
        strategy="",
        total_epochs=-1,
    ) -> None:

        self.total_epochs = total_epochs
        self.strategy = strategy
        self.count_trials = 0
        self.pbar = None
        if total_epochs <= 0:
            logger.warning(
                f"Please set total_epochs > 0 for myLoggerCallback - {total_epochs}"
            )

    def on_trial_start(self, iteration, trials, trial, **info):
        if self.pbar is None:
            if self.total_epochs <= 0:
                self.pbar = ProgressBar().start()
            else:
                self.pbar = ProgressBar(maxval=self.total_epochs).start()

    def on_trial_result(self, iteration, trials, trial, result, **info):
        self.count_trials += 1
        self.pbar.update(self.count_trials)

    def on_experiment_end(self, trials, **info):
        self.pbar.finish()


class ExperimentPlateauStopper(Stopper):
    """Early stop the experiment when a metric plateaued across trials.

    Stops the entire experiment when the metric has plateaued
    for more than the given amount of iterations specified in
    the patience parameter.

    Args:
        metric: The metric to be monitored.
        std: The minimal standard deviation after which
            the tuning process has to stop.
        top: The number of best models to consider.
        mode: The mode to select the top results.
            Can either be "min" or "max".
        patience: Number of epochs to wait for
            a change in the top models.

    Raises:
        ValueError: If the mode parameter is not "min" nor "max".
        ValueError: If the top parameter is not an integer
            greater than 1.
        ValueError: If the standard deviation parameter is not
            a strictly positive float.
        ValueError: If the patience parameter is not
            a strictly positive integer.
    """

    def __init__(
        self,
        metric: str,
        perc: float = 0.001,
        std: float = 0.001,
        top: int = 10,
        mode: str = "min",
        patience: int = 0,
    ):
        if mode not in ("min", "max"):
            raise ValueError("The mode parameter can only be either min or max.")
        if not isinstance(top, int) or top <= 1:
            raise ValueError(
                "Top results to consider must be"
                " a positive integer greater than one."
            )
        if not isinstance(patience, int) or patience < 0:
            raise ValueError("Patience must be a strictly positive integer.")
        if not isinstance(std, float) or std <= 0:
            raise ValueError(
                "The standard deviation must be a strictly positive float number."
            )
        self._mode = mode
        self._metric = metric
        self._patience = patience
        self._iterations_plateau = 0
        self._iterations_noinc = 0
        self._perc = perc if perc > 1 else perc + 1.0
        self._std = std
        self._top = top
        self._top_values = []
        self._best_epoch = 0
        self._current_epoch = 0
        self._best_result = np.inf if mode == "min" else -np.inf
        self._trials_ids = []
        self._last_result = 0
        self.std_value = 0

    def __call__(self, trial_id, result):
        """Return a boolean representing if the tuning has to stop."""
        stop_all = False
        if trial_id not in self._trials_ids:
            self._trials_ids.append(trial_id)
            self._current_epoch += 1
            result_metric = result[self._metric]
            self._last_result = result_metric
            self._top_values.append(result_metric)
            if self._mode == "min":
                self._top_values = sorted(self._top_values)[: self._top]
                if result_metric < self._best_result * self._perc:
                    self._best_result = result_metric
                    self._best_epoch = self._current_epoch
                    self._iterations_noinc = 0
                else:
                    self._iterations_noinc += 1
            else:
                self._top_values = sorted(self._top_values)[-self._top :]
                if result_metric > self._best_result * self._perc:
                    self._best_result = result_metric
                    self._best_epoch = self._current_epoch
                    self._iterations_noinc = 0
                else:
                    self._iterations_noinc += 1

            # If the current iteration has to stop
            has_plateaued = self.has_plateaued()
            no_increase = self.no_increase()
            if has_plateaued:
                # we increment the total counter of iterations
                self._iterations_plateau += 1
            else:
                # otherwise we reset the counter
                self._iterations_plateau = 0

            # and then call the method that re-executes
            # the checks, including the iterations.
            stop_all = (
                has_plateaued and self._iterations_plateau >= self._patience
            ) or no_increase

        return stop_all

    def has_plateaued(self):
        return (
            len(self._top_values) == self._top
            and abs(np.std(self._top_values) / np.mean(self._top_values)) <= self._std
        )

    def no_increase(self):
        return self._current_epoch - self._best_epoch > self._patience

    def stop_all(self):
        """Return whether to stop and prevent trials from starting."""
        stop_all = (
            self.has_plateaued() and self._iterations_plateau >= self._patience
        ) or (self.no_increase() and self._iterations_noinc >= self._patience)
        self.std_value = abs(np.std(self._top_values) / np.mean(self._top_values))
        if stop_all:
            # logger.info(
            #     f"ExperimentPlateauStopper - current_epoch: {self._current_epoch} / best_epoch: {self._best_epoch} / "
            #     f"iterations_plateau: {self._iterations_plateau}/ iterations_noinc: {self._iterations_noinc} / patience: {self._patience} / "
            #     f"has_plateaued: {self.has_plateaued()} / no_increase: {self.no_increase()} / std: {self.std_value} / "
            #     f"last_result: {self._last_result} / best_result: {self._best_result}"
            # )
            print(
                f"ExperimentPlateauStopper - current_epoch: {self._current_epoch} / best_epoch: {self._best_epoch} / "
                f"iterations_plateau: {self._iterations_plateau}/ iterations_noinc: {self._iterations_noinc} / patience: {self._patience} / "
                f"has_plateaued: {self.has_plateaued()} / no_increase: {self.no_increase()} / std: {self.std_value} / "
                f"last_result: {self._last_result} / best_result: {self._best_result}"
            )
        return stop_all


def port_in_use(port):
    try:
        all_connections = psutil.net_connections()
        for conn in all_connections:
            if conn.laddr.port == port:
                return True
    except:  # for os x
        pass
    return False


# print(port_in_use(8265))


def find_first_free_port(port):
    for i in range(100):
        if not port_in_use(port + i):
            return port + i
    return None
