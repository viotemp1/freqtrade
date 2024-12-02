# pragma pylint: disable=too-many-instance-attributes, pointless-string-statement

### TO DO
"""
This module contains the hyperopt logic
"""

import logging
import random
import sys, os
import json
import gc
import warnings
from copy import deepcopy
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import psutil
import time
from time import sleep

import rapidjson

# from joblib import Parallel, cpu_count, delayed, dump, load, wrap_non_picklable_objects
from colorama import init as colorama_init
from joblib import cpu_count, dump, load
from joblib.externals import cloudpickle
from pandas import DataFrame
import pandas as pd

from freqtrade.constants import (
    DATETIME_PRINT_FORMAT,
    FTHYPT_FILEVERSION,
    LAST_BT_RESULT_FN,
    Config,
)
from freqtrade.data.converter import trim_dataframes
from freqtrade.data.history import get_timerange
from freqtrade.data.metrics import calculate_market_change
from freqtrade.enums import HyperoptState
from freqtrade.exceptions import OperationalException
from freqtrade.misc import deep_merge_dicts, file_dump_json, plural
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.optimize.space import Categorical
from freqtrade.optimize.space.decimalspace import SKDecimal
from skopt.space.space import Integer

# Import IHyperOpt and IHyperOptLoss to allow unpickling classes from these modules
from freqtrade.optimize.hyperopt_auto import HyperOptAuto
from freqtrade.optimize.hyperopt_loss_interface import IHyperOptLoss
from freqtrade.optimize.hyperopt_output import HyperoptOutput
from freqtrade.optimize.hyperopt_tools import (
    HyperoptStateContainer,
    HyperoptTools,
    hyperopt_serializer,
)
from freqtrade.optimize.optimize_reports import generate_strategy_stats
from freqtrade.resolvers.hyperopt_resolver import HyperOptLossResolver
from freqtrade.util import get_progress_tracker

from tabulate import tabulate
import numpy as np
from collections import deque

from rich.live import Live
from rich.table import Table
from rich.console import Console
from rich.bar import Bar
from rich.text import Text
from rich.style import Style
import asciichartpy as acp
import setproctitle

# Suppress scikit-learn FutureWarnings from skopt
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from skopt import Optimizer
    from skopt.space import Dimension
    import ray
    from ray import tune, train
    from ray.train import RunConfig
    from ray.tune.search import ConcurrencyLimiter

    # from ray.tune.stopper import ExperimentPlateauStopper
    from ray.util.state import summarize_tasks
    from ray.tune.logger import LoggerCallback
    from ray.tune.stopper.stopper import Stopper


ray_results_table_max_rows = -1  # -1 - half screen
ray_reuse_actors = False

# max_used_memory = 80  # 0 or negative to deactivate, otherwise pause worker

MAX_LOSS = (
    100000  # just a big enough number to be bad result in loss optimization
)

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


def ray_setup_func():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    os.environ["RAY_TQDM"] = "1"
    os.environ["RAY_PROFILING"] = "0"
    os.environ["RAY_DEDUP_LOGS"] = "0"
    # os.environ["RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING"] = "1"
    # os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = (
        f"{max(4,cpu_count()//4)}"  # f"{max(4,cpu_count()//2)}" 2
    )
    # os.environ["FUNCTION_SIZE_WARN_THRESHOLD"] = f"{2 * 10**7}"
    # os.environ["RAY_memory_monitor_refresh_ms"] = "0" # disable memory check
    # os.environ["RAY_memory_usage_threshold"] = "1"

    os.environ["SPT_NOENV"] = "1"

    return logger


logger = ray_setup_func()


# def trial_str_creator(trial):
# return ""


class Hyperopt:
    """
    Hyperopt class, this class contains all the logic to run a hyperopt simulation

    To start a hyperopt run:
    hyperopt = Hyperopt(config)
    hyperopt.start()
    """

    def __init__(self, config: Config) -> None:
        self.buy_space: List[Dimension] = []
        self.sell_space: List[Dimension] = []
        self.protection_space: List[Dimension] = []
        self.roi_space: List[Dimension] = []
        self.stoploss_space: List[Dimension] = []
        self.trailing_space: List[Dimension] = []
        self.max_open_trades_space: List[Dimension] = []
        self.dimensions: Dict = {}

        self._hyper_out: HyperoptOutput = HyperoptOutput(streaming=True)

        self.config = config
        self.min_date: datetime
        self.max_date: datetime

        self.backtesting = Backtesting(self.config)
        self.pairlist = self.backtesting.pairlists.whitelist
        self.custom_hyperopt: HyperOptAuto
        self.analyze_per_epoch = self.config.get("analyze_per_epoch", False)
        HyperoptStateContainer.set_state(HyperoptState.STARTUP)

        if not self.config.get("hyperopt"):
            self.custom_hyperopt = HyperOptAuto(self.config)
        else:
            raise OperationalException(
                "Using separate Hyperopt files has been removed in 2021.9. Please convert "
                "your existing Hyperopt file to the new Hyperoptable strategy interface"
            )

        self.backtesting._set_strategy(self.backtesting.strategylist[0])
        self.custom_hyperopt.strategy = self.backtesting.strategy

        self.hyperopt_pickle_magic(
            self.backtesting.strategy.__class__.__bases__
        )
        self.custom_hyperoptloss: IHyperOptLoss = (
            HyperOptLossResolver.load_hyperoptloss(self.config)
        )
        self.calculate_loss = self.custom_hyperoptloss.hyperopt_loss_function
        time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.strategy_name = str(self.config["strategy"])
        self.results_file: Path = (
            self.config["user_data_dir"]
            / "hyperopt_results"
            / f"strategy_{self.strategy_name}_{time_now}.fthypt"
        )
        self.data_pickle_file = (
            self.config["user_data_dir"]
            / "hyperopt_results"
            / "hyperopt_tickerdata.json"
        )
        self.detail_data_pickle_file = (
            self.config["user_data_dir"]
            / "hyperopt_results"
            / "hyperopt_detail_tickerdata.json"
        )

        self.hyperopt_results_file: Path = (
            Path(self.config["user_data_dir"]).parent
            / "csv"
            / "hyperopt_results.csv"
        )
        self.ray_log_dir = os.path.join("/tmp", "ray", self.strategy_name)

        self.total_epochs = config.get("epochs", 0)
        self.current_best_loss = 100

        self.clean_hyperopt()

        self.market_change = 0.0
        self.num_epochs_saved = 0
        self.current_best_epoch: Optional[Dict[str, Any]] = None

        # Use max_open_trades for hyperopt as well, except --disable-max-market-positions is set
        if not self.config.get("use_max_market_positions", True):
            logger.debug(
                "Ignoring max_open_trades (--disable-max-market-positions was used) ..."
            )
            self.backtesting.strategy.max_open_trades = float("inf")
            config.update(
                {"max_open_trades": self.backtesting.strategy.max_open_trades}
            )

        if HyperoptTools.has_space(self.config, "sell"):
            # Make sure use_exit_signal is enabled
            self.config["use_exit_signal"] = True

        self.print_all = self.config.get("print_all", False)
        # self.print_hyperopt_results = self.config.get("print_hyperopt_results", True)
        self.plot_chart = os.environ.get("RAY_PLOT_CHART", None)
        if self.plot_chart and self.plot_chart.lower() == "false":
            self.plot_chart = False
        else:
            self.plot_chart = self.config.get("plot_chart", True)
        self.print_json = self.config.get("print_json", True)
        if hasattr(self.backtesting.strategy, "plot_metric"):
            self.plot_metric = getattr(self.backtesting.strategy, "plot_metric")
        else:
            self.plot_metric = self.config.get(
                "plot_metric", "Profit"
            )  # Profit Winrate
        self.save_results_to_csv = self.config.get("save_results_to_csv", True)
        self.ray_early_stop_enable = self.config.get(
            "ray_early_stop_enable", True
        )
        self.ray_early_stop_perc = self.config.get(
            "ray_early_stop_perc", 0.005
        )  # 0.001
        self.ray_early_stop_std = self.config.get(
            "ray_early_stop_std", 0.005
        )  # 0.001
        self.ray_early_stop_top = self.config.get("ray_early_stop_top", 10)
        self.ray_early_stop_patience = self.config.get(
            "ray_early_stop_patience", 0.25
        )
        self.ray_dashboard = self.config.get("ray_dashboard", False)
        self.ray_dashboard_port = self.config.get("ray_dashboard_port", 8265)
        self.ray_max_memory_perc = min(
            float(config.get("ray_max_memory", 0.8)),
            float(os.environ.get("RAY_MAX_MEMORY_PERC", 0.8)),
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

        logger.info(f"ray_max_memory: {(self.ray_max_memory):,.2f}. ray_max_memory_perc: {(self.ray_max_memory_perc):,.2f}")

        self.config_jobs = self.config.get("hyperopt_jobs", cpu_count())

        # self.hyperopt_table_header = 0
        # self.print_colorized = self.config.get("print_colorized", False)

        ray_setup_func()

    @staticmethod
    def get_lock_filename(config: Config) -> str:
        return str(config["user_data_dir"] / "hyperopt.lock")

    @staticmethod
    def trial_str_creator(strategy_name, trial):
        return "{}_{}_{}".format(
            strategy_name, trial.trainable_name, trial.trial_id
        )

    def clean_hyperopt(self) -> None:
        """
        Remove hyperopt pickle files to restart hyperopt.
        """
        for f in [
            self.data_pickle_file,
            self.results_file,
            self.detail_data_pickle_file,
        ]:
            p = Path(f)
            if p.is_file():
                logger.info(f"Removing `{p}`.")
                p.unlink()

    def hyperopt_pickle_magic(self, bases) -> None:
        """
        Hyperopt magic to allow strategy inheritance across files.
        For this to properly work, we need to register the module of the imported class
        to pickle as value.
        """
        for modules in bases:
            if modules.__name__ != "IStrategy":
                cloudpickle.register_pickle_by_value(
                    sys.modules[modules.__module__]
                )
                self.hyperopt_pickle_magic(modules.__bases__)

    # def _save_result(self, epoch: Dict) -> None:
    @staticmethod
    def _save_result(epoch: Dict, results_file: Path) -> None:
        """
        Save hyperopt results to file
        Store one line per epoch.
        While not a valid json object - this allows appending easily.
        :param epoch: result dictionary for this epoch.
        """
        epoch[FTHYPT_FILEVERSION] = 2
        with results_file.open("a") as f:
            rapidjson.dump(
                epoch,
                f,
                default=hyperopt_serializer,
                number_mode=rapidjson.NM_NATIVE | rapidjson.NM_NAN,
            )
            f.write("\n")

        # num_epochs_saved += 1
        logger.debug(
            f"{num_epochs_saved} {plural(num_epochs_saved, 'epoch')} "
            f"saved to '{results_file}'."
        )
        # Store hyperopt filename
        latest_filename = Path.joinpath(results_file.parent, LAST_BT_RESULT_FN)
        file_dump_json(
            latest_filename,
            {"latest_hyperopt": str(results_file.name)},
            log=False,
        )

    @staticmethod
    def assign_params(
        backtesting: Backtesting, params_dict: Dict, category: str
    ) -> None:
        """
        Assign hyperoptable parameters
        """
        for attr_name, attr in backtesting.strategy.enumerate_parameters(
            category
        ):
            if attr.optimize:
                # noinspection PyProtectedMember
                attr.value = params_dict[attr_name]

    @staticmethod
    def _get_params_dict(dimensions: {}, raw_params: {}) -> Dict:
        # Ensure the number of dimensions match
        # the number of parameters in the list.
        if len(raw_params) != len(dimensions):
            raise ValueError("Mismatch in number of search-space dimensions.")

        # Return a dict where the keys are the names of the dimensions
        # and the values are taken from the list of parameters.
        # return {d.name: v for d, v in zip(dimensions, raw_params)}
        return raw_params

    def _get_params_details(self, params: Dict) -> Dict:
        """
        Return the params for each space
        """
        result: Dict = {}

        if HyperoptTools.has_space(self.config, "buy"):
            result["buy"] = {p.name: params.get(p.name) for p in self.buy_space}
        if HyperoptTools.has_space(self.config, "sell"):
            result["sell"] = {
                p.name: params.get(p.name) for p in self.sell_space
            }
        if HyperoptTools.has_space(self.config, "protection"):
            result["protection"] = {
                p.name: params.get(p.name) for p in self.protection_space
            }
        if HyperoptTools.has_space(self.config, "roi"):
            result["roi"] = {
                str(k): v
                for k, v in self.custom_hyperopt.generate_roi_table(
                    params
                ).items()
            }
        if HyperoptTools.has_space(self.config, "stoploss"):
            result["stoploss"] = {
                p.name: params.get(p.name) for p in self.stoploss_space
            }
        if HyperoptTools.has_space(self.config, "trailing"):
            result["trailing"] = self.custom_hyperopt.generate_trailing_params(
                params
            )
        if HyperoptTools.has_space(self.config, "trades"):
            result["max_open_trades"] = {
                "max_open_trades": (
                    self.backtesting.strategy.max_open_trades
                    if self.backtesting.strategy.max_open_trades != float("inf")
                    else -1
                )
            }

        return result

    def _get_no_optimize_details(self) -> Dict[str, Any]:
        """
        Get non-optimized parameters
        """
        result = {}

        if not HyperoptTools.has_space(self.config, "roi"):
            result["roi"] = {
                str(k): v
                for k, v in self.backtesting.strategy.minimal_roi.items()
            }
            # print("roi", result["roi"])
        if not HyperoptTools.has_space(self.config, "stoploss"):
            result["stoploss"] = {
                "stoploss": self.backtesting.strategy.stoploss
            }
            # print("stoploss", result["stoploss"])
        if not HyperoptTools.has_space(self.config, "trailing"):
            result["trailing"] = {
                "trailing_stop": self.backtesting.strategy.trailing_stop,
                "trailing_stop_positive": self.backtesting.strategy.trailing_stop_positive,
                "trailing_stop_positive_offset": self.backtesting.strategy.trailing_stop_positive_offset,
                "trailing_only_offset_is_reached": self.backtesting.strategy.trailing_only_offset_is_reached,
            }
            # print("trailing", result["trailing"])
        if not HyperoptTools.has_space(self.config, "trades"):
            result["max_open_trades"] = {
                "max_open_trades": self.backtesting.strategy.max_open_trades
            }
            # print("max_open_trades", result["max_open_trades"])
        # print("_get_no_optimize_details result", result)
        return result

    def init_spaces(self):
        """
        Assign the dimensions in the hyperoptimization space.
        """
        if HyperoptTools.has_space(self.config, "protection"):
            # Protections can only be optimized when using the Parameter interface
            logger.debug("Hyperopt has 'protection' space")
            # Enable Protections if protection space is selected.
            self.config["enable_protections"] = True
            self.backtesting.enable_protections = True
            self.protection_space = self.custom_hyperopt.protection_space()

        if HyperoptTools.has_space(self.config, "buy"):
            logger.debug("Hyperopt has 'buy' space")
            self.buy_space = self.custom_hyperopt.buy_indicator_space()

        if HyperoptTools.has_space(self.config, "sell"):
            logger.debug("Hyperopt has 'sell' space")
            self.sell_space = self.custom_hyperopt.sell_indicator_space()

        if HyperoptTools.has_space(self.config, "roi"):
            logger.debug("Hyperopt has 'roi' space")
            self.roi_space = self.custom_hyperopt.roi_space()

        if HyperoptTools.has_space(self.config, "stoploss"):
            logger.debug("Hyperopt has 'stoploss' space")
            self.stoploss_space = self.custom_hyperopt.stoploss_space()

        if HyperoptTools.has_space(self.config, "trailing"):
            logger.debug("Hyperopt has 'trailing' space")
            self.trailing_space = self.custom_hyperopt.trailing_space()

        if HyperoptTools.has_space(self.config, "trades"):
            logger.debug("Hyperopt has 'trades' space")
            self.max_open_trades_space = (
                self.custom_hyperopt.max_open_trades_space()
            )

        self.dimensions = {}
        dimensions = (
            self.buy_space
            + self.sell_space
            + self.protection_space
            + self.roi_space
            + self.stoploss_space
            + self.trailing_space
            + self.max_open_trades_space
        )
        for original_dim in dimensions:
            # print(original_dim.name, original_dim, type(original_dim))
            if (
                type(original_dim) == Integer
            ):  # isinstance(original_dim, Integer):
                # print("Integer", original_dim.low, original_dim.high)
                if (
                    self.searcher == "bayesopt"
                ):  # 'bayesopt' - does not suport randint
                    logger.info(
                        f"bayesopt does not support Integer. Will convert to tune.uniform. Please change {original_dim.name} to int in your strategy"
                    )
                    self.dimensions[original_dim.name] = tune.uniform(
                        original_dim.low,
                        original_dim.high,
                    )
                else:
                    self.dimensions[original_dim.name] = tune.randint(
                        original_dim.low, original_dim.high
                    )
            elif (
                type(original_dim) == SKDecimal
            ):  # isinstance(original_dim, SKDecimal):
                # print("SKDecimal", original_dim.low_orig, original_dim.high_orig, 1 / pow(10, original_dim.decimals))
                if (
                    self.searcher == "bayesopt"
                ):  # 'bayesopt' - does not suport quniform
                    self.dimensions[original_dim.name] = tune.uniform(
                        original_dim.low_orig,
                        original_dim.high_orig,
                    )
                else:
                    self.dimensions[original_dim.name] = tune.quniform(
                        original_dim.low_orig,
                        original_dim.high_orig,
                        1 / pow(10, original_dim.decimals),
                    )
            elif (
                type(original_dim) == Categorical
            ):  # isinstance(original_dim, Categorical):
                # print("Categorical", list(original_dim.bounds))
                self.dimensions[original_dim.name] = tune.choice(
                    list(original_dim.bounds)
                )
            else:
                # print(f"Unknown search space {original_dim} / {type(original_dim)}")
                raise Exception(
                    f"Unknown search space {original_dim} / {type(original_dim)}"
                )

    def _get_results_dict(
        self,
        backtesting,
        backtesting_results,
        min_date,
        max_date,
        params_dict,
        processed: Dict[str, DataFrame],
    ) -> Dict[str, Any]:
        params_details = self._get_params_details(params_dict)

        strat_stats = generate_strategy_stats(
            self.pairlist,
            backtesting.strategy.get_strategy_name(),
            backtesting_results,
            min_date,
            max_date,
            market_change=self.market_change,
            is_hyperopt=True,
        )
        results_explanation = HyperoptTools.format_results_explanation_string(
            strat_stats, self.config["stake_currency"]
        )

        # print("_get_results_dict strat_stats", strat_stats)
        trade_count = strat_stats["total_trades"]
        total_profit = strat_stats["profit_total"]

        # If this evaluation contains too short amount of trades to be
        # interesting -- consider it as 'bad' (assigned max. loss value)
        # in order to cast this hyperspace point away from optimization
        # path. We do not want to optimize 'hodl' strategies.
        loss: float = MAX_LOSS
        if trade_count >= self.config["hyperopt_min_trades"]:
            loss = self.calculate_loss(
                results=backtesting_results["results"],
                trade_count=trade_count,
                min_date=min_date,
                max_date=max_date,
                config=self.config,
                processed=processed,
                backtest_stats=strat_stats,
            )
        gc.collect()
        return {
            "loss": loss,
            "params_dict": params_dict,
            "params_details": params_details,
            # "params_not_optimized": not_optimized,
            "results_metrics": strat_stats,
            "results_explanation": results_explanation,
            "total_profit": total_profit,
        }

    # searchers: ['variant_generator', 'random', 'hyperopt', 'bohb', 'nevergrad', 'optuna', 'zoopt', 'hebo']
    # 'bayesopt' - not suported - does not suport Integer
    # 'ax' - not working
    # schedulers: ['fifo', 'async_hyperband', 'asynchyperband', 'median_stopping_rule', 'medianstopping', 'hyperband', 'hb_bohb', 'pbt', 'pbt_replay', 'pb2', 'resource_changing']
    def get_search_algo_scheduler(self, dimensions: Dict, config_jobs):
        searcher_orig = self.custom_hyperopt.generate_estimator(
            dimensions=dimensions
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
                                algorithm=self.searcher_param1.replace(
                                    "pymoo_", ""
                                )
                            ),
                        )
                    else:
                        search_algo = NevergradSearch(
                            optimizer=ng.optimizers.registry[
                                self.searcher_param1
                            ],
                        )
                else:
                    searcher_algo = tune.create_searcher(
                        searcher,
                        random_state_seed=self.random_state,
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
                    random_state_seed=self.random_state,
                    budget=self.total_epochs,
                    **zoopt_search_config,
                )
            elif searcher == "bohb":
                searcher_algo = tune.create_searcher(
                    searcher,
                    seed=self.random_state,
                )
            elif searcher == "bayesopt":
                searcher_algo = tune.create_searcher(
                    searcher,
                    random_state=self.random_state,
                )
            elif searcher == "optuna":
                import optuna

                # TPESampler NSGAIIISampler CmaEsSampler GPSampler NSGAIISampler QMCSampler
                if self.searcher_param1:
                    if self.searcher_param1 == "NSGAIIISampler":
                        searcher_algo = tune.create_searcher(
                            searcher,
                            sampler=optuna.samplers.NSGAIIISampler(
                                seed=self.random_state
                            ),
                        )
                    elif self.searcher_param1 == "CmaEsSampler":
                        searcher_algo = tune.create_searcher(
                            searcher,
                            sampler=optuna.samplers.CmaEsSampler(
                                seed=self.random_state,
                                warn_independent_sampling=False,
                            ),
                        )
                    elif self.searcher_param1 == "GPSampler":
                        searcher_algo = tune.create_searcher(
                            searcher,
                            sampler=optuna.samplers.GPSampler(
                                seed=self.random_state,
                                deterministic_objective=True,
                            ),
                        )
                    elif self.searcher_param1 == "NSGAIISampler":
                        searcher_algo = tune.create_searcher(
                            searcher,
                            sampler=optuna.samplers.NSGAIISampler(
                                seed=self.random_state
                            ),
                        )
                    elif self.searcher_param1 == "TPESampler":
                        searcher_algo = tune.create_searcher(
                            searcher,
                            sampler=optuna.samplers.TPESampler(
                                seed=self.random_state, multivariate=True
                            ),  # , group=True
                        )
                    elif self.searcher_param1 == "QMCSampler":
                        searcher_algo = tune.create_searcher(
                            searcher,
                            sampler=optuna.samplers.QMCSampler(
                                seed=self.random_state,
                                warn_independent_sampling=False,
                            ),
                        )
                    elif self.searcher_param1 == "BoTorchSampler":
                        from optuna.exceptions import (
                            ExperimentalWarning as o_ExperimentalWarning,
                        )

                        warnings.filterwarnings(
                            "ignore", category=o_ExperimentalWarning
                        )
                        searcher_algo = tune.create_searcher(
                            searcher,
                            sampler=optuna.integration.BoTorchSampler(
                                seed=self.random_state
                            ),
                        )
                    else:  # default
                        searcher_algo = tune.create_searcher(
                            searcher,
                            sampler=optuna.samplers.TPESampler(
                                seed=self.random_state, multivariate=True
                            ),  # default
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
                        random_state_seed=self.random_state,
                        model_name=self.searcher_param1,
                        scramble_seed=self.random_state,
                    )
                else:  # default
                    searcher_algo = tune.create_searcher(
                        searcher,
                        random_state_seed=self.random_state,
                        model_name="gp",
                        scramble_seed=self.random_state,
                    )
            else:
                searcher_algo = tune.create_searcher(
                    searcher, random_state_seed=self.random_state
                )
        except:
            # searcher = ConcurrencyLimiter(
            #     tune.create_searcher(searcher), max_concurrent=config_jobs
            # )
            logger.warning(
                f"Cannot set random_state_seed {self.random_state} for {self.searcher}"
            )
            searcher_algo = tune.create_searcher(searcher)
            pass

        if isinstance(self.searcher, str) and self.searcher == "bohb":
            scheduler = tune.create_scheduler("hb_bohb")
        else:
            scheduler = tune.create_scheduler("fifo")
        self.scheduler = scheduler
        return searcher_algo, scheduler

    def _set_random_state(self, random_state: Optional[int]) -> int:
        return random_state or random.randint(1, 2**16 - 1)  # noqa: S311

    def advise_and_trim(
        self, data: Dict[str, DataFrame]
    ) -> Dict[str, DataFrame]:
        preprocessed = self.backtesting.strategy.advise_all_indicators(data)

        # Trim startup period from analyzed dataframe to get correct dates for output.
        # This is only used to keep track of min/max date after trimming.
        # The result is NOT returned from this method, actual trimming happens in backtesting.
        trimmed = trim_dataframes(
            preprocessed, self.timerange, self.backtesting.required_startup
        )
        self.min_date, self.max_date = get_timerange(trimmed)
        if not self.market_change:
            self.market_change = calculate_market_change(trimmed, "close")

        # Real trimming will happen as part of backtesting.
        return preprocessed

    def prepare_hyperopt_data(self) -> None:
        HyperoptStateContainer.set_state(HyperoptState.DATALOAD)
        data, self.timerange = self.backtesting.load_bt_data()
        self.backtesting.load_bt_data_detail()
        logger.info("Dataload complete. Calculating indicators")

        if not self.analyze_per_epoch:
            HyperoptStateContainer.set_state(HyperoptState.INDICATORS)

            preprocessed = self.advise_and_trim(data)

            logger.info(
                f"Hyperopting with data from "
                f"{self.min_date.strftime(DATETIME_PRINT_FORMAT)} "
                f"up to {self.max_date.strftime(DATETIME_PRINT_FORMAT)} "
                f"({(self.max_date - self.min_date).days} days).."
            )
            # Store non-trimmed data - will be trimmed after signal generation.
            dump(preprocessed, self.data_pickle_file)
            if self.backtesting.timeframe_detail:
                dump(self.backtesting.detail_data, self.detail_data_pickle_file)
                self.backtesting.detail_data = {}
        else:
            dump(data, self.data_pickle_file)
            if self.backtesting.timeframe_detail:
                dump(self.backtesting.detail_data, self.detail_data_pickle_file)
                self.backtesting.detail_data = {}

    def ray_worker_logging_setup_func(self):
        logger = logging.getLogger("ray")
        logger.setLevel(logging.INFO)
        warnings.simplefilter("always")
        np.random.seed(self.random_state)
        random.seed(self.random_state)

    def start(self) -> None:
        self.random_state = self._set_random_state(
            self.config.get("hyperopt_random_state")
        )
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        logger.info(f"Using optimizer random state: {self.random_state}")
        # self.hyperopt_table_header = -1

        # Searcher
        search_algo, scheduler = self.get_search_algo_scheduler(
            None, self.config_jobs
        )
        # Initialize spaces ...
        self.init_spaces()

        self.prepare_hyperopt_data()

        # We don't need exchange instance anymore while running hyperopt
        self.backtesting.exchange.close()
        self.backtesting.exchange._api = None
        self.backtesting.exchange._api_async = None
        self.backtesting.exchange.loop = None  # type: ignore
        self.backtesting.exchange._loop_lock = None  # type: ignore
        self.backtesting.exchange._cache_lock = None  # type: ignore
        # self.backtesting.exchange = None  # type: ignore
        self.backtesting.pairlists = None  # type: ignore

        cpus = cpu_count()
        logger.info(f"Found {cpus} CPU cores. Let's make them scream!")
        logger.info(f"Number of parallel jobs set as: {self.config_jobs}")

        not_optimized = self.backtesting.strategy.get_no_optimize_params()
        not_optimized = deep_merge_dicts(
            not_optimized, self._get_no_optimize_details()
        )

        ray_setup_func()

        try:
            # print(f"ray.init - {os.getcwd()}")
            # print(self.backtesting.strategy.custom_trade_info)

            # Path("./logs").mkdir(parents=True, exist_ok=True)

            trainable_with_parameters = tune.with_parameters(
                objective,
                config_ft=self.config,
                backtesting=self.backtesting,
                custom_trade_info=(
                    self.backtesting.strategy.custom_trade_info
                    if hasattr(self.backtesting.strategy, "custom_trade_info")
                    else None
                ),
                dimensions_ft=self.dimensions,
                data_pickle_file_ft=self.data_pickle_file,
                detail_data_pickle_file_ft=self.detail_data_pickle_file,
                min_date_ft=self.min_date,
                max_date_ft=self.max_date,
                total_epochs_ft=self.total_epochs,
                custom_hyperopt_ft=self.custom_hyperopt,
                _get_results_dict_ft=self._get_results_dict,
                # _save_result_ft=self._save_result,
                results_file_ft=self.results_file,
                max_memory_per_worker=self.ray_max_memory // self.config_jobs
            )
            if self.ray_max_memory is None:
                trainable_with_resources = tune.with_resources(
                    trainable_with_parameters, {"CPU": cpus // self.config_jobs}
                )
            else:
                trainable_with_resources = tune.with_resources(
                    trainable_with_parameters,
                    tune.PlacementGroupFactory(
                        [
                            {
                                "CPU": cpus // config_jobs,
                                "memory": self.ray_max_memory // self.config_jobs,
                            }
                        ]
                    ),
                    # {
                    #     "cpu": cpus // self.config_jobs,
                    #     "memory": self.ray_max_memory // self.config_jobs,
                    # },
                )
            ray.init(
                ignore_reinit_error=True,
                include_dashboard=self.ray_dashboard,
                dashboard_port=self.ray_dashboard_port,  # None
                object_store_memory=0.05
                * psutil.virtual_memory().total,  # 10**9
                runtime_env={
                    "worker_process_setup_hook": self.ray_worker_logging_setup_func,
                    "env_vars": {
                        "PYTHONPATH": os.path.join(
                            self.config["user_data_dir"], "strategies"
                        )
                    },
                    "worker_process_setup_hook": ray_setup_func,
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

            if (
                self.print_all or self.plot_chart
            ):  # self.print_hyperopt_results or
                r_callbacks = [
                    myLoggerCallback(
                        strategy=self.strategy_name,
                        print_all=self.print_all,
                        total_epochs=self.total_epochs,
                        table_max_rows=ray_results_table_max_rows,
                        plot_metric=self.plot_metric,
                    )
                ]
            else:
                r_callbacks = None

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
                trainable_with_resources,
                tune_config=tune.TuneConfig(
                    metric="loss",
                    mode="min",
                    search_alg=search_algo,
                    scheduler=scheduler,
                    max_concurrent_trials=self.config_jobs,
                    reuse_actors=ray_reuse_actors,
                    num_samples=self.total_epochs,
                    # trial_name_creator=lambda trial: f"{self.strategy_name}_{trial.trainable_name}_{trial.trial_id}",
                ),
                param_space=self.dimensions,
                run_config=RunConfig(
                    # name=self.strategy_name,
                    verbose=0,
                    storage_path=self.ray_log_dir,
                    stop=stop_cb,
                    callbacks=r_callbacks,
                    log_to_file=False,
                ),
            )

            results = tuner.fit()

        except KeyboardInterrupt:
            print("User interrupted..")
            if ray.is_initialized():
                ray.shutdown()

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

        self.total_epochs = results.num_terminated
        logger.info(
            f"fHyperopt finished - OK: {results.num_terminated} / Failed: {results.num_errors}"
        )

        if self.current_best_epoch is None:
            self.current_best_epoch = {}
        self.current_best_epoch["tune_best_result"] = results.get_best_result(
            metric="loss", mode="min"
        )

        # df_results = self.current_best_epoch["tune_best_result"].metrics_dataframe
        df_results = results.get_dataframe(
            filter_metric="loss", filter_mode="min"
        )
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
                self._get_params_details(
                    self.current_best_epoch["tune_best_result"].config
                )
            )
            self.current_best_epoch["params_not_optimized"] = deepcopy(
                not_optimized
            )

            HyperoptTools.try_export_params(
                self.config,
                self.backtesting.strategy.get_strategy_name(),
                self.current_best_epoch,
            )

            HyperoptTools.show_epoch_details(
                self.current_best_epoch, self.total_epochs, self.print_json
            )

            json_results = df_results.to_json(orient="records")
            json_results = json.loads(json_results)[0]
            logger.info(
                f"Best results json:\n {json.dumps(json_results, indent=4)}"
            )

        else:
            logger.info(f"No epochs evaluated yet, no best result.")

        # print(self.current_best_epoch.metrics)
        # {'Trades': '4681', 'Win_Draw_Loss_Win_perc': '3517     0  1164  75.1', 'Avg_profit': '  3.12%', 'Profit': '195340381.096 USDT (19,534,038.11%)', 'Avg_duration': '0 days 21:49:00', 'Objective': '-38,157,864.48667', 'is_profit': True, 'Max_Drawdown_Acct': '  5274021.894 USDT    (5.18%)', 'loss': -38157864.486667246, 'timestamp': 1718690291, 'checkpoint_dir_name': None, 'done': True, 'training_iteration': 1, 'trial_id': '06452780', 'date': '2024-06-18_08-58-11', 'time_this_iter_s': 61.3154194355011, 'time_total_s': 61.3154194355011, 'pid': 1931179, 'hostname': 'vioUbuntu2', 'node_ip': '10.0.0.251', 'config': {'buy_fastk_rsi_patterns': 95, 'buy_max_slippage': 1.075, 'buy_prev_cbuys_count': 3, 'buy_prev_cbuys_rwindow': 5, 'buy_prev_min_close_age': 8, 'buy_prev_min_close_perc': 37.4, 'buy_prev_min_close_rwindow': 5, 'buy_proposed_stake_limit': 3731, 'buy_proposed_stake_limit_margin': 0.208, 'csl_5_step1_SL': 0.052, 'csl_5_step1_time': 591.366, 'csl_5_step2_SL': 0.035, 'csl_5_step2_time': 1625.187, 'csl_5_step3_SL': 0.075, 'csl_5_step3_time': 3717.144, 'csl_5_step4_SL': 0.248, 'sell_order_max_age': 2.8, 'sell_order_min_profit': 0.06, 'stoploss': -0.097}, 'time_since_restore': 61.3154194355011, 'iterations_since_restore': 1, 'experiment_tag': '139_buy_fastk_rsi_patterns=95,buy_max_slippage=1.0750,buy_prev_cbuys_count=3,buy_prev_cbuys_rwindow=5,buy_prev_min_close_age=8,buy_prev_min_close_perc=37.4000,buy_prev_min_close_rwindow=5,buy_proposed_stake_limit=3731,buy_proposed_stake_limit_margin=0.2080,csl_5_step1_SL=0.0520,csl_5_step1_time=591.3660,csl_5_step2_SL=0.0350,csl_5_step2_time=1625.1870,csl_5_step3_SL=0.0750,csl_5_step3_time=3717.1440,csl_5_step4_SL=0.2480,sell_order_max_age=2.8000,sell_order_min_profit=0.0600,stoploss=-0.0970'}


# class myObjective():
#     def __init__(
#         self, config_ft: Dict
#     ) -> None:
#         self.config_ft = config_ft


def objective(
    config: Dict[str, Any],
    config_ft: Dict,
    backtesting: Backtesting,
    custom_trade_info: Dict,
    dimensions_ft: Dict,
    data_pickle_file_ft: str,
    detail_data_pickle_file_ft: str,
    min_date_ft: str,
    max_date_ft: str,
    total_epochs_ft: int,
    custom_hyperopt_ft: Any,
    _get_results_dict_ft: Any,
    # _save_result_ft: Any,
    results_file_ft: Path,
    max_memory_per_worker: float
) -> Dict[str, Any]:
    """
    Used Optimize function.
    Called once per epoch to optimize whatever is configured.
    Keep this function as optimized as possible!
    """

    logger = ray_setup_func()
    # logger.info(f"ray hyperopt objective - ray_available_resources: {ray.available_resources()}")
    mem_available = ray.available_resources().get("memory", 0) # / psutil.virtual_memory().total
    if mem_available > max_memory_per_worker:
        logger.warning(
            f"ray hyperopt objective - low ray available memory: {(mem_available):,.2f}/{(max_memory_per_worker):,.2f}"
        )
    
    obj_id = ray.get_runtime_context().get_task_id()[:10]
    # logger.error(f"""worker_id: {ray.get_runtime_context().get_worker_id()} /
    #     actor_id: {ray.get_runtime_context().get_actor_id()} /
    #     job_id: {ray.get_runtime_context().get_job_id()} /
    #     task_id: {ray.get_runtime_context().get_task_id()}
    #     """)
    # logger.error(f"{setproctitle.getproctitle()} - {setproctitle.getthreadtitle()}")
    strategy_name = backtesting.strategy.get_strategy_name()
    setproctitle.setproctitle(f"ray::{strategy_name}::{obj_id}")
    os.chdir(Path(config_ft["user_data_dir"]).parent.absolute())

    # mem_used = psutil.virtual_memory().percent
    # if max_used_memory > 0 and mem_used > max_used_memory:
    #     logger.warning(f"objective paused - high memory usage {mem_used}")
    #     while psutil.virtual_memory().percent > max_used_memory:
    #         sleep(60)
    #     logger.warning(
    #         f"objective resumed - memory usage {psutil.virtual_memory().percent}"
    #     )

    # print(f"objective start - {os.getcwd()}")
    logger.debug(f"objective start - {os.getcwd()}")
    if custom_trade_info is not None:
        backtesting.strategy.custom_trade_info = custom_trade_info

    HyperoptStateContainer.set_state(HyperoptState.OPTIMIZE)
    backtest_start_time = datetime.now(timezone.utc)
    params_dict = Hyperopt._get_params_dict(dimensions_ft, config)
    # logger.info(f"params_dict - {params_dict}")

    # Apply parameters
    if HyperoptTools.has_space(config_ft, "buy"):
        Hyperopt.assign_params(backtesting, params_dict, "buy")

    if HyperoptTools.has_space(config_ft, "sell"):
        Hyperopt.assign_params(backtesting, params_dict, "sell")

    if HyperoptTools.has_space(config_ft, "protection"):
        Hyperopt.assign_params(backtesting, params_dict, "protection")

    if HyperoptTools.has_space(config_ft, "roi"):
        backtesting.strategy.minimal_roi = (
            custom_hyperopt_ft.generate_roi_table(params_dict)
        )

    if HyperoptTools.has_space(config_ft, "stoploss"):
        backtesting.strategy.stoploss = params_dict["stoploss"]

    if HyperoptTools.has_space(config_ft, "trailing"):
        d = custom_hyperopt_ft.generate_trailing_params(params_dict)
        backtesting.strategy.trailing_stop = d["trailing_stop"]
        backtesting.strategy.trailing_stop_positive = d[
            "trailing_stop_positive"
        ]
        backtesting.strategy.trailing_stop_positive_offset = d[
            "trailing_stop_positive_offset"
        ]
        backtesting.strategy.trailing_only_offset_is_reached = d[
            "trailing_only_offset_is_reached"
        ]

    if HyperoptTools.has_space(config_ft, "trades"):
        if config_ft["stake_amount"] == "unlimited" and (
            params_dict["max_open_trades"] == -1
            or params_dict["max_open_trades"] == 0
        ):
            # Ignore unlimited max open trades if stake amount is unlimited
            params_dict.update(
                {"max_open_trades": config_ft["max_open_trades"]}
            )

        updated_max_open_trades = (
            int(params_dict["max_open_trades"])
            if (
                params_dict["max_open_trades"] != -1
                and params_dict["max_open_trades"] != 0
            )
            else float("inf")
        )

        config_ft.update({"max_open_trades": updated_max_open_trades})

        backtesting.strategy.max_open_trades = updated_max_open_trades

    # logger.warning(f"params_dict - {params_dict}")

    with data_pickle_file_ft.open("rb") as f:
        processed = load(f, mmap_mode="r")
        # if self.analyze_per_epoch:
        #     # Data is not yet analyzed, rerun populate_indicators.
        #     processed = self.advise_and_trim(processed)

    if backtesting.timeframe_detail:
        with detail_data_pickle_file_ft.open("rb") as f:
            backtesting.detail_data = load(f, mmap_mode="r")

    bt_results = backtesting.backtest(
        processed=processed, start_date=min_date_ft, end_date=max_date_ft
    )
    backtest_end_time = datetime.now(timezone.utc)
    bt_results.update(
        {
            "backtest_start_time": int(backtest_start_time.timestamp()),
            "backtest_end_time": int(backtest_end_time.timestamp()),
        }
    )
    result = _get_results_dict_ft(
        backtesting,
        bt_results,
        min_date_ft,
        max_date_ft,
        params_dict,
        processed=processed,
    )
    result["runtime_s"] = int(backtest_end_time.timestamp()) - int(
        backtest_start_time.timestamp()
    )

    ray_result_tmp = HyperoptTools.get_result_dict(
        config_ft,
        result,
        total_epochs_ft,
    )
    # print("objective result", result)
    loss = result["loss"]
    ray_result_tmp["loss"] = [result["loss"]]
    ray_result_tmp["params_dict"] = [str(result["params_dict"])]
    ray_result_tmp["profit_perc"] = [100.0 * result["total_profit"]]

    ray_result = {}
    for key, val in ray_result_tmp.items():
        ray_result[key] = val[0]

    backtesting = None

    gc.collect()

    # print(ray_result)
    # _save_result_ft(result, results_file_ft)

    # train.report(ray_result)
    return ray_result


# https://github.com/Textualize/rich/discussions/482
class myLoggerCallback(LoggerCallback):
    def __init__(
        self,
        strategy="",
        print_all=False,
        total_epochs=-1,
        table_max_rows=-1,
        plot_metric="",
    ) -> None:

        self.console_width = Console().width
        self.console_height = Console().height

        if table_max_rows <= 0:
            table_max_rows = self.console_height // 3

        self.trial_results = deque(maxlen=table_max_rows)  # []
        self.plot_trial_results = (
            []
        )  # deque(maxlen=min(int(0.9*self.console_width), self.console_width-14)) #
        self.plot_trial_results_len = min(
            int(0.9 * self.console_width), self.console_width - 14
        )
        self.best_loss = MAX_LOSS
        self.print_all = print_all
        self.plot_metric = plot_metric
        if total_epochs <= 0:
            logger.warning(
                f"Please set total_epochs > 0 for myLoggerCallback - {total_epochs}"
            )
        self.total_epochs = total_epochs
        self.strategy = strategy
        self.count_trials = 0
        self.best_epoch = "N/A"

        self.live = None
        self.table = Table(expand=True)
        self.table_columns = [
            "Epoch",
            "Trades",
            "Win  Draw  Loss  Win%",
            "Avg profit",
            "Profit",
            "Winrate",
            "Avg duration",
            "Objective",
            "Max Drawdown (Acct)",
            "Time to run",
        ]
        for col in self.table_columns:
            self.table.add_column(col)
        self.table_master = self.generate_empty_table()
        self.refresh_chart = 100

    def generate_empty_table(self) -> Table:
        return Table(
            title=f"{self.strategy} {self.plot_metric} - Epoch: {self.count_trials}/{self.total_epochs} - Best: {self.best_epoch}",
            title_style=Style(color="white", bgcolor="black", bold=True),
            show_header=False,
            padding=(0, 0),
            expand=True,
        )

    def resize_list(self, list_in: [], max_len: int):
        if len(list_in) > max_len:
            list_out = []
            n_averaged_elements = (len(list_in) // max_len) + 1
            for i in range(0, len(list_in), n_averaged_elements):
                slice_from_index = i
                slice_to_index = slice_from_index + n_averaged_elements
                if self.plot_metric in ["Profit", "Winrate"]:
                    list_out.append(
                        np.max(list_in[slice_from_index:slice_to_index])
                    )
                elif self.plot_metric == "loss":
                    list_out.append(
                        np.min(list_in[slice_from_index:slice_to_index])
                    )
                else:
                    list_out.append(
                        np.mean(list_in[slice_from_index:slice_to_index])
                    )
            list_out = list_out[-max_len:]
            return list_out
        else:
            return list_in

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
                if self.plot_metric == "Profit":
                    profit = (
                        result  # [plot_metric_list.index(self.plot_metric)]
                    )
                    try:
                        profit = (
                            profit.split("(")[1]
                            .replace(")", "")
                            .replace("%", "")
                            .replace(",", "")
                        )
                        profit = float(profit)
                        plot_list.append(profit)
                    except:
                        # print(profit)
                        # profit = math.nan
                        pass
                else:
                    result = float(result)
                    plot_list.append(
                        result
                    )  # [plot_metric_list.index(self.plot_metric)]

        # print("plot_metric", self.plot_metric, "len trial_results", len(self.trial_results),  "plot_list", plot_list)
        if plot_list and len(self.plot_trial_results) > 1:
            try:
                # plot_list = plot_list[-int(0.9*self.console_width):]
                plot = acp.plot(
                    self.resize_list(plot_list, self.plot_trial_results_len),
                    {"height": self.console_height // 4, "format": "{:.4e}"},
                )
                self.table_master.add_row(plot)
            except Exception as e:
                logger.error(repr(e))
                pass

        progress = int(
            self.count_trials * self.live.console.width / self.total_epochs
        )
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
                int(
                    self.live.console.width
                    * psutil.virtual_memory().percent
                    / 100.0
                ),
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

    def on_step_begin(self, iteration, trials, **info):  ## too often
        # if self.live is None:
        #     self.live = Live(
        #         self.table_master,
        #         vertical_overflow="ellipsis",
        #         auto_refresh=False,
        #     )  # , screen=True : crop', 'ellipsis', 'visible', , refresh_per_second=0.2, transient=True,
        #     self.live.start(refresh=True)

        start_date = time.time()
        if iteration % self.refresh_chart == 0 and self.live:
            # self.logger.warning(f"myLoggerCallback - on_step_begin - iteration: {iteration}")
            self.generate_table()
            self.live.update(self.table_master, refresh=True)
        if time.time() - start_date > 0.1:
            self.refresh_chart = int(2 * self.refresh_chart)

    def on_trial_start(self, iteration, trials, trial, **info):
        if self.live is None:
            self.live = Live(
                self.table_master,
                vertical_overflow="ellipsis",
                auto_refresh=False,
            )  # , screen=True : crop', 'ellipsis', 'visible', , refresh_per_second=0.2, transient=True,
            self.live.start(refresh=True)
        self.generate_table()
        self.live.update(self.table_master, refresh=True)

    def append_trial_results(self, trial_id, result):
        loss = result["loss"]
        if abs(loss) > 100:
            loss = f"{result['loss']:,.6e}"
        else:
            loss = f"{result['loss']:,.6f}"
        self.trial_results.append(
            (
                f"{trial_id}",
                f"{result['Trades']}",
                f"{result['Win_Draw_Loss_Win_perc']}",
                f"{result['Avg_profit']}",
                f"{result['Profit']}",
                f"{(result['Winrate']):,.2f}",
                f"{result['Avg_duration']}",
                loss,
                f"{result['Max_Drawdown_Acct']}",
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

        self.generate_table()
        self.live.update(self.table_master, refresh=True)

    def on_experiment_end(self, trials, **info):
        if self.live.is_started:
            self.live.stop()


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
            raise ValueError(
                "The mode parameter can only be either min or max."
            )
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

            std_value = abs(
                np.std(self._top_values) / np.mean(self._top_values)
            )

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

            # logger.warning(
            #     f"myExperimentPlateauStopper - {trial_id} {result[self._metric]} - _current_epoch: {self._current_epoch} / _best_epoch: {self._best_epoch} / "
            #     f"_iterations_plateau: {self._iterations_plateau}/_iterations_noinc:{self._iterations_noinc}/{self._patience} / "
            #     f"has_plateaued: {has_plateaued} / no_increase: {no_increase} / stop_all: {stop_all} / "
            #     f"len_top_values: {len(self._top_values)} / std: {std_value}"
            # )

            if stop_all:
                logger.info(
                    f"myExperimentPlateauStopper - current_epoch: {self._current_epoch} / best_epoch: {self._best_epoch} / "
                    f"iterations_plateau: {self._iterations_plateau}/ iterations_noinc: {self._iterations_noinc} / patience: {self._patience} / "
                    f"has_plateaued: {has_plateaued} / no_increase: {no_increase} / stop_all: {stop_all} / std: {std_value} / "
                    f"last_result: {self._last_result} / best_result: {self._best_result}"
                )
        return stop_all

    def has_plateaued(self):
        return (
            len(self._top_values) == self._top
            and abs(np.std(self._top_values) / np.mean(self._top_values))
            <= self._std
        )

    def no_increase(self):
        return self._current_epoch - self._best_epoch > self._patience

    def stop_all(self):
        """Return whether to stop and prevent trials from starting."""
        return (
            self.has_plateaued() and self._iterations_plateau >= self._patience
        ) or (self.no_increase() and self._iterations_noinc >= self._patience)
