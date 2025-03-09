"""
This module contains the hyperopt optimizer class, which needs to be pickled
and will be sent to the hyperopt worker processes.
"""

import logging
import sys
import warnings
from datetime import datetime, timezone
import time
from typing import Any, Dict, List, Optional, Tuple

import os
from joblib import cpu_count, dump, load
from joblib.externals import cloudpickle
import ray
from pandas import DataFrame, json_normalize
from pathlib import Path
import setproctitle
import gc
from tabulate import tabulate

from freqtrade.constants import DATETIME_PRINT_FORMAT, Config
from freqtrade.data.converter import trim_dataframes
from freqtrade.data.history import get_timerange
from freqtrade.data.metrics import calculate_market_change
from freqtrade.enums import HyperoptState
from freqtrade.exceptions import OperationalException
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.optimize.space import Categorical
from freqtrade.optimize.space.decimalspace import SKDecimal
from skopt.space.space import Real
from skopt.space.space import Integer

# Import IHyperOptLoss to allow unpickling classes from these modules
from freqtrade.optimize.hyperopt.hyperopt_auto import HyperOptAuto
from freqtrade.optimize.hyperopt_loss.hyperopt_loss_interface import IHyperOptLoss
from freqtrade.optimize.hyperopt_tools import HyperoptStateContainer, HyperoptTools
from freqtrade.optimize.optimize_reports import generate_strategy_stats
from freqtrade.resolvers.hyperopt_resolver import HyperOptLossResolver
from freqtrade.util.dry_run_wallet import get_dry_run_wallet
from freqtrade.optimize.optimize_reports import generate_wins_draws_losses

import numpy as np

# Suppress scikit-learn FutureWarnings from skopt
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", module="ray.tune.logger.tensorboardx")
    warnings.filterwarnings("ignore", module="ray.tune.callback")
    warnings.filterwarnings("ignore", module="ray.tune.execution.tune_controller")
    logging.getLogger("ray.tune.schedulers.resource_changing_scheduler").setLevel(logging.WARNING)
    from skopt import Optimizer
    from skopt.space import Dimension
    from ray import tune

logger = logging.getLogger(__name__)


MAX_LOSS = 100000  # just a big enough number to be bad result in loss optimization


class HyperOptimizer:
    """
    HyperoptOptimizer class
    This class is sent to the hyperopt worker processes.
    """

    def __init__(self, config: Config) -> None:
        self.buy_space: list[Dimension] = []
        self.sell_space: list[Dimension] = []
        self.protection_space: list[Dimension] = []
        self.roi_space: list[Dimension] = []
        self.stoploss_space: list[Dimension] = []
        self.trailing_space: list[Dimension] = []
        self.max_open_trades_space: list[Dimension] = []
        self.dimensions: list[Dimension] = []

        self.config = config
        self.min_date: datetime
        self.max_date: datetime

        self.backtesting = Backtesting(self.config)
        self.pairlist = self.backtesting.pairlists.whitelist
        self.custom_hyperopt: HyperOptAuto
        self.analyze_per_epoch = self.config.get("analyze_per_epoch", False)

        if not self.config.get("hyperopt"):
            self.custom_hyperopt = HyperOptAuto(self.config)
        else:
            raise OperationalException(
                "Using separate Hyperopt files has been removed in 2021.9. Please convert "
                "your existing Hyperopt file to the new Hyperoptable strategy interface"
            )

        self.backtesting._set_strategy(self.backtesting.strategylist[0])
        self.custom_hyperopt.strategy = self.backtesting.strategy

        self.hyperopt_pickle_magic(self.backtesting.strategy.__class__.__bases__)
        self.custom_hyperoptloss: IHyperOptLoss = (
            HyperOptLossResolver.load_hyperoptloss(self.config)
        )
        self.calculate_loss = self.custom_hyperoptloss.hyperopt_loss_function

        self.market_change = 0.0

        if HyperoptTools.has_space(self.config, "sell"):
            # Make sure use_exit_signal is enabled
            self.config["use_exit_signal"] = True


    def prepare_hyperopt(self, data_pickle_file, detail_data_pickle_file) -> None:
        # Initialize spaces ...
        self.init_spaces()

        self.prepare_hyperopt_data(data_pickle_file, detail_data_pickle_file)

        # We don't need exchange instance anymore while running hyperopt
        self.backtesting.exchange.close()
        self.backtesting.exchange._api = None
        self.backtesting.exchange._api_async = None
        self.backtesting.exchange.loop = None  # type: ignore
        self.backtesting.exchange._loop_lock = None  # type: ignore
        self.backtesting.exchange._cache_lock = None  # type: ignore
        # self.backtesting.exchange = None  # type: ignore
        self.backtesting.pairlists = None  # type: ignore

    def get_strategy_name(self) -> str:
        return self.backtesting.strategy.get_strategy_name()

    def hyperopt_pickle_magic(self, bases: tuple[type, ...]) -> None:
        """
        Hyperopt magic to allow strategy inheritance across files.
        For this to properly work, we need to register the module of the imported class
        to pickle as value.
        """
        for modules in bases:
            if modules.__name__ != "IStrategy":
                if mod := sys.modules.get(modules.__module__):
                    cloudpickle.register_pickle_by_value(mod)
                self.hyperopt_pickle_magic(modules.__bases__)

    def _get_params_details(self, params: dict) -> dict:
        """
        Return the params for each space
        """
        result: dict = {}

        if HyperoptTools.has_space(self.config, "buy"):
            result["buy"] = {p.name: params.get(p.name) for p in self.buy_space}
        if HyperoptTools.has_space(self.config, "sell"):
            result["sell"] = {p.name: params.get(p.name) for p in self.sell_space}
        if HyperoptTools.has_space(self.config, "protection"):
            result["protection"] = {
                p.name: params.get(p.name) for p in self.protection_space
            }
        if HyperoptTools.has_space(self.config, "roi"):
            result["roi"] = {
                str(k): v
                for k, v in self.custom_hyperopt.generate_roi_table(params).items()
            }
        if HyperoptTools.has_space(self.config, "stoploss"):
            result["stoploss"] = {
                p.name: params.get(p.name) for p in self.stoploss_space
            }
        if HyperoptTools.has_space(self.config, "trailing"):
            result["trailing"] = self.custom_hyperopt.generate_trailing_params(params)
        if HyperoptTools.has_space(self.config, "trades"):
            result["max_open_trades"] = {
                "max_open_trades": (
                    self.backtesting.strategy.max_open_trades
                    if self.backtesting.strategy.max_open_trades != float("inf")
                    else -1
                )
            }

        return result

    def _get_no_optimize_details(self) -> dict[str, Any]:
        """
        Get non-optimized parameters
        """
        result: dict[str, Any] = {}
        strategy = self.backtesting.strategy
        if not HyperoptTools.has_space(self.config, "roi"):
            result["roi"] = {str(k): v for k, v in strategy.minimal_roi.items()}
        if not HyperoptTools.has_space(self.config, "stoploss"):
            result["stoploss"] = {"stoploss": strategy.stoploss}
        if not HyperoptTools.has_space(self.config, "trailing"):
            result["trailing"] = {
                "trailing_stop": strategy.trailing_stop,
                "trailing_stop_positive": strategy.trailing_stop_positive,
                "trailing_stop_positive_offset": strategy.trailing_stop_positive_offset,
                "trailing_only_offset_is_reached": strategy.trailing_only_offset_is_reached,
            }
        if not HyperoptTools.has_space(self.config, "trades"):
            result["max_open_trades"] = {"max_open_trades": strategy.max_open_trades}
        return result

    # def init_spaces(self):
    #     """
    #     Assign the dimensions in the hyperoptimization space.
    #     """
    #     if HyperoptTools.has_space(self.config, "protection"):
    #         # Protections can only be optimized when using the Parameter interface
    #         logger.debug("Hyperopt has 'protection' space")
    #         # Enable Protections if protection space is selected.
    #         self.config["enable_protections"] = True
    #         self.backtesting.enable_protections = True
    #         self.protection_space = self.custom_hyperopt.protection_space()

    #     if HyperoptTools.has_space(self.config, "buy"):
    #         logger.debug("Hyperopt has 'buy' space")
    #         self.buy_space = self.custom_hyperopt.buy_indicator_space()

    #     if HyperoptTools.has_space(self.config, "sell"):
    #         logger.debug("Hyperopt has 'sell' space")
    #         self.sell_space = self.custom_hyperopt.sell_indicator_space()

    #     if HyperoptTools.has_space(self.config, "roi"):
    #         logger.debug("Hyperopt has 'roi' space")
    #         self.roi_space = self.custom_hyperopt.roi_space()

    #     if HyperoptTools.has_space(self.config, "stoploss"):
    #         logger.debug("Hyperopt has 'stoploss' space")
    #         self.stoploss_space = self.custom_hyperopt.stoploss_space()

    #     if HyperoptTools.has_space(self.config, "trailing"):
    #         logger.debug("Hyperopt has 'trailing' space")
    #         self.trailing_space = self.custom_hyperopt.trailing_space()

    #     if HyperoptTools.has_space(self.config, "trades"):
    #         logger.debug("Hyperopt has 'trades' space")
    #         self.max_open_trades_space = self.custom_hyperopt.max_open_trades_space()

    #     self.dimensions = (
    #         self.buy_space
    #         + self.sell_space
    #         + self.protection_space
    #         + self.roi_space
    #         + self.stoploss_space
    #         + self.trailing_space
    #         + self.max_open_trades_space
    #     )
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
            self.max_open_trades_space = self.custom_hyperopt.max_open_trades_space()

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

        searcher_orig = self.custom_hyperopt.generate_estimator(
            dimensions=self.dimensions
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

        for original_dim in dimensions:
            # print(original_dim.name, original_dim, type(original_dim))
            if type(original_dim) == Integer:  # isinstance(original_dim, Integer):
                # print("Integer", original_dim.low, original_dim.high)
                if searcher == "bayesopt":  # 'bayesopt' - does not suport randint
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
                if searcher == "bayesopt":  # 'bayesopt' - does not suport quniform
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
                type(original_dim) == Real
            ):
                self.dimensions[original_dim.name] = tune.uniform(
                    original_dim.low,
                    original_dim.high,
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


    def advise_and_trim(self, data: dict[str, DataFrame]) -> dict[str, DataFrame]:
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

    def prepare_hyperopt_data(self, data_pickle_file, detail_data_pickle_file) -> None:
        HyperoptStateContainer.set_state(HyperoptState.DATALOAD)
        data, self.timerange = self.backtesting.load_bt_data()
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
            dump(data, data_pickle_file)
            if self.backtesting.timeframe_detail:
                self.backtesting.load_bt_data_detail()
                dump(self.backtesting.detail_data, detail_data_pickle_file)
                self.backtesting.detail_data = {}
        else:
            dump(data, data_pickle_file)
            if self.backtesting.timeframe_detail:
                self.backtesting.load_bt_data_detail()
                dump(self.backtesting.detail_data, detail_data_pickle_file)
                self.backtesting.detail_data = {}

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
        if trade_count >= self.config.get("hyperopt_min_trades",0):
            loss = self.calculate_loss(
                results=backtesting_results["results"],
                trade_count=trade_count,
                min_date=min_date,
                max_date=max_date,
                config=self.config,
                processed=processed,
                backtest_stats=strat_stats,
            )
        return {
            "loss": loss,
            "params_dict": params_dict,
            "params_details": params_details,
            # "params_not_optimized": not_optimized,
            "results_metrics": strat_stats,
            "results_explanation": results_explanation,
            "total_profit": total_profit,
        }

    @staticmethod
    def assign_params(
        backtesting: Backtesting, params_dict: dict[str, Any], category: str
    ) -> None:
        """
        Assign hyperoptable parameters
        """
        for attr_name, attr in backtesting.strategy.enumerate_parameters(category):
            if attr.optimize:
                # noinspection PyProtectedMember
                attr.value = params_dict[attr_name]

    # def _get_params_dict(
    #     self, dimensions: list[Dimension], raw_params: list[Any]
    # ) -> dict[str, Any]:
    #     # Ensure the number of dimensions match
    #     # the number of parameters in the list.
    #     if len(raw_params) != len(dimensions):
    #         raise ValueError("Mismatch in number of search-space dimensions.")

    #     # Return a dict where the keys are the names of the dimensions
    #     # and the values are taken from the list of parameters.
    #     return {d.name: v for d, v in zip(dimensions, raw_params, strict=False)}

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

    @staticmethod
    def ray_setup_func():
        # try:
        #     from optuna.exceptions import ExperimentalWarning

        #     with warnings.catch_warnings():
        #         warnings.filterwarnings("ignore", category=ExperimentalWarning)
        # except:
        #     pass

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        os.environ["RAY_TQDM"] = "1"
        os.environ["RAY_PROFILING"] = "0"
        os.environ["RAY_DEDUP_LOGS"] = "0"
        # os.environ["RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING"] = "1"
        os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
        os.environ["TUNE_PRINT_ALL_TRIAL_ERRORS"] = "1"
        os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = (
            f"{max(4,cpu_count()//4)}"  # f"{max(4,cpu_count()//2)}" 2
        )
        # os.environ["FUNCTION_SIZE_WARN_THRESHOLD"] = f"{2 * 10**7}"
        # os.environ["RAY_memory_monitor_refresh_ms"] = "0" # disable memory check
        # os.environ["RAY_memory_usage_threshold"] = "1"

        os.environ["SPT_NOENV"] = "1"

        return logger

    @staticmethod
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
        advise_and_trim_ft: Any,
        _get_results_dict_ft: Any,
        results_file_ft: Path,
    ) -> Dict[str, Any]:
        """
        Used Optimize function.
        Called once per epoch to optimize whatever is configured.
        Keep this function as optimized as possible!
        """

        logger = HyperOptimizer.ray_setup_func()
        # logger.info(f"ray hyperopt objective - ray_available_resources: {ray.available_resources()}")
        mem_available = ray.available_resources().get("memory", 0)

        # ray_current_workers = ray.util.state.list_workers(
        #     address=ray.get_runtime_context().gcs_address,
        #     filters=[("is_alive", "=", "True")],
        #     raise_on_missing_output=False,
        # )
        # logger.info(f"ray workers: {len(ray_current_workers)} - {ray_current_workers}")

        # ray_current_tasks = ray.util.state.list_tasks(
        #     address=ray.get_runtime_context().gcs_address,
        #     filters=[("state", "!=", "FINISHED")],
        #     raise_on_missing_output=False,
        # )
        # logger.info(f"ray tasks: {len(ray_current_tasks)} - {ray_current_tasks}")

        obj_id = ray.get_runtime_context().get_task_id()[:10]
        # logger.error(f"""worker_id: {ray.get_runtime_context().get_worker_id()} /
        #     actor_id: {ray.get_runtime_context().get_actor_id()} /
        #     job_id: {ray.get_runtime_context().get_job_id()} /
        #     task_id: {ray.get_runtime_context().get_task_id()}
        #     """)

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
        params_dict = HyperOptimizer._get_params_dict(dimensions_ft, config)
        # logger.info(f"params_dict - {params_dict}")

        # Apply parameters
        if HyperoptTools.has_space(config_ft, "buy"):
            HyperOptimizer.assign_params(backtesting, params_dict, "buy")

        if HyperoptTools.has_space(config_ft, "sell"):
            HyperOptimizer.assign_params(backtesting, params_dict, "sell")

        if HyperoptTools.has_space(config_ft, "protection"):
            HyperOptimizer.assign_params(backtesting, params_dict, "protection")

        if HyperoptTools.has_space(config_ft, "roi"):
            backtesting.strategy.minimal_roi = custom_hyperopt_ft.generate_roi_table(
                params_dict
            )

        if HyperoptTools.has_space(config_ft, "stoploss"):
            backtesting.strategy.stoploss = params_dict["stoploss"]

        if HyperoptTools.has_space(config_ft, "trailing"):
            d = custom_hyperopt_ft.generate_trailing_params(params_dict)
            backtesting.strategy.trailing_stop = d["trailing_stop"]
            backtesting.strategy.trailing_stop_positive = d["trailing_stop_positive"]
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
                params_dict.update({"max_open_trades": config_ft["max_open_trades"]})

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

        data = load(data_pickle_file_ft, mmap_mode="r")
        if backtesting.timeframe_detail:
            backtesting.detail_data = load(detail_data_pickle_file_ft, mmap_mode="r")
        processed = advise_and_trim_ft(data)

        # logger.info(
        #     f"Hyperopting with data from "
        #     f"{min_date_ft} "
        #     f"up to {max_date_ft}"
        # )

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
        # print("objective result", result)
        # ['loss', 'params_dict', 'params_details', 'results_metrics', 'results_explanation', 'total_profit', 'runtime_s']
        # print("objective result", list(result.keys()))
        # ['trades', 'locks', 'best_pair', 'worst_pair', 'results_per_pair', 'results_per_enter_tag',
        # 'exit_reason_summary', 'mix_tag_stats', 'left_open_trades', 'total_trades', 'trade_count_long',
        # 'trade_count_short', 'total_volume', 'avg_stake_amount', 'profit_mean', 'profit_median', 'profit_total', 'profit_total_long', 'profit_total_short', 'profit_total_abs', 'profit_total_long_abs',
        # 'profit_total_short_abs', 'cagr', 'expectancy', 'expectancy_ratio', 'sortino', 'sharpe', 'calmar', 'sqn', 'profit_factor', 'backtest_start', 'backtest_start_ts', 'backtest_end', 'backtest_end_ts', 'backtest_days',
        # 'backtest_run_start_ts', 'backtest_run_end_ts', 'trades_per_day', 'market_change', 'pairlist', 'stake_amount', 'stake_currency', 'stake_currency_decimals', 'starting_balance', 'dry_run_wallet', 'final_balance',
        # 'rejected_signals', 'timedout_entry_orders', 'timedout_exit_orders', 'canceled_trade_entries', 'canceled_entry_orders', 'replaced_entry_orders', 'max_open_trades', 'max_open_trades_setting', 'timeframe',
        # 'timeframe_detail', 'timerange', 'enable_protections', 'strategy_name', 'stoploss', 'trailing_stop', 'trailing_stop_positive', 'trailing_stop_positive_offset', 'trailing_only_offset_is_reached',
        # 'use_custom_stoploss', 'minimal_roi', 'use_exit_signal', 'exit_profit_only', 'exit_profit_offset', 'ignore_roi_if_entry_signal', 'trading_mode', 'margin_mode', 'backtest_best_day', 'backtest_worst_day',
        # 'backtest_best_day_abs', 'backtest_worst_day_abs', 'winning_days', 'draw_days', 'losing_days', 'daily_profit', 'wins', 'losses', 'draws', 'winrate', 'holding_avg', 'holding_avg_s', 'winner_holding_avg',
        # 'winner_holding_avg_s', 'loser_holding_avg', 'loser_holding_avg_s', 'max_consecutive_wins', 'max_consecutive_losses', 'max_drawdown_account', 'max_relative_drawdown', 'max_drawdown_abs', 'drawdown_start',
        # 'drawdown_start_ts', 'drawdown_end', 'drawdown_end_ts', 'max_drawdown_low', 'max_drawdown_high', 'csum_min', 'csum_max']
        # print("objective result", list(result["results_metrics"].keys()))

        # print(tabulate(json_normalize(result, max_level=1), headers='keys', tablefmt='psql'))
        # print("results_explanation", result["results_explanation"])

        # , 'params_dict', 'params_details', , 'results_explanation'
        result_columns = {
            "loss": "Objective",
            "total_profit": "Total_profit1",
            "profit_total": "Total_profit",
            "profit_mean": "Avg_profit",
            "profit_total_abs": "Profit",
            "runtime_s": "TTR",
            "total_trades": "Trades",
            "holding_avg": "Avg_duration",
            "max_drawdown_abs": "Max_drawdown",
            "max_drawdown_account": "Max_Drawdown_Acct",
            "wins": "",
            "draws": "",
            "losses": "",
        }
        trial_result = {
            "loss": result["loss"],
            "profit_perc": 100.0 * result["results_metrics"]["profit_total"],
        }
        for key, value in result.items():
            if key in list(result_columns.keys()):
                if len(result_columns[key]) > 0:
                    trial_result[result_columns[key]] = value
                else:
                    trial_result[key] = value
            elif key == "results_metrics":
                for key1, value1 in result["results_metrics"].items():
                    if key1 in list(result_columns.keys()):
                        if len(result_columns[key1]) > 0:
                            trial_result[result_columns[key1]] = value1
                        else:
                            trial_result[key1] = value1

        trial_result["Win_Draw_Loss_Win_perc"] = generate_wins_draws_losses(
            trial_result["wins"],
            trial_result["draws"],
            trial_result["losses"],
        )

        if trial_result["wins"] > 0 and trial_result["losses"] == 0:
            trial_result["Winrate"] = 100
        elif trial_result["wins"] == 0:
            trial_result["Winrate"] = 0
        else:
            trial_result["Winrate"] = (
                100.0
                / (
                    trial_result["wins"]
                    + trial_result["draws"]
                    + trial_result["losses"]
                )
                * trial_result["wins"]
            )


        backtesting = None

        gc.collect()

        # print(ray_result)
        # _save_result_ft(result, results_file_ft)

        # train.report(ray_result)
        return trial_result
