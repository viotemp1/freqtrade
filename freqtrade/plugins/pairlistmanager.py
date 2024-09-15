"""
PairList manager class
"""

import logging
from functools import partial
from typing import Dict, List, Optional

from cachetools import TTLCache, cached

from freqtrade.constants import Config, ListPairsWithTimeframes
from freqtrade.data.dataprovider import DataProvider
from freqtrade.enums import CandleType
from freqtrade.enums.runmode import RunMode
from freqtrade.exceptions import OperationalException
from freqtrade.exchange.exchange_types import Tickers
from freqtrade.mixins import LoggingMixin
from freqtrade.plugins.pairlist.IPairList import IPairList, SupportsBacktesting
from freqtrade.plugins.pairlist.pairlist_helpers import expand_pairlist
from freqtrade.resolvers import PairListResolver
from freqtrade.enums import RunMode

import os
from datetime import datetime
import cloudpickle

logger = logging.getLogger(__name__)


class PairListManager(LoggingMixin):
    def __init__(
        self, exchange, config: Config, dataprovider: Optional[DataProvider] = None
    ) -> None:
        self._exchange = exchange
        self._config = config
        self._whitelist = self._config["exchange"].get("pair_whitelist")
        self._blacklist = self._config["exchange"].get("pair_blacklist", [])
        self._pairlist_handlers: List[IPairList] = []
        self._tickers_needed = False
        self._dataprovider: Optional[DataProvider] = dataprovider

        if not os.path.isdir("./tmp"):
            os.mkdir("./tmp")
        self.file_cache_ttl = config.get('file_cache_ttl', 86400)
        strategy1 = config.get("strategy", "")
        if strategy1 is not None and len(strategy1) > 0 and strategy1.find("_") >= 0:
            strategy1 = strategy1.split("_")
            if len(strategy1) >= 2:
                strategy1 = strategy1[0] + "_" + strategy1[1]
            else:
                strategy1 = strategy1[0]
        self.pairlist_filename = f"tmp/PairListManager_pairlists_{strategy1}.pkl"

        for pairlist_handler_config in self._config.get('pairlists', []):
            pairlist_handler = PairListResolver.load_pairlist(
                pairlist_handler_config["method"],
                exchange=exchange,
                pairlistmanager=self,
                config=config,
                pairlistconfig=pairlist_handler_config,
                pairlist_pos=len(self._pairlist_handlers),
            )
            self._tickers_needed |= pairlist_handler.needstickers
            self._pairlist_handlers.append(pairlist_handler)

        if not self._pairlist_handlers:
            raise OperationalException("No Pairlist Handlers defined")

        if self._tickers_needed and not self._exchange.exchange_has("fetchTickers"):
            invalid = ". ".join([p.name for p in self._pairlist_handlers if p.needstickers])

            raise OperationalException(
                "Exchange does not support fetchTickers, therefore the following pairlists "
                "cannot be used. Please edit your config and restart the bot.\n"
                f"{invalid}."
            )

        self._check_backtest()

        refresh_period = config.get("pairlist_refresh_period", 3600)
        LoggingMixin.__init__(self, logger, refresh_period)
        self._first_run = True

    def _check_backtest(self) -> None:
        # logger.warning(f"_check_backtest - runmode: {self._config['runmode']}")
        if self._config["runmode"] not in (RunMode.BACKTEST, RunMode.EDGE, RunMode.HYPEROPT):
            return

        pairlist_errors: List[str] = []
        noaction_pairlists: List[str] = []
        biased_pairlists: List[str] = []
        for pairlist_handler in self._pairlist_handlers:
            # logger.warning(f"_check_backtest - pairlist_handler: {pairlist_handler.name} / supports_backtesting: {pairlist_handler.supports_backtesting}")
            if pairlist_handler.supports_backtesting == SupportsBacktesting.NO:
                pairlist_errors.append(pairlist_handler.name)
            if pairlist_handler.supports_backtesting == SupportsBacktesting.NO_ACTION:
                noaction_pairlists.append(pairlist_handler.name)
            if pairlist_handler.supports_backtesting == SupportsBacktesting.BIASED:
                biased_pairlists.append(pairlist_handler.name)

        if noaction_pairlists:
            logger.warning(
                f"Pairlist Handlers {', '.join(noaction_pairlists)} do not generate "
                "any changes during backtesting. While it's safe to leave them enabled, they will "
                "not behave like in dry/live modes. "
            )

        if biased_pairlists:
            logger.warning(
                f"Pairlist Handlers {', '.join(biased_pairlists)} will introduce a lookahead bias "
                "to your backtest results, as they use today's data - which inheritly suffers from "
                "'winner bias'."
            )
        if pairlist_errors:
            raise OperationalException(
                f"Pairlist Handlers {', '.join(pairlist_errors)} do not support backtesting."
            )

    @property
    def whitelist(self) -> List[str]:
        """The current whitelist"""
        return self._whitelist

    @property
    def blacklist(self) -> List[str]:
        """
        The current blacklist
        -> no need to overwrite in subclasses
        """
        return self._blacklist

    @property
    def expanded_blacklist(self) -> List[str]:
        """The expanded blacklist (including wildcard expansion)"""
        return expand_pairlist(self._blacklist, self._exchange.get_markets().keys())

    @property
    def name_list(self) -> List[str]:
        """Get list of loaded Pairlist Handler names"""
        return [p.name for p in self._pairlist_handlers]

    def short_desc(self) -> List[Dict]:
        """List of short_desc for each Pairlist Handler"""
        return [{p.name: p.short_desc()} for p in self._pairlist_handlers]

    @cached(TTLCache(maxsize=1, ttl=1800))
    def _get_cached_tickers(self) -> Tickers:
        return self._exchange.get_tickers()

#    def refresh_pairlist(self) -> None:
#        """Run pairlist through all configured Pairlist Handlers."""
#        # Tickers should be cached to avoid calling the exchange on each call.
#        tickers: Dict = {}
#        if self._tickers_needed:
#            tickers = self._get_cached_tickers()
#
#        # Generate the pairlist with first Pairlist Handler in the chain
#        pairlist = self._pairlist_handlers[0].gen_pairlist(tickers)
#
#        # Process all Pairlist Handlers in the chain
#        # except for the first one, which is the generator.
#        for pairlist_handler in self._pairlist_handlers[1:]:
#            pairlist = pairlist_handler.filter_pairlist(pairlist, tickers)
#
#        # Validation against blacklist happens after the chain of Pairlist Handlers
#        # to ensure blacklist is respected.
#        pairlist = self.verify_blacklist(pairlist, logger.warning)
#
#        self.log_once(f"Whitelist with {len(pairlist)} pairs: {pairlist}", logger.info)
#
#        self._whitelist = pairlist

    def file_too_old(self, filename="", age_seconds=0):
        if not os.path.isfile(filename):
            return True, -1
        t = max(os.path.getmtime(filename), os.path.getctime(filename))
        # (datetime.now()-datetime.fromtimestamp(t)).seconds, datetime.fromtimestamp(t)
        # logger.warning(
        #     f"{datetime.fromtimestamp(os.path.getmtime(filename))} / "
        #     f"{datetime.fromtimestamp(os.path.getctime(filename))} / "
        #     f"{ (datetime.now()-datetime.fromtimestamp(os.path.getmtime(filename))).total_seconds()} / "
        #     f"{ (datetime.now() - datetime.fromtimestamp(os.path.getctime(filename))).total_seconds()}"
        # )
        r_f_age = (datetime.now() - datetime.fromtimestamp(t)).total_seconds()
        r_file_too_old = r_f_age > age_seconds
        # print(r_file_too_old, r_f_age)
        return r_file_too_old, r_f_age

    def refresh_pairlist(self) -> None:
        """Run pairlist through all configured Pairlist Handlers."""
        # Tickers should be cached to avoid calling the exchange on each call.
        tickers: Dict = {}
        if self._tickers_needed:
            tickers = self._get_cached_tickers()

        # Generate the pairlist with first Pairlist Handler in the chain
        pairlist = self._pairlist_handlers[0].gen_pairlist(tickers)
        if self._first_run:
            # logger.warning(
            #     f"{self._pairlist_handlers[0].name} length: {len(pairlist)}"
            # )  #  pairs: {pairlist}
            self.log_once(
                f"{self._pairlist_handlers[0].name} length: {len(pairlist)}",
                logger.info,
            )

        # Process all Pairlist Handlers in the chain
        # except for the first one, which is the generator.
        file_too_old_val, file_age = self.file_too_old(
            filename=self.pairlist_filename, age_seconds=self.file_cache_ttl
        )
        # print("vio -", self._config["runmode"].value, file_too_old_val)
        if (
            self._config["runmode"].value
            in ["hyperopt", "backtest", "util_no_exchange"]
            and file_too_old_val
        ) or (
            self._config["runmode"].value
            not in ["hyperopt", "backtest", "util_no_exchange"]
        ):
            if (
                self._config["runmode"].value
                in ["hyperopt", "backtest", "util_no_exchange"]
                and file_too_old_val
            ):
                # logger.info(
                #     f"vio - need to refresh pairlist - {self._config['runmode'].value} - {self.pairlist_filename}"
                # )
                self.log_once(
                    f"vio - need to refresh pairlist - {self._config['runmode'].value} - {self.pairlist_filename}",
                    logger.info,
                )
            for pairlist_handler in self._pairlist_handlers[1:]:
                pairlist = pairlist_handler.filter_pairlist(pairlist, tickers)
                if self._first_run:
                    # logger.warning(
                    #     f"{pairlist_handler.name} length: {len(pairlist)}"
                    # )
                    self.log_once(
                        f"{pairlist_handler.name} length: {len(pairlist)}",
                        logger.info,
                    )
            if (
                self._config["runmode"].value
                in ["hyperopt", "backtest", "util_no_exchange"]
                and file_too_old_val
                and len(pairlist) > 10
            ):
                with open(self.pairlist_filename, mode="wb") as file:
                    cloudpickle.dump(pairlist, file)
            if len(pairlist) <= 10 and self._first_run:
                # with open(self.pairlist_filename, mode="rb") as file:
                #     pairlist = cloudpickle.load(file)
                logger.warning(f"vio - pairlist issue - len: {len(pairlist)}")
                # self.log_once(f"vio - pairlist issue - len: {len(pairlist)}", logger.warning)
        else:
            with open(self.pairlist_filename, mode="rb") as file:
                pairlist = cloudpickle.load(file)
            # logger.info(
            #     f"vio - will use cached pairlist - {self._config['runmode'].value} - length: {len(pairlist)} - {self.pairlist_filename}"
            # )
            self.log_once(
                f"vio - will use cached pairlist - {self._config['runmode'].value} - length: {len(pairlist)} - {self.pairlist_filename}",
                logger.info,
            )

        # Validation against blacklist happens after the chain of Pairlist Handlers
        # to ensure blacklist is respected.
        pairlist = self.verify_blacklist(pairlist, logger.warning if self._first_run else logger.debug) # warning

        self.log_once(
            f"Whitelist with {len(pairlist)} pairs: {pairlist}", logger.debug
        )

        self._first_run = False

        self._whitelist = pairlist

    def verify_blacklist(self, pairlist: List[str], logmethod) -> List[str]:
        """
        Verify and remove items from pairlist - returning a filtered pairlist.
        Logs a warning or info depending on `aswarning`.
        Pairlist Handlers explicitly using this method shall use
        `logmethod=logger.info` to avoid spamming with warning messages
        :param pairlist: Pairlist to validate
        :param logmethod: Function that'll be called, `logger.info` or `logger.warning`.
        :return: pairlist - blacklisted pairs
        """
        try:
            blacklist = self.expanded_blacklist
        except ValueError as err:
            logger.error(f"Pair blacklist contains an invalid Wildcard: {err}")
            return []
        log_once = partial(self.log_once, logmethod=logmethod)
        for pair in pairlist.copy():
            if pair in blacklist: #  and self._config.get("runmode", None) not in [RunMode.LIVE]
                log_once(f"Pair {pair} in your blacklist. Removing it from whitelist...")
                pairlist.remove(pair)
        return pairlist

    def verify_whitelist(
        self, pairlist: List[str], logmethod, keep_invalid: bool = False
    ) -> List[str]:
        """
        Verify and remove items from pairlist - returning a filtered pairlist.
        Logs a warning or info depending on `aswarning`.
        Pairlist Handlers explicitly using this method shall use
        `logmethod=logger.info` to avoid spamming with warning messages
        :param pairlist: Pairlist to validate
        :param logmethod: Function that'll be called, `logger.info` or `logger.warning`
        :param keep_invalid: If sets to True, drops invalid pairs silently while expanding regexes.
        :return: pairlist - whitelisted pairs
        """
        try:
            whitelist = expand_pairlist(pairlist, self._exchange.get_markets().keys(), keep_invalid)
        except ValueError as err:
            logger.error(f"Pair whitelist contains an invalid Wildcard: {err}")
            return []
        return whitelist

    def create_pair_list(
        self, pairs: List[str], timeframe: Optional[str] = None
    ) -> ListPairsWithTimeframes:
        """
        Create list of pair tuples with (pair, timeframe)
        """
        return [
            (
                pair,
                timeframe or self._config["timeframe"],
                self._config.get("candle_type_def", CandleType.SPOT),
            )
            for pair in pairs
        ]
