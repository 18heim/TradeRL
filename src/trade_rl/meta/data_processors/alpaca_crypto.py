from typing import List, Optional

import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd
import pydantic

from trade_rl.meta.config import TIME_ZONE_SELFDEFINED
# from basic_processor import _Base
from trade_rl.meta.data_processors._base import (
    APIConfig,
    _Base,
    add_tech_indicator,
    calc_time_zone,
    df_to_array,
)


class AlpacaCrypto(_Base):
    def __init__(
        self,
        start_date: str,
        end_date: str,
        time_interval: str,
        api_config: APIConfig,
    ):
        super().__init__("alpacacrypto", start_date, end_date, time_interval, api_config)
        if self.api_config.API is None:
            try:
                self.api = tradeapi.REST(**self.api_config.dict(exclude_none=True),
                                         api_version="v2",
                                         )
            except BaseException:
                raise ValueError("Wrong Account Info!")
        else:
            self.api = self.api_config.API

    def download_data(
        self, ticker_list: List[str]
    ) -> pd.DataFrame:
        """Download historical data from Alpaca.

        Args:
            ticker_list (List[str]): list of strings
        """
        self.time_zone = TIME_ZONE_SELFDEFINED
        start_date = pd.Timestamp(
            self.start_date, tz=self.time_zone).date().isoformat()
        end_date = pd.Timestamp(
            self.end_date, tz=self.time_zone).date().isoformat()
        self.dataframe = self.api.get_crypto_bars(
            ticker_list,
            self.time_interval,
            start=start_date,
            end=end_date,
            exchanges="CBSE"
        ).df
        self.dataframe = self.dataframe.reset_index()
        self.dataframe["time"] = self.dataframe["timestamp"].apply(
            lambda x: x.strftime("%Y-%m-%d %H:%M:%S")
        )
        self.dataframe = self.dataframe.rename(columns={"symbol": "tic"})

    def fetch_latest_data(
        self, ticker_list, time_interval, tech_indicator_list, limit=100
    ) -> pd.DataFrame:
        data_df = self.api.get_crypto_bars(
            ticker_list, time_interval, exchanges='CBSE').df.groupby("symbol").tail(limit)
        data_df = data_df.reset_index()
        data_df["time"] = data_df["timestamp"].apply(
            lambda x: x.strftime("%Y-%m-%d %H:%M:%S")
        )
        data_df = data_df.rename(columns={"symbol": "tic"})
        data_df = data_df.reset_index(drop=True)

        df = add_tech_indicator(data_df, tech_indicator_list)
        df["VIXY"] = 0

        price_array, tech_array, turbulence_array = df_to_array(
            df, tech_indicator_list, if_vix=False
        )
        latest_price = price_array[-1]
        latest_tech = tech_array[-1]
        # TODO: Investigate turbulence / VIXY
        # turb_df = self.api.get_barset(
        #     ["VIXY"], time_interval).df.tail(1)["VIXY"]
        # latest_turb = turb_df["close"].values
        return latest_price, latest_tech, None

    def get_portfolio_history(self, start, end):
        trading_days = self.get_trading_days(start, end)
        df = pd.DataFrame()
        for day in trading_days:
            df = df.append(
                self.api.get_portfolio_history(
                    date_start=day, timeframe="5Min"
                ).df.iloc[:79]
            )
        equities = df.equity.values
        cumu_returns = equities / equities[0]
        cumu_returns = cumu_returns[~np.isnan(cumu_returns)]
        return cumu_returns
