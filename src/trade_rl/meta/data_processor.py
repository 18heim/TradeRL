import os
import pickle
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from trade_rl.meta.data_processors.alpaca_crypto import APIConfig


class DataProcessor:
    def __init__(
        self,
        data_source: str,
        start_date: str,
        end_date: str,
        time_interval: str,
        api_config: Optional[APIConfig] = None,
    ):
        self.data_source = data_source
        self.start_date = start_date
        self.end_date = end_date
        self.time_interval = time_interval
        self.dataframe = pd.DataFrame()

        if self.data_source == "alpaca":
            from trade_rl.meta.data_processors.alpaca import Alpaca

            processor_dict = {self.data_source: Alpaca}

        elif self.data_source == "alpacacrypto":
            from trade_rl.meta.data_processors.alpaca_crypto import AlpacaCrypto

            processor_dict = {self.data_source: AlpacaCrypto}

        elif self.data_source == "binance":
            from trade_rl.meta.data_processors.binance import Binance

            processor_dict = {self.data_source: Binance}

        elif self.data_source == "ccxt":
            from trade_rl.meta.data_processors.ccxt import Ccxt

            processor_dict = {self.data_source: Ccxt}

        elif self.data_source == "yahoofinance":
            from trade_rl.meta.data_processors.yahoofinance import Yahoofinance

            processor_dict = {self.data_source: Yahoofinance}

        else:
            print(f"{self.data_source} is NOT supported yet.")

        self.processor = processor_dict.get(self.data_source)(
            time_interval=time_interval, start_date=start_date,
            end_date=end_date, api_config=api_config,
        )
        print(f"{self.data_source} successfully connected")
        # except:
        #     raise ValueError(
        #         f"Please input correct account info for {self.data_source}!"
        #     )

    def download_data(self, ticker_list):
        self.processor.download_data(ticker_list=ticker_list)
        self.dataframe = self.processor.dataframe

    def clean_data(self):
        self.processor.dataframe = self.dataframe
        self.processor.clean_data()
        self.dataframe = self.processor.dataframe

    def add_technical_indicator(
        self, tech_indicator_list: List[str],
    ):
        self.tech_indicator_list = tech_indicator_list
        self.processor.add_technical_indicator(
            tech_indicator_list
        )
        self.dataframe = self.processor.dataframe

    def add_turbulence(self):
        self.processor.add_turbulence()
        self.dataframe = self.processor.dataframe

    def add_vix(self):
        self.processor.add_vix()
        self.dataframe = self.processor.dataframe

    def df_to_array(self, if_vix: bool) -> np.array:
        price_array, tech_array, turbulence_array = self.processor.df_to_array(
            self.tech_indicator_list, if_vix
        )
        # fill nan with 0 for technical indicators
        tech_nan_positions = np.isnan(tech_array)
        tech_array[tech_nan_positions] = 0

        return price_array, tech_array, turbulence_array

    def data_split(self, df, start, end, target_date_col="date"):
        """
        split the dataset into training or testing using date
        :param data: (df) pandas dataframe, start, end
        :return: (df) pandas dataframe
        """
        data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
        data = data.sort_values([target_date_col, "tic"], ignore_index=True)
        data.index = data[target_date_col].factorize()[0]
        return data

    def run(
        self,
        ticker_list: str,
        technical_indicator_list: List[str],
        if_vix: bool,
        cache: bool = False,
    ):

        if self.time_interval == "1s" and self.data_source != "binance":
            raise ValueError(
                "Currently 1s interval data is only supported with 'binance' as data source"
            )

        cache_filename = (
            "_".join(
                ticker_list
                + [
                    self.data_source,
                    self.start_date,
                    self.end_date,
                    self.time_interval,
                ]
            )
            + ".pickle"
        )
        cache_dir = "./cache"
        cache_path = os.path.join(cache_dir, cache_filename)

        if cache and os.path.isfile(cache_path):
            print(f"Using cached file {cache_path}")
            self.tech_indicator_list = technical_indicator_list
            with open(cache_path, "rb") as handle:
                self.processor.dataframe = pickle.load(handle)
        else:
            self.download_data(ticker_list)
            self.clean_data()
            if cache:
                if not os.path.exists(cache_dir):
                    os.mkdir(cache_dir)
                with open(cache_path, "wb") as handle:
                    pickle.dump(
                        self.dataframe,
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )

        self.add_technical_indicator(
            technical_indicator_list)
        if if_vix:
            self.add_vix()
        price_array, tech_array, turbulence_array = self.df_to_array(if_vix)
        tech_nan_positions = np.isnan(tech_array)
        tech_array[tech_nan_positions] = 0

        return price_array, tech_array, turbulence_array


def test_joinquant():
    # TRADE_START_DATE = "2019-09-01"
    TRADE_START_DATE = "2020-09-01"
    TRADE_END_DATE = "2021-09-11"

    # supported time interval: '1m', '5m', '15m', '30m', '60m', '120m', '1d', '1w', '1M'
    TIME_INTERVAL = "1d"
    TECHNICAL_INDICATOR = [
        "macd",
        "boll_ub",
        "boll_lb",
        "rsi_30",
        "dx_30",
        "close_30_sma",
        "close_60_sma",
    ]

    kwargs = {"username": "xxx", "password": "xxx"}
    p = DataProcessor(
        data_source="joinquant",
        start_date=TRADE_START_DATE,
        end_date=TRADE_END_DATE,
        time_interval=TIME_INTERVAL,
        **kwargs,
    )

    ticker_list = ["000612.XSHE", "601808.XSHG"]

    p.download_data(ticker_list=ticker_list)

    p.clean_data()
    p.add_turbulence()
    p.add_technical_indicator(TECHNICAL_INDICATOR)
    p.add_vix()

    price_array, tech_array, turbulence_array = p.run(
        ticker_list, TECHNICAL_INDICATOR, if_vix=False, cache=True
    )
    pass


# if __name__ == "__main__":
#     # test_joinquant()
#     # test_binance()
#     # test_yahoofinance()
#     test_baostock()
#     # test_quandl()
