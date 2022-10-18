from typing import List

import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd
import pytz

try:
    import exchange_calendars as tc
except:
    print(
        "Cannot import exchange_calendars.",
        "If you are using python>=3.7, please install it.",
    )
    import trading_calendars as tc

    print("Use trading_calendars instead for alpaca processor.")
# from basic_processor import _Base
from trade_rl.meta.data_processors._base import _Base
from trade_rl.meta.data_processors._base import calc_time_zone

from trade_rl.meta.config import (
    TIME_ZONE_SHANGHAI,
    TIME_ZONE_USEASTERN,
    TIME_ZONE_PARIS,
    TIME_ZONE_SELFDEFINED,
    USE_TIME_ZONE_SELFDEFINED,
    BINANCE_BASE_URL,
)


class Alpaca(_Base):
    # def __init__(self, API_KEY=None, API_SECRET=None, API_BASE_URL=None, api=None):
    #     if api is None:
    #         try:
    #             self.api = tradeapi.REST(API_KEY, API_SECRET, API_BASE_URL, "v2")
    #         except BaseException:
    #             raise ValueError("Wrong Account Info!")
    #     else:
    #         self.api = api
    def __init__(
        self,
        data_source: str,
        start_date: str,
        end_date: str,
        time_interval: str,
        **kwargs
    ):
        super().__init__(data_source, start_date, end_date, time_interval, **kwargs)
        if kwargs.get("API") is None:
            try:
                self.api = tradeapi.REST(
                    kwargs["API_KEY"],
                    kwargs["API_SECRET"],
                    kwargs["API_BASE_URL"],
                    "v2",
                )
            except BaseException:
                raise ValueError("Wrong Account Info!")
        else:
            self.api = kwargs["API"]

    def download_data(
        self, ticker_list: List[str]
    ) -> pd.DataFrame:
        self.time_zone = calc_time_zone(
            ticker_list, TIME_ZONE_SELFDEFINED, USE_TIME_ZONE_SELFDEFINED
        )
        start_date = pd.Timestamp(self.start_date, tz=self.time_zone)
        end_date = pd.Timestamp(
            self.end_date, tz=self.time_zone) + pd.Timedelta(days=1)

        date = start_date
        data_df = pd.DataFrame()
        while date != end_date:
            start_time = (date + pd.Timedelta("09:30:00")).isoformat()
            end_time = (date + pd.Timedelta("15:59:00")).isoformat()
            for tic in ticker_list:
                barset = self.api.get_bars(
                    tic,
                    self.time_interval,
                    start=start_time,
                    end=end_time,
                    limit=500,
                ).df
                barset["tic"] = tic
                barset = barset.reset_index()
                data_df = data_df.append(barset)
            print(("Data before ") + end_time + " is successfully fetched")
            # print(data_df.head())
            date = date + pd.Timedelta(days=1)
            if date.isoformat()[-14:-6] == "01:00:00":
                date = date - pd.Timedelta("01:00:00")
            elif date.isoformat()[-14:-6] == "23:00:00":
                date = date + pd.Timedelta("01:00:00")
            if date.isoformat()[-14:-6] != "00:00:00":
                raise ValueError("Timezone Error")

        data_df["time"] = data_df["timestamp"].apply(
            lambda x: x.strftime("%Y-%m-%d %H:%M:%S")
        )
        self.dataframe = data_df

    def clean_data(self):
        df = self.dataframe.copy()
        tic_list = np.unique(df.tic.values)

        trading_days = self.get_trading_days(start=self.start, end=self.end)
        # produce full time index
        times = []
        for day in trading_days:
            current_time = pd.Timestamp(
                day + " 09:30:00").tz_localize(self.time_zone)
            for _ in range(390):
                times.append(current_time)
                current_time += pd.Timedelta(minutes=1)
        # create a new dataframe with full time series
        new_df = pd.DataFrame()
        for tic in tic_list:
            tmp_df = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"], index=times
            )
            tic_df = df[df.tic == tic]
            for i in range(tic_df.shape[0]):
                tmp_df.loc[tic_df.iloc[i]["time"]] = tic_df.iloc[i][
                    ["open", "high", "low", "close", "volume"]
                ]

            # if the close price of the first row is NaN
            if str(tmp_df.iloc[0]["close"]) == "nan":
                print(
                    "The price of the first row for ticker ",
                    tic,
                    " is NaN. ",
                    "It will filled with the first valid price.",
                )
                for i in range(tmp_df.shape[0]):
                    if str(tmp_df.iloc[i]["close"]) != "nan":
                        first_valid_price = tmp_df.iloc[i]["close"]
                        tmp_df.iloc[0] = [
                            first_valid_price,
                            first_valid_price,
                            first_valid_price,
                            first_valid_price,
                            0.0,
                        ]
                        break
            # if the close price of the first row is still NaN (All the prices are NaN in this case)
            if str(tmp_df.iloc[0]["close"]) == "nan":
                print(
                    "Missing data for ticker: ",
                    tic,
                    " . The prices are all NaN. Fill with 0.",
                )
                tmp_df.iloc[0] = [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]

            # forward filling row by row
            for i in range(tmp_df.shape[0]):
                if str(tmp_df.iloc[i]["close"]) == "nan":
                    previous_close = tmp_df.iloc[i - 1]["close"]
                    if str(previous_close) == "nan":
                        raise ValueError
                    tmp_df.iloc[i] = [
                        previous_close,
                        previous_close,
                        previous_close,
                        previous_close,
                        0.0,
                    ]
            tmp_df = tmp_df.astype(float)
            tmp_df["tic"] = tic
            new_df = new_df.append(tmp_df)

        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={"index": "time"})

        print("Data clean finished!")

        self.dataframe = new_df

    def get_trading_days(self, start, end):
        nyse = tc.get_calendar("NYSE")
        df = nyse.sessions_in_range(
            pd.Timestamp(start, tz=pytz.UTC), pd.Timestamp(end, tz=pytz.UTC)
        )
        return [str(day)[:10] for day in df]

    def fetch_latest_data(
        self, ticker_list, time_interval, tech_indicator_list, limit=100
    ) -> pd.DataFrame:

        data_df = pd.DataFrame()
        for tic in ticker_list:
            barset = self.api.get_barset(
                [tic], time_interval, limit=limit).df[tic]
            barset["tic"] = tic
            barset = barset.reset_index()
            data_df = data_df.append(barset)

        data_df = data_df.reset_index(drop=True)
        start_time = data_df.time.min()
        end_time = data_df.time.max()
        times = []
        current_time = start_time
        end = end_time + pd.Timedelta(minutes=1)
        while current_time != end:
            times.append(current_time)
            current_time += pd.Timedelta(minutes=1)

        df = data_df.copy()
        new_df = pd.DataFrame()
        for tic in ticker_list:
            tmp_df = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"], index=times
            )
            tic_df = df[df.tic == tic]
            for i in range(tic_df.shape[0]):
                tmp_df.loc[tic_df.iloc[i]["time"]] = tic_df.iloc[i][
                    ["open", "high", "low", "close", "volume"]
                ]

                if str(tmp_df.iloc[0]["close"]) == "nan":
                    for i in range(tmp_df.shape[0]):
                        if str(tmp_df.iloc[i]["close"]) != "nan":
                            first_valid_close = tmp_df.iloc[i]["close"]
                            tmp_df.iloc[0] = [
                                first_valid_close,
                                first_valid_close,
                                first_valid_close,
                                first_valid_close,
                                0.0,
                            ]
                            break
                if str(tmp_df.iloc[0]["close"]) == "nan":
                    print(
                        "Missing data for ticker: ",
                        tic,
                        " . The prices are all NaN. Fill with 0.",
                    )
                    tmp_df.iloc[0] = [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]

            for i in range(tmp_df.shape[0]):
                if str(tmp_df.iloc[i]["close"]) == "nan":
                    previous_close = tmp_df.iloc[i - 1]["close"]
                    if str(previous_close) == "nan":
                        raise ValueError
                    tmp_df.iloc[i] = [
                        previous_close,
                        previous_close,
                        previous_close,
                        previous_close,
                        0.0,
                    ]
            tmp_df = tmp_df.astype(float)
            tmp_df["tic"] = tic
            new_df = new_df.append(tmp_df)

        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={"index": "time"})

        df = self.add_technical_indicator(new_df, tech_indicator_list)
        df["VIXY"] = 0

        price_array, tech_array, turbulence_array = self.df_to_array(
            df, tech_indicator_list, if_vix=True
        )
        latest_price = price_array[-1]
        latest_tech = tech_array[-1]
        turb_df = self.api.get_barset(
            ["VIXY"], time_interval, limit=1).df["VIXY"]
        latest_turb = turb_df["close"].values
        return latest_price, latest_tech, latest_turb

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
