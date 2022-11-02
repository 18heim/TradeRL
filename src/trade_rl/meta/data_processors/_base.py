
import datetime as dt
import re
from typing import List, Literal, Optional, Any, Dict

import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd
import pydantic
from pydantic import Field
import stockstats

from trade_rl.meta.config import (
    TIME_ZONE_BERLIN,
    TIME_ZONE_JAKARTA,
    TIME_ZONE_PARIS,
    TIME_ZONE_SHANGHAI,
    TIME_ZONE_USEASTERN,
)
from trade_rl.meta.config_tickers import (
    CAC_40_TICKER,
    CSI_300_TICKER,
    DAX_30_TICKER,
    DOW_30_TICKER,
    HSI_50_TICKER,
    LQ45_TICKER,
    MDAX_50_TICKER,
    NAS_100_TICKER,
    SDAX_50_TICKER,
    SP_500_TICKER,
    SSE_50_TICKER,
    TECDAX_TICKER,
)


class TimeIntervalError(Exception):
    """Custom error raised when time interval isn't correct."""

    def __init__(self, time_interval, data_source, time_units):
        self.time_interval = time_interval
        self.data_source = data_source
        self.time_units = time_units
        self.message = f"Error time interval is {time_interval} whereas {data_source} \
                allows only {time_units}"
        super().__init__(self.message)


class APIConfig(pydantic.BaseModel):
    API: Optional[Any]
    key_id:  Optional[str] = Field(alias="API_KEY")
    secret_key: Optional[str] = Field(alias="API_SECRET")
    base_url: Optional[pydantic.HttpUrl] = Field(alias="API_BASE_URL")

    @pydantic.root_validator(pre=True)
    @classmethod
    def check_api_or_key(cls, values):
        """Make sure we have either the secrets or the API."""
        if "API" not in values.keys() and list(values.keys()) != ["API_KEY", "API_SECRET", "API_BASE_URL"]:
            raise KeyError("Missing APIConfig keys")
        return values


class _Base(pydantic.BaseModel,  extra=pydantic.Extra.allow):
    data_source: Literal["alpaca", "alpacacrypto",
                         "ccxt", "binance", "yahoofinance"]
    start_date: Optional[dt.datetime] = None
    end_date: Optional[dt.datetime] = None
    time_interval: str
    api_config: Optional[APIConfig]

    @pydantic.validator("start_date", "end_date", pre=True)
    @classmethod
    def parse_date(cls, value):
        """Parse start_date and end_date."""
        return dt.datetime.fromisoformat(value) if value else None

    @pydantic.validator("api_config", pre=True)
    @classmethod
    def parse_api_config(cls, value):
        """Parse API config dictionnary."""
        value = APIConfig(**value) if value else None
        return value

    @pydantic.validator("time_interval")
    @classmethod
    def check_time_interval(cls, value, values):
        if "alpaca" in values["data_source"]:
            time_units = ["Min", "Hour", "Day", "Month"]
            unit = ''.join(filter(lambda x: x.isalpha(), value))
            if not any([unit in allowed_unit for allowed_unit in time_units]):
                raise TimeIntervalError(
                    value, values["data_source"], time_units)
        if values["data_source"] == "binance":
            time_intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h",
                              "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
            if not value in time_intervals:
                raise TimeIntervalError(
                    value, values["data_source"], time_intervals)
        if values["data_source"] == "yahoofinance":
            time_intervals = ["1m", "2m", "5m", "15m", "30m",
                              "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
            if not value in time_intervals:
                raise TimeIntervalError(
                    value, values["data_source"], time_intervals)
        return value

    def __init__(
        self,
        data_source: str,
        time_interval: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        api_config: Optional[APIConfig] = None,
    ):
        super().__init__(data_source=data_source, start_date=start_date,
                         end_date=end_date, time_interval=time_interval, api_config=api_config)
        self.time_zone = ""
        self.dataframe: pd.DataFrame = pd.DataFrame()
        self.dictnumpy: dict = (
            {}
        )  # e.g., self.dictnumpy["open"] = np.array([1, 2, 3]), self.dictnumpy["close"] = np.array([1, 2, 3])

    def download_data(self, ticker_list: List[str]):
        pass

    def clean_data(self):
        if "date" in self.dataframe.columns.values.tolist():
            self.dataframe.rename(columns={"date": "time"}, inplace=True)
        if "datetime" in self.dataframe.columns.values.tolist():
            self.dataframe.rename(columns={"datetime": "time"}, inplace=True)
        if self.data_source == "ccxt":
            self.dataframe.rename(columns={"index": "time"}, inplace=True)

        self.dataframe.dropna(inplace=True)
        # adjusted_close: adjusted close price
        if "adjusted_close" not in self.dataframe.columns.values.tolist():
            self.dataframe["adjusted_close"] = self.dataframe["close"]
        self.dataframe.sort_values(by=["time", "tic"], inplace=True)
        self.dataframe = self.dataframe[
            [
                "tic",
                "time",
                "open",
                "high",
                "low",
                "close",
                "adjusted_close",
                "volume",
            ]
        ]

    def get_trading_days(self, start: str, end: str) -> List[str]:
        if self.data_source in [
            "binance",
            "ccxt",
        ]:
            print(
                f"Calculate get_trading_days not supported for {self.data_source} yet."
            )
            return None

    def add_technical_indicator(
        self, tech_indicator_list: List[str]
    ):
        """
        calculate technical indicators
        use stockstats package to add technical indicators
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        if "date" in self.dataframe.columns.values.tolist():
            self.dataframe.rename(columns={"date": "time"}, inplace=True)

        if self.data_source == "ccxt":
            self.dataframe.rename(columns={"index": "time"}, inplace=True)

        self.dataframe.reset_index(drop=False, inplace=True)
        if "level_1" in self.dataframe.columns:
            self.dataframe.drop(columns=["level_1"], inplace=True)
        if "level_0" in self.dataframe.columns and "tic" not in self.dataframe.columns:
            self.dataframe.rename(columns={"level_0": "tic"}, inplace=True)
        print("tech_indicator_list: ", tech_indicator_list)
        self.dataframe = add_tech_indicator(
            self.dataframe, tech_indicator_list)
        print("Succesfully add technical indicators")

    def add_turbulence(self):
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        if self.data_source in [
            "binance",
            "ccxt",
        ]:
            print(
                f"Turbulence not supported for {self.data_source} yet. Return original DataFrame."
            )
        if self.data_source in [
            "alpaca",
            "yahoofinance",
        ]:
            turbulence_index = self.calculate_turbulence()
            self.dataframe = self.dataframe.merge(turbulence_index, on="time")
            self.dataframe.sort_values(["time", "tic"], inplace=True).reset_index(
                drop=True, inplace=True
            )

    def calculate_turbulence(self, time_period: int = 252) -> pd.DataFrame:
        """calculate turbulence index based on dow 30"""
        # can add other market assets
        df_price_pivot = self.dataframe.pivot(
            index="time", columns="tic", values="close"
        )
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = self.dataframe["time"].unique()
        # start after a year
        start = time_period
        turbulence_index = [0] * start
        # turbulence_index = [0]
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index ==
                                           unique_date[i]]
            # use one year rolling window to calcualte covariance
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - time_period])
            ]
            # Drop tickers which has number missing values more than the "oldest" ticker
            filtered_hist_price = hist_price.iloc[
                hist_price.isna().sum().min():
            ].dropna(axis=1)

            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[list(filtered_hist_price)] - np.mean(
                filtered_hist_price, axis=0
            )
            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                # avoid large outlier because of the calculation just begins: else turbulence_temp = 0
                turbulence_temp = temp[0][0] if count > 2 else 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)

        turbulence_index = pd.DataFrame(
            {"time": df_price_pivot.index, "turbulence": turbulence_index}
        )
        return turbulence_index

    def add_vix(self):
        """
        add vix from processors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        if self.data_source in [
            "binance",
            "ccxt",
        ]:
            print(
                f"VIX is not applicable for {self.data_source}. Return original DataFrame"
            )
            return None
        elif self.data_source == "yahoofinance":
            ticker = "^VIX"
        elif self.data_source == "alpaca":
            ticker = "VIXY"
        else:
            pass
        df = self.dataframe.copy()
        self.dataframe = [ticker]
        self.download_data(self.start_date, self.end_date, self.time_interval)
        self.clean_data()
        # vix = cleaned_vix[["time", "close"]]
        # vix = vix.rename(columns={"close": "VIXY"})
        cleaned_vix = self.dataframe.rename(columns={ticker: "vix"})

        df = df.merge(cleaned_vix, on="time")
        df = df.sort_values(["time", "tic"]).reset_index(drop=True)
        self.dataframe = df

    def df_to_array(self, tech_indicator_list: List[str], if_vix: bool):
        return df_to_array(self.dataframe, tech_indicator_list, if_vix)

    # "600000.XSHG" -> "sh.600000"
    # "000612.XSHE" -> "sz.000612"
    def transfer_standard_ticker_to_nonstandard(self, ticker: str) -> str:
        return ticker


def time_convert(x: str):
    time_span = int(re.findall(r'^\d+', x)[0])
    unit = re.findall(r'\D+', x)[0].lower()
    if "m" in unit:
        return time_span * 60
    elif "d" in unit:
        return time_span * 24 * 60 * 60
    elif "w" in unit:
        return time_span * 24 * 60 * 60 * 7
    elif "s" in unit:
        return time_span
    else:
        raise ValueError(f"{x} is an incorrect time format")


def add_tech_indicator(dataframe: pd.DataFrame, tech_indicator_list: List[str]):
    stock = stockstats.StockDataFrame.retype(dataframe)
    unique_ticker = stock.tic.unique()
    for indicator in tech_indicator_list:
        print("indicator: ", indicator)
        indicator_df = pd.DataFrame()
        for i in range(len(unique_ticker)):
            try:
                temp_indicator = stock[stock.tic ==
                                       unique_ticker[i]][indicator]
                temp_indicator = pd.DataFrame(temp_indicator)
                temp_indicator["tic"] = unique_ticker[i]
                temp_indicator["time"] = dataframe[
                    dataframe.tic == unique_ticker[i]
                ]["time"].to_list()
                indicator_df = pd.concat(
                    [indicator_df, temp_indicator],
                    axis=0,
                    join="outer",
                    ignore_index=True,
                )
            except Exception as e:
                print(e)
        if not indicator_df.empty:
            dataframe = dataframe.merge(
                indicator_df[["tic", "time", indicator]],
                on=["tic", "time"],
                how="left",
            )
    dataframe.sort_values(by=["time", "tic"], inplace=True)
    time_to_drop = dataframe[dataframe.isna().any(
        axis=1)].time.unique()
    dataframe = dataframe[~dataframe.time.isin(
        time_to_drop)]
    dataframe[tech_indicator_list] = dataframe[tech_indicator_list].backfill()
    return dataframe


def df_to_array(dataframe: pd.DataFrame, tech_indicator_list: List[str], if_vix: bool):
    unique_ticker = dataframe.tic.unique()
    price_array = np.column_stack(
        [dataframe[dataframe.tic == tic].close for tic in unique_ticker]
    )
    common_tech_indicator_list = [
        i
        for i in tech_indicator_list
        if i in dataframe.columns.values.tolist()
    ]
    tech_array = np.hstack(
        [
            dataframe.loc[
                (dataframe.tic == tic), common_tech_indicator_list
            ]
            for tic in unique_ticker
        ]
    )
    if if_vix:
        risk_array = np.column_stack(
            [dataframe[dataframe.tic == tic].vix for tic in unique_ticker]
        )
    else:
        risk_array = (
            np.column_stack(
                [
                    dataframe[dataframe.tic == tic].turbulence
                    for tic in unique_ticker
                ]
            )
            if "turbulence" in dataframe.columns
            else None
        )
    print("Successfully transformed into array")
    return price_array, tech_array, risk_array


def calc_time_zone(
    ticker_list: List[str],
    time_zone_selfdefined: str,
    use_time_zone_selfdefined: int,
) -> str:
    assert isinstance(ticker_list, list)
    ticker_list = ticker_list[0]
    if use_time_zone_selfdefined == 1:
        time_zone = time_zone_selfdefined
    elif ticker_list in HSI_50_TICKER + SSE_50_TICKER + CSI_300_TICKER:
        time_zone = TIME_ZONE_SHANGHAI
    elif ticker_list in DOW_30_TICKER + NAS_100_TICKER + SP_500_TICKER:
        time_zone = TIME_ZONE_USEASTERN
    elif ticker_list == CAC_40_TICKER:
        time_zone = TIME_ZONE_PARIS
    elif ticker_list in DAX_30_TICKER + TECDAX_TICKER + MDAX_50_TICKER + SDAX_50_TICKER:
        time_zone = TIME_ZONE_BERLIN
    elif ticker_list == LQ45_TICKER:
        time_zone = TIME_ZONE_JAKARTA
    else:
        raise ValueError("Time zone is wrong.")
    return time_zone
