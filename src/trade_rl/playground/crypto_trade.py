from trade_rl.meta.data_processors.alpaca_crypto import AlpacaCrypto
from pydantic import BaseModel, ValidationError
import stockstats
import pydantic
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
from typing import List, Literal, Optional, Any
import re
import datetime as dt
import time

from trade_rl.meta.crypto.alpaca_paper_trade_multicrypto import AlpacaPaperTradingMultiCrypto
from trade_rl.meta.crypto.env_multiple_crypto import CryptoEnv
from trade_rl.test import test
from trade_rl.train import train
from pathlib import Path
from trade_rl.meta import constants

# Tester avec binance
# Tester avec AlpacaCrypto
# Tester de trade avec AlpacaCrypto

# 2. Set Parameters
TICKER_LIST = [
    "BTCUSD",
    "ETHUSD",
]

PPO_PARAMS = {"n_steps": 2048, "ent_coef": 0.01,
              "learning_rate": 0.00025, "batch_size": 128}

time_interval = "1d"

TRAIN_START_DATE = "2022-10-26"
TRAIN_END_DATE = "2022-11-01"

TEST_START_DATE = "2021-11-01"
TEST_END_DATE = "2022-01-02"

INDICATORS = [
    "macd",
    "rsi",
    "cci",
    "dx",
]

# 3. Create Multiple Cryptocurrencies Trading Env
initial_capital = 1e6
env = CryptoEnv

# 4. Training
start_time = time.time()

model_params = dict(total_timesteps=1e4,
                    agent_params=PPO_PARAMS, model_name="ppo")

api_config = {'API_KEY': constants.ALPACA_API_KEY,
              'API_SECRET': constants.ALPACA_API_SECRET,
              'API_BASE_URL': "https://paper-api.alpaca.markets"}

processor_config = dict(start_date=TRAIN_START_DATE,
                        end_date=TRAIN_END_DATE,
                        data_source="alpacacrypto",
                        time_interval="1Min",
                        api_config=api_config
                        )

data_config = dict(ticker_list=TICKER_LIST,  technical_indicator_list=INDICATORS,
                   if_vix=False)


train(drl_lib="stable_baselines3",
      processor_config=processor_config,
      data_config=data_config,
      env_class=env,
      cwd=Path("./test_ppo"),
      model_params=model_params,
      initial_capital=10000,
      )

test(drl_lib="stable_baselines3",
     processor_config=processor_config,
     data_config=data_config,
     env_class=env,
     cwd=Path("./test_ppo"),
     model_name="ppo",
     initial_capital=10000,
     )

paper_trading_erl = AlpacaPaperTradingMultiCrypto(ticker_list=TICKER_LIST,
                                                  time_interval='1Min',
                                                  agent_path=Path(
                                                      "./test_ppo/ppo").resolve(),
                                                  agent="ppo",
                                                  action_dim=len(TICKER_LIST),
                                                  api_config=api_config,
                                                  tech_indicator_list=INDICATORS,
                                                  max_stock=1e2)
paper_trading_erl.run()
