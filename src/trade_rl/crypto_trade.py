import time

from trade_rl.meta.crypto.alpaca_paper_trade_multicrypto import AlpacaCrypto
from trade_rl.meta.crypto.env_multiple_crypto import CryptoEnv
from trade_rl.test import test
from trade_rl.train import train

# Tester avec binance
# Tester avec AlpacaCrypto
# Tester de trade avec AlpacaCrypto

# 2. Set Parameters
TICKER_LIST = [
    "BTCUSDT",
    "ETHUSDT",
    "ADAUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "SOLUSDT",
    "DOTUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "UNIUSDT",
]

PPO_PARAMS = {"n_steps": 2048, "ent_coef": 0.01,
              "learning_rate": 0.00025, "batch_size": 128}

time_interval = "1D"

TRAIN_START_DATE = "2020-10-01"
TRAIN_END_DATE = "2021-11-08"

TEST_START_DATE = "2021-11-08"
TEST_END_DATE = "2022-01-22"

INDICATORS = [
    "macd",
    "rsi",
    "cci",
    "dx",
]

net_dimension = 2**9

# ERL_PARAMS = {
#     "learning_rate": 2**-15,
#     "batch_size": 2**11,
#     "gamma": 0.99,
#     "seed": 312,
#     "net_dimension": 2**9,
#     "target_step": 5000,
#     "eval_gap": 30,
#     "eval_times": 1,
# }

# 3. Create Multiple Cryptocurrencies Trading Env
initial_capital = 1e6
env = CryptoEnv

# 4. Training
start_time = time.time()

train(
    start_date=TRAIN_START_DATE,
    end_date=TRAIN_END_DATE,
    ticker_list=TICKER_LIST,
    data_source="binance",
    time_interval=time_interval,
    technical_indicator_list=INDICATORS,
    drl_lib="stable_baselines3",
    env_class=env,
    model_name="ppo",
    cwd="./test_ppo",
    agent_params=PPO_PARAMS,
    if_vix=False,
    total_timesteps=1e4
)

duration_train = round((time.time() - start_time), 2)

# start_time = time.time()

# account_value_erl = test(
#     start_date=TEST_START_DATE,
#     end_date=TEST_END_DATE,
#     ticker_list=TICKER_LIST,
#     data_source="binance",
#     time_interval=time_interval,
#     technical_indicator_list=INDICATORS,
#     drl_lib="stable_baseline3",
#     env=env,
#     model_name="ppo",
#     current_working_dir="./test_ppo",
#     if_vix=False,
# )

# duration_test = round((time.time() - start_time), 2)
