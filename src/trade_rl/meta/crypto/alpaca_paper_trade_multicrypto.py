
import math
import threading
import time
from datetime import datetime
from datetime import timedelta

import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd
import torch

from trade_rl.meta.data_processors.alpaca_crypto import AlpacaCrypto
from trade_rl.meta.data_processors._base import time_convert


class AlpacaPaperTradingMultiCrypto:
    def __init__(
        self,
        ticker_list,
        time_interval,
        agent,
        cwd,
        action_dim,
        API_KEY,
        API_SECRET,
        API_BASE_URL,
        tech_indicator_list,
        max_stock=1e2,
    ):
        # load agent
        # load agent
        if agent == "ppo":
            from stable_baselines3 import PPO

            try:
                # load agent
                self.model = PPO.load(cwd)
                print("Successfully load model", cwd)
            except:
                raise ValueError("Fail to load agent!")
        else:
            raise ValueError("Agent input is NOT supported yet.")

        # connect to Alpaca trading API
        try:
            self.alpaca = tradeapi.REST(
                API_KEY, API_SECRET, API_BASE_URL, "v2")
            print("Connected to Alpaca API!")
        except:
            raise ValueError(
                "Fail to connect Alpaca. Please check account info and internet connection."
            )
        # read trading settings
        self.tech_indicator_list = tech_indicator_list
        self.max_stock = max_stock
        self.previous_candles = 250
        self.lookback = 1
        self.action_dim = action_dim
        self.action_decimals = 2
        self.time_interval = time_interval

        # initialize account
        self.stocks = np.asarray([0] * len(ticker_list))  # stocks holding
        self.stocks_cd = np.zeros_like(self.stocks)
        self.cash = None  # cash record
        self.stocks_df = pd.DataFrame(
            self.stocks, columns=["stocks"], index=ticker_list
        )
        self.asset_list = []
        self.price = np.asarray([0] * len(ticker_list))

        stockUniverse = []
        for stock in ticker_list:
            stock = stock.replace("USDT", "USD")
            stockUniverse.append(stock)

        self.ticker_list = ticker_list
        self.stockUniverse = stockUniverse
        self.equities = []

    def test_latency(self, test_times=10):
        total_time = 0
        for _ in range(test_times):
            time0 = time.time()
            self.get_state()
            time1 = time.time()
            temp_time = time1 - time0
            total_time += temp_time
        latency = total_time / test_times
        print("latency for data processing: ", latency)
        return latency

    def run(self):
        orders = self.alpaca.list_orders(status="open")
        for order in orders:
            self.alpaca.cancel_order(order.id)
        while True:
            print("\n" + "#################### NEW CANDLE ####################")
            print("#################### NEW CANDLE ####################" + "\n")

            trade = threading.Thread(target=self.trade)
            trade.start()
            trade.join()
            last_equity = float(self.alpaca.get_account().last_equity)
            cur_time = time.time()
            self.equities.append([cur_time, last_equity])
            time.sleep(time_convert(self.time_interval))

    def trade(self):
        # Get state
        state = self.get_state()

        # Get action
        action = self.model.predict(state)[0]
        action = (action * self.max_stock).astype(float)

        print("\n" + "ACTION:    ", action, "\n")
        # Normalize action
        action_norm_vector = []
        for price in self.price:
            print("PRICE:    ", price)
            x = math.floor(math.log(price, 10)) - 2
            print("MAG:      ", x)
            action_norm_vector.append(1 / ((10) ** x))
            print("NORM VEC: ", action_norm_vector)

        for i in range(self.action_dim):
            norm_vector_i = action_norm_vector[i]
            action[i] = action[i] * norm_vector_i

        print("\n" + "NORMALIZED ACTION:    ", action, "\n")

        # Trade
        self.stocks_cd += 1
        min_action = 10 ** -(self.action_decimals)  # stock_cd
        for index in np.where(action < -min_action)[0]:  # sell_index:
            sell_num_shares = min(self.stocks[index], -action[index])

            qty = abs(float(sell_num_shares))
            qty = round(qty, self.action_decimals)
            print("SELL, qty:", qty)

            respSO = []
            tSubmitOrder = threading.Thread(
                target=self.submitOrder(
                    qty, self.stockUniverse[index], "sell", respSO)
            )
            tSubmitOrder.start()
            tSubmitOrder.join()
            self.cash = float(self.alpaca.get_account().cash)
            self.stocks_cd[index] = 0

        for index in np.where(action > min_action)[0]:  # buy_index:
            tmp_cash = max(self.cash, 0)
            print("current cash:", tmp_cash)
            # Adjusted part to accept decimal places up to two
            buy_num_shares = min(
                tmp_cash / self.price[index], abs(float(action[index]))
            )

            qty = abs(float(buy_num_shares))
            qty = round(qty, self.action_decimals)
            print("BUY, qty:", qty)

            respSO = []
            tSubmitOrder = threading.Thread(
                target=self.submitOrder(
                    qty, self.stockUniverse[index], "buy", respSO)
            )
            tSubmitOrder.start()
            tSubmitOrder.join()
            self.cash = float(self.alpaca.get_account().cash)
            self.stocks_cd[index] = 0

        print("Trade finished")

    def get_state(self):
        alpaca_proc = AlpacaCrypto(data_source="alpacacrypto",
                                   API=self.alpaca, time_interval=self.time_interval)

        cur_price, cur_tech, _ = alpaca_proc.fetch_latest_data(
            ticker_list=self.stockUniverse,
            time_interval=self.time_interval,
            tech_indicator_list=self.tech_indicator_list,
        )

        print("fetching latest candles..")
        positions = self.alpaca.list_positions()
        stocks = [0] * len(self.stockUniverse)
        self.price = cur_price

        for position in positions:
            ind = self.stockUniverse.index(position.symbol)
            stocks[ind] = abs(int(float(position.qty)))

        stocks = np.asarray(stocks, dtype=float)
        cash = float(self.alpaca.get_account().cash)
        self.cash = cash
        self.stocks = stocks

        # Stack cash and stocks
        state = np.hstack((self.cash * 2**-18, self.stocks * 2**-3))
        normalized_tech = cur_tech * 2**-15
        state = np.hstack((state, normalized_tech)).astype(np.float32)

        print("\n" + "STATE:")
        print(state)

        return state

    def submitOrder(self, qty, stock, side, resp):
        if qty > 0:
            try:
                self.alpaca.submit_order(stock, qty, side, "market", "gtc")
                print(
                    "Market order of | "
                    + str(qty)
                    + " "
                    + stock
                    + " "
                    + side
                    + " | completed."
                )
                resp.append(True)
            except Exception as e:
                print("ALPACA API ERROR: ", e)
                print(
                    "Order of | "
                    + str(qty)
                    + " "
                    + stock
                    + " "
                    + side
                    + " | did not go through."
                )
                resp.append(False)
        else:
            print(
                "Quantity is 0, order of | "
                + str(qty)
                + " "
                + stock
                + " "
                + side
                + " | not completed."
            )
            resp.append(True)

    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh
