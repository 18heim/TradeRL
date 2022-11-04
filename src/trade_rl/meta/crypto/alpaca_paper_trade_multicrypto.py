
import math
import threading
import time

import numpy as np
import pandas as pd

from trade_rl.meta import constants
from trade_rl.meta.data_processors.alpaca_crypto import AlpacaCrypto
from trade_rl.meta.crypto.env_multiple_crypto import generate_action_normalizer
from trade_rl.meta.data_processors._base import time_convert


class AlpacaPaperTradingMultiCrypto:
    def __init__(
        self,
        ticker_list,
        time_interval,
        agent,
        agent_path,
        action_dim,
        api_config,
        tech_indicator_list,
        max_trade=1e3,
        min_trade=20,
    ):
        # load agent
        if agent == "ppo":
            from stable_baselines3 import PPO

            try:
                # load agent
                self.model = PPO.load(agent_path)
                print("Successfully load model", agent_path)
            except:
                raise ValueError("Fail to load agent!")
        else:
            raise ValueError("Agent input is NOT supported yet.")

        # connect to Alpaca trading API
        try:
            self.alpaca = AlpacaCrypto(
                time_interval=time_interval, api_config=api_config)
            print("Connected to Alpaca API!")
        except:
            raise ValueError(
                "Fail to connect Alpaca. Please check account info and internet connection."
            )
        # read trading settings
        self.tech_indicator_list = tech_indicator_list
        self.max_trade = max_trade
        self.min_trade = min_trade
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
        """Test API Latency."""
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
        """Start trading."""
        orders = self.alpaca.api.list_orders(status="open")
        for order in orders:
            self.alpaca.api.cancel_order(order.id)
        while True:
            print("\n" + "#################### NEW CANDLE ####################")
            print("#################### NEW CANDLE ####################" + "\n")

            trade = threading.Thread(target=self.trade)
            trade.start()
            trade.join()
            last_equity = float(self.alpaca.api.get_account().last_equity)
            cur_time = time.time()
            self.equities.append([cur_time, last_equity])
            time.sleep(time_convert(self.time_interval))

    def trade(self):
        """Async trade function.

        Get state.
        Predict action in [-1,1] for each stock.
        Normalize action according to average price of a single stock quantity,
        to take into account the difference in scale of crypto prices.

        """
        # Get state
        state = self.get_state()

        # Get action
        # Normalize action and change to qty.
        action = self.model.predict(state)[0]
        action_cash = action * self.max_trade
        print("\n" + "ACTION CASH:    ", action, "\n")
        for i in range(self.action_dim):
            action[i] = action_cash[i] / self.price[i]

        print("\n" + "ACTION QTY:    ", action, "\n")

        print("\n", "ACTION HELD", self.stocks)

        self.stocks_cd += 1
        # Sell stock
        # sell_index:
        for index in np.where(action_cash < - self.min_trade)[0]:
            sell_num_shares = min(self.stocks[index], -action[index])

            qty = abs(float(sell_num_shares))
            qty = round(qty, self.action_decimals)
            print("SELL, qty:", qty, ",cash:", qty * self.price[index])

            respSO = []
            tSubmitOrder = threading.Thread(
                target=self.submitOrder(
                    qty, self.stockUniverse[index], "sell", respSO)
            )
            tSubmitOrder.start()
            tSubmitOrder.join()
            # Update cash balance.
            self.cash = float(self.alpaca.api.get_account().cash)
            self.stocks_cd[index] = 0
        # Buy stock
        print("current cash:", max(self.cash, 0))
        for index in np.where(action_cash > self.min_trade)[0]:  # buy_index:
            tmp_cash = max(self.cash, 0)
            # Adjusted part to accept decimal places up to two
            buy_num_shares = min(
                tmp_cash / self.price[index], abs(float(action[index]))
            )

            qty = abs(float(buy_num_shares))
            qty = round(qty, self.action_decimals)
            print("BUY, qty:", qty, ",cash:", qty * self.price[index])

            respSO = []
            tSubmitOrder = threading.Thread(
                target=self.submitOrder(
                    qty, self.stockUniverse[index], "buy", respSO)
            )
            tSubmitOrder.start()
            tSubmitOrder.join()
            # Update cash balance.
            self.cash = float(self.alpaca.api.get_account().cash)
            self.stocks_cd[index] = 0

        print("Trade finished")

    def get_state(self):
        """Compute state.

        State comprises:
            - Cash balance
            - Stock qty held
            - (tech_indicators, price) for each stock for
                a certain number of lookback time steps
        """
        # Fetch a window of lookback time steps price & tech.
        print("fetching latest candles..")
        cur_price, cur_tech, _ = self.alpaca.fetch_latest_data(
            ticker_list=self.stockUniverse,
            time_interval=self.time_interval,
            tech_indicator_list=self.tech_indicator_list,
        )
        self.price = cur_price

        # Fetch stock qty held.
        positions = self.alpaca.api.list_positions()
        stocks = [0] * len(self.stockUniverse)
        for position in positions:
            ind = self.stockUniverse.index(position.symbol)
            stocks[ind] = abs(float(position.qty))
        stocks = np.asarray(stocks, dtype=float)
        self.stocks = stocks

        # Fetch cash balance
        cash = float(self.alpaca.api.get_account().cash)
        self.cash = cash

        # Stack cash and stocks
        state = np.hstack((self.cash * constants.CASH_SCALE,
                           self.stocks * constants.STOCK_QTY_SCALE))
        normalized_tech = cur_tech * constants.TECH_SCALE
        normalized_price = cur_price * constants.CASH_SCALE
        state = np.hstack(
            (state, normalized_price, normalized_tech)).astype(np.float32)

        print("\n" + "STATE:")
        print(state)

        return state

    def submitOrder(self, qty, stock, side, resp):
        if qty > 0:
            try:
                self.alpaca.api.submit_order(stock, qty, side, "market", "gtc")
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
