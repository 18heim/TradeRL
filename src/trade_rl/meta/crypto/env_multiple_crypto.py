import math

import numpy as np

import gym
from trade_rl.meta import config

# TODO:
# je veux pouvoir facilement configurer le max de stock qu'on puisse acheter.
# pourquoi est-ce qu'on met pas des montants en € sur l'achat d'action plutôt que
# de dire j'achète 0.3 bitcoin et de normalizer donc des actions qui ne sont pas du tout sur
# la même échelle et d'utiliser un mecanisme chelou pour les standardiser ?
# En plus ça empêche d'avoir un contrôle et de faire des actions avec un faible Apport.


class CryptoEnv(gym.Env):
    """Env for trading multiple Crypto Currencies.

    Inherits gym.Env.
    """

    def __init__(
        self,
        data_config,
        lookback=1,
        initial_capital=1e6,
        buy_cost_pct=1e-3,
        sell_cost_pct=1e-3,
        gamma=0.99,
    ):
        self.lookback = lookback
        self.initial_total_asset = initial_capital
        self.initial_cash = initial_capital
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.max_stock = 1
        self.gamma = gamma
        self.price_array = data_config["price_array"]
        self.tech_array = data_config["tech_array"]
        self._generate_action_normalizer()
        self.crypto_num = self.price_array.shape[1]
        self.max_step = self.price_array.shape[0] - lookback - 1

        # reset
        self.time = lookback - 1
        self.cash = self.initial_cash
        self.current_price = self.price_array[self.time]
        self.current_tech = self.tech_array[self.time]
        self.stocks = np.zeros(self.crypto_num, dtype=np.float32)

        self.total_asset = self.cash + \
            (self.stocks * self.price_array[self.time]).sum()
        self.episode_return = 0.0
        self.gamma_return = 0.0

        """env information"""
        self.env_name = "MulticryptoEnv"
        # Cash + n_stocks_held + (price, tech_array) * n_stocks  * lookback
        self.state_dim = (
            1 + self.price_array.shape[1] +
            (self.price_array.shape[1] + self.tech_array.shape[1]) * lookback
        )
        self.action_dim = self.price_array.shape[1]
        self.if_discrete = False
        self.target_return = 10

        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )

    def reset(self) -> np.ndarray:
        """Reset env.

        Set time to 0, set price at first step etc etc.
        """
        self.time = self.lookback - 1
        self.current_price = self.price_array[self.time]
        self.current_tech = self.tech_array[self.time]
        self.cash = self.initial_cash  # reset()
        self.stocks = np.zeros(self.crypto_num, dtype=np.float32)
        self.total_asset = self.cash + \
            (self.stocks * self.price_array[self.time]).sum()

        return self.get_state()

    def step(self, actions) -> (np.ndarray, float, bool, None):
        """Take a step in MDP.

        Increment time.
        Fetch current price.
        Normalize action according to average price of a single stock quantity,
        to take into account the difference in scale of crypto prices.
        Retrieve stock index to sell. Sell desired quantity or max available.
        Decrement stock quantity.
        Increment cash taking broker fee into account.
        Retrieve stock index to buy. Buy desired quantity or max cash available.
        Increment stock quantity.
        Decrement cash taking broker fee into account.
        Update state.
        Compute reward.
        """
        self.time += 1
        price = self.price_array[self.time]
        # Normalize actions.
        for i in range(self.action_dim):
            norm_vector_i = self.action_norm_vector[i]
            actions[i] = actions[i] * norm_vector_i

        # Sell stock.
        for index in np.where(actions < 0)[0]:
            if price[index] > 0:  # Sell only if current asset is > 0
                sell_num_shares = min(self.stocks[index], -actions[index])
                self.stocks[index] -= sell_num_shares
                self.cash += price[index] * \
                    sell_num_shares * (1 - self.sell_cost_pct)

        # Buy stock
        for index in np.where(actions > 0)[0]:  # buy_index:
            if (
                price[index] > 0
            ):  # Buy only if the price is > 0 (no missing data in this particular date)
                buy_num_shares = min(self.cash // price[index], actions[index])
                self.stocks[index] += buy_num_shares
                self.cash -= price[index] * \
                    buy_num_shares * (1 + self.buy_cost_pct)
        done = self.time == self.max_step
        # Update state
        state = self.get_state()
        # Compute reward
        next_total_asset = self.cash + \
            (self.stocks * self.price_array[self.time]).sum()
        reward = (next_total_asset - self.total_asset) * config.REWARD_SCALE
        self.total_asset = next_total_asset
        self.gamma_return = self.gamma_return * self.gamma + reward
        self.cumu_return = self.total_asset / self.initial_cash
        if done:
            reward = self.gamma_return
            self.episode_return = self.total_asset / self.initial_cash
        return state, reward, done, dict()

    def get_state(self):
        """Compute state.

        State comprises:
            - Cash balance.
            - Stocks qty held.
            - (tech_indicators, price) for each stock for a certain number
                of lookback time steps.
        """
        state = np.hstack((self.cash * config.CASH_SCALE,
                          self.stocks * config.STOCK_QTY_SCALE))
        for i in range(self.lookback):
            tech_i = self.tech_array[self.time - i]
            price_i = self.price_array[self.time - i] * config.CASH_SCALE
            normalized_tech_i = tech_i * config.TECH_SCALE
            state = np.hstack(
                (state, price_i, normalized_tech_i)).astype(np.float32)
        return state

    def close(self):
        pass

    def _generate_action_normalizer(self):
        """normalize action to adjust for large price differences in cryptocurrencies."""
        price_0 = self.price_array[0]  # Use row 0 prices to normalize
        self.action_norm_vector = generate_action_normalizer(price_0)


def generate_action_normalizer(base_price):
    """Utility function standardizing prices."""
    # TODO: Stock qty are scaled but I would like to set a max amount for an action.
    # Or take the initial_capital into account.
    action_norm_vector = []
    for stock_price in base_price:
        x = math.floor(math.log(stock_price, 10))  # the order of magnitude
        action_norm_vector.append(1 / ((10) ** x))

    action_norm_vector = (
        np.asarray(action_norm_vector) * 10000
    )  # roughly control the maximum transaction amount for each action
    return np.asarray(action_norm_vector)
