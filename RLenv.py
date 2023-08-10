import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import ta


class Positions(Enum):
    Short = 0
    Long = 1
    Hold = 2

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long


class Actions(Enum):
    Sell = 0
    Buy = 1
    Hold = 2


class CustomTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size, frame_bound, initial_money):
        assert df.ndim == 2
        assert len(frame_bound) == 2

        self.seed()
        self.df = df
        self.window_size = window_size
        self.frame_bound = frame_bound
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float64)

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None

        self.trade_fee_bid_percent = 0.01  # unit
        self.trade_fee_ask_percent = 0.005  # unit

        self.hold_length = 0

        self.initial_money = initial_money
        self.money = initial_money
        self.number_of_stocks = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.Hold
        self._position_history = (
            self.window_size * [Positions.Hold]) + [self._position]
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.history = {}
        self.hold_length = 0

        self.money = self.initial_money
        self.number_of_stocks = 0

        return self._get_observation()

    def step(self, action):
        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True

        if action == Actions.Buy.value:
            if self._position == Positions.Short and self.money >= self.prices[self._current_tick]:
                self.number_of_stocks = int(
                    self.money / self.prices[self._current_tick])
                self.money -= self.number_of_stocks * \
                    self.prices[self._current_tick]

        elif action == Actions.Sell.value:
            if self._position == Positions.Long and self.number_of_stocks > 0:
                self.money += self.number_of_stocks * \
                    self.prices[self._current_tick]
                self.number_of_stocks = 0

        elif action == Actions.Hold.value:
            self.hold_length += 1

        self._update_profit(action)

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        if action == Actions.Buy.value and (self._position == Positions.Short or self._position == Positions.Hold):
            if self.money >= self.prices[self._current_tick]:
                self.number_of_stocks = int(
                    self.money / self.prices[self._current_tick])
                self.money -= self.number_of_stocks * \
                    self.prices[self._current_tick]
                self._position = Positions.Long
                self._last_trade_tick = self._current_tick
                self.hold_length = 0

        elif action == Actions.Sell.value and (self._position == Positions.Long or self._position == Positions.Hold):
            if self.number_of_stocks > 0:
                self.money += self.number_of_stocks * \
                    self.prices[self._current_tick]
                self.number_of_stocks = 0
                self._position = Positions.Short
                self._last_trade_tick = self._current_tick
                self.hold_length = 0

        else:
            self.hold_length += 1

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            position=self._position.value
        )
        self._update_history(info)

        return observation, step_reward, self._done, info

    def _get_observation(self):
        return self.signal_features[(self._current_tick-self.window_size+1):self._current_tick+1]

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def render(self, mode='human'):
        def _plot_position(position, tick):
            color = None
            if position == Positions.Short:
                color = 'red'
            elif position == Positions.Long:
                color = 'green'
            if color:
                plt.scatter(tick, self.prices[tick], color=color)
        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self._position_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position, self._current_tick)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )
        plt.pause(0.01)

    def render_all(self, mode='human'):
        window_ticks = np.arange(len(self._position_history))
        plt.plot(self.prices)

        buy_ticks = []
        sell_ticks = []
        hold_ticks = []

        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == Positions.Short:
                sell_ticks.append(tick)
            elif self._position_history[i] == Positions.Long:
                buy_ticks.append(tick)
            elif self._position_history[i] == Positions.Hold:
                hold_ticks.append(tick)

        plt.plot(buy_ticks, self.prices[buy_ticks], 'gv', label='Buy')
        plt.plot(sell_ticks, self.prices[sell_ticks], 'r^', label='Sell')
        plt.plot(hold_ticks, self.prices[hold_ticks], 'bo', label='Hold')

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

        plt.legend()  # Add a legend to differentiate Buy and Sell points

        if mode == 'human':
            plt.show()

    def close(self):
        plt.close()

    def _process_data(self):
        prices = self.df['close'].to_numpy()

        prices[self.frame_bound[0] - self.window_size]
        prices = prices[self.frame_bound[0] -
                        self.window_size:self.frame_bound[1]]

        diff = np.insert(np.diff(prices), 0, 0)

        macd_period = 12
        short_ema = self._calculate_ema(prices, macd_period)

        long_ema_period = 26
        long_ema = self._calculate_ema(prices, long_ema_period)

        macd_line = short_ema - long_ema

        signal_features = np.column_stack((prices, diff, macd_line))

        return prices, signal_features

    def _calculate_ema(self, prices, period):
        alpha = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]

        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

        return ema

    def _calculate_reward(self, action):
        step_reward = 0

        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
                (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            price_diff = current_price - last_trade_price

            if self._position == Positions.Long:
                step_reward += price_diff
        else:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            price_diff = current_price - last_trade_price

            if self._position == Positions.Long:
                if price_diff > 0:
                    step_reward += price_diff * self.hold_length

            if self._position == Positions.Hold:
                step_reward += price_diff * self.hold_length

        return step_reward

    def _update_profit(self, action):
        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
                (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade or self._done:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self._position == Positions.Long:
                shares = (self._total_profit *
                          (1 - self.trade_fee_ask_percent)) / last_trade_price
                self._total_profit = (
                    shares * (1 - self.trade_fee_bid_percent)) * current_price

    def max_possible_profit(self):  # trade fees are ignored
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Short
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Long

            if position == Positions.Long:
                current_price = self.prices[current_tick - 1]
                last_trade_price = self.prices[last_trade_tick]
                shares = profit / last_trade_price
                profit = shares * current_price
            last_trade_tick = current_tick - 1

        return profit
