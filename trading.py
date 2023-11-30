import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid

# Alpaca API Configuration
alpaca_api_key = 'your_api_key'
alpaca_secret_key = 'your_secret_key'
base_url = 'https://paper-api.alpaca.markets'  # Use 'https://api.alpaca.markets' for live trading

api = tradeapi.REST(alpaca_api_key, alpaca_secret_key, base_url, api_version='v2')

def fetch_historical_data(symbol, start_date, end_date):
    # Fetch historical stock data
    data = api.get_barset(symbol, 'day', start=start_date, end=end_date).df[symbol]
    return data

def dual_moving_average_strategy(data, short_window, long_window):
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0.0

    # Create short simple moving average
    signals['short_mavg'] = data['close'].rolling(window=short_window, min_periods=1, center=False).mean()

    # Create long simple moving average
    signals['long_mavg'] = data['close'].rolling(window=long_window, min_periods=1, center=False).mean()

    # Generate signals
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)
    signals['positions'] = signals['signal'].diff()

    return signals

def calculate_positions(data, signals, initial_capital=100000, max_position_size=0.02):
    positions = pd.DataFrame(index=data.index).fillna(0.0)
    positions['stock'] = max_position_size * initial_capital * signals['signal']  # Implementing position sizing based on portfolio risk
    portfolio = positions.multiply(data['close'], axis=0)
    cash = initial_capital - np.cumsum(positions.multiply(data['close'], axis=0).sum(axis=1))
    holdings = cash + portfolio.sum(axis=1)
    return holdings

def calculate_returns(holdings):
    returns = holdings.pct_change()
    return returns

def calculate_metrics(returns):
    cumulative_returns = (1 + returns).cumprod()
    daily_returns = returns.copy()
    annualized_return = (cumulative_returns[-1]) ** (1 / (len(returns) / 252.0)) - 1.0
    annualized_volatility = returns.std() * np.sqrt(252.0)
    sharpe_ratio = (annualized_return / annualized_volatility) if annualized_volatility != 0 else 0.0

    return {
        'Cumulative Returns': cumulative_returns[-1],
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': calculate_max_drawdown(holdings),
        'Average Daily Return': daily_returns.mean(),
        'Winning Trades Ratio': calculate_winning_trades_ratio(returns),
        'Losing Trades Ratio': calculate_losing_trades_ratio(returns)
    }

def calculate_max_drawdown(holdings):
    cumulative_returns = (1 + holdings.pct_change()).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    return max_drawdown

def calculate_winning_trades_ratio(returns):
    positive_returns = returns[returns > 0]
    winning_trades_ratio = len(positive_returns) / len(returns) if len(returns) > 0 else 0.0
    return winning_trades_ratio

def calculate_losing_trades_ratio(returns):
    negative_returns = returns[returns < 0]
    losing_trades_ratio = len(negative_returns) / len(returns) if len(returns) > 0 else 0.0
    return losing_trades_ratio

def optimize_parameters(data, parameter_grid):
    best_metrics = None
    best_parameters = None

    for params in ParameterGrid(parameter_grid):
        signals = dual_moving_average_strategy(data, params['short_window'], params['long_window'])
        holdings = calculate_positions(data, signals)
        returns = calculate_returns(holdings)
        metrics = calculate_metrics(returns)

        if best_metrics is None or metrics['Sharpe Ratio'] > best_metrics['Sharpe Ratio']:
            best_metrics = metrics
            best_parameters = params

    return best_parameters

def forward_test(data, best_parameters):
    signals = dual_moving_average_strategy(data, best_parameters['short_window'], best_parameters['long_window'])
    holdings = calculate_positions(data, signals)
    returns = calculate_returns(holdings)
    metrics = calculate_metrics(returns)

    return signals, holdings, metrics

def plot_results(data, signals, holdings):
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 8), sharex=True)

    ax1.plot(data['close'], label='Price')
    ax1.plot(signals['short_mavg'], label='Short MA')
    ax1.plot(signals['long_mavg'], label='Long MA')

    ax1.plot(signals.loc[signals['signal'] == 1.0].index,
             signals['short_mavg'][signals['signal'] == 1.0],
             '^', markersize=10, color='g', label='Buy Signal')

    ax1.plot(signals.loc[signals['signal'] == -1.0].index,
             signals['short_mavg'][signals['signal'] == -1.0],
             'v', markersize=10, color='r', label='Sell Signal')

    ax1.set_title('Dual Moving Average Crossover Strategy')
    ax1.set_ylabel('Price')
    ax1.legend()

    ax2.plot(holdings, label='Portfolio Value', color='c')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Portfolio Value')
    ax2.legend()

    plt.show()

def check_data_quality(data):
    # Implement data quality checks
    if data.isnull().values.any():
        print("Warning: Data contains missing values. Imputing missing values.")
        data.fillna(method='ffill', inplace=True)  # Forward fill missing values

def market_impact_simulation(data, signals, impact_factor=0.01):
    # Simulate market impact by adjusting closing prices based on trading signals
    impact_multiplier = 1 + (impact_factor * signals['signal'].shift(1))
    data['close'] = data['close'] * impact_multiplier

def dynamic_strategy_adaptation(data, signals, adaptation_window=20):
    # Implement dynamic strategy adaptation based on recent market conditions
    volatility = data['close'].pct_change().rolling(window=adaptation_window).std()
    signals['short_window'] = adaptation_window

def backtest(data, parameters):
    signals = dual_moving_average_strategy(data, parameters['short_window'], parameters['long_window'])
    holdings = calculate_positions(data, signals)
    returns = calculate_returns(holdings)
    metrics = calculate_metrics(returns)
    return signals, holdings, returns, metrics

def place_market_order(api, symbol, qty, side):
    try:
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='market',
            time_in_force='gtc'  # Good 'til canceled
        )
        print(f"Market order placed for {qty} shares of {symbol} ({side}).")
    except Exception as e:
        print(f"Error placing order: {e}")

def live_trading(api, signals):
    for index, row in signals.iterrows():
        if row['signal'] == 1.0:
            # Buy signal
            place_market_order(api, symbol='AAPL', qty=10, side='buy')
        elif row['signal'] == -1.0:
            # Sell signal
            place_market_order(api, symbol='AAPL', qty=10, side='sell')

def parse_code():
    # Implement code parsing logic here
    pass

def main():
    # Define stock symbol and date range
    symbol = 'AAPL'
    start_date = '2022-01-01'
    end_date = '2023-01-01'

    # Fetch historical stock data
    stock_data = fetch_historical_data(symbol, start_date, end_date)

    # Check data quality
    check_data_quality(stock_data)

    # Define parameter grid for optimization
    parameter_grid = {
        'short_window': [20, 40, 60],
        'long_window': [80, 100, 120]
    }

    # Optimize parameters
    best_parameters = optimize_parameters(stock_data, parameter_grid)
    print("Optimal Parameters:", best_parameters)

    # Forward test with optimal parameters
    signals, holdings, metrics = forward_test(stock_data, best_parameters)

    # Market impact simulation
    market_impact_simulation(stock_data, signals)

    # Dynamic strategy adaptation
    dynamic_strategy_adaptation(stock_data, signals)

    # Backtest with optimal parameters
    backtest_signals, backtest_holdings, backtest_returns, backtest_metrics = backtest(stock_data, best_parameters)

    # Print performance metrics
    print("Forward Test Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nBacktest Metrics:")
    for metric, value in backtest_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Plot results
    plot_results(stock_data, signals, holdings)

    # Live trading (uncomment the next line when ready for live trading)
    # live_trading(api, signals)

    # Parse code
    parse_code()

if __name__ == "__main__":
    main()
                                                                                                                                                                                                                                                