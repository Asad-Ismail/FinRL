from __future__ import annotations

from cred_config import *


DATA_API_KEY = TRADING_API_KEY= API_KEY
DATA_API_SECRET = TRADING_API_SECRET = API_SECRET
DATA_API_BASE_URL = data_url
TRADING_API_BASE_URL = API_BASE_URL

from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.meta.paper_trading.alpaca import PaperTradingAlpaca
from finrl.config import INDICATORS

# Import Dow Jones 30 Symbols
from finrl.config_tickers import DOW_30_TICKER

ticker_list = DOW_30_TICKER
env = StockTradingEnv
# if you want to use larger datasets (change to longer period), and it raises error, please try to increase "target_step". It should be larger than the episode steps.
ERL_PARAMS = {
    "learning_rate": 3e-6,
    "batch_size": 2048,
    "gamma": 0.985,
    "seed": 312,
    "net_dimension": [128, 64],
    "target_step": 5000,
    "eval_gap": 30,
    "eval_times": 1,
}


action_dim = len(DOW_30_TICKER)
state_dim = (
    1 + 2 + 3 * action_dim + len(INDICATORS) * action_dim
)  # Calculate the DRL state dimension manually for paper trading. amount + (turbulence, turbulence_bool) + (price, shares, cd (holding time)) * stock_dim + tech_dim

paper_trading_erl = PaperTradingAlpaca(
    ticker_list=DOW_30_TICKER,
    time_interval="15Min",
    drl_lib="elegantrl",
    agent="ppo",
    cwd="./papertrading_erl_retrain",
    net_dim=ERL_PARAMS["net_dimension"],
    state_dim=state_dim,
    action_dim=action_dim,
    API_KEY=TRADING_API_KEY,
    API_SECRET=TRADING_API_SECRET,
    API_BASE_URL=TRADING_API_BASE_URL,
    tech_indicator_list=INDICATORS,
    turbulence_thresh=30,
    max_stock=5,
)

paper_trading_erl.run()