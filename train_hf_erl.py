# Disclaimer: Nothing herein is financial advice, and NOT a recommendation to trade real money. Many platforms exist for simulated trading (paper trading) which can be used for building and developing the methods discussed. Please use common sense and always first consult a professional before trading or investing.
# install finrl library
# %pip install --upgrade git+https://github.com/AI4Finance-Foundation/FinRL.git
# Alpaca keys
from __future__ import annotations

from cred_config import *


DATA_API_KEY = TRADING_API_KEY= API_KEY
DATA_API_SECRET = TRADING_API_SECRET = API_SECRET
DATA_API_BASE_URL = data_url
TRADING_API_BASE_URL = API_BASE_URL

print("DATA_API_KEY: ", DATA_API_KEY)
print("DATA_API_SECRET: ", DATA_API_SECRET)
print("DATA_API_BASE_URL: ", DATA_API_BASE_URL)
print("TRADING_API_KEY: ", TRADING_API_KEY)
print("TRADING_API_SECRET: ", TRADING_API_SECRET)
print("TRADING_API_BASE_URL: ", TRADING_API_BASE_URL)

from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.meta.paper_trading.alpaca import PaperTradingAlpaca
from finrl.meta.paper_trading.common import train, test, alpaca_history, DIA_history
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

# Set up sliding window of 6 days training and 2 days testing
import datetime
from pandas.tseries.offsets import BDay  # BDay is business day, not birthday...

today = datetime.datetime.today()

##Train on one month and test on last 5 days
TEST_END_DATE = (today - BDay(1)).to_pydatetime().date()
TEST_END_DATE_Backtest= today.date()
TEST_START_DATE = (TEST_END_DATE - BDay(5)).to_pydatetime().date()
TRAIN_END_DATE = (TEST_START_DATE - BDay(1)).to_pydatetime().date()
TRAIN_START_DATE = (TRAIN_END_DATE - BDay(30)).to_pydatetime().date()
TRAINFULL_START_DATE = TRAIN_START_DATE
TRAINFULL_END_DATE = TEST_END_DATE

TRAIN_START_DATE = str(TRAIN_START_DATE)
TRAIN_END_DATE = str(TRAIN_END_DATE)
TEST_START_DATE = str(TEST_START_DATE)
TEST_END_DATE = str(TEST_END_DATE)
TEST_END_DATE_Backtest= str(TEST_END_DATE_Backtest)
TRAINFULL_START_DATE = str(TRAINFULL_START_DATE)
TRAINFULL_END_DATE = str(TRAINFULL_END_DATE)

print("TRAIN_START_DATE: ", TRAIN_START_DATE)
print("TRAIN_END_DATE: ", TRAIN_END_DATE)
print("TEST_START_DATE: ", TEST_START_DATE)
print("TEST_END_DATE: ", TEST_END_DATE)
print("TRAINFULL_START_DATE: ", TRAINFULL_START_DATE)
print("TRAINFULL_END_DATE: ", TRAINFULL_END_DATE)

train(
    start_date=TRAIN_START_DATE,
    end_date=TRAIN_END_DATE,
    ticker_list=ticker_list,
    data_source="alpaca",
    time_interval="60Min",
    technical_indicator_list=INDICATORS,
    drl_lib="elegantrl",
    env=env,
    model_name="ppo",
    if_vix=True,
    ## Added these arguments
    initial_account=500,
    max_stock=5,
    initial_capital=500,
    buy_cost_pct=1e-1,
    sell_cost_pct=1e-1,
    
    pretrain_path="pretrain",
    
    API_KEY=DATA_API_KEY,
    API_SECRET=DATA_API_SECRET,
    API_BASE_URL=DATA_API_BASE_URL,
    erl_params=ERL_PARAMS,
    cwd="./papertrading_erl",  # current_working_dir
    break_step=1e5,
)

account_value_erl,res_df = test(
    start_date=TEST_START_DATE,
    end_date=TEST_END_DATE,
    ticker_list=ticker_list,
    data_source="alpaca",
    time_interval="60Min",
    technical_indicator_list=INDICATORS,
    drl_lib="elegantrl",
    env=env,
    model_name="ppo",
    if_vix=True,
     ## Added these arguments
    initial_account=500,
    max_stock=5,
    initial_capital=1000,
    buy_cost_pct=1e-1,
    sell_cost_pct=1e-1,
    
    API_KEY=DATA_API_KEY,
    API_SECRET=DATA_API_SECRET,
    API_BASE_URL=DATA_API_BASE_URL,
    cwd="./papertrading_erl",
    net_dimension=ERL_PARAMS["net_dimension"],
)



def backtest(df_test,df_dj,test_account_values,dj_returns):
    # Process df_testing
    # Convert the timestamp column to datetime
    df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])
    # Remove the time component of the timestamp
    df_test['Date'] = df_test['timestamp'].dt.date
    # Remove duplicates based on the timestamp
    df_test = df_test.drop_duplicates(subset='Date')
    #res_df.reset_index(inplace=True)
    df_test['Daily_Return'] = df_test["account"].pct_change()
    
    # Merge both dataframes on Date
    merged_df = pd.merge(df_test, df_dj, on='Date', suffixes=('_test', '_dj'))
    
    # Plotting
    plt.figure(figsize=(14,9))
    plt.plot(merged_df['Date'].values, merged_df['Daily_Return_test'].values,'-o', label='My strategy')
    plt.plot(merged_df['Date'].values, merged_df['Daily_Return_dj'].values,'-o', label='DJ')
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.legend()
    plt.savefig('returns.png')
    
    myreturn=test_account_values[-1]/test_account_values[0]
    
    return myreturn,dj_returns[-1]

df_djia, cumu_djia = DIA_history(start=TEST_START_DATE,end=TEST_END_DATE_Backtest)

algo_return,dj_return=backtest(res_df,df_djia,account_value_erl,cumu_djia)

if dj_return>algo_return:
    print(f"Dow jones results {dj_return} over week is greater than RL agent {algo_return}")
    print(f"It does not make sense to continue exiting!!")
    exit()


print(f"Dow jones results {dj_return} over week is worse than RL agent {algo_return}")
print(f"Now training on whole data!!")
    
train(
    start_date=TRAINFULL_START_DATE,  # After tuning well, retrain on the training and testing sets
    end_date=TRAINFULL_END_DATE,
    ticker_list=ticker_list,
    data_source="alpaca",
    time_interval="60Min",
    technical_indicator_list=INDICATORS,
    drl_lib="elegantrl",
    env=env,
    model_name="ppo",
    if_vix=True,
    
    initial_account=500,
    max_stock=5,
    initial_capital=500,
    buy_cost_pct=1e-1,
    sell_cost_pct=1e-1,
    
    pretrain_path="pretrain",
    
    API_KEY=DATA_API_KEY,
    API_SECRET=DATA_API_SECRET,
    API_BASE_URL=DATA_API_BASE_URL,
    erl_params=ERL_PARAMS,
    cwd="./papertrading_erl_retrain",
    break_step=2e10,
)
