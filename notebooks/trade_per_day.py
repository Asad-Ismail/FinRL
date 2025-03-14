
import datetime
import threading
from finrl.meta.data_processors.processor_alpaca import AlpacaProcessor
import alpaca_trade_api as tradeapi
import time
import pandas as pd
import numpy as np
import torch
import gym
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.data_processor import DataProcessor
from finrl.config import INDICATORS
from finrl.config import RLlib_PARAMS
from train_high_frequency import ActorPPO,AgentPPO
from finrl.meta.preprocessor.preprocessors import FeatureEngineer,data_split
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.env_stock_trading.env_stocktrading import *
import warnings
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from datetime import timedelta


# Ignore all future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)




class AlpacaPaperTrading():

    def __init__(self,df,ticker_list, time_interval, drl_lib, agent, cwd,
                 stock_dim, action_dim, API_KEY, API_SECRET, 
                 API_BASE_URL, tech_indicator_list, turbulence_thresh=30, 
                 max_stock=1e2, latency = None):
        #load agent
        self.drl_lib = drl_lib
        if agent =='ppo':
            if drl_lib == 'elegantrl':              
                agent_class = AgentPPO
                agent = agent_class(net_dim, state_dim, action_dim)
                actor = agent.act
                # load agent
                try:  
                    cwd = cwd + '/actor.pth'
                    print(f"| load actor from: {cwd}")
                    actor.load_state_dict(torch.load(cwd, map_location=lambda storage, loc: storage))
                    self.act = actor
                    self.device = agent.device
                except BaseException:
                    raise ValueError("Fail to load agent!")
                        
            elif drl_lib == 'rllib':
                from ray.rllib.agents import ppo
                from ray.rllib.agents.ppo.ppo import PPOTrainer
                
                config = ppo.DEFAULT_CONFIG.copy()
                config['env'] = StockEnvEmpty
                config["log_level"] = "WARN"
                config['env_config'] = {'state_dim':state_dim,
                            'action_dim':action_dim,}
                trainer = PPOTrainer(env=StockEnvEmpty, config=config)
                trainer.restore(cwd)
                try:
                    trainer.restore(cwd)
                    self.agent = trainer
                    print("Restoring from checkpoint path", cwd)
                except:
                    raise ValueError('Fail to load agent!')
                    
            elif drl_lib == 'stable_baselines3':
                from stable_baselines3 import PPO
                
                try:
                    #load agent
                    print(f"Using Stable baselines PPO")
                    self.model = PPO.load(cwd)
                    print("Successfully load model", cwd)
                except:
                    raise ValueError('Fail to load agent!')
                    
            else:
                raise ValueError('The DRL library input is NOT supported yet. Please check your input.')
               
        else:
            raise ValueError('Agent input is NOT supported yet.')
            
            
            
        #connect to Alpaca trading API
        try:
            self.alpaca = tradeapi.REST(API_KEY,API_SECRET,API_BASE_URL, 'v2')
        except:
            raise ValueError('Fail to connect Alpaca. Please check account info and internet connection.')
        
        #read trading time interval
        if time_interval == '1s':
            self.time_interval = 1
        elif time_interval == '5s':
            self.time_interval = 5
        elif time_interval == '1Min':
            self.time_interval = 60
        elif time_interval == '5Min':
            self.time_interval = 60 * 5
        elif time_interval == '15Min':
            self.time_interval = 60 * 15
        else:
            raise ValueError('Time interval input is NOT supported yet.')
        
        #read trading settings
        self.tech_indicator_list = tech_indicator_list
        self.turbulence_thresh = turbulence_thresh
        self.max_stock = max_stock 
        
        #initialize account
        self.trade_data=df
        self.stocks = np.asarray([0] * len(ticker_list)) #stocks holding
        self.stocks_cd = np.zeros_like(self.stocks) 
        self.cash = None #cash record 
        self.stocks_df = pd.DataFrame(self.stocks, columns=['stocks'], index = ticker_list)
        self.asset_list = []
        self.price = np.asarray([0] * len(ticker_list))
        self.stockUniverse = ticker_list
        self.turbulence_bool = 0
        self.equities = []
        self.stock_dim = stock_dim
        self.action_space=action_dim
        self.buy_cost_pct=0.01
        self.sell_cost_pct=0.01
        self.reward_scaling=1e-4
        
    def test_latency(self, test_times = 10): 
        total_time = 0
        for i in range(0, test_times):
            time0 = time.time()
            self.get_state()
            time1 = time.time()
            temp_time = time1 - time0
            total_time += temp_time
        latency = total_time/test_times
        print('latency for data processing: ', latency)
        return latency
        
    def run(self):
        orders = self.alpaca.list_orders(status="open")
        for order in orders:
          self.alpaca.cancel_order(order.id)
    
        # Wait for market to open.
        print("Waiting for market to open...")
        tAMO = threading.Thread(target=self.awaitMarketOpen)
        tAMO.start()
        tAMO.join()
        print("Market opened.")
        while True:

          # Figure out when the market will close so we can prepare to sell beforehand.
          clock = self.alpaca.get_clock()
          closingTime = clock.next_close.replace(tzinfo=datetime.timezone.utc).timestamp()
          currTime = clock.timestamp.replace(tzinfo=datetime.timezone.utc).timestamp()
          self.timeToClose = closingTime - currTime
    
          if(self.timeToClose < (60)):
            # Close all positions when 1 minutes til market close.
            print("Market closing soon. Stop trading.")
            break
            
            '''# Close all positions when 1 minutes til market close.
            print("Market closing soon.  Closing positions.")
    
            positions = self.alpaca.list_positions()
            for position in positions:
              if(position.side == 'long'):
                orderSide = 'sell'
              else:
                orderSide = 'buy'
              qty = abs(int(float(position.qty)))
              respSO = []
              tSubmitOrder = threading.Thread(target=self.submitOrder(qty, position.symbol, orderSide, respSO))
              tSubmitOrder.start()
              tSubmitOrder.join()
    
            # Run script again after market close for next trading day.
            print("Sleeping until market close (15 minutes).")
            time.sleep(60 * 15)'''
            
          else:
            trade = threading.Thread(target=self.trade)
            trade.start()
            trade.join()
            last_equity = float(self.alpaca.get_account().last_equity)
            cur_time = time.time()
            self.equities.append([cur_time,last_equity])
            # Break if we are done with trading for today
            break
            time.sleep(self.time_interval)
            
    def awaitMarketOpen(self):
        isOpen = self.alpaca.get_clock().is_open
        while(not isOpen):
          clock = self.alpaca.get_clock()
          openingTime = clock.next_open.replace(tzinfo=datetime.timezone.utc).timestamp()
          currTime = clock.timestamp.replace(tzinfo=datetime.timezone.utc).timestamp()
          timeToOpen = int((openingTime - currTime) / 60)
          print(str(timeToOpen) + " minutes til market open.")
          time.sleep(60)
          isOpen = self.alpaca.get_clock().is_open
    
    def trade(self):
        state = self.get_state()
        
        if self.drl_lib == 'elegantrl':
            with torch.no_grad():
                s_tensor = torch.as_tensor((state,), device=self.device)
                a_tensor = self.act(s_tensor)  
                action = a_tensor.detach().cpu().numpy()[0]  
            action = (action * self.max_stock).astype(int)
            
        elif self.drl_lib == 'rllib':
            action = self.agent.compute_single_action(state)
        
        elif self.drl_lib == 'stable_baselines3':
            action = self.model.predict(state)[0]
            action=action.squeeze(0)
            
        else:
            raise ValueError('The DRL library input is NOT supported yet. Please check your input.')
        
        self.stocks_cd += 1
        print(f"Action shape is {action.shape}")
        print(f"Actions are {action}")
        print(f"Action min and max is {action.min(), action.max()}")
        #exit()
        #return
        if self.turbulence_bool == 0:
            min_action = 0  # stock_cd
            for index in np.where(action < -min_action)[0]:  # sell_index:
                sell_num_shares = min(self.stocks[index], -action[index])
                qty =  abs(int(sell_num_shares))
                respSO = []
                tSubmitOrder = threading.Thread(target=self.submitOrder(qty, self.stockUniverse[index], 'sell', respSO))
                tSubmitOrder.start()
                tSubmitOrder.join()
                self.cash = float(self.alpaca.get_account().cash)
                self.stocks_cd[index] = 0

            for index in np.where(action > min_action)[0]:  # buy_index:
                if self.cash < 0:
                    tmp_cash = 0
                else:
                    tmp_cash = self.cash
                buy_num_shares = min(tmp_cash // self.price[index], abs(int(action[index])))
                print(f"Trying to Buy {abs(int(action[index]))} Available cash is {tmp_cash},The stock price is {self.price[index]} " )
                if (buy_num_shares != buy_num_shares): # if buy_num_change = nan
                    qty = 0 # set to 0 quantity
                else:
                    qty = abs(int(buy_num_shares))
                qty = abs(int(buy_num_shares))
                respSO = []
                tSubmitOrder = threading.Thread(target=self.submitOrder(qty, self.stockUniverse[index], 'buy', respSO))
                tSubmitOrder.start()
                tSubmitOrder.join()
                self.cash = float(self.alpaca.get_account().cash)
                self.stocks_cd[index] = 0
                
        else:  # sell all when turbulence
            positions = self.alpaca.list_positions()
            for position in positions:
                if(position.side == 'long'):
                    orderSide = 'sell'
                else:
                    orderSide = 'buy'
                qty = abs(int(float(position.qty)))
                respSO = []
                tSubmitOrder = threading.Thread(target=self.submitOrder(qty, position.symbol, orderSide, respSO))
                tSubmitOrder.start()
                tSubmitOrder.join()
            
            self.stocks_cd[:] = 0
            
    
    def get_state(self):

        alpaca = AlpacaProcessor(api=self.alpaca)
        price, tech, turbulence = alpaca.fetch_latest_data(ticker_list = self.stockUniverse, time_interval='1Min',
                                                     tech_indicator_list=self.tech_indicator_list)
        turbulence_bool = 1 if turbulence >= self.turbulence_thresh else 0
        
        turbulence = (self.sigmoid_sign(turbulence, self.turbulence_thresh) * 2 ** -5).astype(np.float32)
        
        tech = tech * 2 ** -7
        positions = self.alpaca.list_positions()
        stocks = [0] * len(self.stockUniverse)
        for position in positions:
            if position.symbol in self.stockUniverse:
                ind = self.stockUniverse.index(position.symbol)
                stocks[ind] = ( abs(int(float(position.qty))))
        
        #stocks = np.asarray(stocks, dtype = float)
        cash = float(self.alpaca.get_account().cash)


        self.cash = cash
        self.stocks = stocks
        self.turbulence_bool = turbulence_bool 
        self.price = price

        print(f"The value of cash: {cash}, stocks are: {self.price}, turbulace is {self.turbulence_bool}")
        state_space = 1 + 2*self.stock_dim+ len(self.tech_indicator_list)*self.stock_dim

        trade_env = DummyVecEnv(
            [
                lambda: StockTradingEnv(
                    df=self.trade_data,
                    stock_dim=self.stock_dim,
                    hmax=self.max_stock,
                    initial_amount=self.cash,
                    num_stock_shares=stocks,
                    buy_cost_pct=[self.buy_cost_pct] * self.stock_dim,
                    sell_cost_pct=[self.sell_cost_pct] * self.stock_dim,
                    reward_scaling=self.reward_scaling,
                    state_space=state_space,
                    action_space=self.action_space,
                    tech_indicator_list=self.tech_indicator_list,
                    turbulence_threshold=self.turbulence_thresh,
                    initial=True,
                    previous_state=[],
                    model_name="ppo",
                    mode="trade",
                    print_verbosity=1,
                )
            ]
        )

        trade_obs = trade_env.reset()

        return trade_obs
        
    def submitOrder(self, qty, stock, side, resp):
        if(qty > 0):
          try:
            self.alpaca.submit_order(stock, qty, side, "market", "day")
            print("Market order of | " + str(qty) + " " + stock + " " + side + " | completed.")
            resp.append(True)
          except:
            print("Order of | " + str(qty) + " " + stock + " " + side + " | did not go through.")
            resp.append(False)
        else:
          print("Quantity is 0, order of | " + str(qty) + " " + stock + " " + side + " | not completed.")
          resp.append(True)

    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh

trained_ticks=[
  'AAPL',
  'AMGN',
  'AXP',
  'BA',
  'CAT',
  'CRM',
  'CSCO',
  'CVX',
  'DIS',
  'GS',
  'HD',
  'HON',
  'IBM',
  'INTC',
  'JNJ',
  'JPM',
  'KO',
  'MCD',
  'MMM',
  'MRK',
  'MSFT',
  'NKE',
  'PG',
  'TRV',
  'UNH',
  'VZ',
  'WBA',
  'WMT']

stock_dim = len(trained_ticks)
state_space = 1 + 2*stock_dim + len(INDICATORS)*stock_dim


#from datetime import datetime, timedelta

def get_nth_previous_date(n):
    today = datetime.today()
    nth_previous_date = today - timedelta(days=n)
    return nth_previous_date.strftime('%Y-%m-%d')

API_KEY = "PK3EL71L7LP2AJF2D8C9"
API_SECRET = "He2HPgs8axkITDNSwcIl487JOR7twZcnxdvegCOA"
API_BASE_URL = 'https://paper-api.alpaca.markets'
data_url = 'wss://data.alpaca.markets'

#START_DATE = get_nth_previous_date(4)
START_DATE= '2022-01-01'
# Test date is something unreachable in this lifetime
END_DATE = '2080-02-16'

print(f"Start and end Date is {START_DATE}, {END_DATE}")



df = YahooDownloader(start_date = START_DATE,
                     end_date = END_DATE,
                     ticker_list = trained_ticks).fetch_data()

print(f"Initial DF shape is {df.shape}")

fe = FeatureEngineer(use_technical_indicator=True,
                     tech_indicator_list = INDICATORS,
                     use_turbulence=True,
                     user_defined_feature = False)

processed = fe.preprocess_data(df)
processed = processed.copy()
processed = processed.fillna(0)
processed = processed.replace(np.inf,0)

specific_date=df["date"].unique()[-1]
print(f"Specific date is {specific_date}")
# Data split is necessary since it will label all the stocks with same day with same data frame index
processed = data_split(processed,start=specific_date,end=END_DATE,)
print(f"Final DF is {processed.shape}")
assert processed.shape[0]==len(trained_ticks)

paper_trading_erl = AlpacaPaperTrading(df=processed,
                                      ticker_list = trained_ticks, 
                                      time_interval = '15Min', 
                                      drl_lib = 'stable_baselines3', 
                                      agent = 'ppo', 
                                      cwd = './best_trained_model/PPO_6000k.zip', 
                                      stock_dim = stock_dim, 
                                      action_dim= stock_dim, 
                                      API_KEY = API_KEY, 
                                      API_SECRET = API_SECRET, 
                                      API_BASE_URL = API_BASE_URL, 
                                      tech_indicator_list = INDICATORS, 
                                      turbulence_thresh=30, 
                                      max_stock=1)
print(f"Running Trading!!")
paper_trading_erl.run()