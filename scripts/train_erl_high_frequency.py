from __future__ import annotations
from finrl.config_tickers import DOW_30_TICKER
from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.meta.env_stock_trading.env_stock_papertrading import AlpacaPaperTrading
from finrl.meta.data_processor import DataProcessor
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

import numpy as np
import pandas as pd
import os
import time
import gym
import numpy as np
import numpy.random as rd
import torch
import torch.nn as nn
from copy import deepcopy
from torch import Tensor
from torch.distributions.normal import Normal

from finrl.config import ERL_PARAMS
from finrl.config import INDICATORS
from finrl.config import RLlib_PARAMS
from finrl.config import SAC_PARAMS
from finrl.config import TRAIN_END_DATE
from finrl.config import TRAIN_START_DATE
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.data_processor import DataProcessor

from finrl.config import INDICATORS
from finrl.config import RLlib_PARAMS
from finrl.config import TEST_END_DATE
from finrl.config import TEST_START_DATE
from finrl.config_tickers import DOW_30_TICKER

import torch

from helloworld.config import Config
from helloworld.run import train_agent, Evaluator
from helloworld.config import get_gym_env_args
from helloworld.agent import AgentPPO
agent_cls = AgentPPO

OFF_POLICY_MODELS = ["ddpg", "td3", "sac"]
ON_POLICY_MODELS = ["ppo"]
# MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}
#
# NOISE = {
#     "normal": NormalActionNoise,
#     "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
# }


class DRLAgent:
    """Implementations of DRL algorithms
    Attributes
    ----------
        env: gym environment class
            user-defined class
    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(self,env, price_array, tech_array, turbulence_array):
        self.env = env
        self.price_array = price_array
        self.tech_array = tech_array
        self.turbulence_array = turbulence_array

    def get_model(self, model_name, model_kwargs):
        env_config = {
            "price_array": self.price_array,
            "tech_array": self.tech_array,
            "turbulence_array": self.turbulence_array,
            "if_train": True,
        }
        environment = self.env(config=env_config)
        env_args = {'config': env_config,
              'env_name': environment.env_name,
              'state_dim': environment.state_dim,
              'action_dim': environment.action_dim,
              'if_discrete': False}
        model = Config(agent_class=agent_cls, env_class=self.env, env_args=env_args)
        model.if_off_policy = model_name in OFF_POLICY_MODELS
        if model_kwargs is not None:
            try:
                model.learning_rate = model_kwargs["learning_rate"]
                model.batch_size = model_kwargs["batch_size"]
                model.gamma = model_kwargs["gamma"]
                model.seed = model_kwargs["seed"]
                model.net_dims = model_kwargs["net_dimension"]
                model.target_step = model_kwargs["target_step"]
                model.eval_gap = model_kwargs["eval_gap"]
                model.eval_times = model_kwargs["eval_times"]
            except BaseException:
                raise ValueError("Fail to read arguments, please check 'model_kwargs' input.")
        return model

    def train_model(self, model, cwd, total_timesteps=5000):
        model.cwd = cwd
        model.break_step = total_timesteps
        train_agent(model)



def train(
    start_date,
    end_date,
    ticker_list,
    data_source,
    time_interval,
    technical_indicator_list,
    drl_lib,
    env,
    model_name,
    if_vix=True,
    use_preprocess=False,
    **kwargs,
):
    # download data
    dp = DataProcessor(data_source, **kwargs)
    ## Read preprocessed data from pickle file
    if use_preprocess:
        dp.processor.start=start_date
        dp.processor.end=end_date
        dp.processor.time_interval=time_interval
        dp.tech_indicator_list=technical_indicator_list
        data = pd.read_pickle("clean_data_vix.pkl")
        print(data.shape)
        # keep only 200 frames
        #data=data.head(1000)
        print(f"Loaded preprocessed data with shape {data.shape}")
    else:
        data = dp.download_data(ticker_list, start_date, end_date, time_interval)
        data = dp.clean_data(data)
        data = dp.add_technical_indicator(data, technical_indicator_list)
        if if_vix:
            print(f"Adding Vix Indicator to Data {data.shape}")
            data = dp.add_vix(data)
        else:
            print(f"Adding Turbulance Indicator to Data {data.shape}")
            data = dp.add_turbulence(data)
    print(f"Getting Numpy arrays from data!")
    price_array, tech_array, turbulence_array = dp.df_to_array(data, if_vix)
    env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "if_train": True,
    }
    env_instance = env(config=env_config)

    # read parameters
    cwd = kwargs.get("cwd", "./" + str(model_name))

    if drl_lib == "elegantrl":
        DRLAgent_erl = DRLAgent
        break_step = kwargs.get("break_step", 1e6)
        erl_params = kwargs.get("erl_params")
        agent = DRLAgent_erl(
            env=env,
            price_array=price_array,
            tech_array=tech_array,
            turbulence_array=turbulence_array,
        )
        model = agent.get_model(model_name, model_kwargs=erl_params)
        trained_model = agent.train_model(model=model, cwd=cwd, total_timesteps=break_step)



ticker_list = DOW_30_TICKER
action_dim = len(DOW_30_TICKER)

state_dim = 1 + 2 + 3 * action_dim + len(INDICATORS) * action_dim


ERL_PARAMS = {"learning_rate": 3e-6,"batch_size": 2048,"gamma":  0.985,
        "seed":312,"net_dimension":[128,64], "target_step":5000, "eval_gap":5e7,
        "eval_times":500} 

env = StockTradingEnv

if __name__== "__main__":
    print(f"All Imported successfully!!")
    train(start_date = '2022-12-01', 
    end_date = '2023-05-29',
    ticker_list = ticker_list, 
    data_source = 'alpaca',
    time_interval= '15Min', 
    technical_indicator_list= INDICATORS,
    drl_lib='elegantrl', 
    env=env, 
    model_name='ppo',
    if_vix=True, 
    API_KEY = API_KEY, 
    API_SECRET = API_SECRET, 
    API_BASE_URL = API_BASE_URL,
    erl_params=ERL_PARAMS,
    cwd='./erl_trading',
    use_preprocess=True,
    break_step=1e15)