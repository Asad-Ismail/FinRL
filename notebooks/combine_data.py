from finrl.config import TRAIN_END_DATE
from finrl.config import TRAIN_START_DATE
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.data_processor import DataProcessor
from finrl.config import INDICATORS
import pandas as pd

API_KEY = "PKQ6Y2WHFM3CZWU88SVS"
API_SECRET = "DRjZl2fpiqfmSrDHj7hejIpU94FQXgUWUdxccl0d"
API_BASE_URL = 'https://paper-api.alpaca.markets'
data_url = 'wss://data.alpaca.markets'

dp = DataProcessor(data_source = 'alpaca',
                  API_KEY = API_KEY, 
                  API_SECRET = API_SECRET, 
                  API_BASE_URL = API_BASE_URL
                  )

start_date = '2015-01-01'
end_date = '2023-02-16'
ticker_list = DOW_30_TICKER

dp.processor.start=start_date
dp.processor.end=end_date
data = pd.read_pickle("my_data.pkl")

data = dp.clean_data(data)

data = dp.add_technical_indicator(data, INDICATORS)

data.to_pickle("clean_data.pkl")
