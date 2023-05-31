from finrl.config import TRAIN_START_DATE, TRAIN_END_DATE, INDICATORS
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.data_processor import DataProcessor
from cred_config import *

# Create a DataProcessor instance for the Alpaca data source
data_processor = DataProcessor(
    data_source='alpaca',
    API_KEY=API_KEY,
    API_SECRET=API_SECRET,
    API_BASE_URL=API_BASE_URL
)

# Define date range and ticker list for data download
start_date = '2023-01-01'
end_date = '2023-05-01'
ticker_list = DOW_30_TICKER
time_interval = '60Min'

# Download, clean and enhance data with technical indicators and VIX
raw_data = data_processor.download_data(
    start_date=start_date,
    end_date=end_date,
    ticker_list=ticker_list,
    time_interval=time_interval
)
clean_data = data_processor.clean_data(raw_data)
data_with_indicators = data_processor.add_technical_indicator(clean_data, INDICATORS)
enhanced_data = data_processor.add_vix(data_with_indicators)

# Define a filename for the pickle file
filename = "enhanced_dow30_data_with_vix.pkl"

# Save the processed data to a pickle file
enhanced_data.to_pickle(filename)
