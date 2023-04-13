import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta, date
import numpy as np
import pandas as pd
from IPython.display import display, Markdown
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, Naive
from statsforecast.utils import generate_series
from datasetsforecast.losses import rmse

st.set_page_config(
    page_title="Stock app",
    layout="wide"
)


def my_crossvalidation_forecast_function(df, models, freq, len_of_df_test, step_size, n_windows):
    # n_windows : number of forecasting processes in the past you want to explore
    n_jobs = -1  # n_jobs: int, number of jobs used in the parallel processing, use - 1 for all cores.
    sf = StatsForecast(
        df=df,
        models=models,
        freq=freq,
        n_jobs=n_jobs
    )
    # cross validation
    # step_size : How often do you want run the forecast process (read doc crossvalidation)
    crossvalidation_df = sf.cross_validation(
        df=df,
        h=len_of_df_test,
        step_size=step_size,
        n_windows=n_windows
    )
    # print crossvalidation_df tail
    # st.write(crossvalidation_df.tail())
    crossvalidation_df.rename(columns={'y': 'actual'}, inplace=True)  # rename actual values

    return crossvalidation_df


def caluculate_forecast_accuracy(crossvalidation_df):
    forecast_rmse = rmse(crossvalidation_df['actual'], crossvalidation_df['AutoETS'])
    return forecast_rmse


st.title("My Stock app")

stock_col, start_col, end_date_col, interval_col = st.columns(4)

with stock_col:
    stock_selected = st.selectbox(
        "Choose stock",
        ("EURUSD=X", "GBPUSD=X", "JPY=X", "CHF=X", "AUDUSD=X", "CAD=X", "NZDUSD=X",
         "EURGBP=X", "EURCHF=X", "EURCAD=X", "EURAUD=X", "EURNZD=X",
         "EURJPY=X", "GBPJPY=X", "CHFJPY=X", "CADJPY=X", "AUDJPY=X",
         "NZDJPY=X", "GBPCHF=X", "GBPAUD=X", "GBPCAD=X")
    )
    st.write("You selected: ", stock_selected)

with start_col:
    start_date = st.date_input("Choose start date")
    st.write("Start date is: ", start_date)

with end_date_col:
    end_date = st.date_input("Choose end date")
    st.write("End date is: ", end_date)

with interval_col:
    time_interval_selected = st.selectbox(
        "Choose interval",
        ("1h", "1m", "2m", "5m", "15m", "30m", "60m",
         "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo")
    )
    st.write("You selected: ", time_interval_selected)


@st.cache_data
def get_stock_data(ticker, start_at, end_at, time_interval):
    try:
        data = yf.download(ticker, start=str(start_at), end=str(end_at), interval=str(time_interval))
        data.reset_index(inplace=True)
        data.reset_index(inplace=True)
        return data
    except:
        st.write("An exeption occured")


stock_data = get_stock_data(stock_selected, start_date, end_date, time_interval_selected)
st.subheader("Tail data for " + stock_selected)
st.write("Taille des données: ", len(stock_data))
st.write(stock_data.tail())

stock_data_columns = list(stock_data.columns)
# st.write(stock_data_columns, type(stock_data_columns))

if "Date" in stock_data_columns:
    # st.write("Date is in columns")
    df_train = stock_data[['Date', 'Close', "index"]]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y", "index": "unique_id"})
    # we want to select a part of dataframe for test and train

else:
    # st.write("Date is not in columns")
    df_train = stock_data[['Datetime', 'Close', "index"]]
    df_train = df_train.rename(columns={"Datetime": "ds", "Close": "y", "index": "unique_id"})


# see frequency here https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
time_frequencies_min = ["1m", "2m", "5m", "15m", "30m", "60m", "90m"]
time_frequencies_hour = "1h"
time_frequencies_day = ["1d", "5d"]
time_frequencies_week = "1wk"
time_frequencies_month = ["1mo", "3mo"]

forecast_freq = ""
season_length = 0
substract_days = 0

if time_interval_selected in time_frequencies_min:
    forecast_freq = "min"
    season_length = 60
    substract_days = 1

elif time_interval_selected == "1h":
    forecast_freq = "H"
    season_length = 24
    substract_days = 2

elif time_interval_selected in time_frequencies_day:
    forecast_freq = "D"
    season_length = 30
    substract_days = 20

elif time_interval_selected == "1wk":
    forecast_freq = "W"
    season_length = 56
    substract_days = 30

elif time_interval_selected in time_frequencies_month:
    forecast_freq = "M"
    season_length = 12
    substract_days = 24

some_date = end_date - timedelta(substract_days)
st.write("Training data for date before " + str(some_date) + " and use beyond dates for test")

Y_train_df = df_train[df_train.ds <= str(some_date)]
Y_test_df = df_train[df_train.ds > str(some_date)]
st.write("TRAINING data length is " + str(len(Y_train_df)) + " and TEST data length is " + str(len(Y_test_df)))
st.write("Frequence is ", forecast_freq)
st.write("Season lenght is ", season_length)

#forecasting


models = [
    AutoARIMA(season_length=12),
    AutoETS(season_length=12),
    Naive()
]
# Instansiate the StatsForecast class as sf
sf = StatsForecast(
    df=Y_train_df,
    models=models,
    freq= forecast_freq,
    n_jobs=-1
)
# we forecast for the test period to compare test df values with forecast df values
horizon = len(Y_test_df)
Y_hat_df = sf.forecast(horizon)

st.subheader("Forecast data tail")
st.write(Y_hat_df.tail())


