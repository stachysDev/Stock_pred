import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

st.set_page_config(
    page_title="Sklearn forecast",
    layout="wide"
)

st.title("My Stock app")


@st.cache_data
def get_stock_data(ticker, start_at, end_at, time_interval):
    try:
        data = yf.download(ticker, start=str(start_at), end=str(end_at), interval=str(time_interval))
        data.reset_index(inplace=True)
        data.reset_index(inplace=True)
        return data
    except:
        st.write("An exception occurred")


def show_close_open_graph(df):
    list_cols = list(df.columns)
    if "Date" in list_cols:
        fig = px.line(
            df, x='Date',
            y=[list_cols[2], list_cols[5]],
            hover_data={"Date": "|%B %d, %Y"},
            title='Open and Close graphs')
        fig.update_xaxes()
        st.plotly_chart(fig)


    elif "Datetime" in list_cols:
        fig = px.line(
            df, x='Datetime',
            y=[list_cols[2], list_cols[5]],
            hover_data={"Datetime"},
            title='Open and Close graphs')
        fig.update_xaxes()
        st.plotly_chart(fig)

def mape(y_true, y_pred):
    ape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return ape

def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


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

# get stock data
stock_data = get_stock_data(stock_selected, start_date, end_date, time_interval_selected)
stock_data_columns = list(stock_data.columns)

# see frequency here https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
time_frequencies_min = ["1m", "2m", "5m", "15m", "30m", "60m", "90m"]
time_frequencies_hour = ["1h", "2h"]
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

elif time_interval_selected in time_frequencies_hour:
    forecast_freq = "H"
    season_length = 24
    substract_days = 1

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

if len(stock_data != 0):
    st.subheader("Tail data for " + stock_selected)

    st.write(stock_data.tail())
    st.write("Taille des données: ", len(stock_data))

    show_close_open_graph(stock_data)

    stock_data_cols = list(stock_data.columns)
    len_stock_data = len(stock_data)

    some_days_ago = pd.Timestamp('2023-04-04 22:00:00+0100', tz='Europe/London')
    some_days_ago_2 = pd.Timestamp('2023-04-04', tz='Europe/London')
    st.write("Some days ago ", some_days_ago)

    Y_train_df = pd.DataFrame()
    Y_test_df = pd.DataFrame()

    st.write("Some days ago", type(some_days_ago), some_days_ago)
    df_train = pd.DataFrame()
    st.write("Frequency is ", forecast_freq)



    if "Date" in stock_data_cols:
        stock_data = stock_data.rename(columns={"Date": "ds", "Close": "y", "index": "unique_id"})
        last_date = stock_data.ds[len_stock_data - 1]
        some_days_ago_2 = last_date - timedelta(substract_days)
        st.write("Last date is ", type(last_date), last_date)

        Y_train_df = stock_data[stock_data.ds <= some_days_ago_2].copy()
        Y_test_df = stock_data[stock_data.ds > some_days_ago_2].copy()

        #
        Y_train_df['next_y'] = Y_train_df['y'].shift(-1)
        Y_test_df['next_y'] = Y_test_df['y'].shift(-1)
        Y_train_df.dropna()

        st.subheader("Date in df")
        st.write(Y_train_df.tail())
        # period_ind = pd.PeriodIndex(Y_test_df.Date).astype('period[D]')

        y_true = Y_train_df['next_y']
        y_pred = Y_train_df['y']

        st.write("mape ", mape(y_true, y_pred))
        st.write("wmape ", wmape(y_true, y_pred))

        m = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=2)
        m.fit(Y_test_df['ds'], Y_train_df['y'])

        st.write(m)






    elif "Datetime" in stock_data_cols:
        stock_data = stock_data.rename(columns={"Datetime": "ds", "Close": "y", "index": "unique_id"})
        last_date = stock_data.ds[len_stock_data - 1]
        st.write("Last date is ", last_date, "And its type is", type(last_date))
        st.subheader(last_date > some_days_ago)
        some_days_ago = last_date - timedelta(substract_days)

        Y_train_df = stock_data[stock_data.ds <= some_days_ago].copy()
        Y_test_df = stock_data[stock_data.ds > some_days_ago].copy()

        st.subheader("Datetime in df")
        st.write(Y_train_df.tail())



    Y_train_len = len(Y_train_df)
    Y_test_len = len(Y_test_df)

    st.write("TRAINING data length is " + str(Y_train_len) + " and TEST data length is " + str(Y_test_len))



