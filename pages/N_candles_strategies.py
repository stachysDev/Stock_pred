import pandas as pd
import streamlit as st
from datetime import date
import my_functions as myf

st.set_page_config(
    page_title="Stratégies des N bougies",
    layout="wide"
)
st.title("Stratégies des 4 dernières bougies")


def get_symbols_group_infos(symbol_group_selected, start_date, end_date, time_interval_selected):
    symbol_group = ""
    if symbol_group_selected == "Cryptos":
        symbol_group = myf.cryptos
    elif symbol_group_selected == "Forex_Minors":
        symbol_group = myf.forex_minors
    elif symbol_group_selected == "Forex_Majors":
        symbol_group = myf.forex_majors

    if symbol_group != "":
        for ticker in symbol_group:
            txt = ticker + " - " + time_interval_selected
            st.subheader(txt)
            data = myf.get_stock_data(ticker, start_date, end_date, time_interval_selected)
            if len(data) != 0:
                st.write(data.tail(3))
                t = list(data.tail(3))
                if t[2] > t[1]:
                    st.subheader("True")
                else:
                    st.subheader("False")
            else:
                st.write("No data found")




cryptos = myf.cryptos
forex_minors = myf.forex_minors
forex_majors = myf.forex_majors
time_intervals = myf.time_intervals

data = pd.DataFrame()

symbols = [
    "Cryptos", "Forex_Minors", "Forex_Majors",
]

symbols_col, start_date_col, end_date_col, time_interval_col = st.columns(4)

with symbols_col:
    symbol_selected = st.selectbox(
        "Choisir un groupe de monnaies",
        symbols
    )

with start_date_col:
    start_date = st.date_input("Choose start date")

with end_date_col:
    end_date = st.date_input(
        "Date de début",
        date.today()
    )

with time_interval_col:
    time_interval_selected = st.selectbox(
        "Choisir un intervalle de temps",
        time_intervals
    )

get_symbols_group_infos(symbol_selected, start_date, end_date, time_interval_selected)

