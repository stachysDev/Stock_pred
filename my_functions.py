import streamlit as st
import yfinance as yf

cryptos = [
    "BNB-USD", "BTC-USD", "DSH-USD", "EOS-USD",
    "ETH-USD", "LTC-USD", "XMR-USD", "XRP-USD",
    "ZEC-USD", "IOT-USD", "NEO-USD", "OMG-USD",
    "TRX-USD", "XLM-USD", "BTC-ETH", "BTC-LTC",
    "ADA-USD", "ALG-USD", "BAT-USD", "DOGE-USD",
    "BCH-USD", "DOT-USD", "VEC-USD", "ETC-USD",
    "FIL-USD", "LNK-USD", "MKR-USD", "MTC-USD",
    "SOL-USD", "UNI-USD", "XTZ-USD", "AVA-USD",
    "AAV-USD", "APE-USD", "ATM-USD", "ENJ-USD",
    "FTT-USD", "GMT-USD", "KNC-USD", "LPT-USD",
    "LRC-USD", "MAN-USD", "SHB-USD"
]

# symbols must be verified
# add -X to minors and majors
forex_minors = [
    "AUDCHF=X",
    "AUDNZD=X",
    "CADCHF=X",
    "CADJPY=X",
    "CHFJPY=X",
    "EURNOK=X",
    "EURNZD=X",
    "EURPLN=X",
    "EURSEK=X",
    "GBPCAD=X",
    "GBPCHF=X",
    "AUDCAD=X",
    "GBPNOK=X",
    "USDZAR=X",
    "GBPNZD=X",
    "GBPSEK=X",
    "NZDCAD=X",
    "NZDJPY=X",
    "NZDUSD=X",
    "CNH=X",
    "MXN=X",
    "NOK=X",
    "PLN=X",
    "SEK=X"
]

forex_majors = [
    "AUDUSD=X",
    "AURAUD=X",
    "EURCAD=X",
    "EURCHF=X",
    "EURGBP=X",
    "EURJPY=X",
    "AUDJPY=X",
    "EURUSD=X",
    "USDJPY=X",
    "GBPAUD=X",
    "GBPJPY=X",
    "GBPUSD=X",
    "USDCAD=X",
    "USDCHF=X"
]

time_intervals = [
    "1h", "1m", "2m", "5m", "15m", "30m", "60m",
    "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"
]


@st.cache_data
def get_stock_data(ticker, start_at, end_at, time_interval):
    try:
        data = yf.download(ticker, start=str(start_at), end=str(end_at), interval=str(time_interval))
        data.reset_index(inplace=True)
        data.reset_index(inplace=True)
        return data
    except:
        st.write("An exception occurred")


