import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from datetime import datetime

# Opció: candlestick és technikai indikátorok
HAS_MPLFINANCE = False
HAS_TA = False
try:
    import mplfinance as mpf
    HAS_MPLFINANCE = True
except Exception:
    HAS_MPLFINANCE = False

try:
    import ta  # technical analysis library (RSI, MACD, etc.)
    HAS_TA = True
except Exception:
    HAS_TA = False

# --- Konfiguráció ---
data = "consolidated_coin_data.csv"  # változtasd, ha máshol van
output = "crypto_analysis_outputs"
os.makedirs(output, exist_ok=True)






# --- Segédfüggvények és előfeldolgozás ---

def load_and_prepare(path=data):
    """
    Betölti a CSV-t, datetime típussá alakítja a Date oszlopot,
    és a numerikus oszlopokat konvertálja float-ra.
    Visszatér egy pandas DataFrame-fel.
    """
    df = pd.read_csv(path)
    # Date oszlop feldolgozása
    df['Date'] = pd.to_datetime(df['Date'])
    # Alapvető típusok
    oszlopok = ['Open','High','Low','Close','Volume','Market Cap']
    for c in oszlopok:
        # Ha van százalék vagy egyéb, eltávolítjuk (biztonsági lépés)
        df[c] = pd.to_numeric(df[c], errors='coerce')
    # Rendezés dátum szerint
    df = df.sort_values(['Currency','Date']).reset_index(drop=True)
    #print(df)
    return df

df = load_and_prepare()



#def basic_ohlc_stats(df, currency):
def alap_statisztikak(df, currency):
    """
    Minimum, maximum, átlag, medián az Open, High, Low, Close oszlopokra egy adott coinra.
    Visszaad egy DataFrame-et summary formában.
    """
    copy = df[df['Currency'] == currency].copy()
    oszlopok = ['Open','High','Low','Close']
    statisztika = {}
    for o in oszlopok:
        statisztika[o] = {
            'min': copy[o].min(),
            'max': copy[o].max(),
            'mean': copy[o].mean(),
            'median': copy[o].median(),
            'std': copy[o].std(),
            'var': copy[o].var()
        }

    eredmeny = pd.DataFrame(statisztika).T
    print(eredmeny)
    return eredmeny

alap_statisztikak(df,"tezos")



# --- Heti elérhető átlagos profit (medián áron vásárlunk és maxon adunk el) ---

def weekly_max_minus_median(df, currency):
    """
    Heti bontásban kiszámolja: weekly_max - weekly_median (a te leírásod szerint).
    Ez lehet interpretálható 'átlagos árnál befektetett nyereség/veszteség'.
    """
    copy = df[df['Currency'] == currency].copy()
    copy.set_index('Date', inplace=True)

    weekly = copy.resample('W-MON').agg({'High':'max','Close':'median'}).dropna()
    weekly['max_minus_median'] = weekly['High'] - weekly['Close']
    weekly = weekly.rename(columns={'High':'weekly_max','Close':'weekly_median'})
    print(weekly[['weekly_max','weekly_median','max_minus_median']])
    return weekly[['weekly_max','weekly_median','max_minus_median']]
weekly_max_minus_median(df,"tezos")





def compute_returns(df, freq='D'):
    """
    Számolja a log hozamokat (log returns) napi/het/hónap alapján.
    freq: 'D' napi, 'W' heti, 'M' havi stb.
    Visszaad egy DataFrame-et: Date, Currency, ret_log, ret_simple
    """
    out = []
    for coin, g in df.groupby('Currency'):
        tmp = g[['Date','Close']].set_index('Date').resample(freq).last().dropna()
        # egyszeri hozam: (P_t / P_{t-1} - 1)
        tmp['ret_simple'] = tmp['Close'].pct_change()
        tmp['ret_log'] = np.log(tmp['Close']).diff()
        tmp['Currency'] = coin
        tmp = tmp.dropna()
        out.append(tmp.reset_index())
    print(pd.concat(out, ignore_index=True))
    return pd.concat(out, ignore_index=True)

compute_returns(df, freq='D')




def volatility(returns_series):
    """
    Volatilitás: egyszerűen a hozamok szórása.
    Ha a hozamok napiak, akkor annualizálhatjuk: sigma * sqrt(252).
    Visszaad napi szórást és annualizáltat is.
    """
    daily_std = returns_series.std()
    annualized = daily_std * np.sqrt(252)  # 252 kereskedési nap közelítése
    #print(daily_std, annualized)
    return daily_std, annualized

print("diiwdjiwdjowjdjwdjiowojpdwodwd")
# Először kiszámoljuk a hozamokat
returns_df = compute_returns(df, freq='D')
# Szűrjük az adott coinra, pl. "tezos"
tezos_returns = returns_df[returns_df['Currency'] == 'tezos']
# Átadjuk a log hozamokat a volatility függvénynek
volatility(tezos_returns['ret_log'])
volatility(tezos_returns['ret_simple'])