import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import numpy as np

import glob  # Fájlok kereséséhez
import os  # Fájlnevek kezeléséhez

plt.style.use('ggplot')
# seaborn stílus
sns.set(style="whitegrid")

# --- MÓDOSÍTOTT ADATBEOLVASÁS (MEME COIN ADATKÉSZLET) ---
print("Adatbeolvasás és előkészítés (Meme Coin adatkészlet, több CSV)...")
try:
    # 1. Keressük meg az ÖSSZES CSV fájlt a mappában
    all_files = glob.glob("*.csv")

    if not all_files:
        print("BAAAJ VAAAN - Nem találhatók CSV fájlok a mappában.")
        exit()

    print(f"Talált {len(all_files)} db CSV fájl. Szűrés és beolvasás...")

    li = []  # Ebben a listában gyűjtjük a DataFrame-eket

    # Meghatározzuk az elvárt oszlopokat
    required_cols = {'Date', 'Open', 'High', 'Low', 'Close'}

    # 2. Ciklus az összes megtalált fájlon
    for filename in all_files:
        try:
            # Először ellenőrizzük a fejlécet (első 0 sort)
            header_df = pd.read_csv(filename, nrows=0)

            # 3. SZŰRÉS: Csak azokat olvassuk be, amik adat CSV-nek tűnnek
            # (Megvan bennük az összes elvárt oszlop)
            if required_cols.issubset(header_df.columns):
                print(f"Beolvasás: {filename}")
                temp_df = pd.read_csv(filename)

                # 4. LÉTREHOZZUK A 'Currency' OSZLOPOT A FÁJLNÉVBŐL
                # "Dogecoin.csv" -> "Dogecoin" -> "dogecoin"
                # "Akita Inu.csv" -> "Akita Inu" -> "akita inu"
                currency_name = os.path.splitext(os.path.basename(filename))[0]
                temp_df['Currency'] = currency_name.lower()

                li.append(temp_df)
            else:
                # Csendesen kihagyjuk azokat a CSV-ket, amik nem adatok
                # (pl. ha valaki Excelben megnyitja és menti a python szkriptet)
                # print(f"Kihagyva: {filename} (nem tűnik adatfájlnak)")
                pass
        except Exception as e:
            # Hiba egy adott fájl olvasása közben, de a többit megpróbáljuk
            print(f"Hiba a(z) {filename} fájl olvasása közben (kihagyás): {e}")

    if not li:
        print("Hiba: Nem sikerült egyetlen érvényes adat CSV-t sem beolvasni.")
        print("Ellenőrizd, hogy a CSV-k tartalmazzák-e: Date, Open, High, Low, Close")
        exit()

    # összefűzés
    df = pd.concat(li, axis=0, ignore_index=True)

    print("Az összes valuta adat sikeresen összefűzve egy DataFrame-be.")

    # --- INNENTŐL A V2-ES KÓDOD VÁLTOZATLANUL KÖVETKEZIK ---

    # 5. OSZLOPNÉV JAVÍTÁSA (ha 'Marketcap' lenne 'Market Cap' helyett)
    # A te kódod ezt már kezeli. Ha 'Market Cap' van (ahogy ebben a datasetben),
    # az 'if' hamis lesz, és minden megy tovább helyesen.
    if 'Marketcap' in df.columns:
        df = df.rename(columns={'Marketcap': 'Market Cap'})

    # Oszlopok, amiket numerikussá kell tenni
    oszlopok = ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']

    # Oszlopok numerikussá tétele
    print("Oszlopok numerikussá alakítása...")
    for oszlop in oszlopok:
        if oszlop in df.columns and df[oszlop].dtype == 'object':
            df[oszlop] = df[oszlop].str.replace(',', '', regex=False)
            df[oszlop] = pd.to_numeric(df[oszlop], errors='coerce')
        elif oszlop not in df.columns:
            print(f"Figyelem: A '{oszlop}' oszlop hiányzik, kihagyás...")

    # Hiányzó adatok (NaN) kezelése
    df['Volume'] = df['Volume'].fillna(0)
    df['Market Cap'] = df['Market Cap'].fillna(0)
    arak = ['Open', 'High', 'Low', 'Close']
    df[arak] = df[arak].fillna(method='ffill')

    # "Date" -> datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # sor azonositoja a date
    df = df.set_index('Date')

    # datum szerint novekvo sorrend
    df = df.sort_index()

    # Tisztítás a ffill után (ha az első sorok NaN-ok maradtak)
    df = df.dropna(subset=arak)

    print("Adatbeolvasás és -tisztítás sikeres.")
    # ------- Adatbeolvasas vege ---------

except FileNotFoundError:
    print("AAAAAAAAAAAAAAAAAAAAAAAA, BAJ VAN (Fájl nem található)")
    exit()  # Kilépés a szkriptből, ha nincs adat
except Exception as e:
    print(f"AAAAAAAAAAAAAAAAAAAAAAAA, BAJ VAN AZ ADATBEOLVASÁS SORÁN: {e}")
    exit()

# --- alap leirostat. ---
print("df.info kezdete \n")
df.info()
print("df.info vege \n")

print("df.describe eleje \n")
print(df.describe())
print("df.describe vege \n")

egyedi = df['Currency'].nunique()
print(f"Az adatkészlet {egyedi} kriptovalutát tartalmaz.")

# elemezni kívánt coin kiválasztása
coin_to_analyze = 'dogecoin'

print(f"\n Egyedi Valuta Elemzése: {coin_to_analyze.capitalize()}")

# Al-DataFrame létrehozása
df_coin = df[df['Currency'] == coin_to_analyze].copy()

if df_coin.empty:
    print(f"Nincs '{coin_to_analyze}' nevű Coin a datasetben (ellenőrizd a kis- és nagybetűket, szóközöket)")
else:
    # Árfolyam (Close) alakulása vonaldiagrammal
    plt.figure(figsize=(14, 7))
    df_coin['Close'].plot()
    plt.title(f'{coin_to_analyze.capitalize()} Árfolyam (Close) Alakulása')
    plt.xlabel('Dátum')
    plt.ylabel('Ár (USD)')
    plt.tight_layout()

    # Kereskedési volumen (kereskedesi forgalom) alakulása
    plt.figure(figsize=(14, 7))
    df_coin['Volume'].plot(color='blue')
    plt.title(f'{coin_to_analyze.capitalize()} Kereskedési Volumen Alakulása')
    plt.xlabel('Dátum')
    plt.ylabel('Volumen (USD)')
    plt.tight_layout()

    # Mozgóátlagok - Moving Averages (MA)
    # 50 napos trend
    df_coin['MA50'] = df_coin['Close'].rolling(window=50).mean()
    # 200 napos trend
    df_coin['MA200'] = df_coin['Close'].rolling(window=200).mean()

    plt.figure(figsize=(14, 7))
    plt.plot(df_coin.index, df_coin['Close'], label='Záróár (Close)', alpha=0.7)
    plt.plot(df_coin.index, df_coin['MA50'], label='50 napos mozgóátlag (MA50)', color='orange')
    plt.plot(df_coin.index, df_coin['MA200'], label='200 napos mozgóátlag (MA200)', color='red')
    plt.title(f'{coin_to_analyze.capitalize()} Árfolyam Mozgóátlagokkal')
    plt.xlabel('Dátum')
    plt.ylabel('Ár (USD)')
    plt.legend()
    plt.tight_layout()

    # A napi hozam (záróár változása százalékban)
    df_coin['Daily_Return'] = df_coin['Close'].pct_change()
    print("A napi hozamok átlaga:", df_coin['Daily_Return'].mean())
    print("A napi hozamok mediánja:", df_coin['Daily_Return'].median())
    print("\n")

    # napi hozamok eloszlása
    plt.figure(figsize=(10, 6))
    sns.histplot(df_coin['Daily_Return'].dropna(), bins=100, kde=True,
                 color='green')
    plt.title(f'{coin_to_analyze.capitalize()} Napi Hozamok Eloszlása')
    plt.xlabel('Napi Hozam (%)')
    plt.ylabel('Gyakoriság')
    plt.tight_layout()

    # Volatilitás (szórás)
    daily_volatility = df_coin['Daily_Return'].std()
    print(f"Napi volatilitás (szórás): {daily_volatility:.4f}")
    annualized_volatility = daily_volatility * np.sqrt(365)
    print(f"Évesített volatilitás: {annualized_volatility:.4f}")

    # Gyertyás diagram (MŰKÖDIK EZZEL A DATASETTEL!)
    df_coin_100 = df_coin.tail(100)

    fig = go.Figure(data=[go.Candlestick(x=df_coin_100.index,
                                         open=df_coin_100['Open'],
                                         high=df_coin_100['High'],
                                         low=df_coin_100['Low'],
                                         close=df_coin_100['Close'])])
    fig.update_layout(
        title=f'{coin_to_analyze.capitalize()} Gyertya Diagram (Utolsó 100 nap)',
        yaxis_title='Ár (USD)',
        xaxis_rangeslider_visible=False
    )
    # fig.show() # Kikommentelve, hogy ne nyissa meg futáskor, de a plt.show() a végén működik

    # <---- IDŐ ALAPÚ ELEMZÉSEK ----->

    df_coin['Day_of_Week'] = df_coin.index.dayofweek
    df_coin['Month'] = df_coin.index.month
    napok = ['Hétfő', 'Kedd', 'Szerda', 'Csütörtök', 'Péntek', 'Szombat', 'Vasárnap']
    honapok = ['Jan', 'Feb', 'Már', 'Ápr', 'Máj', 'Jún', 'Júl', 'Aug', 'Szep', 'Okt', 'Nov', 'Dec']

    # átlagos hozam a hét napjai szerint
    plt.figure(figsize=(10, 6))
    avg_return_day = df_coin.groupby('Day_of_Week')['Daily_Return'].mean()
    avg_return_day.index = avg_return_day.index.map(lambda x: napok[x])
    avg_return_day = avg_return_day.reindex(napok)

    avg_return_day.plot(kind='bar', color='purple')
    plt.title(f'{coin_to_analyze.capitalize()} Átlagos Napi Hozam a Hét Napjai Szerint')
    plt.xlabel('Hét Napja')
    plt.ylabel('Átlagos Napi Hozam')
    plt.xticks(rotation=0)
    plt.tight_layout()

    # átlagos volumen a hét napjai szerint
    plt.figure(figsize=(10, 6))
    avg_volume_day = df_coin.groupby('Day_of_Week')['Volume'].mean()
    avg_volume_day.index = avg_volume_day.index.map(lambda x: napok[x])
    avg_volume_day = avg_volume_day.reindex(napok)

    avg_volume_day.plot(kind='bar', color='orange')
    plt.title(f'{coin_to_analyze.capitalize()} Átlagos Kereskedési Volumen a Hét Napjai Szerint')
    plt.xlabel('Hét Napja')
    plt.ylabel('Átlagos Volumen (USD)')
    plt.xticks(rotation=0)
    plt.tight_layout()

    # átlagos hozam a hónapok szerint
    plt.figure(figsize=(10, 6))
    avg_return_month = df_coin.groupby('Month')['Daily_Return'].mean()
    avg_return_month.index = avg_return_month.index.map(
        lambda x: honapok[x - 1])
    avg_return_month = avg_return_month.reindex(honapok)

    avg_return_month.plot(kind='bar', color='cyan')
    plt.title(f'{coin_to_analyze.capitalize()} Átlagos Napi Hozam a Hónapok Szerint')
    plt.xlabel('Hónap')
    plt.ylabel('Átlagos Napi Hozam')
    plt.xticks(rotation=0)
    plt.tight_layout()

    # napi ártartomány (MŰKÖDIK EZZEL A DATASETTEL!)
    df_coin['Daily_Range'] = df_coin['High'] - df_coin['Low']
    df_coin['Daily_Range_Pct'] = (df_coin['Daily_Range'] / df_coin['Close']) * 100

    print("Átlagos napi ártartomány (USD):", df_coin['Daily_Range'].mean())
    print("Átlagos napi ártartomány (Záróár %-a):", df_coin['Daily_Range_Pct'].mean())
    print("\n")

    # napi ártartomány eloszlása (MŰKÖDIK EZZEL A DATASETTEL!)
    plt.figure(figsize=(14, 7))
    sns.boxplot(x='Day_of_Week', y='Daily_Range_Pct', data=df_coin)
    plt.title(f'{coin_to_analyze.capitalize()} Napi Ártartomány (Intraday Volatilitás , High - Low) Eloszlása')
    plt.xticks(ticks=range(7), labels=napok)
    plt.xlabel('Hét Napja')
    plt.ylabel('Napi Ártartomány (a Záróár %-ában)')
    plt.tight_layout()

# <--- TÖBB VALUTA ÖSSZEHASONÉÍTÁSA --->

# top 10 valuta a datasetben
latest_date = df.index.max()
df_latest = df.loc[latest_date]

if not df_latest.empty:
    if isinstance(df_latest, pd.Series):
        df_latest = df_latest.to_frame().T
        df_latest['Market Cap'] = pd.to_numeric(df_latest['Market Cap'], errors='coerce')

    if isinstance(df_latest, pd.DataFrame):
        top_10_market_cap = df_latest.nlargest(10, 'Market Cap')
    else:
        top_10_market_cap = df.nlargest(10, 'Market Cap')

    top_10_coins = top_10_market_cap['Currency'].tolist()
    top_10_coins = [coin for coin in top_10_coins if pd.notna(coin)]

    print(f"Top 10 valuta: {top_10_coins}")
    print("\n")

    df_top10 = df[df['Currency'].isin(top_10_coins)]

    if df_top10.empty:
        print("BAJ VAN A TOP 10el - Nem található adat a Top 10 valutához.")
    else:
        # Korrelációs mátrix -> close alapján
        df_pivot = df_top10.pivot_table(index='Date', columns='Currency', values='Close')

        df_pivot_filled = df_pivot.fillna(method='ffill').fillna(method='bfill')
        df_returns = df_pivot_filled.pct_change()
        correlation_matrix = df_returns.corr()

        # HEATMAP
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Top 10 Valuta Napi Hozamainak Korrelációja')
        plt.tight_layout()

        # Korrelációs mátrix printelése
        print("Korrelációs mátrix:\n")
        print("1 = tökéletes együttmozgás, -1 = tökéletes ellentétes mozgás, 0 = nincs összefüggés")
        print(correlation_matrix)

        # Kördiagramm - Market Cap eloszlás
        plt.figure(figsize=(10, 8))
        plt.pie(top_10_market_cap['Market Cap'], labels=top_10_market_cap['Currency'], autopct='%1.1f%%', startangle=90)
        plt.title('Top 10 Valuta Piaci Kapitalizációjának Eloszlása (Legutolsó Adat)')
        plt.axis('equal')
        plt.tight_layout()

        # Napi volatilitás összehasonlítása (Top 10)
        volatility_comparison = df_returns.std().sort_values(ascending=False)

        plt.figure(figsize=(12, 6))
        sns.barplot(x=volatility_comparison.index, y=volatility_comparison.values)
        plt.title('Napi Volatilitás (Szórás) Összehasonlítása (Top 10)')
        plt.xlabel('Valuta')
        plt.ylabel('Napi Volatilitás (Szórás)')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Normalizált árfolyamok -> relatív teljesítmény
        df_pivot_filled = df_pivot.fillna(method='ffill').dropna()

        if not df_pivot_filled.empty:
            df_normalized = (df_pivot_filled / df_pivot_filled.iloc[0] * 100)

            plt.figure(figsize=(14, 7))
            df_normalized.plot(ax=plt.gca())
            plt.title('Normalizált Árfolyamok (Relatív Teljesítmény)')
            plt.xlabel('Dátum')
            plt.ylabel('Normalizált Ár (Kezdőérték = 100)')
            plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
            plt.tight_layout()
        else:
            print("A normalizált árfolyamok ábrája nem készíthető el -> túl sok a hiányzó adat.")

else:
    print("BAJ VAN A TOP 10el")


plt.show()