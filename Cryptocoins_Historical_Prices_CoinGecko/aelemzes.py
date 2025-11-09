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


print("Adatbeolvasás és előkészítés (CoinGecko adatkészlet, több CSV)...")
try:
    # 1. Keressük meg az összes CSV fájlt a mappában
    all_files = glob.glob("*.csv")

    if not all_files:
        print("Hiba: Nem találhatók '*.csv' fájlok a mappában.")
        print("Győződj meg róla, hogy a szkript és az összes CoinGecko CSV egy mappában van.")
        exit()

    li = []  # Ebben a listában gyűjtjük a DataFrame-eket
    print(f"Talált {len(all_files)} db CSV fájl. Szűrés és beolvasás...")

    # 2. Ciklus az összes megtalált fájlon
    for filename in all_files:
        try:
            # Beolvassuk az egyedi CSV-t (csak az első sort, hogy ellenőrizzük)
            # Ez gyorsabb, mintha az egészet beolvasnánk feleslegesen
            header_df = pd.read_csv(filename, nrows=0)

            # 3. ELLENŐRZÉS: Ez a megfelelő adatkészlet?
            # Az új (CoinGecko) datasetben 'coin_name' és 'price' van.
            if 'coin_name' in header_df.columns and 'price' in header_df.columns:
                print(f"Beolvasás: {filename}")
                temp_df = pd.read_csv(filename)

                # 4. OSZLOPNEVEK EGYSÉGESÍTÉSE (hogy a régi kód működjön)
                # 'coin_name' -> 'Currency' (kisbetűvel)
                temp_df['Currency'] = temp_df['coin_name'].str.lower()

                # Oszlopok átnevezése
                temp_df = temp_df.rename(columns={
                    'date': 'Date',
                    'price': 'Close',  # A 'price'-t 'Close'-nak (záróárnak) tekintjük
                    'total_volume': 'Volume',
                    'market_cap': 'Market Cap'
                })

                li.append(temp_df)
            else:
                # print(f"Kihagyva: {filename} (nem tűnik CoinGecko adatfájlnak)")
                pass  # Csendesen kihagyjuk a nem megfelelő fájlokat

        except Exception as e:
            print(f"Hiba a(z) {filename} fájl olvasása közben: {e}")

    if not li:
        print("Hiba: Nem sikerült egyetlen érvényes CoinGecko adatfájlt sem beolvasni.")
        print("Ellenőrizd, hogy a CSV-k tartalmazzák-e a 'coin_name' és 'price' oszlopokat.")
        exit()

    # 5. Az összes DataFrame összefűzése eggyé
    df = pd.concat(li, axis=0, ignore_index=True)

    print("Az összes valuta adat sikeresen összefűzve egy DataFrame-be.")

    # --- INNENTŐL A RÉGI KÓDOD (SZINTE) VÁLTOZATLANUL KÖVETKEZIK ---

    # Oszlopok, amiket numerikussá kell tenni (Open, High, Low már nincs)
    oszlopok = ['Close', 'Volume', 'Market Cap']

    # Oszlopok numerikussá tétele
    print("Oszlopok numerikussá alakítása...")
    for oszlop in oszlopok:
        if oszlop in df.columns and df[oszlop].dtype == 'object':
            df[oszlop] = df[oszlop].str.replace(',', '', regex=False)
            df[oszlop] = pd.to_numeric(df[oszlop], errors='coerce')
        elif oszlop not in df.columns:
            print(f"Figyelem: A '{oszlop}' oszlop hiányzik, kihagyás...")



    df['Volume'] = df['Volume'].fillna(0)
    df['Market Cap'] = df['Market Cap'].fillna(0)

    price_cols = ['Close']
    df[price_cols] = df[price_cols].fillna(method='ffill')

    # "Date" -> datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # sor azonositoja a date
    df = df.set_index('Date')

    # datum szerint novekvo sorrend
    df = df.sort_index()




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
coin_to_analyze = 'bitcoin'

print(f"\n Egyedi Valuta Elemzése: {coin_to_analyze.capitalize()}")

# Al-DataFrame létrehozása
df_coin = df[df['Currency'] == coin_to_analyze].copy()

if df_coin['Close'].isnull().any():
    print(f"Figyelem: A '{coin_to_analyze}' adatok hiányosak (NaN értékek maradtak) a tisztítás után is.")
    print("A hiányos sorok eldobása az elemzéshez...")
    df_coin = df_coin.dropna(subset=['Close'])  # Már csak a Close-ra kell ellenőrizni

if df_coin.empty:
    print("Nincs ilyen nevű Coin a datasetben (vagy nincs hozzá adat)")
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
    df_coin['MA50'] = df_coin['Close'].rolling(window=50).mean()
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



    # <---- IDŐ ALAPÚ ELEMZÉSEK ----->
    print(f"\n--- {coin_to_analyze.capitalize()} Idő-alapú Elemzései (11-13) ---")

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




# top 10 valuta a datasetben
print("Top 10 valuta meghatározása a legutolsó piaci kapitalizáció alapján...")
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
    print("BAJ VAN A TOP 10el - Nem található adat a legutolsó dátumra.")

print("\n--- Elemzés Befejezve! ---")
print("Az elemzés lefutott. Az ábrák felugró ablak(ok)ban jelennek meg.")
plt.show()