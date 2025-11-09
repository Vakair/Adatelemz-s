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

# --- MÓDOSÍTOTT ADATBEOLVASÁS (Top 100 Dataset) ---
try:
    # VÁLTOZTATÁS 1: Minden .csv fájlt keresünk
    all_files = glob.glob("*.csv")

    # VÁLTOZTATÁS 2: Eltávolítjuk a felesleges fájlokat, ahogy kérted
    files_to_remove = []
    for f in all_files:
        # Hozzáadjuk a listához az összes fájlt, amit ki akarunk hagyni
        basename = os.path.basename(f)
        if basename == "100 List.csv" or basename == "Top 100 Cryptos.ipynb":
            files_to_remove.append(f)

    for f in files_to_remove:
        print(f"Kihagyás: {os.path.basename(f)}")
        all_files.remove(f)

    if not all_files:
        print("BAAAJ VAAAN")  # A V2-es kódod printje
        exit()

    print(f"{len(all_files)} db file")  # A V2-es kódod printje

    li = []  # Ebben a listában gyűjtjük a DataFrame-eket

    # 2. Ciklus az összes megtalált fájlon
    for filename in all_files:
        # Beolvassuk az egyedi CSV-t
        temp_df = pd.read_csv(filename)

        # 3. LÉTREHOZZUK A HIÁNYZÓ 'Currency' OSZLOPOT
        # Megtartjuk a V2-es kódod logikáját
        if 'Name' in temp_df.columns:
            temp_df['Currency'] = temp_df['Name'].str.lower()
        else:
            # VÁLTOZTATÁS 3: Az 'else' ágat módosítjuk, hogy a fájlnévből
            # helyesen nyerje ki a nevet (a 'coin_' replace nélkül)
            # "Bitcoin.csv" -> "Bitcoin" -> "bitcoin"
            currency_name = os.path.splitext(os.path.basename(filename))[0]
            temp_df['Currency'] = currency_name.lower()

        li.append(temp_df)

    # összefűzés
    df = pd.concat(li, axis=0, ignore_index=True)

    print("Az összes valuta adat sikeresen összefűzve egy DataFrame-be.")

    # Átnevezzük, hogy a kód működjön.
    if 'Marketcap' in df.columns:
        df = df.rename(columns={'Marketcap': 'Market Cap'})

    # Oszlopok, amiket numerikussá kell tenni
    # Ebben a datasetben az O,H,L,C már float, de a V, MC string vesszőkkel
    oszlopok = ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']

    # Oszlopok numerikussá tétele
    print("Oszlopok numerikussá alakítása...")
    for oszlop in oszlopok:
        if oszlop in df.columns and df[oszlop].dtype == 'object':
            # Eltávolítjuk a vesszőket (ha vannak)
            df[oszlop] = df[oszlop].str.replace(',', '', regex=False)

            # A '-' karaktereket is cseréljük NaN-ra, hogy a pd.to_numeric kezelni tudja
            # (Néhány datasetben a hiányzó adat '-' jel)
            df[oszlop] = df[oszlop].replace('-', np.nan)

            df[oszlop] = pd.to_numeric(df[oszlop], errors='coerce')
        elif oszlop not in df.columns:
            print(f"Figyelem: A '{oszlop}' oszlop hiányzik, kihagyás...")

    # Hiányzó adatok (NaN) kezelése
    df['Volume'] = df['Volume'].fillna(0)
    df['Market Cap'] = df['Market Cap'].fillna(0)
    arak = ['Open', 'High', 'Low', 'Close']
    df[arak] = df[arak].fillna(method='ffill')

    # "Date" -> datetime
    print("Dátum oszlop (Date) átalakítása datetime objektummá...")

    # A "year 42093 is out of range" hiba azt jelzi, hogy Excel dátumformátumok (számok)
    # keverednek a szöveges dátumokkal (pl. "Sep 22, 2017").

    # 1. kísérlet: Átalakítás szövegként (errors='coerce' a hibásakat NaT-ra állítja)
    # Ez kezeli a "Sep 22, 2017" és "2017-09-22" formátumokat.
    dates_from_string = pd.to_datetime(df['Date'], errors='coerce')

    # 2. kísérlet: Átalakítás számként (Excel formátum)
    # Először numerikussá alakítjuk (errors='coerce' a szövegeket NaN-ra állítja)
    dates_as_numeric = pd.to_numeric(df['Date'], errors='coerce')
    # Ezután átalakítjuk az Excel dátumokat (az 1899-12-30 az Excel "nulladik" napja)
    dates_from_excel = pd.to_datetime(dates_as_numeric, unit='D', origin='1899-12-30')

    # 3. Összefésülés:
    # Ahol a szöveges átalakítás sikerült (nem NaT), azt használjuk.
    # Ahol nem sikerült (NaT), ott megpróbáljuk az Excel-ből konvertáltat.
    df['Date'] = dates_from_string.fillna(dates_from_excel)

    # Ellenőrzés, hogy maradt-e hiba
    if df['Date'].isnull().any():
        print("Figyelem: A dátumok átalakítása után is maradtak érvénytelen (NaN) dátumok.")
        print("Az érvénytelen dátumú sorok eldobása...")
        df = df.dropna(subset=['Date'])

    print("Dátum átalakítás sikeres.")

    # sor azonositoja a date
    df = df.set_index('Date')

    # datum szerint novekvo sorrend
    df = df.sort_index()

    # VÁLTOZTATÁS 4 (Biztonsági): Eltávolítjuk azokat a sorokat (főleg az elejéről),
    # ahol a 'ffill' után is NaN maradt
    df = df.dropna(subset=arak)

    # ------- Adatbeolvasas vege ---------

except FileNotFoundError:

    print("AAAAAAAAAAAAAAAAAAAAAAAA, BAJ VAN")
    exit()  # Kilépés a szkriptből, ha nincs adat
except Exception as e:
    print(f"AAAAAAAAAAAAAAAAAAAAAAAA, BAJ VAN: {e}")
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

# Al-DataFrame létrehozása csak a kiválasztott valuta adataival
# A .copy() használata segít elkerülni a pandas 'SettingWithCopyWarning' figyelmeztetését
df_coin = df[df['Currency'] == coin_to_analyze].copy()

if df_coin.empty:
    # Frissített hibaüzenet, hogy informatívabb legyen
    print(f"Nincs '{coin_to_analyze}' nevű Coin a datasetben (ellenőrizd a kis- és nagybetűket, szóközöket)")
else:
    # Árfolyam (Close) alakulása vonaldiagrammal
    plt.figure(figsize=(14, 7))
    df_coin['Close'].plot()
    plt.title(f'{coin_to_analyze.capitalize()} Árfolyam (Close) Alakulása')
    plt.xlabel('Dátum')
    plt.ylabel('Ár (USD)')
    plt.tight_layout()
    # plt.savefig('1_btc_close_price.png') # Ábra mentése fájlba

    # Kereskedési volumen (kereskedesi forgalom) alakulása
    plt.figure(figsize=(14, 7))
    df_coin['Volume'].plot(color='blue')
    plt.title(f'{coin_to_analyze.capitalize()} Kereskedési Volumen Alakulása')
    plt.xlabel('Dátum')
    plt.ylabel('Volumen (USD)')
    plt.tight_layout()
    # plt.savefig('2_btc_volume.png')

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
    # plt.savefig('3_btc_moving_averages.png')

    # A napi hozam (záróár változása százalékban) -> százalékos növekedés vagy csokkenes
    # (Mai ár - Tegnapi ár) / Tegnapi ár
    df_coin['Daily_Return'] = df_coin['Close'].pct_change()
    print("A napi hozamok átlaga:", df_coin['Daily_Return'].mean())
    print("A napi hozamok mediánja:", df_coin['Daily_Return'].median())
    print("\n")

    # napi hozamok eloszlása -> napi pozitiv/negativ változások gyakorisága
    plt.figure(figsize=(10, 6))
    # A .dropna() eltávolítja az első napot ahol nincs hozam -> NaN érték van
    sns.histplot(df_coin['Daily_Return'].dropna(), bins=100, kde=True,
                 color='green')
    plt.title(f'{coin_to_analyze.capitalize()} Napi Hozamok Eloszlása')
    plt.xlabel('Napi Hozam (%)')
    plt.ylabel('Gyakoriság')
    plt.tight_layout()
    # plt.savefig('4_btc_daily_returns_hist.png')

    # Volatilitás (szórás) , kiszamoljuk egy napra jellemző átlagos ármozgás mértékét.
    daily_volatility = df_coin['Daily_Return'].std()
    print(f"Napi volatilitás (szórás): {daily_volatility:.4f}")
    annualized_volatility = daily_volatility * np.sqrt(365)
    print(f"Évesített volatilitás: {annualized_volatility:.4f}")

    # Gyertyás diagram
    # utolsó 100 napot
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
    # A V2-es kódodban ez be volt kapcsolva.
    fig.show()

    # <---- IDŐ ALAPÚ ELEMZÉSEK ----->

    df_coin['Day_of_Week'] = df_coin.index.dayofweek  # 0=Hétfő, 1=Kedd, ..., 6=Vasárnap
    df_coin['Month'] = df_coin.index.month  # 1=Január, ..., 12=December
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
    # plt.savefig('10_btc_return_by_day.png')

    # átlagos volumen a hét napjai szerint -> coinban lévő pénz mennyiség
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
    # plt.savefig('11_btc_volume_by_day.png')

    # átlagos hozam a hónapok szerint -> mennyit emelkedik az ár átlagosan egy adott hónapban
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
    # plt.savefig('12_btc_return_by_month.png')

    # napi ártartomány -> high - low
    df_coin['Daily_Range'] = df_coin['High'] - df_coin['Low']
    df_coin['Daily_Range_Pct'] = (df_coin['Daily_Range'] / df_coin['Close']) * 100

    print("Átlagos napi ártartomány (USD):", df_coin['Daily_Range'].mean())
    print("Átlagos napi ártartomány (Záróár %-a):", df_coin['Daily_Range_Pct'].mean())
    print("\n")

    # napi ártartomány százalékos eloszlása a hét napjai szerint
    # Box Plot
    plt.figure(figsize=(14, 7))
    sns.boxplot(x='Day_of_Week', y='Daily_Range_Pct', data=df_coin)
    plt.title(f'{coin_to_analyze.capitalize()} Napi Ártartomány (Intraday Volatilitás , High - Low) Eloszlása')
    plt.xticks(ticks=range(7), labels=napok)
    plt.xlabel('Hét Napja')
    plt.ylabel('Napi Ártartomány (a Záróár %-ában)')
    plt.tight_layout()
    # plt.savefig('13_btc_daily_range_boxplot.png')

# <--- TÖBB VALUTA ÖSSZEHASONLÍTÁSA --->

latest_date = df.index.max()

df_latest = df.loc[latest_date]

# Ha egy napon több valuta is van (ami valószínű), akkor megkeressük a 10 legnagyobbat
if not df_latest.empty:

    # Biztonsági ellenőrzés és átalakítás, ha az utolsó napon csak 1 coin van
    if isinstance(df_latest, pd.Series):
        df_latest = df_latest.to_frame().T

    if isinstance(df_latest, pd.DataFrame):
        # Győződjünk meg róla, hogy a Market Cap numerikus, mielőtt a .nlargest-et hívjuk
        df_latest['Market Cap'] = pd.to_numeric(df_latest['Market Cap'], errors='coerce')
        top_10_market_cap = df_latest.nlargest(10, 'Market Cap')
    else:
        # Fallback, ha a df_latest valamiért mégsem DataFrame
        df['Market Cap'] = pd.to_numeric(df['Market Cap'], errors='coerce')
        top_10_market_cap = df.nlargest(10, 'Market Cap')

    top_10_coins = top_10_market_cap['Currency'].tolist()
    # Tisztítás (ha pl. NaN vagy None kerül a listába)
    top_10_coins = [coin for coin in top_10_coins if pd.notna(coin)]

    print(f"Top 10 valuta: {top_10_coins}")
    print("\n")

    if not top_10_coins:
        print("BAJ VAN A TOP 10el.")
        df_top10 = pd.DataFrame()  # Üres df, hogy a kód ne álljon le
    else:
        df_top10 = df[df['Currency'].isin(top_10_coins)]

    if df_top10.empty:
        print("BAJ VAN A TOP 10el")
    else:
        # Korrelációs mátrix -> close alapján
        df_pivot = df_top10.pivot_table(index='Date', columns='Currency', values='Close')

        # VÁLTOZTATÁS (Biztonsági): A hiányzó adatok (NaN) kezelése
        # (ha egy valuta később indult)
        df_pivot_filled = df_pivot.fillna(method='ffill').fillna(method='bfill')

        # napi hozamok számítása a pivotált táblán
        df_returns = df_pivot_filled.pct_change()

        correlation_matrix = df_returns.corr()

        # HEATMAP
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Top 10 Valuta Napi Hozamainak Korrelációja')
        plt.tight_layout()
        # plt.savefig('6_correlation_matrix.png')

        # Korrelációs mátrix printelése
        print("Korrelációs mátrix:\n")
        print("1 = tökéletes együttmozgás, -1 = tökéletes ellentétes mozgás, 0 = nincs összefüggés")
        print(correlation_matrix)

        # Kördiagramm - Market Cap eloszlás
        # (Print hozzáadva a V2-es kódod alapján)
        plt.figure(figsize=(10, 8))
        # Kördiagram (pie chart)
        plt.pie(top_10_market_cap['Market Cap'], labels=top_10_market_cap['Currency'], autopct='%1.1f%%', startangle=90)
        plt.title('Top 10 Valuta Piaci Kapitalizációjának Eloszlása (Legutolsó Adat)')
        plt.axis('equal')
        plt.tight_layout()
        # plt.savefig('7_market_cap_pie.png')

        # Napi volatilitás összehasonlítása (Top 10)
        volatility_comparison = df_returns.std().sort_values(ascending=False)

        plt.figure(figsize=(12, 6))
        sns.barplot(x=volatility_comparison.index, y=volatility_comparison.values)
        plt.title('Napi Volatilitás (Szórás) Összehasonlítása (Top 10)')
        plt.xlabel('Valuta')
        plt.ylabel('Napi Volatilitás (Szórás)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        # plt.savefig('8_volatility_comparison.png')

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
            # plt.savefig('9_normalized_prices.png', bbox_inches='tight')
        else:
            print("A normalizált árfolyamok ábrája nem készíthető el -> túl sok a hiányzó adat.")

else:
    print("BAJ VAN A TOP 10el")

plt.show()