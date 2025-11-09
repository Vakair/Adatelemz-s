import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import numpy as np

plt.style.use('ggplot')
# seaborn stílus
sns.set(style="whitegrid")

#Adatbeolvasás
print("Adatok beolvasása és előkészítése...")
try:
    # Az adatkészlet beolvasása egy pandas DataFrame-be (táblázatba)
    # A 'consolidated_coin_data.csv' fájlnak ugyanabban a mappában kell lennie, mint a szkriptnek
    df = pd.read_csv('consolidated_coin_data.csv')

    # *** HIBAJAVÍTÁS KEZDETE ***
    # A df.info() kimenet (lásd traceback) alapján az ároszlopok (Open, High, stb.)
    # 'object' (szöveg) típusként lettek beolvasva, valószínűleg a bennük lévő vesszők (,) miatt.
    # Ezeket numerikussá kell alakítani.

    # Oszlopok, amiknek numerikusnak kellene lenniük
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']

    print("Oszlopok numerikussá alakítása (vesszők eltávolítása)...")
    for col in numeric_cols:
        # Először ellenőrizzük, hogy az oszlop létezik-e és 'object' típusú-e
        if col in df.columns and df[col].dtype == 'object':
            # Eltávolítjuk a vesszőket (ez a leggyakoribb hibaok)
            # A regex=False gyorsítja a műveletet, mivel sima szöveges cserét végzünk
            df[col] = df[col].str.replace(',', '', regex=False)

            # Átalakítjuk numerikus típussá (float).
            # errors='coerce' -> ha valamit (pl. '-') nem tud átalakítani, NaN értéket ad neki,
            # ahelyett, hogy hibát dobna. Ez a statisztikákhoz megfelelő.
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print("Numerikus átalakítás kész.")
    # *** HIBAJAVÍTÁS VÉGE ***





    # A 'Date' oszlop átalakítása 'datetime' objektummá, ami elengedhetetlen az idősoros elemzéshez
    # A 'format' argumentum segít, ha az alapértelmezett formátumtól eltérő a dátum
    # (feltételezve, hogy 'YYYY-MM-DD' vagy hasonló, amit a pandas felismer)
    df['Date'] = pd.to_datetime(df['Date'])
    # Ha a beolvasás hibát jelez, a formátumot pontosítani kell, pl. format='%Y-%m-%d %H:%M:%S'

    # A 'Date' oszlop beállítása a DataFrame indexévé (sor azonosítójává)
    # Ez megkönnyíti az idősoros műveleteket, pl. mozgóátlag számítást
    df = df.set_index('Date')

    # A DataFrame rendezése az index (dátum) szerint növekvő sorrendbe
    df = df.sort_index()

    print("Adatbeolvasás sikeres.")

except FileNotFoundError:
    # Hibaüzenet, ha a fájl nem található
    print("Hiba: A 'consolidated_coin_data.csv' fájl nem található.")
    print("Kérlek, másold a fájlt a szkript mappájába és futtasd újra.")
    exit()  # Kilépés a szkriptből, ha nincs adat
except Exception as e:
    # Általános hibaüzenet, pl. ha a dátum formátuma nem megfelelő
    print(f"Hiba történt az adatbeolvasás során: {e}")
    exit()

# --- 3. Alapvető Leíró Statisztikák ---
print("\n--- Alapvető Leíró Statisztikák (1, 2, 3) ---")

# 1. Általános információk a DataFrame-ről
print("\n[1. Adatkeret Információk (Info)]")
# Kiírja az oszlopneveket, az oszlopok adattípusát, és a nem-null (nem hiányzó) értékek számát
# Ez segít gyorsan azonosítani a hiányzó adatokat és a rossz adattípusokat (pl. ha a 'Volume' szövegként van tárolva)
df.info()

# 2. Leíró statisztikák a numerikus oszlopokról
print("\n[2. Leíró Statisztikák (Describe)]")
# Kiírja a főbb statisztikai mutatókat:
# count: adatok száma, mean: átlag, std: szórás (volatilitás mérőszáma)
# min: minimum érték, 25%: 1. kvartilis, 50%: medián, 75%: 3. kvartilis, max: maximum érték
print(df.describe())

# 3. Egyedi valuták számának meghatározása
print("\n[3. Egyedi Valuták Száma]")
# Megszámolja, hány különböző valuta (pl. 'bitcoin', 'ethereum') található az 'Currency' oszlopban
unique_currencies = df['Currency'].nunique()
print(f"Az adatkészlet {unique_currencies} különböző kriptovalutát tartalmaz.")

# --- 4. Egyedi Valuta Elemzése ---
# Koncentráljunk egyetlen, jól ismert valutára a részletesebb elemzéshez
# (A botod valószínűleg egy-egy párra fog fókuszálni)

# ÁLLÍTSD BE AZ ELEMEZNI KÍVÁNT VALUTÁT ITT:
coin_to_analyze = 'bitcoin'

print(f"\n--- Egyedi Valuta Elemzése: {coin_to_analyze.capitalize()} (4-15) ---")

# Al-DataFrame létrehozása csak a kiválasztott valuta adataival
# A .copy() használata segít elkerülni a pandas 'SettingWithCopyWarning' figyelmeztetését
df_coin = df[df['Currency'] == coin_to_analyze].copy()

if df_coin.empty:
    print(f"Hiba: Nincsenek '{coin_to_analyze}' adatok az adatkészletben. Az egyedi elemzés kihagyva.")
else:
    # 4. Árfolyam alakulása (Záróár)
    print(f"\n[4. Vizualizáció: {coin_to_analyze.capitalize()} Záróár (Close) Alakulása]")
    plt.figure(figsize=(14, 7))  # Ábra méretének beállítása
    df_coin['Close'].plot()  # A 'Close' oszlop kirajzolása (vonaldiagram)
    plt.title(f'{coin_to_analyze.capitalize()} Árfolyam (Close) Alakulása')  # Cím
    plt.xlabel('Dátum')  # X tengely felirata
    plt.ylabel('Ár (USD)')  # Y tengely felirata
    plt.tight_layout()  # Elrendezés javítása
    # plt.savefig('1_btc_close_price.png') # Ábra mentése fájlba
    print("Ábra '1_coin_close_price' elkészítve a megjelenítéshez.")

    # 5. Kereskedési volumen alakulása
    print(f"\n[5. Vizualizáció: {coin_to_analyze.capitalize()} Kereskedési Volumen Alakulása]")
    # Mivel túl sok adatpont van egy oszlopdiagramhoz, vonaldiagramot használunk
    plt.figure(figsize=(14, 7))
    df_coin['Volume'].plot(color='blue')
    plt.title(f'{coin_to_analyze.capitalize()} Kereskedési Volumen Alakulása')
    plt.xlabel('Dátum')
    plt.ylabel('Volumen (USD)')
    plt.tight_layout()
    # plt.savefig('2_btc_volume.png')
    print("Ábra '2_coin_volume' elkészítve a megjelenítéshez.")

    # 6. Mozgóátlagok (Moving Averages - MA)
    print(f"\n[6. Vizualizáció: {coin_to_analyze.capitalize()} Árfolyam Mozgóátlagokkal (MA50, MA200)]")
    # A mozgóátlagok segítenek a trendek azonosításában (pl. aranykereszt, halálkereszt)
    # 50 napos mozgóátlag (rövid távú trend)
    df_coin['MA50'] = df_coin['Close'].rolling(window=50).mean()
    # 200 napos mozgóátlag (hosszú távú trend)
    df_coin['MA200'] = df_coin['Close'].rolling(window=200).mean()

    plt.figure(figsize=(14, 7))
    plt.plot(df_coin.index, df_coin['Close'], label='Záróár (Close)', alpha=0.7)
    plt.plot(df_coin.index, df_coin['MA50'], label='50 napos mozgóátlag (MA50)', color='orange')
    plt.plot(df_coin.index, df_coin['MA200'], label='200 napos mozgóátlag (MA200)', color='red')
    plt.title(f'{coin_to_analyze.capitalize()} Árfolyam Mozgóátlagokkal')
    plt.xlabel('Dátum')
    plt.ylabel('Ár (USD)')
    plt.legend()  # Jelmagyarázat hozzáadása
    plt.tight_layout()
    # plt.savefig('3_btc_moving_averages.png')
    print("Ábra '3_coin_moving_averages' elkészítve a megjelenítéshez.")

    # 7. Napi hozamok kiszámítása
    print(f"\n[7. Statisztika: {coin_to_analyze.capitalize()} Napi Hozamok (Daily Returns) Számítása]")
    # A napi hozam (záróár változása százalékban) alapvető a volatilitás és stratégia teszteléséhez
    # (Mai ár - Tegnapi ár) / Tegnapi ár
    df_coin['Daily_Return'] = df_coin['Close'].pct_change()
    print("A napi hozamok átlaga:", df_coin['Daily_Return'].mean())
    print("A napi hozamok mediánja:", df_coin['Daily_Return'].median())

    # 8. Napi hozamok eloszlása (Hisztogram)
    print(f"\n[8. Vizualizáció: {coin_to_analyze.capitalize()} Napi Hozamok Eloszlása (Hisztogram)]")
    # Megmutatja, milyen gyakoriak a különböző mértékű napi (pozitív és negatív) változások
    plt.figure(figsize=(10, 6))
    # A .dropna() eltávolítja az első napot (ahol nincs hozam, NaN érték van)
    sns.histplot(df_coin['Daily_Return'].dropna(), bins=100, kde=True,
                 color='green')  # kde=True rárajzol egy sűrűségfüggvényt
    plt.title(f'{coin_to_analyze.capitalize()} Napi Hozamok Eloszlása')
    plt.xlabel('Napi Hozam (%)')
    plt.ylabel('Gyakoriság')
    plt.tight_layout()
    # plt.savefig('4_btc_daily_returns_hist.png')
    print("Ábra '4_coin_daily_returns_hist' elkészítve a megjelenítéshez.")

    # 9. Volatilitás (Szórás)
    print(f"\n[9. Statisztika: {coin_to_analyze.capitalize()} Napi Volatilitás (Napi Hozamok Szórása)]")
    # A volatilitás a kockázat egyik fő mérőszáma. Magasabb volatilitás = nagyobb kockázat/potenciális nyereség
    daily_volatility = df_coin['Daily_Return'].std()
    print(f"Napi volatilitás (szórás): {daily_volatility:.4f}")
    # Évesített volatilitás (feltéve, hogy 365 napos a piac)
    annualized_volatility = daily_volatility * np.sqrt(365)
    print(f"Évesített volatilitás: {annualized_volatility:.4f}")

    # 10. Gyertya diagram (Candlestick Chart)
    print(f"\n[10. Vizualizáció: {coin_to_analyze.capitalize()} Gyertya Diagram (Plotly, utolsó 100 nap)]")
    # A gyertya diagram a trading alapvető eszköze, mutatja a nyitó, záró, minimum és maximum árat

    # Vegyük az utolsó 100 napot
    df_coin_100 = df_coin.tail(100)

    fig = go.Figure(data=[go.Candlestick(x=df_coin_100.index,
                                         open=df_coin_100['Open'],
                                         high=df_coin_100['High'],
                                         low=df_coin_100['Low'],
                                         close=df_coin_100['Close'])])

    fig.update_layout(
        title=f'{coin_to_analyze.capitalize()} Gyertya Diagram (Utolsó 100 nap)',
        yaxis_title='Ár (USD)',
        xaxis_rangeslider_visible=False  # Csúszka kikapcsolása az X tengelyen
    )
    # A fig.show() megnyitja az ábrát böngészőben.
    # fig.show() # Ez megnyitná a böngészőben

    # --- Idő-alapú Elemzések (Szezonalitás) - ÁTHELYEZVE IDE ---

    print(f"\n--- {coin_to_analyze.capitalize()} Idő-alapú Elemzései (11-15) ---")

    # Új oszlopok létrehozása a hét napjának és a hónapnak
    df_coin['Day_of_Week'] = df_coin.index.dayofweek  # 0=Hétfő, 1=Kedd, ..., 6=Vasárnap
    df_coin['Month'] = df_coin.index.month  # 1=Január, ..., 12=December
    day_names = ['Hétfő', 'Kedd', 'Szerda', 'Csütörtök', 'Péntek', 'Szombat', 'Vasárnap']
    month_names = ['Jan', 'Feb', 'Már', 'Ápr', 'Máj', 'Jún', 'Júl', 'Aug', 'Szep', 'Okt', 'Nov', 'Dec']

    # 11. Átlagos hozam a hét napjai szerint
    print(f"\n[11. Vizualizáció: {coin_to_analyze.capitalize()} Átlagos Napi Hozam a Hét Napjai Szerint]")
    plt.figure(figsize=(10, 6))
    avg_return_day = df_coin.groupby('Day_of_Week')['Daily_Return'].mean()
    avg_return_day.index = avg_return_day.index.map(lambda x: day_names[x])
    avg_return_day = avg_return_day.reindex(day_names)  # Sorrend biztosítása

    avg_return_day.plot(kind='bar', color='purple')
    plt.title(f'{coin_to_analyze.capitalize()} Átlagos Napi Hozam a Hét Napjai Szerint')
    plt.xlabel('Hét Napja')
    plt.ylabel('Átlagos Napi Hozam')
    plt.xticks(rotation=0)
    plt.tight_layout()
    # plt.savefig('10_btc_return_by_day.png')
    print("Ábra '10_coin_return_by_day' elkészítve a megjelenítéshez.")

    # 12. Átlagos volumen a hét napjai szerint
    print(f"\n[12. Vizualizáció: {coin_to_analyze.capitalize()} Átlagos Kereskedési Volumen a Hét Napjai Szerint]")
    plt.figure(figsize=(10, 6))
    avg_volume_day = df_coin.groupby('Day_of_Week')['Volume'].mean()
    avg_volume_day.index = avg_volume_day.index.map(lambda x: day_names[x])
    avg_volume_day = avg_volume_day.reindex(day_names)  # Sorrend biztosítása

    avg_volume_day.plot(kind='bar', color='orange')
    plt.title(f'{coin_to_analyze.capitalize()} Átlagos Kereskedési Volumen a Hét Napjai Szerint')
    plt.xlabel('Hét Napja')
    plt.ylabel('Átlagos Volumen (USD)')
    plt.xticks(rotation=0)
    plt.tight_layout()
    # plt.savefig('11_btc_volume_by_day.png')
    print("Ábra '11_coin_volume_by_day' elkészítve a megjelenítéshez.")

    # 13. Átlagos hozam a hónapok szerint
    print(f"\n[13. Vizualizáció: {coin_to_analyze.capitalize()} Átlagos Napi Hozam a Hónapok Szerint]")
    plt.figure(figsize=(10, 6))
    avg_return_month = df_coin.groupby('Month')['Daily_Return'].mean()
    avg_return_month.index = avg_return_month.index.map(
        lambda x: month_names[x - 1])  # x-1, mert a hónapok 1-től indulnak
    avg_return_month = avg_return_month.reindex(month_names)  # Sorrend biztosítása

    avg_return_month.plot(kind='bar', color='cyan')
    plt.title(f'{coin_to_analyze.capitalize()} Átlagos Napi Hozam a Hónapok Szerint')
    plt.xlabel('Hónap')
    plt.ylabel('Átlagos Napi Hozam')
    plt.xticks(rotation=0)
    plt.tight_layout()
    # plt.savefig('12_btc_return_by_month.png')
    print("Ábra '12_coin_return_by_month' elkészítve a megjelenítéshez.")

    # 14. Napi ártartomány (Intraday Volatilitás) kiszámítása
    print(f"\n[14. Statisztika: {coin_to_analyze.capitalize()} Napi Ártartomány (High - Low) Számítása]")
    # Ez a napi volatilitás egy másik mérőszáma: a napon belüli legmagasabb és legalacsonyabb ár különbsége
    df_coin['Daily_Range'] = df_coin['High'] - df_coin['Low']
    # Napi ártartomány százalékos aránya a záróárhoz képest (jobban összehasonlítható)
    df_coin['Daily_Range_Pct'] = (df_coin['Daily_Range'] / df_coin['Close']) * 100

    print("Átlagos napi ártartomány (USD):", df_coin['Daily_Range'].mean())
    print("Átlagos napi ártartomány (Záróár %-a):", df_coin['Daily_Range_Pct'].mean())

    # 15. Napi ártartomány (százalékos) eloszlása a hét napjai szerint
    print(
        f"\n[15. Vizualizáció: {coin_to_analyze.capitalize()} Napi Ártartomány Eloszlása (Box Plot, Hét Napjai Szerint)]")
    # A Box Plot (doboz-ábra) megmutatja az eloszlást (medián, kvartilisek, kiugró értékek)
    plt.figure(figsize=(14, 7))
    # A 'Daily_Range_Pct' eloszlásának ábrázolása a hét napjai szerint
    sns.boxplot(x='Day_of_Week', y='Daily_Range_Pct', data=df_coin)
    plt.title(f'{coin_to_analyze.capitalize()} Napi Ártartomány (Intraday Volatilitás) Eloszlása')
    # A tengelyfeliratok beállítása a napok nevére
    plt.xticks(ticks=range(7), labels=day_names)
    plt.xlabel('Hét Napja')
    plt.ylabel('Napi Ártartomány (a Záróár %-ában)')
    plt.tight_layout()
    # plt.savefig('13_btc_daily_range_boxplot.png')
    print("Ábra '13_coin_daily_range_boxplot' elkészítve a megjelenítéshez.")

# --- 5. Több Valuta Összehasonlítása ---
print("\n--- Több Valuta Összehasonlítása (16-20) ---")

# Válasszuk ki a 10 legnagyobb valutát az utolsó elérhető piaci kapitalizáció alapján
# Először lekérjük az utolsó dátumot
latest_date = df.index.max()
# Lekérjük az adatokat ezen a napon
df_latest = df.loc[latest_date]

# Ha egy napon több valuta is van (ami valószínű), akkor megkeressük a 10 legnagyobbat
if not df_latest.empty:
    if isinstance(df_latest, pd.DataFrame):  # Ha a legutolsó indexre több sor esik
        top_10_market_cap = df_latest.nlargest(10, 'Market Cap')
    else:  # Ha csak egy sor (nem valószínű a mi adatunkkal, de kezeljük le)
        top_10_market_cap = df.nlargest(10, 'Market Cap')  # Ez esetben a valaha volt 10 legnagyobbat veszi

    top_10_coins = top_10_market_cap['Currency'].tolist()
    print(f"Top 10 valuta (piaci kap. alapján): {top_10_coins}")

    # Szűrjük az eredeti DataFrame-et csak erre a 10 valutára
    df_top10 = df[df['Currency'].isin(top_10_coins)]

    # 16. Korrelációs mátrix (Napi hozamok alapján)
    print("\n[16. Vizualizáció: Korrelációs Mátrix (Napi Hozamok)]")
    # A korreláció megmutatja, mennyire mozognak együtt a különböző valuták árai
    # Ehhez át kell alakítani a táblát: oszlopok a valuták, sorok a dátumok, érték a záróár
    df_pivot = df_top10.pivot_table(index='Date', columns='Currency', values='Close')
    # Napi hozamok számítása a pivotált táblán
    df_returns = df_pivot.pct_change()

    # Korrelációs mátrix számítása
    correlation_matrix = df_returns.corr()

    plt.figure(figsize=(10, 8))
    # Hőtérkép (heatmap) a korrelációs mátrixról
    # annot=True: kiírja az értékeket, fmt='.2f': két tizedesjegyre kerekítve
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Top 10 Valuta Napi Hozamainak Korrelációja')
    plt.tight_layout()
    # plt.savefig('6_correlation_matrix.png')
    print("Ábra '6_correlation_matrix' elkészítve a megjelenítéshez.")

    # 17. Korrelációs mátrix kiírása
    print("\n[17. Statisztika: Korrelációs Mátrix Értékei]")
    print(correlation_matrix)
    print("Magyarázat: 1 = tökéletes együttmozgás, -1 = tökéletes ellentétes mozgás, 0 = nincs összefüggés")

    # 18. Piaci kapitalizáció eloszlása (Top 10)
    print("\n[18. Vizualizáció: Piaci Kapitalizáció Eloszlása (Top 10)]")
    plt.figure(figsize=(10, 8))
    # Kördiagram (pie chart)
    plt.pie(top_10_market_cap['Market Cap'], labels=top_10_market_cap['Currency'], autopct='%1.1f%%', startangle=90)
    plt.title('Top 10 Valuta Piaci Kapitalizációjának Eloszlása (Legutolsó Adat)')
    plt.axis('equal')  # Biztosítja, hogy a kör kör alakú legyen
    plt.tight_layout()
    # plt.savefig('7_market_cap_pie.png')
    print("Ábra '7_market_cap_pie' elkészítve a megjelenítéshez.")

    # 19. Volatilitás összehasonlítása (Top 10)
    print("\n[19. Vizualizáció: Napi Volatilitás Összehasonlítása (Top 10)]")
    # A korábban számított (pivotált) hozamok szórása (volatilitása)
    volatility_comparison = df_returns.std().sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=volatility_comparison.index, y=volatility_comparison.values)
    plt.title('Napi Volatilitás (Szórás) Összehasonlítása (Top 10)')
    plt.xlabel('Valuta')
    plt.ylabel('Napi Volatilitás (Szórás)')
    plt.xticks(rotation=45)  # Tengelyfeliratok elforgatása, ha átfedik egymást
    plt.tight_layout()
    # plt.savefig('8_volatility_comparison.png')
    print("Ábra '8_volatility_comparison' elkészítve a megjelenítéshez.")

    # 20. Normalizált árfolyamok (Relatív teljesítmény)
    print("\n[20. Vizualizáció: Normalizált Árfolyamok (Relatív Teljesítmény)]")
    # Megmutatja, hogyan teljesítettek a valuták egymáshoz képest, ha mind 100-ról indultak volna
    # A pivotált táblát (df_pivot) használjuk, de először kezelni kell a hiányzó adatokat
    # (pl. ha egy valuta később indult)
    # Töltsük fel a hiányzó értékeket az előző ismert értékkel, majd dobjuk el a sorokat, ahol még mindig van NaN
    df_pivot_filled = df_pivot.fillna(method='ffill').dropna()

    if not df_pivot_filled.empty:
        # Normalizálás az első elérhető dátumhoz képest (ahol már mind a 10-nek van ára)
        df_normalized = (df_pivot_filled / df_pivot_filled.iloc[0] * 100)

        plt.figure(figsize=(14, 7))
        df_normalized.plot(ax=plt.gca())  # ax=plt.gca() -> a jelenlegi ábrára rajzoljon
        plt.title('Normalizált Árfolyamok (Relatív Teljesítmény)')
        plt.xlabel('Dátum')
        plt.ylabel('Normalizált Ár (Kezdőérték = 100)')
        plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))  # Jelmagyarázat az ábrán kívülre
        plt.tight_layout()
        # plt.savefig('9_normalized_prices.png', bbox_inches='tight') # bbox_inches='tight' -> mentéskor ne vágja le a jelmagyarázatot
        print("Ábra '9_normalized_prices' elkészítve a megjelenítéshez.")
    else:
        print("A normalizált árfolyamok ábrája nem készíthető el (túl sok a hiányzó adat).")

else:
    print("Top 10 valuta nem található, a több-valutás elemzés (16-20) kihagyva.")

print("\n--- Elemzés Befejezve! ---")
print("Az elemzés lefutott. Az ábrák felugró ablak(ok)ban jelennek meg.")

# Ez a parancs megjeleníti az összes elkészült Matplotlib ábrát, ha a szkriptet interaktívan futtatod
# (Spyder, Jupyter, vagy terminálból)
# Ha a szkript lefutása után automatikusan be akarod zárni, kommenteld ki.
plt.show()

