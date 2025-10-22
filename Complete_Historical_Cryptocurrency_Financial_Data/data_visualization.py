import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Betöltés és előkészítés
df = pd.read_csv("consolidated_coin_data.csv", parse_dates=["Date"])
df.sort_values(by=['Currency', 'Date'], inplace=True)

# 🧹 Tisztítás: szöveges számok konvertálása numerikusra
numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']
for col in numeric_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(',', '', regex=False)
        .str.replace('$', '', regex=False)
    )
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Dátum fixálása
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
print(f"Bitcoin dátumtartomány: {df[df['Currency'] == 'bitcoin']['Date'].min()} - {df[df['Currency'] == 'bitcoin']['Date'].max()}")

# Hozam és volatilitás számítás
df['returns'] = df.groupby('Currency')['Close'].pct_change()
df['volatility'] = (df['High'] - df['Low']) / df['Open']

# Leíró statisztika – pl. Bitcoin
btc = df[df['Currency'] == 'bitcoin']
print(btc.describe())

# 1️⃣ Bitcoin árfolyam
plt.figure(figsize=(10, 4))
sns.lineplot(data=btc, x='Date', y='Close')
plt.title('Bitcoin árfolyam alakulása (USD)')
plt.xlabel('Dátum')
plt.ylabel('Záróár')
plt.tight_layout()
plt.show()

# 2️⃣ Bitcoin napi hozam eloszlása
plt.figure(figsize=(8, 4))
sns.histplot(btc['returns'].dropna(), kde=True, bins=60)
plt.title('Bitcoin napi hozam eloszlása')
plt.xlabel('Napi hozam')
plt.tight_layout()
plt.show()

# 3️⃣ Bitcoin 30 napos volatilitás
plt.figure(figsize=(10, 4))
sns.lineplot(data=btc, x='Date', y=btc['volatility'].rolling(30).mean())
plt.title('Bitcoin 30 napos volatilitás')
plt.xlabel('Dátum')
plt.ylabel('Volatilitás (30 napos gördülő átlag)')
plt.tight_layout()
plt.show()

# 4️⃣ Top 5 coin – záróár alakulás
top5 = df['Currency'].value_counts().head(5).index
plt.figure(figsize=(12, 6))
sns.lineplot(data=df[df['Currency'].isin(top5)], x='Date', y='Close', hue='Currency')
plt.title('Top 5 kriptovaluta árfolyam-idősora')
plt.yscale('log')
plt.tight_layout()
plt.show()

# 5️⃣ Átlagos volatilitás vs. piaci kapitalizáció
avg_vol = df.groupby('Currency')['volatility'].mean()
avg_cap = df.groupby('Currency')['Market Cap'].mean()
vol_cap = pd.DataFrame({'avg_volatility': avg_vol, 'avg_marketcap': avg_cap}).dropna()

plt.figure(figsize=(7, 5))
sns.scatterplot(data=vol_cap, x='avg_marketcap', y='avg_volatility')
plt.xscale('log')
plt.title('Volatilitás vs. Piaci kapitalizáció')
plt.xlabel('Piaci kapitalizáció (log skála)')
plt.ylabel('Átlagos volatilitás')
plt.tight_layout()
plt.show()

# 6️⃣ Korrelációs mátrix a hozamokra
returns_pivot = df.pivot_table(index='Date', columns='Currency', values='returns', aggfunc='mean')
corr = returns_pivot.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title('Kriptovaluták közötti hozamkorreláció')
plt.tight_layout()
plt.show()

# 7️⃣ Top 10 volatilis coin
top_vol = df.groupby('Currency')['volatility'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(8, 4))
sns.barplot(x=top_vol.values, y=top_vol.index, orient='h')
plt.title('Top 10 legvolatilisabb kriptovaluta')
plt.xlabel('Átlagos volatilitás')
plt.tight_layout()
plt.show()
