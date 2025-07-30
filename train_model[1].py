import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import joblib

df = yf.download('AAPL', start='2020-01-01', end='2023-01-01')
df['SMA'] = df['Close'].rolling(window=14).mean()
delta = df['Close'].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = pd.Series(gain).rolling(window=14).mean()
avg_loss = pd.Series(loss).rolling(window=14).mean()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))
df.dropna(inplace=True)

X = df[['SMA', 'RSI']]
y = df['Close']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LinearRegression()
model.fit(X_scaled, y)
preds = model.predict(X_scaled)
score = r2_score(y, preds)
print(f"R² Score: {score:.2f}")

joblib.dump(model, 'model/model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')