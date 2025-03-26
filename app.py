from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 学習済みモデル読み込み（Booster型でもOK）
model = joblib.load("stock_up_model.pkl")

# 対象銘柄（必要に応じて100銘柄まで拡張）
tickers = [
    "7203.T", "6758.T", "9984.T", "8306.T", "8035.T",
    "6861.T", "9432.T", "8766.T", "9020.T", "4502.T",
    "8591.T", "6098.T", "2802.T", "4063.T", "9983.T",
    "4452.T", "7974.T", "2502.T", "9433.T", "2413.T"
]

# 特徴量計算関数（エラー回避込み）
def compute_features(df):
    df = df.copy()

    close_series = df['Close'].squeeze()  # ← Seriesとして強制的に取得

    df['ma5'] = close_series.rolling(window=5).mean()
    df['ma25'] = close_series.rolling(window=25).mean()
    df['dis_ma5'] = (close_series - df['ma5']) / df['ma5']
    df['dis_ma25'] = (close_series - df['ma25']) / df['ma25']

    rolling_std = close_series.rolling(window=20).std()
    df['bb_width'] = (rolling_std * 2) / close_series

    df['price_range'] = (df['High'].rolling(window=20).max() - df['Low'].rolling(window=20).min()) / df['Low'].rolling(window=20).min()
    df['vol_ratio'] = df['Volume'].rolling(window=10).mean() / df['Volume'].rolling(window=20).mean()

    if len(close_series) > 1:
        X = np.arange(len(close_series)).reshape(-1, 1)
        y = close_series.values.reshape(-1, 1)
        reg = LinearRegression().fit(X, y)
        df['trend_slope'] = reg.coef_[0][0]
    else:
        df['trend_slope'] = np.nan

    return df
@app.get("/", response_class=HTMLResponse)
def show_ranking(request: Request):
    results = []

    for ticker in tickers:
        try:
            print("処理中:", ticker)
            df = yf.download(ticker, period="60d", interval="1d")
            if df.empty:
                continue

            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df = compute_features(df)
            df = df.dropna(subset=[
                'trend_slope', 'dis_ma5', 'dis_ma25', 'bb_width', 'price_range', 'vol_ratio'
            ])
            if df.empty:
                continue

            X = df[['trend_slope', 'dis_ma5', 'dis_ma25', 'bb_width', 'price_range', 'vol_ratio']].iloc[-1:]
            prob = model.predict(X)[0]  # Booster型の場合 predict_proba は使わない
            results.append({"ticker": ticker, "probability": round(prob * 100, 2)})

        except Exception as e:
            print(f"エラー（{ticker}）: {e}")
            continue

    # 上昇確率の高い順にソートしてTOP5を抽出
    top5 = sorted(results, key=lambda x: x["probability"], reverse=True)[:5]
    print("予測結果:", top5)

    return templates.TemplateResponse("ranking.html", {
        "request": request,
        "ranking": top5
    })