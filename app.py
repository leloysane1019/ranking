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

# モデル読み込み
model = joblib.load("stock_up_model.pkl")

# 銘柄リスト（100件まで拡張可）
tickers = [
    "7203.T", "6758.T", "9984.T", "8306.T", "8035.T",
    "6861.T", "9432.T", "8766.T", "9020.T", "4502.T",
    "8591.T", "6098.T", "2802.T", "4063.T", "9983.T",
    "4452.T", "7974.T", "2502.T", "9433.T", "2413.T"
]

# 特徴量を計算する関数
def compute_features(df):
    df = df.copy()
    close = df["Close"].squeeze()  # Seriesとして扱う

    df["ma5"] = close.rolling(window=5).mean()
    df["ma25"] = close.rolling(window=25).mean()
    df["dis_ma5"] = (close - df["ma5"]) / df["ma5"]
    df["dis_ma25"] = (close - df["ma25"]) / df["ma25"]
    rolling_std = close.rolling(window=20).std()
    df["bb_width"] = (rolling_std * 2) / close
    df["price_range"] = (df["High"].rolling(window=20).max() - df["Low"].rolling(window=20).min()) / df["Low"].rolling(window=20).min()
    df["vol_ratio"] = df["Volume"].rolling(window=10).mean() / df["Volume"].rolling(window=20).mean()

    if len(close) > 1:
        X = np.arange(len(close)).reshape(-1, 1)
        y = close.values.reshape(-1, 1)
        reg = LinearRegression().fit(X, y)
        df["trend_slope"] = reg.coef_[0][0]
    else:
        df["trend_slope"] = np.nan

    return df

@app.get("/", response_class=HTMLResponse)
def show_ranking(request: Request):
    results = []
    required_cols = ['trend_slope', 'dis_ma5', 'dis_ma25', 'bb_width', 'price_range', 'vol_ratio']

    for ticker in tickers:
        try:
            print("処理中:", ticker)
            df = yf.download(ticker, period="120d", interval="1d")  # データ期間を延長
            if df.empty:
                print(f"{ticker}: データなし")
                continue

            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df = compute_features(df)

            # 欠損列のチェックと表示（改良済み）
            missing_cols = df[required_cols].isna().any()
            if missing_cols.any():
                missing_list = [col for col, is_missing in missing_cols.items() if is_missing]
                print(f"{ticker} 欠損列: {missing_list}")

            df = df.dropna(subset=required_cols)
            if df.empty:
                print(f"{ticker}: 有効な行がありません")
                continue

            X = df[required_cols].iloc[-1:]
            prob = model.predict(X)[0]
            results.append({"ticker": ticker, "probability": round(prob * 100, 2)})

        except Exception as e:
            print(f"エラー（{ticker}）: {e}")
            continue

    top5 = sorted(results, key=lambda x: x["probability"], reverse=True)[:5]
    print("予測結果:", top5)
    return templates.TemplateResponse("ranking.html", {"request": request, "ranking": top5})