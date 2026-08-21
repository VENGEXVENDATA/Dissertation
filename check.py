import pandas as pd

df = pd.read_csv("data/processed/master_methodology_trends.csv")
df["date"] = pd.to_datetime(df["date"])

df = df[
    ~((df["market"] == "sp500") & (df["date"] < "2014-01-01"))
]

for market in df["market"].unique():
    n = (
        df[df["market"] == market]
        [["date", "ari_stability", "market_volatility"]]
        .drop_duplicates()
        .dropna()
        .shape[0]
    )
    print(market, "observations:", n)