# ======================================
# STEP 1-6 SAME AS BEFORE (LOAD, CLEAN, NORMALIZE)
# ======================================

import pandas as pd
import numpy as np

file_path = "Tourism_Long_Format.csv"
df = pd.read_csv(file_path)

df.columns = df.columns.str.strip()
df = df.drop_duplicates()
df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"], errors="coerce")
df = df.dropna(subset=["TIME_PERIOD"])
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

# Keep monthly data
df = df[df["freq"] == "M"]

# Include both domestic and foreign, all sectors
# Encode later if needed
# df = df[df["c_resid"].isin(["DOM","FOR"])]
# df = df[df["nace_r2"].notna()]

df = df.sort_values(["geo","TIME_PERIOD"])

df["Value"] = df.groupby("geo")["Value"].ffill()
df["Value"] = df.groupby("geo")["Value"].bfill()

df["Value_scaled"] = df.groupby("geo")["Value"].transform(
    lambda x: (x - x.min()) / (x.max() - x.min())
)

df["month"] = df["TIME_PERIOD"].dt.month
df["quarter"] = df["TIME_PERIOD"].dt.quarter
df["year"] = df["TIME_PERIOD"].dt.year
df["peak_season"] = df["month"].isin([6,7,8]).astype(int)

# ======================================
# STEP 7: SAFE LAGS
# ======================================

min_length = df.groupby("geo").size().min()
lags = [1,2,3] if min_length >= 3 else [1]

for lag in lags:
    df[f"lag_{lag}"] = df.groupby("geo")["Value_scaled"].shift(lag)

# ======================================
# STEP 8: ROLLING FEATURES
# ======================================

df["rolling_mean_3"] = df.groupby("geo")["Value_scaled"].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)

df["rolling_std_3"] = df.groupby("geo")["Value_scaled"].transform(
    lambda x: x.rolling(window=3, min_periods=1).std().fillna(0)
)

df["trend_diff"] = df.groupby("geo")["Value_scaled"].diff().fillna(0)
df["growth_rate"] = df.groupby("geo")["Value_scaled"].pct_change().fillna(0)

# ======================================
# STEP 9: CREATE TARGET FOR NEXT MONTH
# ======================================

df["target"] = df.groupby("geo")["Value_scaled"].shift(-1)

# Remove only rows without target
df = df.dropna(subset=["target"]).reset_index(drop=True)

# ======================================
# STEP 10: SELECT FEATURES
# ======================================

feature_columns = [
    "lag_1","lag_2","lag_3",
    "rolling_mean_3","rolling_std_3",
    "trend_diff","growth_rate",
    "month","quarter","peak_season"
]

X = df[feature_columns]
y = df["target"]

print("Final dataset shape:", df.shape)
print("Number of rows:", len(df))
print("Features preview:\n", X.head())

df.to_csv("tourism_model_ready_more_data.csv", index=False)
print("✅ More data feature-engineered dataset saved.")
