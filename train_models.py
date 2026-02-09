import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

#Build training dataset

train_df = df.copy()

# Targets
y_dom = train_df["days_on_market_capped"]
y_price = train_df["current_price"]

# Drop columns we don't want as features
drop_cols = [
    "days_on_market", "days_on_market_capped",
    "listing_id", "address", "street", "mls_number",
    "agent_id", "office_id",
    "removed_date", "last_seen_ts", "status",
]

X = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])

# Replace inf and fill missing
X = X.replace([np.inf, -np.inf], np.nan)

num_cols_all = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols_all = X.select_dtypes(include=["object", "category", "string"]).columns

X[num_cols_all] = X[num_cols_all].fillna(X[num_cols_all].median())
X[cat_cols_all] = X[cat_cols_all].fillna("Unknown")

# IMPORTANT: remove raw date strings if still present
date_cols = ["listed_date", "created_date"]
X = X.drop(columns=[c for c in date_cols if c in X.columns])

# Identify cols again after drop
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns

print("Numeric:", len(num_cols), "Categorical:", len(cat_cols))
print("Categorical columns:", cat_cols.tolist())

#Preprocessor

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ]
)


#  Models 

dom_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", ExtraTreesRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    ))
])

price_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", ExtraTreesRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    ))
])


# Train/test split 
X_train, X_test, y_dom_train, y_dom_test = train_test_split(
    X, y_dom, test_size=0.2, random_state=42
)
_, _, y_price_train, y_price_test = train_test_split(
    X, y_price, test_size=0.2, random_state=42
)

# Fit
dom_model.fit(X_train, y_dom_train)
price_model.fit(X_train, y_price_train)

# Evaluate DOM
dom_pred = dom_model.predict(X_test)
mae = mean_absolute_error(y_dom_test, dom_pred)
rmse = np.sqrt(mean_squared_error(y_dom_test, dom_pred))
r2 = r2_score(y_dom_test, dom_pred)
print("\nDOM model:")
print(f"MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.3f}")

# Evaluate PRICE (use log scale if you want later; for now raw)
price_pred = price_model.predict(X_test)
p_mae = mean_absolute_error(y_price_test, price_pred)
p_rmse = np.sqrt(mean_squared_error(y_price_test, price_pred))
p_r2 = r2_score(y_price_test, price_pred)
print("\nPRICE model:")
print(f"MAE={p_mae:,.0f}, RMSE={p_rmse:,.0f}, R2={p_r2:.3f}")


# 5) Save models + feature columns

os.makedirs("../models", exist_ok=True)

joblib.dump(dom_model, "../models/dom_model.joblib")
joblib.dump(price_model, "../models/price_model.joblib")

# Save the feature column order so Streamlit can build rows correctly
joblib.dump(list(X.columns), "../models/feature_columns.joblib")

print("\n Saved:")
print(" - ../models/dom_model.joblib")
print(" - ../models/price_model.joblib")
print(" - ../models/feature_columns.joblib")
