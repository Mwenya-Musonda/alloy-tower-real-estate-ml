import glob
import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="AlloyTower Dashboard", layout="wide")


# PATHS (robust for /app folder)

APP_DIR = os.path.dirname(os.path.abspath(__file__))     
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))  

LISTINGS_DIR = os.path.join(ROOT_DIR, "ml", "data")
RISK_CSV_PATH = os.path.join(ROOT_DIR, "reports", "dom_risk_scoring_output.csv")


@st.cache_data
def load_latest_listings(data_dir=LISTINGS_DIR):
    files = sorted(glob.glob(os.path.join(data_dir, "clean_sales_listings_*.csv")))
    if not files:
        return None, None
    path = files[-1]
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    return df, path


@st.cache_data
def load_risk_output(path=RISK_CSV_PATH):
    if os.path.exists(path):
        out = pd.read_csv(path)
        out.columns = [c.lower() for c in out.columns]
        return out
    return None


def kpi_card(label, value):
    st.markdown(
        f"""
        <div style="padding:14px;border-radius:12px;background:#111827;color:white">
            <div style="font-size:12px;opacity:0.85">{label}</div>
            <div style="font-size:28px;font-weight:700">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# LOAD DATA

df, listings_path = load_latest_listings()

st.title("AlloyTower Real Estate Dashboard (Listings + DOM Risk)")

if df is None:
    st.error(f"No listings CSV found in: {LISTINGS_DIR}\n\nRun ingest first (to generate clean_sales_listings_*.csv).")
    st.stop()

st.caption(f"Loaded: {listings_path}")

# Ensure key columns exist + clean types
for col in ["county", "city", "property_type", "listing_id"]:
    if col not in df.columns:
        df[col] = np.nan

# IMPORTANT: make listing_id always a STRING (fixes merge issues later)
df["listing_id"] = df["listing_id"].astype(str).str.strip()

df["listed_date"] = pd.to_datetime(df.get("listed_date"), errors="coerce")
df["days_on_market"] = pd.to_numeric(df.get("days_on_market"), errors="coerce")
df["current_price"] = pd.to_numeric(df.get("current_price"), errors="coerce")
df["hoa_fee"] = pd.to_numeric(df.get("hoa_fee"), errors="coerce")
df["price_per_sq_ft"] = pd.to_numeric(df.get("price_per_sq_ft"), errors="coerce")


# SIDEBAR FILTERS

st.sidebar.header("Filters")

counties = ["All"] + sorted(df["county"].dropna().astype(str).unique().tolist())
county = st.sidebar.selectbox("County", counties)

filtered = df.copy()
if county != "All":
    filtered = filtered[filtered["county"].astype(str) == county]

cities = ["All"] + sorted(filtered["city"].dropna().astype(str).unique().tolist())
city = st.sidebar.selectbox("City", cities)

if city != "All":
    filtered = filtered[filtered["city"].astype(str) == city]

ptype = ["All"] + sorted(filtered["property_type"].dropna().astype(str).unique().tolist())
property_type = st.sidebar.selectbox("Property Type", ptype)

if property_type != "All":
    filtered = filtered[filtered["property_type"].astype(str) == property_type]

# Price slider safety (if NaNs exist)
min_price = float(np.nanmin(filtered["current_price"].values)) if filtered["current_price"].notna().any() else 0.0
max_price = float(np.nanmax(filtered["current_price"].values)) if filtered["current_price"].notna().any() else 0.0

if min_price > max_price:
    min_price, max_price = 0.0, 0.0

price_range = st.sidebar.slider(
    "Current Price Range",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price)
)

filtered = filtered[
    (filtered["current_price"].fillna(0) >= price_range[0]) &
    (filtered["current_price"].fillna(0) <= price_range[1])
]


# KPIs

total_listings = len(filtered)
num_counties = filtered["county"].nunique()

avg_price = filtered["current_price"].mean()
avg_dom = filtered["days_on_market"].mean()
median_ppsf = filtered["price_per_sq_ft"].median()
avg_hoa = filtered["hoa_fee"].mean()

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: kpi_card("Total Listings", f"{total_listings:,}")
with c2: kpi_card("Unique Counties", f"{num_counties:,}")
with c3: kpi_card("Avg Price", f"${avg_price:,.0f}" if not np.isnan(avg_price) else "N/A")
with c4: kpi_card("Avg DOM", f"{avg_dom:,.0f} days" if not np.isnan(avg_dom) else "N/A")
with c5: kpi_card("Median PPSF", f"${median_ppsf:,.0f}" if not np.isnan(median_ppsf) else "N/A")
with c6: kpi_card("Avg HOA Fee", f"${avg_hoa:,.0f}" if not np.isnan(avg_hoa) else "N/A")

st.divider()


# CHARTS

left, right = st.columns(2)

with left:
    st.subheader("Top Counties by Listing Count")
    county_counts = filtered["county"].astype(str).value_counts().head(10)
    fig = plt.figure()
    plt.bar(county_counts.index.astype(str), county_counts.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Listings")
    st.pyplot(fig)

with right:
    st.subheader("Average Price by Property Type")
    avg_price_pt = (
        filtered.groupby(filtered["property_type"].astype(str))["current_price"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    fig = plt.figure()
    plt.bar(avg_price_pt.index.astype(str), avg_price_pt.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Avg Price")
    st.pyplot(fig)

st.divider()


# RISK OUTPUT
# -----------------------------
# RISK OUTPUT
# -----------------------------
st.subheader("High-Risk Listings (DOM Prediction Risk)")

risk_out = load_risk_output()

if risk_out is None:
    st.warning(f"No risk file found yet at:\n{RISK_CSV_PATH}\n\nRun the risk scoring notebook cell and save it.")
else:
    # Normalize column names
    risk_out.columns = [c.lower().strip() for c in risk_out.columns]

    # Try to detect the ID column (common variations)
    possible_id_cols = ["listing_id", "listingid", "id", "listing id", "listing-id"]
    id_col = None
    for c in possible_id_cols:
        if c in risk_out.columns:
            id_col = c
            break

    if id_col is None:
        st.error(
            "Risk file is missing a listing id column.\n\n"
            f"Columns found in risk file: {risk_out.columns.tolist()}\n\n"
            "Fix: make sure your risk scoring CSV includes listing_id."
        )
        st.stop()

    # Rename detected id column to listing_id for consistency
    if id_col != "listing_id":
        risk_out = risk_out.rename(columns={id_col: "listing_id"})

    # FORCE listing_id to string on both sides (prevents merge dtype errors)
    risk_out["listing_id"] = risk_out["listing_id"].astype(str).str.strip()
    df["listing_id"] = df["listing_id"].astype(str).str.strip()

    # Merge: bring county/city/type/price into risk_out
    meta_cols = ["listing_id", "county", "city", "property_type", "current_price"]
    merged = risk_out.merge(df[meta_cols], on="listing_id", how="left")

    # Apply same filters
    if county != "All":
        merged = merged[merged["county"].astype(str) == county]
    if city != "All":
        merged = merged[merged["city"].astype(str) == city]
    if property_type != "All":
        merged = merged[merged["property_type"].astype(str) == property_type]

    merged = merged[
        (merged["current_price"].fillna(0) >= price_range[0]) &
        (merged["current_price"].fillna(0) <= price_range[1])
    ]

    # Sort by risk score if exists
    if "risk_score_0_100" in merged.columns:
        merged = merged.sort_values("risk_score_0_100", ascending=False)

    # Display table (only show columns that actually exist)
    base_cols = ["listing_id", "county", "city", "property_type", "current_price", "pred_dom", "risk_level"]
    optional_cols = ["risk_score_0_100", "recommended_action"]
    show_cols = [c for c in base_cols if c in merged.columns] + [c for c in optional_cols if c in merged.columns]

    st.dataframe(merged[show_cols].head(50))

    #...........
    import os
import glob
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AlloyTower Predictions", layout="wide")


# Helpers

@st.cache_data
def load_latest_listings(data_dir="ml/data/"):
    files = sorted(glob.glob(os.path.join(data_dir, "clean_sales_listings_*.csv")))
    if not files:
        return None, None
    path = files[-1]
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    return df, path

@st.cache_resource
def load_models():
    dom_model = joblib.load("models/dom_model.joblib")
    price_model = joblib.load("models/price_model.joblib")
    feature_cols = joblib.load("models/feature_columns.joblib")
    return dom_model, price_model, feature_cols

def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default

# Load data + models

df, listings_path = load_latest_listings()

st.title("AlloyTower Live Predictions Dashboard")

if df is None:
    st.error("No listings CSV found in ml/data/. Run ingest first.")
    st.stop()

st.caption(f"Loaded listings file: {listings_path}")

# Models
try:
    dom_model, price_model, feature_cols = load_models()
except Exception as e:
    st.error("Models not found. Train and save models first into /models.")
    st.code(str(e))
    st.stop()

# Basic cleanup
for c in ["current_price", "hoa_fee", "price_per_sq_ft", "square_footage", "lot_size", "bedrooms", "bathrooms", "year_built", "zip_code", "latitude", "longitude"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")


# Sidebar: Choose input method

st.sidebar.header("Prediction Mode")

mode = st.sidebar.radio(
    "How do you want to create inputs?",
    ["Pick an existing listing", "Manual input (scenario)"]
)

st.sidebar.divider()

# Build input row
input_row = {}

if mode == "Pick an existing listing":
    # Filters to narrow down
    county = st.sidebar.selectbox("County", ["All"] + sorted(df["county"].dropna().unique().tolist()))
    filtered = df.copy()
    if county != "All":
        filtered = filtered[filtered["county"] == county]

    city = st.sidebar.selectbox("City", ["All"] + sorted(filtered["city"].dropna().unique().tolist()))
    if city != "All":
        filtered = filtered[filtered["city"] == city]

    ptype = st.sidebar.selectbox("Property Type", ["All"] + sorted(filtered["property_type"].dropna().unique().tolist()))
    if ptype != "All":
        filtered = filtered[filtered["property_type"] == ptype]

    st.sidebar.write(f"Matching listings: {len(filtered):,}")

    # Pick a listing
    if len(filtered) == 0:
        st.warning("No listings match your filters.")
        st.stop()

    listing_id = st.sidebar.selectbox("Choose listing_id", filtered["listing_id"].head(500).tolist())

    row = filtered[filtered["listing_id"] == listing_id].iloc[0].to_dict()

    # Convert listed_date → listed_year/month/dayofweek if your model expects them
    listed_date = pd.to_datetime(row.get("listed_date"), errors="coerce")
    row["listed_year"] = getattr(listed_date, "year", np.nan)
    row["listed_month"] = getattr(listed_date, "month", np.nan)
    row["listed_dayofweek"] = getattr(listed_date, "dayofweek", np.nan)

    input_row = row

else:
    st.sidebar.subheader("Manual scenario inputs")

    input_row["county"] = st.sidebar.text_input("County (e.g., Travis)", "Travis")
    input_row["city"] = st.sidebar.text_input("City (e.g., Austin)", "Austin")
    input_row["property_type"] = st.sidebar.selectbox("Property Type", sorted(df["property_type"].dropna().unique().tolist()))
    input_row["listing_type"] = st.sidebar.selectbox("Listing Type", sorted(df["listing_type"].dropna().unique().tolist()))
    input_row["unit"] = st.sidebar.text_input("Unit (or 'Unknown')", "Unknown")

    input_row["zip_code"] = st.sidebar.number_input("Zip Code", value=75000, step=1)
    input_row["latitude"] = st.sidebar.number_input("Latitude", value=30.0)
    input_row["longitude"] = st.sidebar.number_input("Longitude", value=-97.0)

    input_row["bedrooms"] = st.sidebar.number_input("Bedrooms", value=3, step=1)
    input_row["bathrooms"] = st.sidebar.number_input("Bathrooms", value=2, step=1)
    input_row["square_footage"] = st.sidebar.number_input("Square Footage", value=1800, step=10)
    input_row["lot_size"] = st.sidebar.number_input("Lot Size", value=7000, step=10)
    input_row["year_built"] = st.sidebar.number_input("Year Built", value=2005, step=1)

    input_row["hoa_fee"] = st.sidebar.number_input("HOA Fee", value=0, step=10)
    input_row["price_per_sq_ft"] = st.sidebar.number_input("PPSF (optional)", value=0.0)

    # This is your “scenario year”
    input_row["listed_year"] = st.sidebar.number_input("Listed Year (scenario)", value=2026, step=1)
    input_row["listed_month"] = st.sidebar.slider("Listed Month", 1, 12, 1)
    input_row["listed_dayofweek"] = st.sidebar.slider("Listed Day of Week", 0, 6, 0)

    # For price model we still need a price feature? We DO NOT.
    # But for DOM model you included log_price earlier; in the saved model code we didn't require it unless it exists in feature_cols.
    # If feature_cols requires current_price, provide it:
    input_row["current_price"] = st.sidebar.number_input("Current Price (if required by model features)", value=400000, step=1000)


# Build prediction DataFrame

# Create a single-row df with EXACT feature columns used in training
X_pred = pd.DataFrame([{c: input_row.get(c, np.nan) for c in feature_cols}])

# Replace inf and fill missing like training
X_pred = X_pred.replace([np.inf, -np.inf], np.nan)

num_cols_pred = X_pred.select_dtypes(include=["int64", "float64"]).columns
cat_cols_pred = X_pred.select_dtypes(include=["object", "category", "string"]).columns

X_pred[num_cols_pred] = X_pred[num_cols_pred].fillna(X_pred[num_cols_pred].median())
X_pred[cat_cols_pred] = X_pred[cat_cols_pred].fillna("Unknown")


# Predict

colA, colB = st.columns(2)

with colA:
    st.subheader("Predicted Days on Market (DOM)")
    pred_dom = float(dom_model.predict(X_pred)[0])
    st.metric("Predicted DOM", f"{pred_dom:.0f} days")

    # Simple risk label
    if pred_dom <= 60:
        risk = "Low"
    elif pred_dom <= 120:
        risk = "Medium"
    else:
        risk = "High"
    st.write(f"Risk level: **{risk}**")

with colB:
    st.subheader("Predicted Price (Expected)")
    pred_price = float(price_model.predict(X_pred)[0])
    st.metric("Predicted Price", f"${pred_price:,.0f}")

st.divider()

st.subheader("Input used for prediction")
st.dataframe(X_pred)

st.divider()

st.subheader("Quick KPIs (current dataset)")
k1, k2, k3, k4 = st.columns(4)
with k1: st.metric("Total Listings", f"{len(df):,}")
with k2: st.metric("Counties", f"{df['county'].nunique():,}")
with k3: st.metric("Avg Current Price", f"${df['current_price'].mean():,.0f}")
with k4: st.metric("Avg DOM", f"{pd.to_numeric(df['days_on_market'], errors='coerce').mean():,.0f}")

