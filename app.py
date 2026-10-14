from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingRegressor

# ============ Custom classes (previously in model_utils.py) ============
STATIC_NUM = [
    "distance", "log_distance", "weight", "weight_missing",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "bearing_ew", "bearing_ns",
]
STATIC_CAT = ["pickup", "delivery", "equipment"]
T0 = pd.Timestamp("2025-01-01")


class FreightFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["dow"] = df["date"].dt.dayofweek
        for c in ["distance", "weight", "market_index",
                  "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["weight_missing"] = df["weight"].isna().astype(int)
        df["weight"] = df["weight"].abs().clip(upper=47500)
        day_mean = df.groupby("date")["market_index"].transform("mean")
        df["market_index"] = df["market_index"].fillna(day_mean)
        df["log_distance"] = np.log(df["distance"])
        df["bearing_ew"] = df["delivery_lon"] - df["pickup_lon"]
        df["bearing_ns"] = df["delivery_lat"] - df["pickup_lat"]
        return df


class TwoStageRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, block=14, carry_days=60, iters=2, seed=42):
        self.block = block
        self.carry_days = carry_days
        self.iters = iters
        self.seed = seed

    def _static(self, d):
        return d[STATIC_NUM + STATIC_CAT]

    def fit(self, X, y):
        fe = FreightFeatureEngineer()
        self.fe_ = fe
        d = fe.transform(X)
        y = np.log(np.asarray(y, float))
        self.prep_ = ColumnTransformer([
            ("num", "passthrough", STATIC_NUM),
            ("cat", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-1), STATIC_CAT),
        ])
        Z = self.prep_.fit_transform(self._static(d))
        cat_mask = [False] * len(STATIC_NUM) + [True] * len(STATIC_CAT)
        days = (d["date"] - T0).dt.days.to_numpy()
        blk = days // self.block
        B = (blk[:, None] == np.unique(blk)[None, :]).astype(float)
        m = d["market_index"].to_numpy()[:, None]
        A = np.c_[m, B]
        lvl = np.zeros_like(y)
        for _ in range(self.iters):
            self.gbm_ = HistGradientBoostingRegressor(
                categorical_features=cat_mask,
                max_iter=300, learning_rate=0.05,
                min_samples_leaf=40, l2_regularization=1.0,
                early_stopping=False, random_state=self.seed,
            ).fit(Z, y - lvl)
            r = y - self.gbm_.predict(Z)
            coef = np.linalg.lstsq(A, r, rcond=None)[0]
            lvl = A @ coef
        self.slope_ = float(coef[0])
        resid_level = r - self.slope_ * m[:, 0]
        self.offset_ = float(
            resid_level[days >= days.max() - self.carry_days].mean()
        )
        return self

    def predict(self, X):
        d = self.fe_.transform(X)
        Z = self.prep_.transform(self._static(d))
        return np.exp(
            self.gbm_.predict(Z)
            + self.slope_ * d["market_index"].to_numpy()
            + self.offset_
        )


def build_model():
    return TwoStageRegressor()


# ============ Streamlit app ============
st.set_page_config(page_title="Freight Rate Predictor", page_icon="🚚", layout="wide")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "models" / "final_model.joblib"

EQUIPMENT_TYPES = ["Dry Van", "Reefer", "Flatbed", "Tanker", "Step Deck", "Power Only"]


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_city_lookup():
    path = DATA_DIR / "validation.csv"
    if not path.exists():
        return {"pickup": {}, "delivery": {}}
    df = pd.read_csv(path)
    p = (df.dropna(subset=["pickup_lat", "pickup_lon"])
           .drop_duplicates("pickup")
           .set_index("pickup")[["pickup_lat", "pickup_lon"]])
    d = (df.dropna(subset=["delivery_lat", "delivery_lon"])
           .drop_duplicates("delivery")
           .set_index("delivery")[["delivery_lat", "delivery_lon"]])
    return {"pickup": p.to_dict("index"), "delivery": d.to_dict("index")}


model = load_model()
city_lookup = load_city_lookup()


def make_input_row(pickup, delivery, pickup_lat, pickup_lon,
                   delivery_lat, delivery_lon, distance, equipment,
                   weight, market_index, load_date):
    return pd.DataFrame([{
        "pickup": pickup, "delivery": delivery,
        "pickup_lat": pickup_lat, "pickup_lon": pickup_lon,
        "delivery_lat": delivery_lat, "delivery_lon": delivery_lon,
        "distance": distance, "equipment": equipment,
        "weight": weight, "market_index": market_index,
        "date": pd.to_datetime(load_date),
    }])


st.sidebar.title("🚚 Freight Rate Predictor")
page = st.sidebar.radio("Navigate",
    ["Single Prediction", "Batch Upload", "December Chart", "About"])

if model is None:
    st.sidebar.error("Model not found at models/final_model.joblib")
else:
    st.sidebar.success("Model loaded ✅")
    st.sidebar.caption(
        f"Slope: **{getattr(model, 'slope_', float('nan')):.3f}**  \n"
        f"Offset: **{getattr(model, 'offset_', float('nan')):.4f}**"
    )


if page == "Single Prediction":
    st.title("Single Load Rate Prediction")
    col1, col2 = st.columns(2)
    with col1:
        pickup = st.text_input("Pickup city", "Richmond")
        delivery = st.text_input("Delivery city", "Baltimore")
        p_lat = st.number_input("Pickup latitude", value=38.09122, format="%.5f")
        p_lon = st.number_input("Pickup longitude", value=-76.78906, format="%.5f")
        d_lat = st.number_input("Delivery latitude", value=38.16908, format="%.5f")
        d_lon = st.number_input("Delivery longitude", value=-72.74564, format="%.5f")
        if st.button("Auto-fill coordinates"):
            p = city_lookup.get("pickup", {}).get(pickup)
            d = city_lookup.get("delivery", {}).get(delivery)
            if p: p_lat, p_lon = p["pickup_lat"], p["pickup_lon"]
            if d: d_lat, d_lon = d["delivery_lat"], d["delivery_lon"]
            st.success(f"Filled (pickup={bool(p)}, delivery={bool(d)})")
    with col2:
        distance = st.number_input("Distance (miles)", min_value=1.0, value=274.3)
        equipment = st.selectbox("Equipment", EQUIPMENT_TYPES)
        weight = st.number_input("Weight (lbs)", min_value=0.0, value=30658.0)
        market_index = st.number_input("Market index", value=0.95684, format="%.5f")
        load_date = st.date_input("Load date", value=date(2025, 5, 15))

    if st.button("Predict rate", type="primary") and model is not None:
        row = make_input_row(pickup, delivery, p_lat, p_lon, d_lat, d_lon,
                             distance, equipment, weight, market_index, load_date)
        try:
            pred = float(model.predict(row)[0])
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted rate", f"${pred:,.2f}")
            c2.metric("Rate per mile", f"${pred/distance:,.2f}")
            c3.metric("Distance", f"{distance:,.0f} mi")
        except Exception as e:
            st.exception(e)


elif page == "Batch Upload":
    st.title("Batch Prediction")
    uploaded = st.file_uploader("Choose CSV file", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        req = ["pickup", "delivery", "pickup_lat", "pickup_lon",
               "delivery_lat", "delivery_lon", "distance",
               "equipment", "weight", "date", "market_index"]
        missing = [c for c in req if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        elif model is None:
            st.error("Model not loaded.")
        else:
            preds = model.predict(df[req])
            out = df.copy()
            out["predicted_rate"] = np.round(preds, 2)
            st.dataframe(out.head(50), use_container_width=True)
            st.download_button("Download CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv", mime="text/csv")


elif page == "December Chart":
    st.title("December 2025 Predicted Load Rate")
    dec_path = DATA_DIR / "december_chart_inputs_filled.csv"
    if not dec_path.exists():
        st.warning("Place december_chart_inputs_filled.csv in data/")
    else:
        dec = pd.read_csv(dec_path)
        dec["date"] = pd.to_datetime(dec["date"])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(dec["date"], dec["predicted_rate"], marker="o", linewidth=2)
        ax.set_ylabel("rate ($)")
        ax.set_title("Candidate: December 2025 Predicted Load Rate")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        st.dataframe(dec, use_container_width=True)


else:
    st.title("About")
    st.markdown("""
    **Model:** Two-stage regressor — HGB on static features + market-index slope + time-block level.

    MAE ≈ 57.09 · RMSE ≈ 139.11 · R² ≈ 0.989 · MAPE ≈ 2.36%

    `quote_signal` is excluded by design.
    """)
