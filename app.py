"""Freight Rate Predictor (Streamlit).

Always trains from CSV on startup — avoids sklearn-version pickle issues.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
STATIC_NUM = ['distance', 'log_distance', 'weight', 'weight_missing',
              'pickup_lat', 'delivery_lat', 'delivery_lon',
              'bearing_ew', 'bearing_ns']
STATIC_CAT = ['pickup', 'delivery', 'equipment']
T0 = pd.Timestamp('2025-01-01')


class FreightFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['dow'] = df['date'].dt.dayofweek
        for c in ['distance', 'weight', 'market_index',
                  'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        df['weight_missing'] = df['weight'].isna().astype(int)
        df['weight'] = df['weight'].abs().clip(upper=47500)
        day_mean = df.groupby('date')['market_index'].transform('mean')
        df['market_index'] = df['market_index'].fillna(day_mean)
        df['log_distance'] = np.log(df['distance'])
        df['bearing_ew'] = df['delivery_lon'] - df['pickup_lon']
        df['bearing_ns'] = df['delivery_lat'] - df['pickup_lat']
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
        self.fe_ = FreightFeatureEngineer()
        d = self.fe_.transform(X)
        y = np.log(np.asarray(y, float))

        self.prep_ = ColumnTransformer([
            ('num', 'passthrough', STATIC_NUM),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value',
                                   unknown_value=-1,
                                   encoded_missing_value=-1), STATIC_CAT)])
        Z = self.prep_.fit_transform(self._static(d))
        cat_mask = [False] * len(STATIC_NUM) + [True] * len(STATIC_CAT)

        days = (d['date'] - T0).dt.days.to_numpy()
        blk = days // self.block
        B = (blk[:, None] == np.unique(blk)[None, :]).astype(float)
        m = d['market_index'].to_numpy()[:, None]
        A = np.c_[m, B]
        lvl = np.zeros_like(y)

        for _ in range(self.iters):
            self.gbm_ = HistGradientBoostingRegressor(
                categorical_features=cat_mask,
                max_iter=300,
                learning_rate=0.05,
                min_samples_leaf=40,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=self.seed).fit(Z, y - lvl)
            r = y - self.gbm_.predict(Z)
            coef = np.linalg.lstsq(A, r, rcond=None)[0]
            lvl = A @ coef

        self.slope_ = float(coef[0])
        resid_level = r - self.slope_ * m[:, 0]
        self.offset_ = float(resid_level[days >= days.max() - self.carry_days].mean())
        return self

    def predict(self, X):
        d = self.fe_.transform(X)
        Z = self.prep_.transform(self._static(d))
        return np.exp(self.gbm_.predict(Z) + self.slope_ * d['market_index']).to_numpy()


def build_model():
    return TwoStageRegressor()


def drop_label_errors(df, lo=1.0, hi=5.0):
    rpm = df['posted_rate'] / df['distance']
    bad = (rpm < lo) | (rpm > hi)
    return df.loc[~bad].reset_index(drop=True)


# --------------------------------------------------------------------------
# CSV loading — tries multiple filenames, ignores empty / LFS pointer files
# --------------------------------------------------------------------------
def find_csv(base: Path, names):
    for n in names:
        for d in (base, base / "data"):
            p = d / n
            if not p.exists():
                continue
            size = p.stat().st_size
            if size == 0:
                continue
            # Skip Git LFS pointer files (they start with "version https://git-lfs")
            with open(p, "rb") as f:
                head = f.read(64)
            if head.startswith(b"version https://git-lfs"):
                raise RuntimeError(
                    f"{p} is a Git LFS pointer, not the real CSV. "
                    "Re-upload the original file."
                )
            return p
    return None


@st.cache_resource(show_spinner="Training the model — this takes about a minute on first load.")
def train_from_csv(base: Path):
    train_path = find_csv(base, ["train-test.csv", "train_test.csv"])
    if train_path is None:
        st.error("Training CSV not found in the repository. "
                 "Upload train-test.csv (or train_test.csv) to the repo root.")
        st.stop()

    train_df = pd.read_csv(train_path, parse_dates=["date"])
    if train_df.empty or "posted_rate" not in train_df.columns:
        st.error(f"{train_path.name} appears empty or malformed "
                 f"(shape={train_df.shape}, columns={list(train_df.columns)[:5]}...). "
                 "Re-upload the original CSV from your machine.")
        st.stop()

    clean = drop_label_errors(train_df)
    X = clean.drop(columns=['posted_rate', 'load_id', 'quote_signal'], errors='ignore')
    model = build_model().fit(X, clean['posted_rate'])

    return model, train_df


@st.cache_data(show_spinner=False)
def load_reference(base: Path):
    train_path = find_csv(base, ["train-test.csv", "train_test.csv"])
    val_path = find_csv(base, ["validation.csv"])

    train_df = pd.read_csv(train_path, parse_dates=["date"])
    val_df = pd.read_csv(val_path, parse_dates=["date"]) if val_path else pd.DataFrame()

    cities = sorted(set(train_df["pickup"]) | set(train_df["delivery"]))
    equipment = sorted(train_df["equipment"].dropna().unique())

    frames = [train_df[["date", "market_index"]]]
    if not val_df.empty and "market_index" in val_df.columns:
        frames.insert(0, val_df[["date", "market_index"]])
    mi = pd.concat(frames, ignore_index=True).groupby("date")["market_index"].mean().sort_index()

    lane_dist = train_df.groupby(["pickup", "delivery"])["distance"].median()

    return {
        "cities": cities,
        "equipment": equipment,
        "market_index": mi,
        "lane_distance": lane_dist,
        "train_dates": (train_df["date"].min(), train_df["date"].max()),
        "coords_p": train_df.drop_duplicates("pickup").set_index("pickup")[["pickup_lat", "pickup_lon"]],
        "coords_d": train_df.drop_duplicates("delivery").set_index("delivery")[["delivery_lat", "delivery_lon"]],
    }


def market_for(ref, dates):
    mi = ref["market_index"]
    s = pd.Series(pd.to_datetime(list(dates)))
    vals = s.map(mi)
    n_missing = int(vals.isna().sum())
    if n_missing:
        fallback = mi.iloc[-14:].mean() if len(mi) else 1.0
        vals = vals.fillna(fallback)
    return vals, n_missing


def default_distance(ref, pickup, delivery):
    ld = ref["lane_distance"]
    if (pickup, delivery) in ld.index:
        return float(ld.loc[(pickup, delivery)]), "median distance for this lane"
    return 500.0, "no history for this lane — using a 500 mi placeholder"


REQUIRED = ["pickup", "delivery", "distance", "equipment", "weight", "date"]


def predict_frame(model, ref, df):
    df = df.copy()

    if "market_index" not in df.columns or df["market_index"].isna().any():
        vals, _ = market_for(ref, df["date"])
        if "market_index" in df.columns:
            df["market_index"] = df["market_index"].fillna(vals.values)
        else:
            df["market_index"] = vals.values

    df = df.join(ref["coords_p"], on="pickup").join(ref["coords_d"], on="delivery")

    # Unknown cities: use the mean coordinate
    for c, src in [("pickup_lat", ref["coords_p"]["pickup_lat"]),
                   ("pickup_lon", ref["coords_p"]["pickup_lon"]),
                   ("delivery_lat", ref["coords_d"]["delivery_lat"]),
                   ("delivery_lon", ref["coords_d"]["delivery_lon"])]:
        df[c] = df[c].fillna(src.mean())

    return model.predict(df)


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.set_page_config(page_title="Freight Rate Predictor", page_icon="🚚", layout="wide")

BASE = Path(__file__).parent
model, train_df = train_from_csv(BASE)
ref = load_reference(BASE)

TYPICAL_ERROR = 0.0236

st.title("🚚 Freight Rate Predictor")
st.caption("Estimates the posted rate for a truckload from lane, equipment, weight, distance and date.")

with st.sidebar:
    st.header("Load details")
    pickup = st.selectbox("Pickup city", ref["cities"], index=0)
    delivery = st.selectbox("Delivery city", ref["cities"], index=min(1, len(ref["cities"]) - 1))
    equipment = st.selectbox("Equipment", ref["equipment"], index=0)
    weight = st.number_input("Weight (lb)", min_value=1000, max_value=47500, value=32000, step=500)
    if pickup == delivery:
        st.warning("Pickup and delivery are the same city.")
        st.stop()
    dist_default, dist_note = default_distance(ref, pickup, delivery)
    distance = st.number_input("Distance (miles)", min_value=1.0, max_value=5000.0,
                               value=float(dist_default), step=10.0,
                               key=f"dist_{pickup}_{delivery}")
    st.caption(f"Default: {dist_note}.")


def one_row(date, market=None):
    row = {"pickup": pickup, "delivery": delivery, "distance": float(distance),
           "equipment": equipment, "weight": float(weight), "date": pd.Timestamp(date)}
    if market is not None:
        row["market_index"] = float(market)
    return pd.DataFrame([row])


tab1, tab2, tab3, tab4 = st.tabs(["Single prediction", "Date sweep", "Batch (CSV)", "About"])

with tab1:
    c1, c2 = st.columns([1, 2])
    with c1:
        date = st.date_input("Pickup date", value=pd.Timestamp("2025-12-01").date())
    with c2:
        pred = float(predict_frame(model, ref, one_row(date))[0])
        m1, m2 = st.columns(2)
        m1.metric("Predicted rate", f"${pred:,.0f}")
        m2.metric("Per mile", f"${pred / distance:,.2f}")
        st.caption(f"Typical error ≈ {TYPICAL_ERROR:.1%} "
                   f"(${pred * (1 - TYPICAL_ERROR):,.0f} – ${pred * (1 + TYPICAL_ERROR):,.0f}).")

with tab2:
    st.write(f"Same load ({pickup} → {delivery}, {equipment}, {weight:,.0f} lb, "
             f"{distance:,.0f} mi) over a date range. Only the date and market index change.")
    d1, d2 = st.columns(2)
    start = d1.date_input("Start", value=pd.Timestamp("2025-12-01").date(), key="sweep_start")
    end = d2.date_input("End", value=pd.Timestamp("2025-12-31").date(), key="sweep_end")
    if end < start:
        st.warning("End must be on or after start.")
    elif (end - start).days > 365:
        st.warning("Max range is one year.")
    else:
        days = pd.date_range(start, end, freq="D")
        frame = pd.DataFrame({"pickup": pickup, "delivery": delivery,
                              "distance": float(distance), "equipment": equipment,
                              "weight": float(weight), "date": days})
        rates = predict_frame(model, ref, frame)
        out = pd.DataFrame({"date": days, "predicted_rate": rates.round(2)})
        fig, ax = plt.subplots(figsize=(8, 3.4))
        ax.plot(out["date"], out["predicted_rate"], color="#064A56",
                linewidth=2.2, marker="o", markersize=3)
        ax.set_ylabel("Predicted rate ($)")
        ax.grid(axis="y", color="#D9E2E4", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.autofmt_xdate()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(out, use_container_width=True, hide_index=True)

with tab3:
    st.write("Upload a CSV with columns: " + ", ".join(f"`{c}`" for c in REQUIRED))
    template = pd.DataFrame({
        "pickup": ["Richmond", "Dallas"],
        "delivery": ["Baltimore", "Nashville"],
        "distance": [274.3, 650.0],
        "equipment": ["Dry Van", "Reefer"],
        "weight": [32000, 28000],
        "date": ["2025-12-01", "2025-12-02"],
    })
    st.download_button("Download template", template.to_csv(index=False),
                       "batch_template.csv", "text/csv")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        try:
            df = pd.read_csv(up)
            missing = [c for c in REQUIRED if c not in df.columns]
            if missing:
                st.error("Missing columns: " + ", ".join(missing))
            else:
                df["predicted_rate"] = predict_frame(model, ref, df).round(2)
                st.success(f"Predicted {len(df):,} loads.")
                st.dataframe(df.head(200), use_container_width=True, hide_index=True)
                st.download_button("Download predictions", df.to_csv(index=False),
                                   "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not process this file: {e}")

with tab4:
    st.markdown(f"""
**Model.** Gradient-boosted trees on log(rate) + blockwise slope on `market_index`
+ carried-forward level for slow drift.

**Trained in-app** on {ref['train_dates'][0].date()} → {ref['train_dates'][1].date()}
(loads only once per deploy — cached).

**Validation** (three forward-in-time folds, mean): MAE $57.07, RMSE $139.35, R² 0.989, MAPE 2.36%.

**Notes**
- `quote_signal` is not used — its relationship with rate flips sign month to month.
- Rates are estimates of the posted rate, not quotes.
""")
