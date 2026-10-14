from pathlib import Path
from datetime import date
import warnings

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import InconsistentVersionWarning

# ============ Model classes (same as the notebook) ============
STATIC_NUM = ['distance', 'log_distance', 'weight', 'weight_missing', 'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon', 'bearing_ew', 'bearing_ns']
STATIC_CAT = ['pickup', 'delivery', 'equipment']
T0 = pd.Timestamp('2025-01-01')



class FreightFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['dow'] = df['date'].dt.dayofweek
        for c in ['distance', 'weight', 'market_index', 'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon']:
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
        fe = FreightFeatureEngineer()
        self.fe_ = fe
        d = fe.transform(X)
        y = np.log(np.asarray(y, float))
        self.prep_ = ColumnTransformer([('num', 'passthrough', STATIC_NUM), ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-1), STATIC_CAT)])
        Z = self.prep_.fit_transform(self._static(d))
        cat_mask = [False] * len(STATIC_NUM) + [True] * len(STATIC_CAT)
        days = (d['date'] - T0).dt.days.to_numpy()
        blk = days // self.block
        B = (blk[:, None] == np.unique(blk)[None, :]).astype(float)
        m = d['market_index'].to_numpy()[:, None]
        A = np.c_[m, B]
        lvl = np.zeros_like(y)
        for _ in range(self.iters):
            self.gbm_ = HistGradientBoostingRegressor(categorical_features=cat_mask, max_iter=300, learning_rate=0.05, min_samples_leaf=40, l2_regularization=1.0, early_stopping=False, random_state=self.seed).fit(Z, y - lvl)
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
        return np.exp(self.gbm_.predict(Z) + self.slope_ * d['market_index'].to_numpy() + self.offset_)


def build_model():
    return TwoStageRegressor()


# ============ Data helpers ============
BASE_DIR = Path(__file__).parent
REQUIRED = ["pickup", "delivery", "distance", "equipment", "weight", "date"]
TYPICAL_ERROR = 0.0236   # about the mean MAPE over the three forward-in-time folds


def find_file(*names):
    """Data may sit in data/ or next to app.py, with _ or - in the file name."""
    for n in names:
        for p in (BASE_DIR / "data" / n, BASE_DIR / n):
            if p.exists():
                return p
    return None


def drop_label_errors(df, lo=1.0, hi=5.0):
    rpm = df["posted_rate"] / df["distance"]
    return df.loc[~((rpm < lo) | (rpm > hi))].reset_index(drop=True)


def haversine_miles(lat1, lon1, lat2, lon2):
    a, b, c, d = map(np.radians, (lat1, lon1, lat2, lon2))
    x = np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(x))


@st.cache_data(show_spinner=False)
def load_reference():
    """City coordinates, typical lane distances, equipment types and the daily market index."""
    tp, vp = find_file("train_test.csv", "train-test.csv"), find_file("validation.csv")
    if tp is None:
        return None
    train = pd.read_csv(tp, parse_dates=["date"])
    frames = [train] + ([pd.read_csv(vp, parse_dates=["date"])] if vp else [])

    def roles(df):
        return pd.concat([
            df[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1),
            df[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1)])
    coords = pd.concat([roles(f) for f in frames]).groupby("city")[["lat", "lon"]].median()
    hv = haversine_miles(train.pickup_lat, train.pickup_lon, train.delivery_lat, train.delivery_lon)
    mk = pd.concat([f[["date", "market_index"]] for f in frames]).groupby("date")["market_index"].mean()
    return {
        "cities": sorted(coords.index),
        "trained_cities": sorted(set(train.pickup) | set(train.delivery)),
        "coords": {c: (float(r.lat), float(r.lon)) for c, r in coords.iterrows()},
        "lane_distance": train.groupby(["pickup", "delivery"])["distance"].median().round(0).to_dict(),
        "road_factor": float((train["distance"] / hv.replace(0, np.nan)).median()),
        "equipment": sorted(train["equipment"].unique()),
        "market": {k.strftime("%Y-%m-%d"): float(v) for k, v in mk.items()},
        "market_fallback": float(mk.iloc[-14:].mean()),
    }


def market_for(ref, dates):
    d = pd.to_datetime(pd.Series(dates)).dt.strftime("%Y-%m-%d")
    vals = d.map(ref["market"])
    return vals.fillna(ref["market_fallback"]).astype(float), int(vals.isna().sum())


def complete_frame(ref, df):
    """Add coordinates and market_index when they are not supplied."""
    f = df.copy()
    f["date"] = pd.to_datetime(f["date"])
    unknown = sorted((set(f["pickup"]) | set(f["delivery"])) - set(ref["coords"]))
    if unknown:
        raise ValueError("Unknown cities: " + ", ".join(map(str, unknown)))
    for role in ("pickup", "delivery"):
        for k, name in enumerate(("lat", "lon")):
            col = f"{role}_{name}"
            looked = f[role].map(lambda c, k=k: ref["coords"][c][k])
            f[col] = pd.to_numeric(f[col], errors="coerce").fillna(looked) if col in f.columns else looked
    if "market_index" not in f.columns:
        f["market_index"] = np.nan
    auto, _ = market_for(ref, f["date"])
    f["market_index"] = f["market_index"].fillna(auto)
    return f


def default_distance(ref, pickup, delivery):
    d = ref["lane_distance"].get((pickup, delivery))
    if d:
        return float(d)
    (la1, lo1), (la2, lo2) = ref["coords"][pickup], ref["coords"][delivery]
    return float(round(haversine_miles(la1, lo1, la2, lo2) * ref["road_factor"]))


# ============ Model loading ============
def _smoke(model):
    row = pd.DataFrame([{"pickup": "Lexington", "delivery": "Fort Wayne", "distance": 360.0,
                         "equipment": "Dry Van", "weight": 32000.0, "date": "2025-12-01",
                         "pickup_lat": 37.99, "pickup_lon": -84.5, "delivery_lat": 41.3,
                         "delivery_lon": -85.3, "market_index": 0.95}])
    p = float(model.predict(row)[0])
    if not np.isfinite(p) or not (100 < p < 5000):
        raise RuntimeError("smoke test failed")


@st.cache_resource(show_spinner="Preparing the model (the first load can take a minute)...")
def load_model():
    """Use a saved model if it loads cleanly; otherwise train the same model from the data."""
    reason = "no saved model found"
    for path in (BASE_DIR / "models" / "final_model.joblib", BASE_DIR / "freight_rate_model.joblib"):
        if not path.exists():
            continue
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = joblib.load(path)
            if any(issubclass(w.category, InconsistentVersionWarning) for w in caught):
                reason = f"{path.name} was saved with a different scikit-learn version"
                continue
            _smoke(model)
            return model, f"loaded from {path.name}"
        except Exception:
            reason = f"{path.name} could not be read on this server"
    tp = find_file("train_test.csv", "train-test.csv")
    if tp is None:
        return None, "no usable saved model and no training data found"
    train = pd.read_csv(tp, parse_dates=["date"])
    clean = drop_label_errors(train)
    model = build_model().fit(
        clean.drop(columns=["posted_rate", "load_id", "quote_signal"], errors="ignore"), clean["posted_rate"])
    return model, f"trained from the training data when the app started ({reason})"


# ============ Streamlit app ============
st.set_page_config(page_title="Freight Rate Predictor", page_icon="🚚", layout="wide")

ref = load_reference()
model, model_status = load_model()

st.sidebar.title("🚚 Freight Rate Predictor")
page = st.sidebar.radio("Navigate", ["Single Prediction", "Batch Upload", "December Chart", "About"])

if model is None or ref is None:
    st.sidebar.error("Model could not be prepared. Check that train-test.csv (or train_test.csv) is in the repo.")
    st.error("Model could not be prepared: " + model_status)
    st.stop()

st.sidebar.success("Model ready ✅")
st.sidebar.caption(f"Slope: **{model.slope_:.3f}**  \nOffset: **{model.offset_:.4f}**")


def predict_rows(df):
    return model.predict(complete_frame(ref, df))


if page == "Single Prediction":
    st.title("Single Load Rate Prediction")
    cities = ref["cities"]
    col1, col2 = st.columns(2)
    with col1:
        pickup = st.selectbox("Pickup city", cities, index=cities.index("Lexington") if "Lexington" in cities else 0)
        delivery = st.selectbox("Delivery city", cities, index=cities.index("Fort Wayne") if "Fort Wayne" in cities else 1)
        new = [c for c in (pickup, delivery) if c not in ref["trained_cities"]]
        if new:
            st.info(f"{', '.join(new)}: not in the training data. The estimate relies on coordinates and distance, so it may be a little less accurate.")
        equipment = st.selectbox("Equipment", ref["equipment"],
                                 index=ref["equipment"].index("Dry Van") if "Dry Van" in ref["equipment"] else 0)
        weight = st.number_input("Weight (lbs)", min_value=1000.0, max_value=47500.0, value=32000.0, step=500.0)
    if pickup == delivery:
        st.warning("Pickup and delivery are the same city. Pick two different cities.")
        st.stop()
    with col2:
        distance = st.number_input("Distance (miles)", min_value=1.0, max_value=5000.0,
                                   value=default_distance(ref, pickup, delivery), step=10.0,
                                   key=f"dist_{pickup}_{delivery}")
        st.caption("Starts at the typical distance for this lane; you can change it.")
        load_date = st.date_input("Load date", value=date(2025, 12, 1))
        auto_mkt, n_out = market_for(ref, [load_date])
        override = st.checkbox("Override the market index")
        market = st.number_input("Market index", min_value=0.5, max_value=2.0, value=float(round(auto_mkt.iloc[0], 3)),
                                 step=0.01, format="%.3f") if override else float(auto_mkt.iloc[0])
        if n_out and not override:
            st.info("This date is outside the data, so the latest 14-day average market index is used.")
    row = pd.DataFrame([{"pickup": pickup, "delivery": delivery, "distance": float(distance),
                         "equipment": equipment, "weight": float(weight), "date": pd.Timestamp(load_date),
                         "market_index": market}])
    pred = float(predict_rows(row)[0])
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted rate", f"${pred:,.2f}")
    c2.metric("Rate per mile", f"${pred / distance:,.2f}")
    c3.metric("Market index used", f"{market:.3f}")
    st.caption(f"Typical error in validation is about {TYPICAL_ERROR:.1%}, so roughly "
               f"\\${pred * (1 - TYPICAL_ERROR):,.0f} to \\${pred * (1 + TYPICAL_ERROR):,.0f}. This is a guide, not a guarantee.")


elif page == "Batch Upload":
    st.title("Batch Prediction")
    st.write("Upload a CSV with columns: " + ", ".join(f"`{c}`" for c in REQUIRED) +
             ". Optional: `market_index` and the four latitude/longitude columns (they are filled in automatically).")
    template = pd.DataFrame({"pickup": ["Lexington", "Dallas"], "delivery": ["Fort Wayne", "Nashville"],
                             "distance": [360, 650], "equipment": ["Dry Van", "Reefer"],
                             "weight": [32000, 28000], "date": ["2025-12-01", "2025-12-02"]})
    st.download_button("Download template", template.to_csv(index=False), "batch_template.csv", "text/csv")
    uploaded = st.file_uploader("Choose CSV file", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            missing = [c for c in REQUIRED if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                bad = df[REQUIRED].isna().any(axis=1) | ~df["equipment"].isin(ref["equipment"])
                if bad.any():
                    st.warning(f"{int(bad.sum())} row(s) have missing values or an unknown equipment type and were skipped.")
                good = df.loc[~bad].copy()
                good["predicted_rate"] = np.round(predict_rows(good), 2)
                st.success(f"Predicted {len(good):,} loads.")
                st.dataframe(good.head(200), use_container_width=True, hide_index=True)
                st.download_button("Download CSV", good.to_csv(index=False).encode("utf-8"),
                                   "predictions.csv", "text/csv")
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Could not process this file: {e}")


elif page == "December Chart":
    st.title("December 2025 Predicted Load Rate")
    st.write("Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb. Only the date changes.")
    days = pd.date_range("2025-12-01", "2025-12-31", freq="D")
    frame = pd.DataFrame({"pickup": "Lexington", "delivery": "Fort Wayne", "distance": 360.0,
                          "equipment": "Dry Van", "weight": 32000.0, "date": days})
    dec = pd.DataFrame({"date": days, "predicted_rate": np.round(predict_rows(frame), 2)})
    lo, hi = float(dec.predicted_rate.min()), float(dec.predicted_rate.max())
    pad = max(5.0, (hi - lo) * 0.2)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dec["date"], dec["predicted_rate"], color="#064A56", marker="o", markersize=3, linewidth=2.2)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_ylabel("Predicted rate ($)")
    ax.set_title("December 2025 Predicted Load Rate")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    a, b, c = st.columns(3)
    a.metric("Lowest", f"${lo:,.0f}")
    b.metric("Average", f"${dec.predicted_rate.mean():,.0f}")
    c.metric("Highest", f"${hi:,.0f}")
    st.dataframe(dec, use_container_width=True, hide_index=True)
    st.download_button("Download CSV", dec.to_csv(index=False), "december_predictions.csv", "text/csv")


else:
    st.title("About")
    st.markdown(f"""
**Model:** two-stage regressor. Gradient boosting on log(rate) learns load, lane and equipment effects; a linear
market-index term and a carried-forward level handle the slow drift in rates.

**Validation** (three forward-in-time folds): MAE about \\$57, RMSE about \\$138-139, R² 0.989, MAPE about 2.4%.

**Model status:** {model_status}.

`quote_signal` is not used: its relationship with the rate changes from month to month.

**Things to know**
- The market index for a date comes from the data files; dates outside them use the latest 14-day average.
- Distances start at the typical distance for the lane and can be edited.
- Cities that are not in the training data are estimated from coordinates and may be a little less accurate.
- The data ends in October, so holiday effects in late December are not learned.
- Rates are estimates of the posted rate, not quotes.
""")
