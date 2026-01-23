import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# Page config 
# =========================
st.set_page_config(page_title="Solar Power Generation Dashboard", layout="wide")

BG = "#F6F8FC"
CARD = "#FFFFFF"
TEXT = "#111827"
MUTED = "rgba(17,24,39,0.65)"
BORDER = "rgba(15,23,42,0.08)"

st.markdown(
    f"""
<style>
html, body, [data-testid="stAppViewContainer"] {{
    background: {BG};
}}
.block-container {{
    padding-top: 1.1rem;
    padding-bottom: 1.5rem;
}}
#MainMenu {{visibility:hidden;}}
footer {{visibility:hidden;}}

section[data-testid="stSidebar"] > div {{
    border-right: 1px solid {BORDER};
}}

.card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 16px 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
}}
.small {{
    color: {MUTED};
    font-size: 12px;
}}
</style>
""",
    unsafe_allow_html=True,
)


def kpi_card(title, value, subtitle=""):
    st.markdown(
        f"""
<div class="card">
  <div style="color:{MUTED}; font-weight:600; font-size:14px;">{title}</div>
  <div style="color:{TEXT}; font-weight:800; font-size:28px; margin-top:6px;">{value}</div>
  <div style="color:{MUTED}; font-size:12px; margin-top:6px;">{subtitle}</div>
</div>
""",
        unsafe_allow_html=True,
    )


# =========================
# Data loading (LOCAL first)
# =========================
DATA_PATH = Path(__file__).parent / "data.csv"

@st.cache_data(show_spinner=False)
def load_local_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


# =========================
# Model helpers
# =========================
def build_model(random_state: int, n_estimators: int, max_depth: int | None):
    # auto-select numeric + categorical
    numeric_selector = selector(dtype_include=np.number)
    categorical_selector = selector(dtype_exclude=np.number)

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_selector),
            ("cat", categorical_transformer, categorical_selector),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        max_depth=max_depth,
    )

    return Pipeline(steps=[("prep", preprocessor), ("model", model)])


@st.cache_resource(show_spinner=False)
def train_model_cached(df: pd.DataFrame, target_col: str, test_size: float, random_state: int,
                       n_estimators: int, max_depth: int | None):
    df = df.copy().dropna(axis=0, how="all")

    # keep only rows where target is numeric + not null
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipe = build_model(random_state=random_state, n_estimators=n_estimators, max_depth=max_depth)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    # Feature importance
    fi = pd.DataFrame({"feature": [], "importance": []})
    try:
        feat_names = pipe.named_steps["prep"].get_feature_names_out()
        importances = pipe.named_steps["model"].feature_importances_
        fi = (
            pd.DataFrame({"feature": feat_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    except Exception:
        pass

    return pipe, X_test, y_test, y_pred, mae, rmse, r2, fi


# =========================
# Header
# =========================
st.markdown("## Solar Power Generation Dashboard")
st.caption(
    "A clean, story-driven dashboard that explains **what drives solar generation** and how well the model predicts it — "
    "for non-technical and technical viewers."
)
st.write("")


# =========================
# Sidebar
# =========================
st.sidebar.title("Controls")
page = st.sidebar.radio("Navigate", ["Overview", "EDA", "Model", "Predict"], index=0)

st.sidebar.divider()
st.sidebar.subheader("Data")

use_local = st.sidebar.checkbox("Use built-in repo dataset (data.csv)", value=True)
upload = None
if not use_local:
    upload = st.sidebar.file_uploader("Upload another CSV (optional)", type=["csv"])

# Load data
if use_local:
    if not DATA_PATH.exists():
        st.error("I couldn’t find `data.csv` next to `app.py`. Put the file in the same folder and redeploy.")
        st.stop()
    df = load_local_csv(DATA_PATH)
else:
    if upload is None:
        st.info("Upload a CSV in the sidebar or turn on the built-in dataset option.")
        st.stop()
    df = load_uploaded_csv(upload)

st.sidebar.divider()
st.sidebar.subheader("Target + model")

# Target column chooser (auto-detect common target)
default_target = "generated_power_kw" if "generated_power_kw" in df.columns else df.columns[-1]
target_col = st.sidebar.selectbox("Target column", options=df.columns.tolist(), index=df.columns.tolist().index(default_target))

test_size = st.sidebar.slider("Test split", 0.1, 0.4, 0.2, step=0.05)
random_state = st.sidebar.number_input("Random state", min_value=0, max_value=10_000, value=42)
n_estimators = st.sidebar.slider("Trees (n_estimators)", 50, 500, 200, step=50)
depth_choice = st.sidebar.selectbox("Max depth", ["None", "10", "20", "30"], index=0)
max_depth = None if depth_choice == "None" else int(depth_choice)

run_train = st.sidebar.button("Train / Refresh model", type="primary")


# =========================
# Basic stats
# =========================
n_rows, n_cols = df.shape
missing_pct = (df.isna().sum().sum() / (n_rows * n_cols)) * 100 if n_rows and n_cols else 0.0

y = pd.to_numeric(df[target_col], errors="coerce").dropna()
y_mean = y.mean() if len(y) else np.nan
y_max = y.max() if len(y) else np.nan


# =========================
# OVERVIEW
# =========================
if page == "Overview":
    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        kpi_card("Rows", f"{n_rows:,}", "In dataset")
    with c2:
        kpi_card("Columns", f"{n_cols:,}", "Features + target")
    with c3:
        kpi_card("Avg generated power", f"{y_mean:.2f}" if np.isfinite(y_mean) else "—", f"Target: {target_col}")
    with c4:
        kpi_card("Missing data", f"{missing_pct:.1f}%", "Across all cells")

    st.write("")
    left, right = st.columns([1.15, 1.0], gap="large")
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Project Summary")
        st.write(
            """
- **How solar output varies** across conditions in the dataset  
- **Which features matter most** (feature importance)  
- **How accurate** the predictions are (MAE / RMSE / R²)  
- A simple **“try it yourself”** predictor for non-technical viewers
            """.strip()
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Data preview")
        st.dataframe(df.head(40), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# =========================
# EDA
# =========================
elif page == "EDA":
    st.markdown("### Exploratory Data Analysis")

    left, right = st.columns([1.2, 1.0], gap="large")
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Target distribution")
        fig = px.histogram(pd.to_numeric(df[target_col], errors="coerce").dropna(), nbins=60)
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Missing values (top columns)")
        miss = df.isna().sum().sort_values(ascending=False)
        miss = miss[miss > 0].head(12)
        if len(miss) == 0:
            st.success("No missing values detected")
        else:
            miss_df = miss.reset_index()
            miss_df.columns = ["column", "missing_count"]
            fig2 = px.bar(miss_df, x="missing_count", y="column", orientation="h")
            fig2.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Relationship Explorer")
    st.caption("Pick one feature to compare with the target. This is the most explainable chart for non-technical viewers.")

    feature_candidates = [c for c in df.columns if c != target_col]
    feat = st.selectbox("Choose a feature", feature_candidates)

    tmp = df[[feat, target_col]].copy()
    tmp[target_col] = pd.to_numeric(tmp[target_col], errors="coerce")
    tmp = tmp.dropna(subset=[target_col])

    if pd.api.types.is_numeric_dtype(df[feat]):
        tmp[feat] = pd.to_numeric(tmp[feat], errors="coerce")
        tmp = tmp.dropna(subset=[feat])
        fig3 = px.scatter(tmp, x=feat, y=target_col, opacity=0.45)
    else:
        tmp[feat] = tmp[feat].astype(str)
        fig3 = px.box(tmp, x=feat, y=target_col)

    fig3.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# MODEL
# =========================
elif page == "Model":
    st.markdown("### Model Performance & Drivers")

    if not run_train:
        st.info("Click **Train / Refresh model** in the sidebar.")
        st.stop()

    with st.spinner("Training model (cached)…"):
        pipe, X_test, y_test, y_pred, mae, rmse, r2, fi = train_model_cached(
            df=df,
            target_col=target_col,
            test_size=float(test_size),
            random_state=int(random_state),
            n_estimators=int(n_estimators),
            max_depth=max_depth,
        )

    m1, m2, m3, m4 = st.columns(4, gap="large")
    with m1:
        kpi_card("MAE", f"{mae:.2f}", "Avg absolute error")
    with m2:
        kpi_card("RMSE", f"{rmse:.2f}", "Penalises big errors")
    with m3:
        kpi_card("R²", f"{r2:.2f}", "Closer to 1 is better")
    with m4:
        kpi_card("Test rows", f"{len(y_test):,}", "Holdout size")

    st.write("")
    left, right = st.columns([1.35, 1.0], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Actual vs Predicted")
        actual = pd.Series(y_test).reset_index(drop=True)
        pred = pd.Series(y_pred).reset_index(drop=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=actual, name="Actual"))
        fig.add_trace(go.Scatter(y=pred, name="Predicted"))
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Key takeaways")
        st.write(
            f"""
- Typical error is about **{mae:.2f}** in the target units.
- **R² = {r2:.2f}** means the model explains a solid chunk of variation.
- Next: see **which features matter most** below.
            """.strip()
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Feature importance (top drivers)")

    if fi.empty:
        st.warning("Feature importance couldn’t be extracted (preprocessing names issue).")
    else:
        topn = st.slider("Show top N features", 5, 30, 15)
        fi_top = fi.head(topn)
        fig2 = px.bar(fi_top[::-1], x="importance", y="feature", orientation="h")
        fig2.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(fi_top, use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# PREDICT
# =========================
else:
    st.markdown("### Predict Solar Generation (Try it yourself)")
    st.caption("This is the section that sells the project: change inputs and instantly see the predicted output.")

    if not run_train:
        st.info("Click **Train / Refresh model** in the sidebar first.")
        st.stop()

    with st.spinner("Training model (cached)…"):
        pipe, X_test, y_test, y_pred, mae, rmse, r2, fi = train_model_cached(
            df=df,
            target_col=target_col,
            test_size=float(test_size),
            random_state=int(random_state),
            n_estimators=int(n_estimators),
            max_depth=max_depth,
        )

    feature_cols = [c for c in df.columns if c != target_col]
    input_row = {}

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("1) Choose inputs")

    cols = st.columns(3, gap="large")
    for i, col in enumerate(feature_cols):
        box = cols[i % 3]
        series = df[col]

        if pd.api.types.is_numeric_dtype(series):
            s = pd.to_numeric(series, errors="coerce").dropna()
            if len(s) == 0:
                input_row[col] = 0.0
                continue
            vmin, vmax = float(s.quantile(0.01)), float(s.quantile(0.99))
            vmed = float(s.median())
            input_row[col] = box.slider(col, min_value=float(vmin), max_value=float(vmax), value=float(vmed))
        else:
            vals = series.dropna().astype(str).unique().tolist()
            vals = sorted(vals)[:200] if len(vals) > 200 else sorted(vals)
            input_row[col] = box.selectbox(col, vals)

    st.markdown("</div>", unsafe_allow_html=True)

    X_new = pd.DataFrame([input_row])
    pred_val = float(pipe.predict(X_new)[0])

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("2) Predicted output")
    st.success(f"Predicted **{target_col}**: **{pred_val:.2f}**")
    st.markdown("<div class='small'>Tip: change one input at a time to see what influences the prediction.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
