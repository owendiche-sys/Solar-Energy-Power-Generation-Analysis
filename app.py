import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# Page config (LIGHT only)
# =========================
st.set_page_config(page_title="Solar Power Generation Dashboard", layout="wide")

BG = "#F6F8FC"
CARD = "#FFFFFF"
TEXT = "#111827"
MUTED = "rgba(17,24,39,0.65)"
BORDER = "rgba(15,23,42,0.08)"


def apply_styles():
    st.markdown(
        f"""
<style>
html, body, [data-testid="stAppViewContainer"] {{
    background: {BG};
}}
.block-container {{
    padding-top: 2.6rem;  /* avoids title clipping + keeps sidebar toggle usable */
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


apply_styles()


# =========================
# Data loading (LOCAL first)
# =========================
APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data.csv"


@st.cache_data(show_spinner=False)
def load_csv_with_fallback(path: Path) -> pd.DataFrame:
    # A bit more robust if the file has non-utf8 characters
    for enc in ("utf-8", "ISO-8859-1", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    for enc in ("utf-8", "ISO-8859-1", "cp1252", "latin1"):
        try:
            return pd.read_csv(uploaded_file, encoding=enc)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            continue
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file)


def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# =========================
# Model helpers
# =========================
def build_model(random_state: int, n_estimators: int, max_depth: int | None) -> Pipeline:
    # Dataset is fully numeric, so preprocessing is simple and fast
    prep = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        random_state=int(random_state),
        n_jobs=-1,
        max_depth=max_depth,
    )

    return Pipeline(steps=[("prep", prep), ("model", model)])


@st.cache_resource(show_spinner=False)
def train_model_cached(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    random_state: int,
    n_estimators: int,
    max_depth: int | None,
):
    d = df.copy().dropna(axis=0, how="all")
    d = ensure_numeric(d)

    d[target_col] = pd.to_numeric(d[target_col], errors="coerce")
    d = d.dropna(subset=[target_col])

    X = d.drop(columns=[target_col])
    y = d[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(test_size), random_state=int(random_state)
    )

    pipe = build_model(random_state=random_state, n_estimators=n_estimators, max_depth=max_depth)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    # Feature importance
    fi = pd.DataFrame({"feature": X.columns, "importance": pipe.named_steps["model"].feature_importances_})
    fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)

    # Residuals (for insight diagnostics)
    resid = pd.Series(y_test.values - y_pred, name="residual").reset_index(drop=True)
    abs_err = pd.Series(np.abs(resid.values), name="abs_error").reset_index(drop=True)

    return pipe, X_test.reset_index(drop=True), y_test.reset_index(drop=True), pd.Series(y_pred).reset_index(drop=True), mae, rmse, r2, fi, resid, abs_err


# =========================
# Insight helpers (data-driven + model-driven)
# =========================
def fmt_num(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    if abs(x) >= 1e6:
        return f"{x/1e6:.2f}M"
    if abs(x) >= 1e3:
        return f"{x/1e3:.2f}K"
    return f"{x:.2f}"


def corr_table(df: pd.DataFrame, target_col: str, top_n: int = 10) -> pd.DataFrame:
    d = ensure_numeric(df)
    cols = [c for c in d.columns if c != target_col]
    out = []
    for c in cols:
        s = d[[c, target_col]].dropna()
        if len(s) < 20:
            continue
        corr = s[c].corr(s[target_col])
        if np.isfinite(corr):
            out.append((c, float(corr)))
    res = pd.DataFrame(out, columns=["feature", "correlation"]).sort_values("correlation", ascending=False)
    top_pos = res.head(top_n)
    top_neg = res.tail(top_n).sort_values("correlation", ascending=True)
    return top_pos.reset_index(drop=True), top_neg.reset_index(drop=True)


def top_generation_ranges(df: pd.DataFrame, target_col: str, features: list[str], top_pct: float = 0.10) -> pd.DataFrame:
    d = ensure_numeric(df).dropna(subset=[target_col])
    if d.empty:
        return pd.DataFrame()

    cutoff = d[target_col].quantile(1 - top_pct)
    top = d[d[target_col] >= cutoff].copy()
    rows = []
    for f in features:
        if f not in top.columns:
            continue
        s = top[f].dropna()
        if len(s) == 0:
            continue
        rows.append(
            {
                "feature": f,
                "top_10pct_min": float(s.min()),
                "top_10pct_median": float(s.median()),
                "top_10pct_max": float(s.max()),
            }
        )
    return pd.DataFrame(rows)


def bin_effect(df: pd.DataFrame, feature: str, target_col: str, bins: int = 8) -> pd.DataFrame:
    d = ensure_numeric(df[[feature, target_col]]).dropna()
    if d.empty:
        return pd.DataFrame()

    try:
        d["bin"] = pd.qcut(d[feature], q=bins, duplicates="drop")
    except ValueError:
        d["bin"] = pd.cut(d[feature], bins=bins)

    out = (
        d.groupby("bin", observed=True)
        .agg(
            n=(target_col, "size"),
            target_mean=(target_col, "mean"),
            feat_median=(feature, "median"),
            feat_min=(feature, "min"),
            feat_max=(feature, "max"),
        )
        .reset_index(drop=True)
    )
    out = out.sort_values("feat_median")
    return out


def build_actionable_insights(df: pd.DataFrame, target_col: str, focus_features: list[str]) -> list[str]:
    d = ensure_numeric(df).dropna(subset=[target_col])
    if d.empty:
        return ["No insights available (target column has no valid numeric data)."]

    y = d[target_col]
    insights = []

    # Base stats
    insights.append(f"Typical generated power is around {fmt_num(float(y.median()))} kW (median).")
    insights.append(f"High generation days reach up to {fmt_num(float(y.quantile(0.95)))} kW (95th percentile).")

    # Correlation-based drivers
    top_pos, top_neg = corr_table(d, target_col, top_n=5)
    if not top_pos.empty:
        f = top_pos.iloc[0]["feature"]
        c = top_pos.iloc[0]["correlation"]
        insights.append(f"Strongest positive driver in this dataset is {f} (correlation {c:+.2f}).")
    if not top_neg.empty:
        f = top_neg.iloc[0]["feature"]
        c = top_neg.iloc[0]["correlation"]
        insights.append(f"Strongest negative driver in this dataset is {f} (correlation {c:+.2f}).")

    # High-output operating window (top 10%)
    ranges = top_generation_ranges(d, target_col, focus_features, top_pct=0.10)
    if not ranges.empty:
        # Pick up to 3 most interpretable ones
        sample = ranges.head(3).copy()
        for _, r in sample.iterrows():
            insights.append(
                f"Top 10% generation typically occurs when {r['feature']} is between {r['top_10pct_min']:.2f} and {r['top_10pct_max']:.2f} "
                f"(median {r['top_10pct_median']:.2f})."
            )

    return insights


# =========================
# Header
# =========================
st.markdown("## Solar Power Generation Dashboard")
st.caption(
    "This dashboard explains what drives solar generation, summarises key patterns, and evaluates a Random Forest regression model."
)
st.write("")


# =========================
# Sidebar
# =========================
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

st.sidebar.title("Controls")

page = st.sidebar.radio("Navigate", ["Dashboard Summary", "EDA", "Insights", "Model", "Predict"], index=0)

st.sidebar.divider()
st.sidebar.subheader("Data")

use_local = st.sidebar.checkbox("Use repo dataset (data.csv)", value=True)
upload = None
if not use_local:
    upload = st.sidebar.file_uploader("Upload another CSV (optional)", type=["csv"])

# Load data
if use_local:
    if not DATA_PATH.exists():
        st.error("Could not find data.csv next to app.py. Place the file in the same folder and redeploy.")
        st.stop()
    df = load_csv_with_fallback(DATA_PATH)
else:
    if upload is None:
        st.info("Upload a CSV in the sidebar or enable the repo dataset option.")
        st.stop()
    df = load_uploaded_csv(upload)

df = ensure_numeric(df)

st.sidebar.divider()
st.sidebar.subheader("Target + model")

default_target = "generated_power_kw" if "generated_power_kw" in df.columns else df.columns[-1]
target_col = st.sidebar.selectbox(
    "Target column",
    options=df.columns.tolist(),
    index=df.columns.tolist().index(default_target),
)

test_size = st.sidebar.slider("Test split", 0.1, 0.4, 0.2, step=0.05)
random_state = st.sidebar.number_input("Random state", min_value=0, max_value=10_000, value=42)
n_estimators = st.sidebar.slider("Trees (n_estimators)", 50, 500, 200, step=50)
depth_choice = st.sidebar.selectbox("Max depth", ["None", "10", "20", "30"], index=0)
max_depth = None if depth_choice == "None" else int(depth_choice)

run_train = st.sidebar.button("Train / Refresh model", type="primary")
if run_train:
    st.session_state.model_trained = True


# =========================
# Basic stats
# =========================
n_rows, n_cols = df.shape
missing_pct = (df.isna().sum().sum() / (n_rows * n_cols)) * 100 if n_rows and n_cols else 0.0

y = pd.to_numeric(df[target_col], errors="coerce").dropna()
y_mean = float(y.mean()) if len(y) else np.nan
y_med = float(y.median()) if len(y) else np.nan
y_max = float(y.max()) if len(y) else np.nan


# =========================
# DASHBOARD SUMMARY
# =========================
if page == "Dashboard Summary":
    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        kpi_card("Rows", f"{n_rows:,}", "In dataset")
    with c2:
        kpi_card("Columns", f"{n_cols:,}", "Features + target")
    with c3:
        kpi_card("Average generated power", f"{fmt_num(y_mean)}", f"Target: {target_col}")
    with c4:
        kpi_card("Missing data", f"{missing_pct:.1f}%", "Across all cells")

    st.write("")
    left, right = st.columns([1.2, 1.0], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Project Summary")
        st.write(
            """
This project focuses on understanding which environmental and geometric factors drive solar power generation.
The dashboard provides:
- Summary metrics and target distribution
- Feature-to-target relationships for interpretation
- Model performance (MAE, RMSE, R²)
- A prediction tool to test scenarios and compare outcomes
            """.strip()
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Data preview")
        st.dataframe(df.head(40), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Target distribution")
    fig = px.histogram(pd.to_numeric(df[target_col], errors="coerce").dropna(), nbins=60, labels={"value": target_col})
    fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
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
        fig = px.histogram(pd.to_numeric(df[target_col], errors="coerce").dropna(), nbins=60, labels={"value": target_col})
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Missing values (top columns)")
        miss = df.isna().sum().sort_values(ascending=False)
        miss = miss[miss > 0].head(12)
        if len(miss) == 0:
            st.success("No missing values detected.")
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
    st.caption("Select a feature to see how it relates to the target. This is designed to be interpretable for non-technical viewers.")

    feature_candidates = [c for c in df.columns if c != target_col]
    feat = st.selectbox("Feature", feature_candidates)

    tmp = df[[feat, target_col]].copy()
    tmp[target_col] = pd.to_numeric(tmp[target_col], errors="coerce")
    tmp[feat] = pd.to_numeric(tmp[feat], errors="coerce")
    tmp = tmp.dropna(subset=[feat, target_col])

    fig3 = px.scatter(tmp, x=feat, y=target_col, opacity=0.45)
    fig3.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<div class='small'>Tip: patterns in this chart describe association in the dataset, not guaranteed cause-and-effect.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# INSIGHTS (auto-generated)
# =========================
elif page == "Insights":
    st.markdown("### Insights")
    st.caption("This section converts the charts into plain-language takeaways. Insights are computed directly from the dataset and model outputs.")

    # Focus features chosen for interpretability in solar datasets
    focus_features = [
        "shortwave_radiation_backwards_sfc",
        "total_cloud_cover_sfc",
        "zenith",
        "angle_of_incidence",
        "temperature_2_m_above_gnd",
        "relative_humidity_2_m_above_gnd",
    ]
    focus_features = [c for c in focus_features if c in df.columns]

    # Data-only insights (always available)
    insights = build_actionable_insights(df, target_col, focus_features)

    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        kpi_card("Median generated power", fmt_num(y_med), "Typical output")
    with c2:
        kpi_card("Average generated power", fmt_num(y_mean), "Mean output")
    with c3:
        kpi_card("Peak generated power", fmt_num(y_max), "Maximum observed")
    with c4:
        q95 = float(y.quantile(0.95)) if len(y) else np.nan
        kpi_card("High-output threshold", fmt_num(q95), "95th percentile")

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Actionable insights")
    for item in insights[:10]:
        st.write(f"- {item}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    top_pos, top_neg = corr_table(df, target_col, top_n=10)

    left, right = st.columns([1.0, 1.0], gap="large")
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Top positive relationships")
        if top_pos.empty:
            st.write("Not enough data to compute correlations.")
        else:
            figp = px.bar(top_pos.iloc[::-1], x="correlation", y="feature", orientation="h")
            figp.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(figp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Top negative relationships")
        if top_neg.empty:
            st.write("Not enough data to compute correlations.")
        else:
            fign = px.bar(top_neg.iloc[::-1], x="correlation", y="feature", orientation="h")
            fign.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fign, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("High-output operating window (top 10% of generation)")
    st.caption("Ranges below summarise feature values commonly present when generation is in the top 10% of the dataset.")
    ranges = top_generation_ranges(df, target_col, focus_features, top_pct=0.10)
    if ranges.empty:
        st.write("Not enough data to compute operating windows.")
    else:
        ranges_show = ranges.copy()
        st.dataframe(ranges_show, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Feature impact by range")
    st.caption("This view bins a feature and shows how average generation changes across the bins.")

    if len(focus_features) == 0:
        st.write("No focus features available in this dataset.")
    else:
        feat = st.selectbox("Feature to bin", focus_features, index=0)
        bins = st.slider("Bins", 5, 12, 8)

        bt = bin_effect(df, feat, target_col, bins=bins)
        if bt.empty:
            st.write("Not enough valid values for this feature.")
        else:
            fig = px.line(
                bt,
                x="feat_median",
                y="target_mean",
                markers=True,
                labels={"feat_median": "Bin median", "target_mean": f"Average {target_col}"},
            )
            fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

            bt_show = bt.copy()
            bt_show["range"] = bt_show.apply(lambda r: f"{r['feat_min']:.2f} to {r['feat_max']:.2f}", axis=1)
            bt_show = bt_show[["range", "n", "target_mean"]].rename(columns={"target_mean": f"avg_{target_col}"})
            st.dataframe(bt_show, use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Optional model insights (only if trained)
    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Model-driven insights")
    st.caption("These insights require a trained model. Use the Train / Refresh model button in the sidebar.")

    if not st.session_state.model_trained:
        st.info("Train the model to unlock feature importance and error diagnostics.")
    else:
        with st.spinner("Loading trained model (cached)..."):
            pipe, X_test, y_test, y_pred, mae, rmse, r2, fi, resid, abs_err = train_model_cached(
                df=df,
                target_col=target_col,
                test_size=float(test_size),
                random_state=int(random_state),
                n_estimators=int(n_estimators),
                max_depth=max_depth,
            )

        a, b, c = st.columns(3, gap="large")
        with a:
            kpi_card("MAE", f"{mae:.2f}", "Average absolute error")
        with b:
            kpi_card("RMSE", f"{rmse:.2f}", "Penalises larger errors")
        with c:
            kpi_card("R²", f"{r2:.2f}", "Explained variance")

        st.write("")
        topn = 12
        fi_top = fi.head(topn).iloc[::-1]
        figfi = px.bar(fi_top, x="importance", y="feature", orientation="h")
        figfi.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(figfi, use_container_width=True)

        st.write("")
        st.subheader("Error distribution")
        err_fig = px.histogram(abs_err, nbins=50, labels={"value": "Absolute error"})
        err_fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(err_fig, use_container_width=True)

        # Where errors are largest (top 5%)
        cutoff = float(abs_err.quantile(0.95))
        worst_idx = abs_err[abs_err >= cutoff].index
        worst = X_test.loc[worst_idx].copy()
        worst["actual"] = y_test.loc[worst_idx].values
        worst["predicted"] = y_pred.loc[worst_idx].values
        worst["abs_error"] = abs_err.loc[worst_idx].values

        st.subheader("Cases with largest errors (top 5%)")
        st.caption("This helps identify conditions where the model is less reliable.")
        st.dataframe(worst.sort_values("abs_error", ascending=False).head(25), use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# MODEL
# =========================
elif page == "Model":
    st.markdown("### Model Performance & Drivers")

    if not st.session_state.model_trained:
        st.info("Click Train / Refresh model in the sidebar.")
        st.stop()

    with st.spinner("Training model (cached)..."):
        pipe, X_test, y_test, y_pred, mae, rmse, r2, fi, resid, abs_err = train_model_cached(
            df=df,
            target_col=target_col,
            test_size=float(test_size),
            random_state=int(random_state),
            n_estimators=int(n_estimators),
            max_depth=max_depth,
        )

    m1, m2, m3, m4 = st.columns(4, gap="large")
    with m1:
        kpi_card("MAE", f"{mae:.2f}", "Average absolute error")
    with m2:
        kpi_card("RMSE", f"{rmse:.2f}", "Penalises larger errors")
    with m3:
        kpi_card("R²", f"{r2:.2f}", "Closer to 1 is better")
    with m4:
        kpi_card("Test rows", f"{len(y_test):,}", "Holdout size")

    st.write("")
    left, right = st.columns([1.35, 1.0], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Actual vs Predicted")

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_test, name="Actual"))
        fig.add_trace(go.Scatter(y=y_pred, name="Predicted"))
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Model Summary")
        st.write(
            f"""
- Typical error (MAE) is about {mae:.2f} in target units.
- R² is {r2:.2f}, indicating how much of the variation is explained.
- Feature importance below highlights which inputs the model relies on most.
            """.strip()
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Feature importance (top drivers)")

    topn = st.slider("Show top N features", 5, min(30, len(fi)), 15)
    fi_top = fi.head(topn)
    fig2 = px.bar(fi_top.iloc[::-1], x="importance", y="feature", orientation="h")
    fig2.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(fi_top, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Residuals (actual - predicted)")
    fig_r = px.histogram(resid, nbins=60, labels={"value": "Residual"})
    fig_r.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_r, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# PREDICT
# =========================
else:
    st.markdown("### Prediction Tool")
    st.caption("Adjust inputs to see the predicted solar generation output.")

    if not st.session_state.model_trained:
        st.info("Click Train / Refresh model in the sidebar first.")
        st.stop()

    with st.spinner("Loading trained model (cached)..."):
        pipe, X_test, y_test, y_pred, mae, rmse, r2, fi, resid, abs_err = train_model_cached(
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
    st.subheader("Inputs")

    cols = st.columns(3, gap="large")
    for i, col in enumerate(feature_cols):
        box = cols[i % 3]
        series = pd.to_numeric(df[col], errors="coerce").dropna()

        if len(series) == 0:
            input_row[col] = 0.0
            continue

        vmin = float(series.quantile(0.01))
        vmax = float(series.quantile(0.99))
        vmed = float(series.median())

        # Prevent slider failures when vmin == vmax
        if np.isclose(vmin, vmax):
            input_row[col] = box.number_input(col, value=float(vmed))
        else:
            input_row[col] = box.slider(col, min_value=float(vmin), max_value=float(vmax), value=float(vmed))

    st.markdown("</div>", unsafe_allow_html=True)

    X_new = pd.DataFrame([input_row])
    pred_val = float(pipe.predict(X_new)[0])

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Prediction result")
    st.success(f"Predicted {target_col}: {pred_val:.2f}")
    st.markdown(
        "<div class='small'>Tip: adjust one input at a time to see which conditions increase or decrease the prediction.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
