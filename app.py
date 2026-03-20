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


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Solar Power Generation Dashboard", layout="wide")

BG = "#F6F8FC"
CARD = "#FFFFFF"
TEXT = "#111827"
MUTED = "rgba(17,24,39,0.68)"
BORDER = "rgba(15,23,42,0.08)"
ACCENT = "#2563EB"


# =========================================================
# STYLING
# =========================================================
def apply_styles():
    st.markdown(
        f"""
<style>
html, body, [data-testid="stAppViewContainer"] {{
    background: {BG};
}}

.block-container {{
    padding-top: 1.8rem;
    padding-bottom: 2rem;
    max-width: 1360px;
}}

#MainMenu {{
    visibility: hidden;
}}

footer {{
    visibility: hidden;
}}

section[data-testid="stSidebar"] > div {{
    border-right: 1px solid {BORDER};
}}

.card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 20px;
    padding: 18px 18px;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
}}

.kpi-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 20px;
    padding: 16px 18px;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
    min-height: 118px;
}}

.kpi-title {{
    color: {MUTED};
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 10px;
}}

.kpi-value {{
    color: {TEXT};
    font-size: 30px;
    font-weight: 800;
    line-height: 1.05;
}}

.kpi-subtitle {{
    color: {MUTED};
    font-size: 12px;
    margin-top: 8px;
}}

.hero {{
    background: linear-gradient(135deg, rgba(37,99,235,0.08), rgba(255,255,255,0.94));
    border: 1px solid {BORDER};
    border-radius: 24px;
    padding: 24px 24px 18px 24px;
    margin-bottom: 1rem;
}}

.hero-title {{
    color: {TEXT};
    font-size: 2.15rem;
    font-weight: 800;
    line-height: 1.05;
    margin-bottom: 0.55rem;
}}

.hero-sub {{
    color: {MUTED};
    font-size: 1rem;
    line-height: 1.75;
    max-width: 980px;
}}

.section-label {{
    color: {ACCENT};
    font-size: 0.82rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}}

.small {{
    color: {MUTED};
    font-size: 12px;
}}

.metric-chip {{
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    background: rgba(15,23,42,0.05);
    border: 1px solid rgba(15,23,42,0.06);
    font-size: 12px;
    font-weight: 700;
    color: {TEXT};
    margin-right: 8px;
    margin-bottom: 8px;
}}

.insight-box {{
    background: rgba(37,99,235,0.04);
    border: 1px solid rgba(37,99,235,0.10);
    border-radius: 16px;
    padding: 14px 16px;
}}
</style>
""",
        unsafe_allow_html=True,
    )


apply_styles()


# =========================================================
# UI HELPERS
# =========================================================
def kpi_card(title: str, value: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(label: str, title: str, caption: str | None = None) -> None:
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)
    st.markdown(f"### {title}")
    if caption:
        st.caption(caption)


def badge_row(items: list[str]) -> None:
    html = "".join([f'<span class="metric-chip">{item}</span>' for item in items])
    st.markdown(html, unsafe_allow_html=True)


def fmt_num(x: float | int | None, digits: int = 2) -> str:
    if x is None:
        return "—"
    try:
        if not np.isfinite(float(x)):
            return "—"
    except Exception:
        return "—"
    x = float(x)
    if abs(x) >= 1e6:
        return f"{x/1e6:.2f}M"
    if abs(x) >= 1e3:
        return f"{x/1e3:.2f}K"
    return f"{x:.{digits}f}"


def fmt_pct(x: float | None, digits: int = 1) -> str:
    if x is None:
        return "—"
    try:
        if not np.isfinite(float(x)):
            return "—"
    except Exception:
        return "—"
    return f"{float(x):.{digits}f}%"


# =========================================================
# DATA
# =========================================================
APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data.csv"


@st.cache_data(show_spinner=False)
def load_csv_with_fallback(path: Path) -> pd.DataFrame:
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
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


# =========================================================
# MODEL
# =========================================================
def build_model(random_state: int, n_estimators: int, max_depth: int | None) -> Pipeline:
    prep = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        random_state=int(random_state),
        n_jobs=-1,
        max_depth=max_depth,
    )

    return Pipeline(
        steps=[
            ("prep", prep),
            ("model", model),
        ]
    )


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
        X,
        y,
        test_size=float(test_size),
        random_state=int(random_state),
    )

    pipe = build_model(
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    fi = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": pipe.named_steps["model"].feature_importances_,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    comparison = pd.DataFrame(
        {
            "actual": y_test.reset_index(drop=True),
            "predicted": pd.Series(y_pred).reset_index(drop=True),
        }
    )
    comparison["residual"] = comparison["actual"] - comparison["predicted"]
    comparison["abs_error"] = np.abs(comparison["residual"])

    return {
        "pipe": pipe,
        "X_test": X_test.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "y_pred": pd.Series(y_pred).reset_index(drop=True),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "fi": fi,
        "comparison": comparison,
    }


# =========================================================
# INSIGHT HELPERS
# =========================================================
def corr_table(df: pd.DataFrame, target_col: str, top_n: int = 10):
    d = ensure_numeric(df)
    cols = [c for c in d.columns if c != target_col]
    rows = []

    for col in cols:
        s = d[[col, target_col]].dropna()
        if len(s) < 20:
            continue
        corr = s[col].corr(s[target_col])
        if np.isfinite(corr):
            rows.append((col, float(corr)))

    if not rows:
        empty = pd.DataFrame(columns=["feature", "correlation"])
        return empty, empty

    res = pd.DataFrame(rows, columns=["feature", "correlation"]).sort_values("correlation", ascending=False)
    top_pos = res.head(top_n).reset_index(drop=True)
    top_neg = res.tail(top_n).sort_values("correlation", ascending=True).reset_index(drop=True)
    return top_pos, top_neg


def top_generation_ranges(
    df: pd.DataFrame,
    target_col: str,
    features: list[str],
    top_pct: float = 0.10,
) -> pd.DataFrame:
    d = ensure_numeric(df).dropna(subset=[target_col])
    if d.empty:
        return pd.DataFrame()

    cutoff = d[target_col].quantile(1 - top_pct)
    top = d[d[target_col] >= cutoff].copy()

    rows = []
    for feature in features:
        if feature not in top.columns:
            continue
        s = top[feature].dropna()
        if len(s) == 0:
            continue
        rows.append(
            {
                "feature": feature,
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
        .sort_values("feat_median")
    )
    return out


def build_actionable_insights(df: pd.DataFrame, target_col: str, focus_features: list[str]) -> list[str]:
    d = ensure_numeric(df).dropna(subset=[target_col])
    if d.empty:
        return ["No insights available because the target column has no valid numeric values."]

    y = d[target_col]
    insights = []

    insights.append(f"Typical generated power is around **{fmt_num(float(y.median()))}** at the median operating point.")
    insights.append(f"High-output observations begin around **{fmt_num(float(y.quantile(0.95)))}**, which defines the upper generation band.")

    top_pos, top_neg = corr_table(d, target_col, top_n=5)

    if not top_pos.empty:
        feature = top_pos.iloc[0]["feature"]
        corr = top_pos.iloc[0]["correlation"]
        insights.append(f"The strongest positive relationship with output is **{feature}** with correlation **{corr:+.2f}**.")

    if not top_neg.empty:
        feature = top_neg.iloc[0]["feature"]
        corr = top_neg.iloc[0]["correlation"]
        insights.append(f"The strongest negative relationship with output is **{feature}** with correlation **{corr:+.2f}**.")

    ranges = top_generation_ranges(d, target_col, focus_features, top_pct=0.10)
    if not ranges.empty:
        sample = ranges.head(3).copy()
        for _, row in sample.iterrows():
            insights.append(
                f"Top 10% generation commonly occurs when **{row['feature']}** sits between **{row['top_10pct_min']:.2f}** and **{row['top_10pct_max']:.2f}**, with median **{row['top_10pct_median']:.2f}**."
            )

    return insights


def compute_summary_stats(df: pd.DataFrame, target_col: str):
    y = pd.to_numeric(df[target_col], errors="coerce").dropna()
    summary = {
        "mean": float(y.mean()) if len(y) else np.nan,
        "median": float(y.median()) if len(y) else np.nan,
        "max": float(y.max()) if len(y) else np.nan,
        "q95": float(y.quantile(0.95)) if len(y) else np.nan,
        "min": float(y.min()) if len(y) else np.nan,
        "std": float(y.std(ddof=0)) if len(y) else np.nan,
        "missing_pct": float((df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100) if df.shape[0] and df.shape[1] else 0.0,
    }
    return summary, y


# =========================================================
# SESSION STATE
# =========================================================
if "model_results" not in st.session_state:
    st.session_state.model_results = None

if "trained_signature" not in st.session_state:
    st.session_state.trained_signature = None


# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Solar Power Generation Dashboard</div>
        <div class="hero-sub">
            An interactive dashboard for understanding what drives solar power generation,
            identifying high-output operating conditions, and evaluating a Random Forest regression model.
            This version is designed to surface what the data is actually saying rather than just describing the file.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("Controls")

page = st.sidebar.radio(
    "Navigate",
    ["Dashboard Summary", "EDA", "Insights", "Model", "Predict"],
    index=0,
)

st.sidebar.divider()
st.sidebar.subheader("Data")

use_local = st.sidebar.checkbox("Use repo dataset (data.csv)", value=True)
uploaded = None

if not use_local:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if use_local:
    if not DATA_PATH.exists():
        st.error("Could not find data.csv next to app.py. Place the file beside app.py and redeploy.")
        st.stop()
    df = load_csv_with_fallback(DATA_PATH)
else:
    if uploaded is None:
        st.info("Upload a CSV in the sidebar or enable the repo dataset option.")
        st.stop()
    df = load_uploaded_csv(uploaded)

df = ensure_numeric(df)

st.sidebar.divider()
st.sidebar.subheader("Target and model")

default_target = "generated_power_kw" if "generated_power_kw" in df.columns else df.columns[-1]

target_col = st.sidebar.selectbox(
    "Target column",
    options=df.columns.tolist(),
    index=df.columns.tolist().index(default_target),
)

test_size = st.sidebar.slider("Test split", 0.10, 0.40, 0.20, step=0.05)
random_state = st.sidebar.number_input("Random state", min_value=0, max_value=10_000, value=42)
n_estimators = st.sidebar.slider("Trees (n_estimators)", 50, 500, 200, step=50)
depth_choice = st.sidebar.selectbox("Max depth", ["None", "10", "20", "30"], index=0)
max_depth = None if depth_choice == "None" else int(depth_choice)

train_signature = (
    target_col,
    float(test_size),
    int(random_state),
    int(n_estimators),
    max_depth,
    tuple(df.columns.tolist()),
    int(len(df)),
)

if st.sidebar.button("Train / Refresh model", type="primary"):
    with st.spinner("Training model and preparing outputs..."):
        st.session_state.model_results = train_model_cached(
            df=df,
            target_col=target_col,
            test_size=float(test_size),
            random_state=int(random_state),
            n_estimators=int(n_estimators),
            max_depth=max_depth,
        )
        st.session_state.trained_signature = train_signature
    st.sidebar.success("Model outputs refreshed.")

model_ready = (
    st.session_state.model_results is not None
    and st.session_state.trained_signature == train_signature
)

summary_stats, y = compute_summary_stats(df, target_col)

focus_features = [
    "shortwave_radiation_backwards_sfc",
    "total_cloud_cover_sfc",
    "zenith",
    "angle_of_incidence",
    "temperature_2_m_above_gnd",
    "relative_humidity_2_m_above_gnd",
]
focus_features = [c for c in focus_features if c in df.columns]


# =========================================================
# DASHBOARD SUMMARY
# =========================================================
if page == "Dashboard Summary":
    section_header(
        "Overview",
        "Generation snapshot",
        "This page is built to show the generation story first: normal output, high-output conditions, strongest drivers, and current model quality.",
    )

    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        kpi_card("Typical Output", fmt_num(summary_stats["median"]), "Median generated power")
    with c2:
        kpi_card("Average Output", fmt_num(summary_stats["mean"]), "Mean generated power")
    with c3:
        kpi_card("High-Output Threshold", fmt_num(summary_stats["q95"]), "95th percentile of generation")
    with c4:
        kpi_card("Peak Output", fmt_num(summary_stats["max"]), "Maximum observed generation")

    st.write("")

    top_pos, top_neg = corr_table(df, target_col, top_n=3)
    ranges = top_generation_ranges(df, target_col, focus_features, top_pct=0.10)

    c5, c6, c7, c8 = st.columns(4, gap="large")
    with c5:
        strongest_up = top_pos.iloc[0]["feature"] if not top_pos.empty else "—"
        kpi_card("Strongest Positive Driver", strongest_up, "Highest positive correlation")
    with c6:
        strongest_down = top_neg.iloc[0]["feature"] if not top_neg.empty else "—"
        kpi_card("Strongest Negative Driver", strongest_down, "Strongest inverse relationship")
    with c7:
        kpi_card("Output Variability", fmt_num(summary_stats["std"]), "Standard deviation of generated power")
    with c8:
        kpi_card("Missing Data", fmt_pct(summary_stats["missing_pct"]), "Across all cells")

    st.write("")

    left, right = st.columns([1.15, 1.0], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("What the data is telling us")
        notes = [
            f"- Typical generation is around **{fmt_num(summary_stats['median'])}**, while strong production begins around **{fmt_num(summary_stats['q95'])}**.",
            f"- Peak observed generation reaches **{fmt_num(summary_stats['max'])}**, showing the upper operating range in the dataset.",
        ]
        if not top_pos.empty:
            notes.append(
                f"- The strongest upward driver is **{top_pos.iloc[0]['feature']}** with correlation **{top_pos.iloc[0]['correlation']:+.2f}**."
            )
        if not top_neg.empty:
            notes.append(
                f"- The strongest downward driver is **{top_neg.iloc[0]['feature']}** with correlation **{top_neg.iloc[0]['correlation']:+.2f}**."
            )
        if not ranges.empty:
            row = ranges.iloc[0]
            notes.append(
                f"- High-output conditions are commonly associated with **{row['feature']}** values between **{row['top_10pct_min']:.2f}** and **{row['top_10pct_max']:.2f}**."
            )

        st.write("\n".join(notes))
        st.write("")
        badge_row(
            [
                f"Target: {target_col}",
                "Insight-led dashboard",
                "Model: Random Forest Regressor",
            ]
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Target distribution")
        fig = px.histogram(
            pd.to_numeric(df[target_col], errors="coerce").dropna(),
            nbins=60,
            labels={"value": target_col},
        )
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    bottom_left, bottom_right = st.columns([1.05, 1.0], gap="large")

    with bottom_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Driver summary")
        st.caption("These charts show the strongest positive and negative feature relationships with generated power.")

        if top_pos.empty and top_neg.empty:
            st.info("Not enough valid data to compute feature-target correlations.")
        else:
            plot_df = pd.concat([top_pos.head(5), top_neg.head(5)], axis=0).drop_duplicates("feature")
            fig_corr = px.bar(
                plot_df.sort_values("correlation"),
                x="correlation",
                y="feature",
                orientation="h",
            )
            fig_corr.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with bottom_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("High-output operating window")
        st.caption("Feature ranges commonly seen when generation is in the top 10% of observations.")

        if ranges.empty:
            st.info("Not enough data to compute high-output operating windows.")
        else:
            st.dataframe(ranges.head(6), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if model_ready:
        st.write("")
        section_header(
            "Model snapshot",
            "Current predictive performance",
            "These metrics reflect the current sidebar model settings.",
        )

        mr = st.session_state.model_results
        m1, m2, m3, m4 = st.columns(4, gap="large")
        with m1:
            kpi_card("MAE", fmt_num(mr["mae"]), "Average absolute error")
        with m2:
            kpi_card("RMSE", fmt_num(mr["rmse"]), "Penalises larger misses")
        with m3:
            kpi_card("R²", f"{mr['r2']:.2f}", "Explained variance")
        with m4:
            top_model_driver = mr["fi"].iloc[0]["feature"] if not mr["fi"].empty else "—"
            kpi_card("Top Model Driver", top_model_driver, "Most important feature in the model")


# =========================================================
# EDA
# =========================================================
elif page == "EDA":
    section_header(
        "Exploration",
        "Exploratory analysis",
        "Use this page to inspect distributions, missingness, raw associations, and binned feature effects.",
    )

    left, right = st.columns([1.15, 1.0], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Target distribution")
        fig = px.histogram(
            pd.to_numeric(df[target_col], errors="coerce").dropna(),
            nbins=60,
            labels={"value": target_col},
        )
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Missing values")
        missing = df.isna().sum().sort_values(ascending=False)
        missing = missing[missing > 0].head(12)

        if len(missing) == 0:
            st.success("No missing values detected.")
        else:
            missing_df = missing.reset_index()
            missing_df.columns = ["column", "missing_count"]
            fig2 = px.bar(missing_df, x="missing_count", y="column", orientation="h")
            fig2.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Relationship explorer")
    st.caption("Choose a feature to examine both its raw scatter pattern and its average effect across bins.")

    feature_candidates = [c for c in df.columns if c != target_col]
    feature = st.selectbox("Feature", feature_candidates)

    temp = df[[feature, target_col]].copy()
    temp[feature] = pd.to_numeric(temp[feature], errors="coerce")
    temp[target_col] = pd.to_numeric(temp[target_col], errors="coerce")
    temp = temp.dropna(subset=[feature, target_col])

    rel_left, rel_right = st.columns([1.15, 1.0], gap="large")

    with rel_left:
        fig3 = px.scatter(temp, x=feature, y=target_col, opacity=0.45)
        fig3.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig3, use_container_width=True)

    with rel_right:
        bt = bin_effect(df, feature, target_col, bins=8)
        if bt.empty:
            st.info("Not enough valid values to compute a binned effect view.")
        else:
            fig4 = px.line(
                bt,
                x="feat_median",
                y="target_mean",
                markers=True,
                labels={
                    "feat_median": "Feature bin median",
                    "target_mean": f"Average {target_col}",
                },
            )
            fig4.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig4, use_container_width=True)

    st.markdown(
        "<div class='small'>The binned view usually tells the story more clearly than the raw scatter because it shows how average generation changes across feature ranges.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# INSIGHTS
# =========================================================
elif page == "Insights":
    section_header(
        "Interpretation",
        "Insights",
        "This page is structured with data-driven insights first, then model-driven insights after training.",
    )

    insights = build_actionable_insights(df, target_col, focus_features)

    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        kpi_card("Typical Output", fmt_num(summary_stats["median"]), "Median generated power")
    with c2:
        kpi_card("Average Output", fmt_num(summary_stats["mean"]), "Mean generated power")
    with c3:
        kpi_card("High-Output Threshold", fmt_num(summary_stats["q95"]), "95th percentile")
    with c4:
        kpi_card("Peak Output", fmt_num(summary_stats["max"]), "Maximum observed")

    st.write("")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Data-driven insights")
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
    st.subheader("High-output operating window")
    st.caption("These ranges summarise values commonly observed when generation is in the top 10% of the dataset.")
    ranges = top_generation_ranges(df, target_col, focus_features, top_pct=0.10)
    if ranges.empty:
        st.write("Not enough data to compute operating windows.")
    else:
        st.dataframe(ranges, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Feature impact by range")
    st.caption("This view bins a feature and shows how average generation changes across those bins.")

    if len(focus_features) == 0:
        st.write("No predefined focus features are available in this dataset.")
    else:
        feature = st.selectbox("Feature to bin", focus_features, index=0)
        bins = st.slider("Bins", 5, 12, 8)

        bt = bin_effect(df, feature, target_col, bins=bins)
        if bt.empty:
            st.write("Not enough valid values for this feature.")
        else:
            fig = px.line(
                bt,
                x="feat_median",
                y="target_mean",
                markers=True,
                labels={
                    "feat_median": "Bin median",
                    "target_mean": f"Average {target_col}",
                },
            )
            fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

            bt_show = bt.copy()
            bt_show["range"] = bt_show.apply(
                lambda row: f"{row['feat_min']:.2f} to {row['feat_max']:.2f}",
                axis=1,
            )
            bt_show = bt_show[["range", "n", "target_mean"]].rename(
                columns={"target_mean": f"avg_{target_col}"}
            )
            st.dataframe(bt_show, use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Model-driven insights")
    st.caption("These insights require a trained model from the sidebar.")

    if not model_ready:
        st.info("Train the model to unlock feature importance and error diagnostics.")
    else:
        mr = st.session_state.model_results

        a, b, c = st.columns(3, gap="large")
        with a:
            kpi_card("MAE", fmt_num(mr["mae"]), "Average absolute error")
        with b:
            kpi_card("RMSE", fmt_num(mr["rmse"]), "Penalises larger misses")
        with c:
            kpi_card("R²", f"{mr['r2']:.2f}", "Explained variance")

        st.write("")

        figfi = px.bar(
            mr["fi"].head(12).iloc[::-1],
            x="importance",
            y="feature",
            orientation="h",
        )
        figfi.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(figfi, use_container_width=True)

        st.write("")

        st.subheader("Where the model struggles")
        err_fig = px.histogram(mr["comparison"]["abs_error"], nbins=50, labels={"value": "Absolute error"})
        err_fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(err_fig, use_container_width=True)

        cutoff = float(mr["comparison"]["abs_error"].quantile(0.95))
        worst = mr["comparison"][mr["comparison"]["abs_error"] >= cutoff].copy()
        st.dataframe(
            worst.sort_values("abs_error", ascending=False).head(25),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# MODEL
# =========================================================
elif page == "Model":
    section_header(
        "Evaluation",
        "Model performance and drivers",
        "Detailed model metrics, prediction quality, feature importance, and residual diagnostics.",
    )

    if not model_ready:
        st.info("Click Train / Refresh model in the sidebar.")
        st.stop()

    mr = st.session_state.model_results

    m1, m2, m3, m4 = st.columns(4, gap="large")
    with m1:
        kpi_card("MAE", fmt_num(mr["mae"]), "Average absolute error")
    with m2:
        kpi_card("RMSE", fmt_num(mr["rmse"]), "Penalises larger errors")
    with m3:
        kpi_card("R²", f"{mr['r2']:.2f}", "Closer to 1 is better")
    with m4:
        kpi_card("Top Model Driver", mr["fi"].iloc[0]["feature"] if not mr["fi"].empty else "—", "Highest feature importance")

    st.write("")

    left, right = st.columns([1.15, 1.0], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Actual vs predicted")
        comp = mr["comparison"].copy()

        fig = px.scatter(
            comp,
            x="actual",
            y="predicted",
            opacity=0.55,
            labels={"actual": "Actual", "predicted": "Predicted"},
        )
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Model summary")
        st.write(
            f"""
- Typical prediction error is about **{mr['mae']:.2f}** in target units.
- RMSE is **{mr['rmse']:.2f}**, which gives more weight to larger misses.
- R² is **{mr['r2']:.2f}**, indicating how much of the output variation is explained.
- The feature-importance view shows which inputs the model relies on most.
            """.strip()
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    a, b = st.columns([1.05, 0.95], gap="large")

    with a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Feature importance")
        topn = st.slider("Show top N features", 5, min(30, len(mr["fi"])), 15)
        fi_top = mr["fi"].head(topn)

        fig2 = px.bar(fi_top.iloc[::-1], x="importance", y="feature", orientation="h")
        fig2.update_layout(height=500, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(fi_top, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Residuals")
        fig_r = px.histogram(mr["comparison"]["residual"], nbins=60, labels={"value": "Residual"})
        fig_r.update_layout(height=500, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_r, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Prediction sequence")
    seq_fig = go.Figure()
    seq_fig.add_trace(go.Scatter(y=mr["y_test"], name="Actual"))
    seq_fig.add_trace(go.Scatter(y=mr["y_pred"], name="Predicted"))
    seq_fig.update_layout(
        height=380,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h"),
        xaxis_title="Test observation",
        yaxis_title=target_col,
    )
    st.plotly_chart(seq_fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# PREDICT
# =========================================================
else:
    section_header(
        "Scenario testing",
        "Prediction tool",
        "Adjust feature values to estimate solar generation under different operating conditions.",
    )

    if not model_ready:
        st.info("Click Train / Refresh model in the sidebar first.")
        st.stop()

    mr = st.session_state.model_results
    pipe = mr["pipe"]

    # show only most important features first
    important_features = mr["fi"]["feature"].head(min(12, len(mr["fi"]))).tolist()
    all_features = [c for c in df.columns if c != target_col]
    remaining = [c for c in all_features if c not in important_features]
    feature_cols = important_features + remaining

    input_row = {}

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Inputs")
    st.caption("The most important features are shown first to make scenario testing more useful.")

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

        if np.isclose(vmin, vmax):
            input_row[col] = box.number_input(col, value=float(vmed))
        else:
            input_row[col] = box.slider(
                col,
                min_value=float(vmin),
                max_value=float(vmax),
                value=float(vmed),
            )

    st.markdown("</div>", unsafe_allow_html=True)

    X_new = pd.DataFrame([input_row])
    pred_val = float(pipe.predict(X_new)[0])

    st.write("")

    left, right = st.columns([1.0, 1.0], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Prediction result")
        st.success(f"Predicted {target_col}: {pred_val:.2f}")

        comparison_label = "typical"
        if np.isfinite(summary_stats["q95"]) and pred_val >= summary_stats["q95"]:
            comparison_label = "high-output"
        elif np.isfinite(summary_stats["median"]) and pred_val < summary_stats["median"]:
            comparison_label = "below-typical"

        st.markdown(
            f"""
<div class="insight-box">
This scenario is currently estimated as a <strong>{comparison_label}</strong> generation case based on the distribution of observed output values.
</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("How to interpret")
        st.write(
            """
- Start from the default median values to represent a typical operating condition.
- Change one variable at a time to see which inputs move the prediction most.
- Compare the predicted value against the median and 95th percentile to judge whether the scenario is typical, weak, or high-output.
            """.strip()
        )
        st.markdown("</div>", unsafe_allow_html=True)