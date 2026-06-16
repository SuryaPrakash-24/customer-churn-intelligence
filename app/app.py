import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Customer Churn Intelligence", layout="wide", page_icon="📉"
)

CONFIG = {
    "dataset_path": "customer_churn.csv",
    "new_data_path": "new_customers_1.csv",
    "target_column": "Churn",
    "random_state": 77,
    "test_size": 0.20,
    "val_size": 0.20,
}


# ---------------------------------------------------------------------------
# Data + model pipeline (ported from customer_churn_analysis_improved.ipynb)
# ---------------------------------------------------------------------------


@st.cache_data
def load_raw_data():
    df = pd.read_csv(Path(__file__).parent / CONFIG["dataset_path"])
    return df


@st.cache_data
def engineer_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    df[CONFIG["target_column"]] = df[CONFIG["target_column"]].astype(str).str.strip()
    df[CONFIG["target_column"]] = (
        df[CONFIG["target_column"]].map({"0": 0, "1": 1}).astype(int)
    )

    onboard_dt = pd.to_datetime(df["Onboard_date"], errors="coerce")
    reference_date = onboard_dt.max()
    df["days_since_onboard"] = (reference_date - onboard_dt).dt.days
    df["onboard_year"] = onboard_dt.dt.year.astype("Int64").astype(str)
    df["onboard_month"] = onboard_dt.dt.month.astype("Int64").astype(str)
    df["onboard_quarter"] = onboard_dt.dt.quarter.astype("Int64").astype(str)

    df["state_code"] = (
        df["Location"]
        .astype(str)
        .str.extract(r",?\s*([A-Z]{2})\s+\d{5}", expand=False)
        .fillna("UNK")
    )
    df["company_initial"] = df["Company"].astype(str).str[0].str.upper().fillna("UNK")

    df = df.drop(columns=["Names", "Onboard_date", "Location", "Company"])
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def summarize_binary_performance(y_true, y_prob, threshold=0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
    }


@st.cache_resource
def train_model(df: pd.DataFrame):
    X = df.drop(columns=[CONFIG["target_column"]])
    y = df[CONFIG["target_column"]]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"],
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=CONFIG["val_size"],
        random_state=CONFIG["random_state"],
        stratify=y_train_val,
    )

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", RobustScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=CONFIG["random_state"],
        n_jobs=-1,
    )
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train_val, y_train_val)

    val_prob = pipe.predict_proba(X_val)[:, 1]
    threshold_grid = np.arange(0.10, 0.91, 0.05)
    threshold_rows = [
        summarize_binary_performance(y_val, val_prob, threshold=t)
        for t in threshold_grid
    ]
    threshold_df = pd.DataFrame(threshold_rows)
    selected_threshold = threshold_df.sort_values(
        ["recall", "pr_auc", "precision"], ascending=False
    ).iloc[0]["threshold"]

    test_prob = pipe.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= selected_threshold).astype(int)
    test_metrics = summarize_binary_performance(
        y_test, test_prob, threshold=selected_threshold
    )

    cm = confusion_matrix(y_test, test_pred)
    cm_norm = confusion_matrix(y_test, test_pred, normalize="true")

    fpr, tpr, _ = roc_curve(y_val, val_prob)
    prec, rec, _ = precision_recall_curve(y_val, val_prob)
    prob_true, prob_pred = calibration_curve(y_val, val_prob, n_bins=10)

    perm = permutation_importance(
        pipe,
        X_val,
        y_val,
        n_repeats=8,
        random_state=CONFIG["random_state"],
        scoring="recall",
    )
    perm_df = (
        pd.DataFrame(
            {
                "feature": pipe.feature_names_in_,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .head(15)
    )

    artifacts = {
        "pipe": pipe,
        "selected_threshold": float(selected_threshold),
        "threshold_df": threshold_df,
        "test_metrics": test_metrics,
        "cm": cm,
        "cm_norm": cm_norm,
        "roc": (fpr, tpr, roc_auc_score(y_val, val_prob)),
        "pr": (rec, prec, average_precision_score(y_val, val_prob)),
        "calibration": (prob_pred, prob_true),
        "perm_df": perm_df,
        "feature_columns": X.columns.tolist(),
    }
    return artifacts


@st.cache_data
def score_new_customers(_pipe, threshold, feature_columns):
    new_path = Path(__file__).parent / CONFIG["new_data_path"]
    if not new_path.exists():
        return None
    new_raw = pd.read_csv(new_path)
    new_raw_with_target = new_raw.assign(Churn="0")
    new_eng = engineer_features(new_raw_with_target).drop(
        columns=[CONFIG["target_column"]]
    )
    new_eng = new_eng[feature_columns]
    new_prob = _pipe.predict_proba(new_eng)[:, 1]
    scored = new_raw.copy()
    scored["churn_probability"] = new_prob
    scored["predicted_churn"] = (new_prob >= threshold).astype(int)
    scored["state_code"] = new_eng["state_code"].values
    return scored.sort_values("churn_probability", ascending=False).reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# Load + train
# ---------------------------------------------------------------------------

raw_df = load_raw_data()
df = engineer_features(raw_df)
artifacts = train_model(df)
pipe = artifacts["pipe"]

scored_new = score_new_customers(
    pipe, artifacts["selected_threshold"], artifacts["feature_columns"]
)

# Score the full known population too, for the "At-Risk Customers" tab
X_all = df.drop(columns=[CONFIG["target_column"]])
all_prob = pipe.predict_proba(X_all)[:, 1]
scored_existing = raw_df.copy()
scored_existing["churn_probability"] = all_prob
scored_existing["state_code"] = df["state_code"].values


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

st.sidebar.title("📉 Customer Churn Intelligence")
st.sidebar.markdown("Filter the customer base shown across all tabs.")

am_options = ["All", "Has Account Manager", "No Account Manager"]
selected_am = st.sidebar.selectbox("Account Manager", am_options)

state_options = ["All"] + sorted([s for s in df["state_code"].unique() if s != "UNK"])
selected_state = st.sidebar.selectbox("State", state_options)

years_min, years_max = float(df["Years"].min()), float(df["Years"].max())
selected_years = st.sidebar.slider(
    "Tenure (Years)", years_min, years_max, (years_min, years_max), step=0.5
)

quarter_options = ["All"] + sorted(df["onboard_quarter"].unique().tolist())
selected_quarter = st.sidebar.selectbox("Onboard quarter", quarter_options)

st.sidebar.markdown("---")
threshold = st.sidebar.slider(
    "Churn probability threshold",
    0.10,
    0.90,
    float(artifacts["selected_threshold"]),
    step=0.05,
    help="Customers above this probability are flagged as predicted churn.",
)
st.sidebar.caption(
    f"Model-tuned default threshold: **{artifacts['selected_threshold']:.2f}**"
)


def apply_filters(raw_data, eng_data, scored_data):
    mask = pd.Series(True, index=raw_data.index)
    if selected_am == "Has Account Manager":
        mask &= raw_data["Account_Manager"] == 1
    elif selected_am == "No Account Manager":
        mask &= raw_data["Account_Manager"] == 0
    if selected_state != "All":
        mask &= eng_data["state_code"] == selected_state
    mask &= eng_data["Years"].between(*selected_years)
    if selected_quarter != "All":
        mask &= eng_data["onboard_quarter"] == selected_quarter
    return raw_data[mask], eng_data[mask], scored_data[mask]


filtered_raw, filtered_eng, filtered_scored = apply_filters(raw_df, df, scored_existing)

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(filtered_raw):,}** of {len(raw_df):,} customers")
st.sidebar.caption("Model: Random Forest (class-balanced)")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_atrisk, tab_perf = st.tabs(
    ["📊 Churn Overview", "🚨 At-Risk Customers", "🎯 Threshold & Performance"]
)

# --- Churn Overview ----------------------------------------------------------
with tab_overview:
    st.header("Churn Overview")

    churn_rate = (
        filtered_eng[CONFIG["target_column"]].mean() if len(filtered_eng) > 0 else 0
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers (filtered)", f"{len(filtered_raw):,}")
    c2.metric("Churn Rate", f"{churn_rate:.1%}")
    c3.metric("Avg Total Purchase", f"${filtered_raw['Total_Purchase'].mean():,.0f}")
    c4.metric("Avg Tenure (Years)", f"{filtered_raw['Years'].mean():.1f}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Class Balance")
        class_counts = (
            filtered_eng[CONFIG["target_column"]]
            .value_counts()
            .reindex([0, 1], fill_value=0)
        )
        fig_bal = px.bar(
            x=["No Churn", "Churn"],
            y=class_counts.values,
            color=["No Churn", "Churn"],
            color_discrete_map={"No Churn": "steelblue", "Churn": "tomato"},
        )
        fig_bal.update_layout(
            showlegend=False, height=350, xaxis_title="", yaxis_title="Count"
        )
        st.plotly_chart(fig_bal, width="stretch")

    with col2:
        st.subheader("Churn Rate by Account Manager Status")
        am_churn = (
            filtered_eng.assign(
                am_label=filtered_eng["Account_Manager"].map({0: "No AM", 1: "Has AM"})
            )
            .groupby("am_label")[CONFIG["target_column"]]
            .mean()
            .reset_index()
        )
        fig_am = px.bar(
            am_churn,
            x="am_label",
            y=CONFIG["target_column"],
            color_discrete_sequence=["#fd8d3c"],
        )
        fig_am.update_layout(
            height=350, xaxis_title="", yaxis_title="Churn rate", yaxis_tickformat=".0%"
        )
        st.plotly_chart(fig_am, width="stretch")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Total_Purchase by Churn Class")
        fig_violin1 = px.violin(
            filtered_eng,
            x=CONFIG["target_column"],
            y="Total_Purchase",
            box=True,
            color=CONFIG["target_column"],
            color_discrete_map={0: "steelblue", 1: "tomato"},
        )
        fig_violin1.update_xaxes(
            tickvals=[0, 1], ticktext=["No Churn", "Churn"], title=""
        )
        fig_violin1.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig_violin1, width="stretch")

    with col4:
        st.subheader("Num_Sites by Churn Class")
        fig_violin2 = px.violin(
            filtered_eng,
            x=CONFIG["target_column"],
            y="Num_Sites",
            box=True,
            color=CONFIG["target_column"],
            color_discrete_map={0: "steelblue", 1: "tomato"},
        )
        fig_violin2.update_xaxes(
            tickvals=[0, 1], ticktext=["No Churn", "Churn"], title=""
        )
        fig_violin2.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig_violin2, width="stretch")

    st.subheader("Churn Rate by State (Top 10 by customer count)")
    state_summary = (
        filtered_eng.groupby("state_code")
        .agg(
            churn_rate=(CONFIG["target_column"], "mean"),
            customers=(CONFIG["target_column"], "count"),
        )
        .query("state_code != 'UNK'")
        .sort_values("customers", ascending=False)
        .head(10)
        .reset_index()
    )
    fig_state = px.bar(
        state_summary.sort_values("churn_rate"),
        x="churn_rate",
        y="state_code",
        orientation="h",
        color_discrete_sequence=["#756bb1"],
    )
    fig_state.update_layout(
        height=400,
        xaxis_title="Churn rate",
        yaxis_title="State",
        xaxis_tickformat=".0%",
    )
    st.plotly_chart(fig_state, width="stretch")

    st.subheader("Top Churn Drivers (Permutation Importance)")
    fig_perm = px.bar(
        artifacts["perm_df"].sort_values("importance_mean"),
        x="importance_mean",
        y="feature",
        error_x="importance_std",
        orientation="h",
        color_discrete_sequence=["#2ca25f"],
    )
    fig_perm.update_layout(
        height=450, xaxis_title="Importance (recall increase)", yaxis_title=""
    )
    st.plotly_chart(fig_perm, width="stretch")


# --- At-Risk Customers --------------------------------------------------------
with tab_atrisk:
    st.header("At-Risk Customers")

    n_flagged = (filtered_scored["churn_probability"] >= threshold).sum()
    st.metric(
        "Customers flagged at current threshold",
        f"{n_flagged:,} of {len(filtered_scored):,}",
    )

    st.subheader("Existing Customers Ranked by Churn Probability")
    search_term = st.text_input("Search by name or company", "")

    display_existing = filtered_scored.copy()
    if search_term:
        mask = display_existing["Names"].str.contains(
            search_term, case=False, na=False
        ) | display_existing["Company"].str.contains(search_term, case=False, na=False)
        display_existing = display_existing[mask]

    display_existing = display_existing.sort_values(
        "churn_probability", ascending=False
    )
    display_existing["flagged"] = display_existing["churn_probability"] >= threshold

    st.dataframe(
        display_existing[
            [
                "Names",
                "Company",
                "state_code",
                "Total_Purchase",
                "Years",
                "Num_Sites",
                "Account_Manager",
                "churn_probability",
                "flagged",
            ]
        ]
        .head(50)
        .style.format(
            {
                "churn_probability": "{:.1%}",
                "Total_Purchase": "${:,.0f}",
                "Years": "{:.1f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Churn Probability Distribution (Filtered Segment)")
    fig_dist = px.histogram(
        filtered_scored,
        x="churn_probability",
        nbins=30,
        color_discrete_sequence=["#e6550d"],
    )
    fig_dist.add_vline(
        x=threshold, line_dash="dash", line_color="black", annotation_text="Threshold"
    )
    fig_dist.update_layout(
        height=350, xaxis_title="Predicted churn probability", yaxis_title="Count"
    )
    st.plotly_chart(fig_dist, width="stretch")

    st.markdown("---")
    st.subheader("New Customer Scoring")
    if scored_new is not None:
        scored_new_display = scored_new.copy()
        scored_new_display["flagged"] = (
            scored_new_display["churn_probability"] >= threshold
        )
        st.dataframe(
            scored_new_display[
                [
                    "Names",
                    "Company",
                    "state_code",
                    "Total_Purchase",
                    "Years",
                    "Num_Sites",
                    "Account_Manager",
                    "churn_probability",
                    "flagged",
                ]
            ].style.format(
                {
                    "churn_probability": "{:.1%}",
                    "Total_Purchase": "${:,.0f}",
                    "Years": "{:.1f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            f"Scored from `{CONFIG['new_data_path']}` using the current threshold."
        )
    else:
        st.info(
            f"`{CONFIG['new_data_path']}` not found — new-customer scoring unavailable."
        )

    st.markdown("""
**Recommended actions**
- Prioritize outreach (account manager call, renewal incentive) for the highest-probability accounts above,
  ordered by `Total_Purchase` to focus on highest-value at-risk customers first.
- If unmanaged accounts (`No AM`) show notably higher churn in the Overview tab, consider assigning account
  managers to high-value unmanaged customers in high-churn states.
- For low `Num_Sites` / low tenure customers flagged here, consider onboarding or engagement programs to
  increase product usage before the churn window.
""")


# --- Threshold & Performance --------------------------------------------------
with tab_perf:
    st.header("Threshold Tuning & Model Performance")

    st.subheader("Precision / Recall / F1 vs Threshold")
    tdf = artifacts["threshold_df"]
    fig_thresh = go.Figure()
    for col, color in [
        ("precision", "#3182bd"),
        ("recall", "#e6550d"),
        ("f1", "#31a354"),
    ]:
        fig_thresh.add_trace(
            go.Scatter(
                x=tdf["threshold"],
                y=tdf[col],
                mode="lines+markers",
                name=col,
                line=dict(color=color),
            )
        )
    fig_thresh.add_vline(
        x=threshold, line_dash="dash", line_color="black", annotation_text="Selected"
    )
    fig_thresh.update_layout(height=400, xaxis_title="Threshold", yaxis_title="Score")
    st.plotly_chart(fig_thresh, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ROC Curve (Validation)")
        fpr, tpr, auc_val = artifacts["roc"]
        fig_roc = go.Figure()
        fig_roc.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"ROC AUC = {auc_val:.3f}",
                line=dict(color="#3182bd"),
            )
        )
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color="gray"),
                showlegend=False,
            )
        )
        fig_roc.update_layout(
            height=380,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
        )
        st.plotly_chart(fig_roc, width="stretch")

    with col2:
        st.subheader("Precision-Recall Curve (Validation)")
        rec, prec, pr_auc_val = artifacts["pr"]
        fig_pr = go.Figure()
        fig_pr.add_trace(
            go.Scatter(
                x=rec,
                y=prec,
                mode="lines",
                name=f"PR AUC = {pr_auc_val:.3f}",
                line=dict(color="#e6550d"),
            )
        )
        fig_pr.update_layout(height=380, xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig_pr, width="stretch")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Calibration Curve (Validation)")
        prob_pred, prob_true = artifacts["calibration"]
        fig_cal = go.Figure()
        fig_cal.add_trace(
            go.Scatter(
                x=prob_pred,
                y=prob_true,
                mode="lines+markers",
                name="Model",
                line=dict(color="#756bb1"),
            )
        )
        fig_cal.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color="gray"),
                name="Perfectly calibrated",
            )
        )
        fig_cal.update_layout(
            height=380,
            xaxis_title="Mean predicted probability",
            yaxis_title="Observed positive rate",
        )
        st.plotly_chart(fig_cal, width="stretch")

    with col4:
        st.subheader("Confusion Matrix (Test Set, at deployment threshold)")
        cm = artifacts["cm"]
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            x=["Pred: No Churn", "Pred: Churn"],
            y=["Actual: No Churn", "Actual: Churn"],
        )
        fig_cm.update_layout(height=380)
        st.plotly_chart(fig_cm, width="stretch")

    st.subheader("Test Set Summary Metrics (at deployment threshold)")
    tm = artifacts["test_metrics"]
    metric_cols = st.columns(6)
    metric_cols[0].metric("Accuracy", f"{tm['accuracy']:.3f}")
    metric_cols[1].metric("Balanced Acc.", f"{tm['balanced_accuracy']:.3f}")
    metric_cols[2].metric("Precision", f"{tm['precision']:.3f}")
    metric_cols[3].metric("Recall", f"{tm['recall']:.3f}")
    metric_cols[4].metric("F1", f"{tm['f1']:.3f}")
    metric_cols[5].metric("ROC AUC", f"{tm['roc_auc']:.3f}")

    st.markdown(f"""
**Notes**
- Deployment threshold used for the test-set metrics above is **{artifacts['selected_threshold']:.2f}**
  (tuned on validation data, prioritizing recall, then PR AUC, then precision).
- The sidebar threshold slider lets you explore the precision/recall trade-off interactively; metrics in the
  At-Risk Customers tab update live with the slider, while the confusion matrix and summary metrics above reflect
  the fixed deployment threshold from the trained model.
- PR AUC: **{tm['pr_auc']:.3f}**, Brier score: **{tm['brier']:.3f}** (lower is better for calibration).
""")
