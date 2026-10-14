from pathlib import Path
import warnings

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=UserWarning)
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Thales_Group_Manufacturing.csv"
TARGET = "Efficiency_Status"

st.set_page_config(
    page_title="Manufacturing Efficiency AI",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"Required target column '{TARGET}' was not found.")
    return df

@st.cache_resource
def train_model(df):
    # Keep the target and input columns explicit; exclude Error_Rate_% because
    # it may be mechanically related to the target-generation process.
    cat_candidates = ["Machine_ID", "Operation_Mode"]
    num_candidates = [
        "Temperature_C", "Vibration_Hz", "Power_Consumption_kW",
        "Network_Latency_ms", "Packet_Loss_%",
        "Quality_Control_Defect_Rate_%",
        "Production_Speed_units_per_hr", "Predictive_Maintenance_Score",
    ]
    cat = [c for c in cat_candidates if c in df.columns]
    num = [c for c in num_candidates if c in df.columns]
    features = cat + num
    clean = df[features + [TARGET]].dropna().copy()
    X, y = clean[features], clean[TARGET].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    transformers = []
    if cat:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore"), cat))
    if num:
        transformers.append(("numeric", "passthrough", num))
    prep = ColumnTransformer(transformers=transformers)
    model = Pipeline([
        ("prep", prep),
        ("rf", RandomForestClassifier(
            n_estimators=160, random_state=42, class_weight="balanced",
            n_jobs=-1, min_samples_leaf=2
        ))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "features": features, "categorical": cat, "numeric": num,
        "X_test": X_test, "y_test": y_test, "predictions": pred,
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
        "report": classification_report(y_test, pred, output_dict=True, zero_division=0),
        "labels": sorted(y.unique()),
        "confusion": confusion_matrix(y_test, pred, labels=sorted(y.unique())),
    }
    return model, metrics

st.title("🏭 Manufacturing Efficiency AI")
st.caption("Sensor, production, quality and network analytics for smart manufacturing")
try:
    df = load_data()
    model, metrics = train_model(df)
except Exception as exc:
    st.error(f"The dataset or model could not be loaded: {exc}")
    st.stop()

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Choose a page",
        ["Overview", "Model performance", "Predict efficiency", "Explore data"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Dataset: Thales Group Manufacturing CSV")
    st.caption(f"{len(df):,} records · {len(df.columns)} columns")

if page == "Overview":
    st.subheader("Manufacturing overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total records", f"{len(df):,}")
    c2.metric("Machines", f"{df['Machine_ID'].nunique():,}" if "Machine_ID" in df else "N/A")
    c3.metric("Efficiency classes", f"{df[TARGET].nunique()}")
    c4.metric("Balanced accuracy", f"{metrics['balanced_accuracy']:.1%}")
    left, right = st.columns(2)
    with left:
        counts = df[TARGET].astype(str).value_counts().rename_axis("Efficiency status").reset_index(name="Records")
        st.plotly_chart(px.bar(counts, x="Efficiency status", y="Records", title="Efficiency status distribution",
                               color="Efficiency status"), use_container_width=True)
    with right:
        if "Operation_Mode" in df:
            mode_counts = df["Operation_Mode"].astype(str).value_counts().rename_axis("Operation mode").reset_index(name="Records")
            st.plotly_chart(px.bar(mode_counts, x="Operation mode", y="Records", title="Operation mode distribution"),
                            use_container_width=True)
        elif "Machine_ID" in df:
            machine_counts = df["Machine_ID"].astype(str).value_counts().head(15).rename_axis("Machine").reset_index(name="Records")
            st.plotly_chart(px.bar(machine_counts, x="Machine", y="Records", title="Records by machine"), use_container_width=True)
    st.subheader("Dataset preview")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)
    st.info("The dataset contains network measurements such as latency and packet loss. These are used as network-performance indicators; this dashboard does not connect to or operate a real 6G network.")

elif page == "Model performance":
    st.subheader("Model performance")
    st.caption("Random Forest classifier · stratified 80/20 train/test split · class-weighted training")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    c2.metric("Balanced accuracy", f"{metrics['balanced_accuracy']:.1%}")
    c3.metric("Macro F1-score", f"{metrics['macro_f1']:.1%}")
    st.write("Balanced accuracy and macro F1 help show performance across all classes when class counts differ.")
    labels = metrics["labels"]
    cm_df = pd.DataFrame(metrics["confusion"], index=[f"Actual {x}" for x in labels],
                         columns=[f"Predicted {x}" for x in labels])
    st.plotly_chart(px.imshow(cm_df, text_auto=True, aspect="auto", title="Confusion matrix",
                              labels=dict(x="Predicted class", y="Actual class", color="Count")),
                    use_container_width=True)
    report_df = pd.DataFrame(metrics["report"]).T
    st.subheader("Classification report")
    st.dataframe(report_df.round(3), use_container_width=True)
    st.subheader("Features used")
    st.write(", ".join(metrics["features"]))
    st.warning("Error_Rate_%, Date and Timestamp are excluded to reduce possible target leakage or timestamp-only signals. The target-generation method should be verified before making real-world performance claims.")

elif page == "Predict efficiency":
    st.subheader("Predict efficiency")
    st.write("Enter values below to estimate an efficiency class using the model trained on the uploaded dataset.")
    with st.form("prediction_form"):
        values = {}
        cols = st.columns(2)
        for i, col in enumerate(metrics["features"]):
            with cols[i % 2]:
                if col in metrics["categorical"]:
                    options = sorted(df[col].dropna().astype(str).unique().tolist())
                    values[col] = st.selectbox(col.replace("_", " "), options=options)
                else:
                    series = pd.to_numeric(df[col], errors="coerce").dropna()
                    low, high = float(series.min()), float(series.max())
                    default = float(series.median())
                    values[col] = st.number_input(
                        col.replace("_", " "), min_value=low, max_value=high,
                        value=min(max(default, low), high), format="%.4f"
                    )
        submitted = st.form_submit_button("Predict efficiency", type="primary")
    if submitted:
        row = pd.DataFrame([{c: values[c] for c in metrics["features"]}])
        prediction = model.predict(row)[0]
        st.success(f"Predicted efficiency status: **{prediction}**")
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(row)[0]
            prob_df = pd.DataFrame({"Class": model.classes_, "Probability": probs})
            st.plotly_chart(px.bar(prob_df, x="Class", y="Probability", range_y=[0, 1],
                                   title="Model class probabilities"), use_container_width=True)
        st.caption("This is a model estimate for demonstration, not a certified industrial control or safety decision.")

elif page == "Explore data":
    st.subheader("Explore dataset")
    filtered = df.copy()
    if TARGET in filtered:
        classes = sorted(filtered[TARGET].dropna().astype(str).unique())
        chosen = st.multiselect("Filter efficiency status", classes, default=classes)
        filtered = filtered[filtered[TARGET].astype(str).isin(chosen)]
    if "Machine_ID" in filtered:
        machines = sorted(filtered["Machine_ID"].dropna().astype(str).unique())
        selected_machines = st.multiselect("Filter machine", machines, default=machines)
        filtered = filtered[filtered["Machine_ID"].astype(str).isin(selected_machines)]
    st.caption(f"Showing {len(filtered):,} of {len(df):,} records")
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered data as CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="manufacturing_filtered_data.csv",
        mime="text/csv",
    )
    numeric_columns = [c for c in filtered.select_dtypes(include="number").columns if c != TARGET]
    if numeric_columns:
        chosen_col = st.selectbox("Choose a numeric column to inspect", numeric_columns)
        st.plotly_chart(px.histogram(filtered, x=chosen_col, title=f"Distribution of {chosen_col}"),
                        use_container_width=True)
