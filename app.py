from pathlib import Path
import io
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Manufacturing Efficiency AI", page_icon="🏭", layout="wide")
ROOT=Path(__file__).resolve().parent
TARGET="Efficiency_Status"
CAT=["Machine_ID","Operation_Mode"]
NUM=["Temperature_C","Vibration_Hz","Power_Consumption_kW","Network_Latency_ms","Packet_Loss_%","Quality_Control_Defect_Rate_%","Predictive_Maintenance_Score"]
FEATURES=CAT+NUM

st.title("🏭 Manufacturing Efficiency AI")
st.caption("Classify manufacturing efficiency from machine and sensor readings.")
with st.sidebar:
    st.header("Data")
    upload=st.file_uploader("Upload your manufacturing CSV (optional)", type=["csv"])
    page=st.radio("Page",["Overview","Model performance","Predict efficiency","Explore data"])

@st.cache_data
def demo_data():
    return pd.read_csv(ROOT/"data"/"manufacturing_demo.csv")

try:
    raw=pd.read_csv(io.BytesIO(upload.getvalue())) if upload else demo_data()
    missing=[c for c in FEATURES+[TARGET] if c not in raw.columns]
    if missing:
        st.error("Missing required columns: "+", ".join(missing)); st.stop()
    data=raw[FEATURES+[TARGET]].copy()
    for c in NUM: data[c]=pd.to_numeric(data[c],errors="coerce")
    data[TARGET]=data[TARGET].astype(str).str.strip().str.title()
    data=data.dropna()
    data=data[data[TARGET].isin(["High","Medium","Low"])]
    if len(data)<30 or data[TARGET].nunique()<2:
        st.error("Please provide at least 30 complete rows and at least two target classes."); st.stop()
except Exception as e:
    st.error(f"Could not read dataset: {e}"); st.stop()

@st.cache_resource
def fit_model(frame):
    X=frame[FEATURES]; y=frame[TARGET]
    strat=y if y.value_counts().min()>=2 else None
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=strat)
    prep=ColumnTransformer([("categorical",OneHotEncoder(handle_unknown="ignore"),CAT),("numeric","passthrough",NUM)])
    model=Pipeline([("prep",prep),("rf",RandomForestClassifier(n_estimators=140,random_state=42,class_weight="balanced",n_jobs=-1,min_samples_leaf=2))])
    model.fit(Xtr,ytr); pred=model.predict(Xte)
    return model,yte,pred,accuracy_score(yte,pred)
model,y_test,y_pred,acc=fit_model(data)

if page=="Overview":
    a,b,c,d=st.columns(4)
    a.metric("Records",f"{len(data):,}"); b.metric("Machines",data.Machine_ID.nunique())
    c.metric("Test accuracy",f"{acc:.1%}"); d.metric("Classes",data[TARGET].nunique())
    left,right=st.columns(2)
    with left:
        st.subheader("Efficiency status distribution")
        counts=data[TARGET].value_counts().reindex(["High","Medium","Low"]).dropna()
        fig,ax=plt.subplots(); counts.plot(kind="bar",ax=ax); ax.set_xlabel("Status"); ax.set_ylabel("Records"); ax.tick_params(axis="x",rotation=0); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with right:
        st.subheader("Sensor readings by status")
        sensor=st.selectbox("Sensor",NUM)
        fig,ax=plt.subplots()
        for label in ["High","Medium","Low"]:
            vals=data.loc[data[TARGET]==label,sensor]
            if len(vals): ax.hist(vals,bins=24,alpha=.5,label=label)
        ax.set_xlabel(sensor); ax.set_ylabel("Records"); ax.legend(); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.info("The included CSV is synthetic demo data. Upload an approved real dataset before reporting results as real-world performance.")
elif page=="Model performance":
    st.metric("Held-out test accuracy",f"{acc:.2%}")
    labels=["High","Medium","Low"]
    cm=confusion_matrix(y_test,y_pred,labels=labels)
    fig,ax=plt.subplots(); im=ax.imshow(cm); ax.set_xticks(range(3),labels); ax.set_yticks(range(3),labels); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix")
    for i in range(3):
        for j in range(3): ax.text(j,i,str(cm[i,j]),ha="center",va="center")
    fig.colorbar(im,ax=ax); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.subheader("Classification report")
    st.dataframe(pd.DataFrame(classification_report(y_test,y_pred,labels=labels,output_dict=True,zero_division=0)).transpose().round(3),use_container_width=True)
elif page=="Predict efficiency":
    st.subheader("Try a machine reading")
    with st.form("predict"):
        a,b,c=st.columns(3)
        machine=a.selectbox("Machine ID",sorted(data.Machine_ID.astype(str).unique()))
        mode=b.selectbox("Operation mode",sorted(data.Operation_Mode.astype(str).unique()))
        temp=a.number_input("Temperature (°C)",0.0,200.0,float(data.Temperature_C.median()))
        vib=b.number_input("Vibration (Hz)",0.0,100.0,float(data.Vibration_Hz.median()))
        power=c.number_input("Power (kW)",0.0,500.0,float(data.Power_Consumption_kW.median()))
        lat=a.number_input("Network latency (ms)",0.0,1000.0,float(data.Network_Latency_ms.median()))
        loss=b.number_input("Packet loss (%)",0.0,100.0,float(data["Packet_Loss_%"].median()))
        defect=c.number_input("Defect rate (%)",0.0,100.0,float(data["Quality_Control_Defect_Rate_%"].median()))
        maint=st.slider("Predictive maintenance score",0.0,100.0,float(data.Predictive_Maintenance_Score.median()))
        submit=st.form_submit_button("Predict efficiency",type="primary")
    if submit:
        row=pd.DataFrame([{"Machine_ID":machine,"Operation_Mode":mode,"Temperature_C":temp,"Vibration_Hz":vib,"Power_Consumption_kW":power,"Network_Latency_ms":lat,"Packet_Loss_%":loss,"Quality_Control_Defect_Rate_%":defect,"Predictive_Maintenance_Score":maint}],columns=FEATURES)
        pred=model.predict(row)[0]; probs=model.predict_proba(row)[0]
        st.success(f"Predicted efficiency status: **{pred}**")
        st.bar_chart(pd.DataFrame({"Probability":probs},index=model.named_steps["rf"].classes_))
        st.caption("Predictions are estimates for educational demonstration, not safety-critical decisions.")
else:
    st.subheader("Explore data")
    st.write(f"{len(data):,} rows × {len(data.columns)} columns")
    term=st.text_input("Filter by machine ID")
    view=data if not term else data[data.Machine_ID.astype(str).str.contains(term,case=False,na=False)]
    st.dataframe(view.head(500),use_container_width=True)
    st.download_button("Download filtered CSV",view.to_csv(index=False).encode(), "manufacturing_data.csv","text/csv")
st.divider()
st.caption("Manufacturing Efficiency AI • Educational machine-learning project")
