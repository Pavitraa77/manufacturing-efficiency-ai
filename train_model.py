from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

df=pd.read_csv(Path(__file__).resolve().parent/"data"/"manufacturing_demo.csv")
cat=["Machine_ID","Operation_Mode"]
num=["Temperature_C","Vibration_Hz","Power_Consumption_kW","Network_Latency_ms","Packet_Loss_%","Quality_Control_Defect_Rate_%","Predictive_Maintenance_Score"]
X=df[cat+num]; y=df["Efficiency_Status"]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
prep=ColumnTransformer([("categorical",OneHotEncoder(handle_unknown="ignore"),cat),("numeric","passthrough",num)])
model=Pipeline([("prep",prep),("rf",RandomForestClassifier(n_estimators=140,random_state=42,class_weight="balanced",n_jobs=-1,min_samples_leaf=2))])
model.fit(Xtr,ytr); pred=model.predict(Xte)
print(f"Rows: {len(df):,}")
print(f"Accuracy: {accuracy_score(yte,pred):.2%}")
print(classification_report(yte,pred,zero_division=0))
