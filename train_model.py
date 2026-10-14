from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Thales_Group_Manufacturing.csv"
TARGET = "Efficiency_Status"

df = pd.read_csv(DATA_PATH)
cat = [c for c in ["Machine_ID", "Operation_Mode"] if c in df.columns]
num = [c for c in [
    "Temperature_C", "Vibration_Hz", "Power_Consumption_kW",
    "Network_Latency_ms", "Packet_Loss_%",
    "Quality_Control_Defect_Rate_%", "Production_Speed_units_per_hr",
    "Predictive_Maintenance_Score"
] if c in df.columns]
features = cat + num
data = df[features + [TARGET]].dropna().copy()
X, y = data[features], data[TARGET].astype(str)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
transformers = []
if cat:
    transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore"), cat))
if num:
    transformers.append(("numeric", "passthrough", num))
model = Pipeline([
    ("prep", ColumnTransformer(transformers=transformers)),
    ("rf", RandomForestClassifier(
        n_estimators=160, random_state=42, class_weight="balanced",
        n_jobs=-1, min_samples_leaf=2
    ))
])
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(f"Dataset: {DATA_PATH.name}")
print(f"Rows used: {len(data):,}")
print(f"Features used: {', '.join(features)}")
print(f"Accuracy: {accuracy_score(y_test, pred):.2%}")
print(f"Balanced accuracy: {balanced_accuracy_score(y_test, pred):.2%}")
print(f"Macro F1-score: {f1_score(y_test, pred, average='macro', zero_division=0):.2%}")
print("\\nClassification report:")
print(classification_report(y_test, pred, zero_division=0))
labels = sorted(y.unique())
print("Confusion matrix (true rows, predicted columns); class order:", labels)
print(confusion_matrix(y_test, pred, labels=labels))
print("\\nNote: Error_Rate_%, Date, and Timestamp are excluded to reduce possible target leakage.")
