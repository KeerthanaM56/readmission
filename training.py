import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap
import joblib
import os

# ==============================
# 1. Load Data
# ==============================
def load_data(path="data/hospital_readmissions_30k.csv"):
    return pd.read_csv(path)

# ==============================
# 3. Preprocessing + Feature Engineering
# ==============================
def preprocess_data(df):
    # Blood pressure split
    df[['systolic_bp', 'diastolic_bp']] = df['blood_pressure'].str.split('/', expand=True).astype(float)
    df.drop(['blood_pressure', 'patient_id'], axis=1, inplace=True)

    # Binary encoding
    for col in ['diabetes', 'hypertension']:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # Label encoding
    le_gender, le_discharge, le_bmi = LabelEncoder(), LabelEncoder(), LabelEncoder()
    df['gender'] = le_gender.fit_transform(df['gender'])
    df['discharge_destination'] = le_discharge.fit_transform(df['discharge_destination'])

    # Target encoding
    df['readmitted_30_days'] = df['readmitted_30_days'].map({'Yes': 1, 'No': 0})

    # New features
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    df['high_bp_flag'] = ((df['systolic_bp'] > 140) | (df['diastolic_bp'] > 90)).astype(int)
    df['comorbidity_index'] = df['diabetes'] + df['hypertension']
    df['bmi_category'] = pd.cut(df['bmi'], bins=[0,18.5,24.9,29.9,100],
                                labels=['Underweight','Normal','Overweight','Obese'])
    df['bmi_category'] = le_bmi.fit_transform(df['bmi_category'])

    # Risk stratification
    def assign_risk(row):
        if row['readmitted_30_days'] == 1 and row['high_bp_flag'] == 1:
            return "High"
        elif row['readmitted_30_days'] == 1:
            return "Medium"
        else:
            return "Low"
    df['risk_level'] = df.apply(assign_risk, axis=1)

    return df, le_gender, le_discharge, le_bmi


# ==============================
# 4. Scale + SMOTE
# ==============================
def scale_and_balance(X, y):
    scaler = StandardScaler()
    X_scaled = X.copy()
    num_cols = X_scaled.select_dtypes(include=["int64", "float64"]).columns
    X_scaled[num_cols] = scaler.fit_transform(X_scaled[num_cols])

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    return X_resampled, y_resampled, scaler, num_cols


# ==============================
# 5. Define Models
# ==============================
def get_models():
    return {
        "random_forest": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=300, max_depth=-1, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42
        )
    }


# ==============================
# 6. Train + Evaluate
# ==============================
def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        results.append((name, acc, auc))

        print("="*40)
        print(f"📌 {name.upper()} Results")
        print("Accuracy:", acc)
        print("ROC-AUC:", auc)
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        print("✅ Testing Accuracy:", acc)
    return models, results


# ==============================
# 7. SHAP Explainability
# ==============================
def explain_with_shap(model, X_test):
    print("📊 Running SHAP Explainability...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test, show=True)
    shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:], matplotlib=True)


# ==============================
# 8. Save Models
# ==============================
def save_artifacts(models, scaler, le_gender, le_discharge, le_bmi, num_cols, feature_names):
    os.makedirs("models", exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, f"models/{name}_model.pkl")
        print(f"💾 Saved {name} model as models/{name}_model.pkl")

    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(feature_names, "models/feature_names.pkl")
    joblib.dump(num_cols, "models/num_cols.pkl")
    joblib.dump(le_gender, "models/le_gender.pkl")
    joblib.dump(le_discharge, "models/le_discharge.pkl")
    joblib.dump(le_bmi, "models/le_bmi.pkl")
    print("✅ All models + preprocessors saved in models/")


# ==============================
# MAIN PIPELINE
# ==============================
if __name__ == "__main__":
    df = load_data()
    df, le_gender, le_discharge, le_bmi = preprocess_data(df)

    X = df.drop(["readmitted_30_days", "risk_level"], axis=1)
    y = df["readmitted_30_days"]

    X_resampled, y_resampled, scaler, num_cols = scale_and_balance(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
    )

    models = get_models()
    models, results = train_and_evaluate(models, X_train, X_test, y_train, y_test)

    explain_with_shap(models["xgboost"], X_test)

    save_artifacts(models, scaler, le_gender, le_discharge, le_bmi, num_cols, X.columns.tolist())
