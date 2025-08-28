# app.py — Integrated app (prediction + dashboards + PDF + email + clinical summary)

import os, io, csv, datetime, smtplib, joblib, json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, redirect, url_for, send_file, flash, jsonify
from dotenv import load_dotenv
from email.mime.text import MIMEText
import plotly.graph_objs as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

# PDF libs
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# =====================
# Optional: Hugging Face summarizer (from mail.py)
# =====================
summarizer = None
try:
    # Lazy import so the app still works if transformers/torch aren’t installed
    from transformers import pipeline  #
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")
    print("✅ Hugging Face summarizer loaded.")
except Exception as e:
    print("ℹ️ Summarizer disabled (transformers not available):", e)

app = Flask(__name__)
app.secret_key = "supersecret"   # Needed for flash messages

# Set matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# =====================
# Load Model + Encoders (from original app.py)
# =====================
try:
    model = joblib.load("models/xgboost_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    num_cols = joblib.load("models/num_cols.pkl")
    le_gender = joblib.load("models/le_gender.pkl")
    le_discharge = joblib.load("models/le_discharge.pkl")
    le_bmi = joblib.load("models/le_bmi.pkl")
    print("✅ Models loaded successfully")
except Exception as e:
    print("❌ Missing model/scaler/encoder files:", e)

# =====================
# Email Setup
# =====================
load_dotenv()
FROM_EMAIL = os.getenv("EMAIL_USER", "")
EMAIL_PASS = os.getenv("EMAIL_PASS", "")
# =====================
# Configurable Risk Settings
# =====================
USE_OVERRIDES = os.getenv("USE_OVERRIDES", "True") == "True"
HIGH_THRESHOLD = float(os.getenv("HIGH_THRESHOLD", 0.70))
MEDIUM_THRESHOLD = float(os.getenv("MEDIUM_THRESHOLD", 0.40))

# =====================
# Global Colors (for charts, tables, reports)
# =====================
COLOR_MAP = {
    "High": "#ff4757",   # Red
    "Medium": "#ffa502", # Orange
    "Low": "#2ed573"     # Green
}

def send_email(to_email: str, subject: str, body: str) -> str:
    """Simple Gmail SMTP sender reused by /predict."""
    if not FROM_EMAIL or not EMAIL_PASS:
        return "Email disabled: set EMAIL_USER and EMAIL_PASS in .env"
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = FROM_EMAIL
        msg["To"] = to_email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as server:
            server.login(FROM_EMAIL, EMAIL_PASS)
            server.send_message(msg)
        return "Sent"
    except Exception as e:
        return f"Failed: {e}"

# =====================
# Clinical Summary (integrated + generalized from mail.py)
# =====================
def generate_summary(patient_id, doctor_name, features_dict, prob, risk, context_notes=None):
    """
    Builds a concise clinical narrative then summarizes it with HF summarizer if available.
    Falls back to plain narrative if summarizer is unavailable.  
    """
    age = features_dict.get("age", "N/A")
    bmi = features_dict.get("bmi", "N/A")
    chol = features_dict.get("cholesterol", "N/A")
    meds = features_dict.get("medication_count", "N/A")
    los = features_dict.get("length_of_stay", "N/A")
    dia = features_dict.get("diabetes", "N/A")
    htn = features_dict.get("hypertension", "N/A")
    pulse_pressure = features_dict.get("pulse_pressure", "N/A")

    context = f" Additional notes: {context_notes}." if context_notes else ""

    if risk == "High":
        action = "Immediate post-discharge planning and close monitoring are strongly recommended."
    elif risk == "Medium":
        action = "Consider scheduling follow-up and reviewing medication adherence."
    else:
        action = "Routine follow-up may suffice, but continue monitoring for changes."

    base = (
        f"Patient {patient_id} under Dr. {doctor_name} has a predicted readmission probability of {prob:.2f} "
        f"with risk level {risk}. Age {age}, BMI {bmi}, cholesterol {chol}, medications {meds}, "
        f"length of stay {los} days, diabetes {dia}, hypertension {htn}, pulse pressure {pulse_pressure}. "
        f"{action}{context}"
    )

    if summarizer:
        try:
            out = summarizer(base, max_length=120, min_length=30, do_sample=False)[0]['summary_text']
            return out
        except Exception as e:
            print("ℹ️ Summarizer failed, using base text:", e)
    return base

# =====================
# PDF Helpers (from original app.py)
# =====================
def _probability_bar(prob: float) -> bytes:
    buf = io.BytesIO()
    fig, ax = plt.subplots(figsize=(4.5, 0.6))

    if prob >= 0.70:
        color = '#ff4757'   # red for High
    elif prob >= 0.30:
        color = '#ffa502'   # orange for Medium
    else:
        color = '#2ed573'   # green for Low

    ax.barh([0], [prob], height=0.6, color=color, alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Probability", fontweight='bold')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def generate_patient_report(patient_id, doctor_name, doctor_email, features_dict, prob, risk) -> bytes:
    os.makedirs("reports", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"reports/Patient_{patient_id}_{timestamp}.pdf"

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=36, bottomMargin=36, leftMargin=42, rightMargin=42)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Hospital Readmission Risk Report", styles["Title"]))
    story.append(Spacer(1, 12))

    pt_tbl = Table([
        ["Patient ID", patient_id],
        ["Doctor Name", doctor_name],
        ["Doctor Email", doctor_email],
        ["Report Generated", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    ], colWidths=[120, 350])
    pt_tbl.setStyle(TableStyle([
        ("BOX", (0,0), (-1,-1), 0.5, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey)
    ]))
    story.append(pt_tbl)
    story.append(Spacer(1, 12))

    if risk == "High":
        color = colors.red
    elif risk == "Medium":
        color = colors.orange
    else:
        color = colors.green


    sum_tbl = Table([
        ["Predicted Risk", risk],
        ["Probability", f"{prob:.2f}"],
    ], colWidths=[120, 350])
    sum_tbl.setStyle(TableStyle([
        ("TEXTCOLOR", (1,0), (1,0), color),
        ("FONTSIZE", (0,0), (-1,-1), 12),
        ("BOX", (0,0), (-1,-1), 0.5, colors.grey)
    ]))
    story.append(sum_tbl)
    story.append(Spacer(1, 12))

    try:
        bar_bytes = _probability_bar(prob)
        story.append(Image(io.BytesIO(bar_bytes), width=5.2*inch, height=0.8*inch))
    except:
        pass

    story.append(Spacer(1, 12))
    story.append(Paragraph("Clinical Recommendations", styles["Heading3"]))

    if risk == "High":
        recommendations = [
            "Telehealth within 48 hours",
            "Home nurse call within 72 hours",
            "Medication reconciliation & pillbox setup",
            "Early labs & vitals review",
            "Care coordinator & social support outreach"
        ]
    elif risk == "Medium":
        recommendations = [
            "Phone check 5–7 days post-discharge",
            "Pharmacist medication review",
            "Dietician counseling for risk factors",
            "Automated SMS reminders to patient"
        ]
    else:
        recommendations = [
            "Standard discharge education",
            "PCP follow-up in 2–4 weeks",
            "Self-monitoring log (BP/BG/Weight)"
        ]
    for rec in recommendations:
        story.append(Paragraph(f"• {rec}", styles["Normal"]))
    story.append(Spacer(1, 12))

    rows = [["Feature", "Value"]] + [[k.replace("_", " ").title(), str(v)] for k, v in features_dict.items()]
    feat_tbl = Table(rows, colWidths=[200, 270])
    feat_tbl.setStyle(TableStyle([
        ("BOX", (0,0), (-1,-1), 0.5, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.lightblue),
        ("FONTSIZE", (0,0), (-1,-1), 10)
    ]))
    story.append(Paragraph("Model Inputs", styles["Heading3"]))
    story.append(feat_tbl)

    doc.build(story)

    with open(filename, "wb") as f:
        f.write(buf.getvalue())
    buf.seek(0)
    return buf.getvalue()

# =====================
# Analytics + Risk (from original app.py)
# =====================
def get_dashboard_stats():
    if not os.path.exists(LOG_FILE):
        return {
            'total_predictions': 0,
            'high_risk_count': 0,
            'low_risk_count': 0,
            'high_risk_percentage': 0,
            'avg_age': 0,
            'avg_length_stay': 0
        }
    df = pd.read_csv(LOG_FILE)
    return {
        'total_predictions': len(df),
        'high_risk_count': len(df[df['risk'] == 'High']),
        'medium_risk_count': len(df[df['risk'] == 'Medium']),
        'low_risk_count': len(df[df['risk'] == 'Low']),
        'high_risk_percentage': round((len(df[df['risk'] == 'High']) / len(df)) * 100, 1) if len(df) > 0 else 0,
        'avg_age': round(df['age'].mean(), 1) if len(df) > 0 else 0,
        'avg_length_stay': round(df['length_of_stay'].mean(), 1) if len(df) > 0 else 0,
        'recent_predictions': df.tail(5).to_dict('records') if len(df) > 0 else []
    }
def create_risk_distribution_chart():
    if not os.path.exists(LOG_FILE):
        return json.dumps({}, cls=PlotlyJSONEncoder)

    df = pd.read_csv(LOG_FILE)
    risk_counts = df['risk'].value_counts()

    labels = [r for r in ["High", "Medium", "Low"] if r in risk_counts]
    values = [risk_counts[r] for r in labels]
    colors = [COLOR_MAP[r] for r in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=colors)
    )])
    fig.update_layout(title="Risk Distribution", font=dict(size=14), showlegend=True)
    return json.dumps(fig, cls=PlotlyJSONEncoder)

def create_age_risk_chart():
    if not os.path.exists(LOG_FILE):
        return json.dumps({}, cls=PlotlyJSONEncoder)

    df = pd.read_csv(LOG_FILE)
    fig = px.scatter(
        df, x='age', y='probability',
        color='risk',
        color_discrete_map=COLOR_MAP,   # ✅ fixed
        title='Age vs Risk Probability',
        labels={'probability': 'Risk Probability', 'age': 'Age'}
    )
    fig.update_layout(font=dict(size=12))
    return json.dumps(fig, cls=PlotlyJSONEncoder)

def create_predictions_over_time_chart():
    if not os.path.exists(LOG_FILE):
        return json.dumps({}, cls=PlotlyJSONEncoder)

    df = pd.read_csv(LOG_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    recent_df = df[df['timestamp'] > (pd.Timestamp.now() - pd.Timedelta(days=30))]

    fig = px.histogram(
        recent_df, x='timestamp', color='risk',
        color_discrete_map=COLOR_MAP,   # ✅ fixed
        nbins=30,
        title="Predictions Over Time"
    )
    fig.update_layout(font=dict(size=12), barmode='stack')
    return json.dumps(fig, cls=PlotlyJSONEncoder)

def adjust_probability(probability, override_flag=False):
    """
    Apply a universal bump to probability, and a bigger bump if override is triggered.
    """
    prob_percent = probability * 100  # convert to %

    # ✅ Base bump for everyone
    prob_percent += 10

    # ✅ Extra bump if override applies
    if override_flag:
        prob_percent += 20

    # ✅ Cap max at 95%
    prob_percent = min(prob_percent, 95)
    return prob_percent


def classify_risk(probability, override_flag=False):
    """
    Classify risk based on adjusted probability.
    """
    prob_percent = adjust_probability(probability, override_flag)

    if prob_percent < 30:
        return "Low"
    elif prob_percent < 70:
        return "Medium"
    else:
        return "High"



# =====================
# Logging (extended to include summary)
# =====================
LOG_FILE = "prediction_log.csv"
LOG_HEADER = [
    "timestamp", "patient_id", "doctor_name", "doctor_email",
    "age","gender","cholesterol","bmi","diabetes","hypertension",
    "medication_count","length_of_stay","discharge_destination",
    "systolic_bp","diastolic_bp","pulse_pressure","high_bp_flag",
    "comorbidity_index","bmi_category","discharge_risk",
    "meds_per_day","meds_per_comorbidity",
    "probability","risk","override_applied","override_reasons","email_status",
    "summary"  # NEW
]
def log_prediction_row(row: list):
    newfile = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)   # ✅ force quoting
        if newfile:
            w.writerow(LOG_HEADER)
        w.writerow(row)


# =====================
# Routes
# =====================
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            form = request.form
            patient_id = form["patient_id"]
            age = int(form["age"])
            gender = form["gender"]
            bp = form["blood_pressure"]
            cholesterol = float(form["cholesterol"])
            bmi = float(form["bmi"])
            diabetes = 1 if form["diabetes"] == "Yes" else 0
            hypertension = 1 if form["hypertension"] == "Yes" else 0
            medication_count = int(form["medication_count"])
            length_of_stay = int(form["length_of_stay"])
            discharge = form["discharge_destination"].replace(" ", "_")
            doctor_name = form["doctor_name"]
            doctor_email = form["doctor_email"]

            try:
                systolic_bp, diastolic_bp = map(float, bp.split("/"))
            except:
                flash("⚠️ Invalid BP format, use systolic/diastolic", "danger")
                return redirect(url_for("predict"))

            # Prepare features (kept from original)
            data = {
                "age": age,
                "gender": le_gender.transform([gender])[0],
                "cholesterol": cholesterol,
                "bmi": bmi,
                "diabetes": diabetes,
                "hypertension": hypertension,
                "medication_count": medication_count,
                "length_of_stay": length_of_stay,
                "discharge_destination": le_discharge.transform([discharge])[0],
                "systolic_bp": systolic_bp,
                "diastolic_bp": diastolic_bp,
            }
            data["pulse_pressure"] = systolic_bp - diastolic_bp
            data["high_bp_flag"] = 1 if systolic_bp > 140 or diastolic_bp > 90 else 0
            data["comorbidity_index"] = diabetes + hypertension

            if bmi < 18.5: bmi_cat = "Underweight"
            elif bmi < 25: bmi_cat = "Normal"
            elif bmi < 30: bmi_cat = "Overweight"
            else: bmi_cat = "Obese"
            data["bmi_category"] = le_bmi.transform([bmi_cat])[0]

            data["discharge_risk"] = data["discharge_destination"]
            data["meds_per_day"] = medication_count / (length_of_stay + 1)
            data["meds_per_comorbidity"] = medication_count / (data["comorbidity_index"] + 1)

            input_df = pd.DataFrame([data]).reindex(columns=feature_names, fill_value=0)
            input_df[num_cols] = scaler.transform(input_df[num_cols])

            # Prediction
            prediction = model.predict(input_df)[0]
            prediction_proba = float(model.predict_proba(input_df)[0][1])
            
            # -------------------------------
            # Clinical overrides (shortened list)
            # -------------------------------
            override_flag, reasons = False, []

            if age >= 85:
                override_flag, reasons = True, reasons + ["Age ≥ 85"]
            elif age >= 75 and diabetes:
                override_flag, reasons = True, reasons + ["Elderly with diabetes"]
            elif data["comorbidity_index"] >= 2 and age >= 65:
                override_flag, reasons = True, reasons + ["Multiple comorbidities"]
            elif bmi >= 35 and diabetes:
                override_flag, reasons = True, reasons + ["Severe obesity with diabetes"]
            elif discharge in ["Nursing_Facility", "Rehab"]:
                override_flag, reasons = True, reasons + ["Discharge to nursing/rehab facility"]
            # ✅ Apply final adjusted probability (base bump + override bump if any)
            adjusted_prob = adjust_probability(prediction_proba, override_flag)

            # For UI/PDF/logging
            display_prob = round(adjusted_prob, 2)

            # Classify based on adjusted probability
            risk_text = classify_risk(prediction_proba, override_flag)


            # ===== Clinical Summary (new) =====
            # Build a human-readable features dict for summary (use raw values where possible)
            summary_features = {
                "age": age,
                "bmi": bmi,
                "cholesterol": cholesterol,
                "medication_count": medication_count,
                "length_of_stay": length_of_stay,
                "diabetes": "Yes" if diabetes else "No",
                "hypertension": "Yes" if hypertension else "No",
                "pulse_pressure": round(data["pulse_pressure"], 2)
            }
            summary_text = generate_summary(patient_id, doctor_name, summary_features, prediction_proba, risk_text)

            # Email (now includes the summary; uses helper)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            email_status = "Skipped"
            if doctor_email and risk_text == "High":
                email_body = f"""
            Patient ID: {patient_id}
            Probability: {prediction_proba:.2f}
            Risk Level: {risk_text}
            Override Applied: {override_flag}
            Reasons: {', '.join(reasons) if reasons else 'None'}

🧠 Clinical Summary:
{summary_text}
"""
                subject = f"Patient {patient_id} Risk Report"
                email_status = send_email(doctor_email, subject, email_body)
                print("📧 Email status:", email_status)

            # Log Row (extended with summary at the end)
            log_row = [
                timestamp, patient_id, doctor_name, doctor_email,
                age, gender, cholesterol, bmi, diabetes, hypertension,
                medication_count, length_of_stay, discharge,
                systolic_bp, diastolic_bp, data["pulse_pressure"], data["high_bp_flag"],
                data["comorbidity_index"], data["bmi_category"], data["discharge_risk"],
                round(data["meds_per_day"],6), round(data["meds_per_comorbidity"],6),
                round(adjusted_prob, 6), risk_text, int(override_flag), "|".join(reasons),
                email_status, summary_text
            ]
            log_prediction_row(log_row)

            # PDF Report
            pdf_features = {k: (round(v, 3) if isinstance(v,(int,float)) else v) for k,v in data.items()}
            _ = generate_patient_report(patient_id, doctor_name, doctor_email, pdf_features, adjusted_prob / 100, risk_text)

            # Result page
            return render_template(
                "result.html",
                patient_id=patient_id,
                probability=display_prob,
                risk=risk_text,
                override_flag=override_flag,
                reasons=reasons,
                timestamp=timestamp,
                summary=summary_text,
                doctor_name=doctor_name,       # ✅ Added
                doctor_email=doctor_email      # ✅ Added# you can display this in your template if desired
            )

        except Exception as e:
            print("❌ ERROR in /predict:", e)
            flash(f"Error during prediction: {e}", "danger")
            return redirect(url_for("predict"))

    return render_template("predict.html")

@app.route("/dashboard")
def dashboard():
    stats = get_dashboard_stats()
    return render_template("dashboard.html", stats=stats)

# ---------------------
# Dashboard JSON APIs
# ---------------------
@app.route("/api/risk-distribution")
def api_risk_distribution():
    if not os.path.exists(LOG_FILE):
        return jsonify({"data": []})
    df = pd.read_csv(LOG_FILE)
    risk_counts = df['risk'].value_counts().to_dict()
    data = [{
        "labels": list(risk_counts.keys()),
        "values": list(risk_counts.values()),
        "type": "pie",
        "name": "Risk Mix"
    }]
    return jsonify({"data": data})

@app.route("/api/age-risk")
def api_age_risk():
    if not os.path.exists(LOG_FILE):
        return jsonify({"data": []})
    df = pd.read_csv(LOG_FILE)
    data = [{
        "x": df["age"].tolist(),
        "y": df["probability"].tolist(),
        "mode": "markers",
        "type": "scatter",
        "name": "Age vs Risk"
    }]
    return jsonify({"data": data})

@app.route("/api/predictions-timeline")
def api_predictions_timeline():
    if not os.path.exists(LOG_FILE):
        return jsonify({"data": []})
    df = pd.read_csv(LOG_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format="%Y-%m-%d_%H-%M-%S", errors="coerce")
    df['date'] = df['timestamp'].dt.strftime("%Y-%m-%d")
    grouped = df.groupby(["date", "risk"]).size().unstack(fill_value=0)
    traces = []
    if "High" in grouped.columns:
        traces.append({"x": grouped.index.tolist(), "y": grouped["High"].tolist(),
                    "type": "bar", "name": "High", "marker": {"color": "#ff4757"}})
    if "Medium" in grouped.columns:
        traces.append({"x": grouped.index.tolist(), "y": grouped["Medium"].tolist(),
                    "type": "bar", "name": "Medium", "marker": {"color": "#ffa502"}})
    if "Low" in grouped.columns:
        traces.append({"x": grouped.index.tolist(), "y": grouped["Low"].tolist(),
                    "type": "bar", "name": "Low", "marker": {"color": "#2ed573"}})
    return jsonify({"data": traces})
@app.route("/patients")
def patients():
    if not os.path.exists(LOG_FILE):
        flash("No predictions logged yet.", "info")
        return render_template("patients.html", rows=[], colors={})
    df = pd.read_csv(LOG_FILE)
    df = df.sort_values('timestamp', ascending=False)
    rows = df.to_dict(orient="records")
    return render_template("patients.html", rows=rows, colors=COLOR_MAP)  # ✅ added colors
@app.route("/analytics")
def analytics():
    return render_template("analytics.html")

@app.route("/download_pdf/<patient_id>/<timestamp>")
def download_pdf(patient_id, timestamp):
    if not os.path.exists(LOG_FILE):
        flash("No logs found.", "danger")
        return redirect(url_for("dashboard"))
    df = pd.read_csv(LOG_FILE)
    print(f"🔍 Requested download: patient_id={patient_id}, timestamp={timestamp}")
    row = df[(df["patient_id"].astype(str) == str(patient_id)) & (df["timestamp"] == timestamp)]
    if row.empty:
        print("❌ No matching record found!")
        print(df[df["patient_id"].astype(str) == str(patient_id)][["patient_id", "timestamp"]])
        flash("Prediction not found.", "warning")
        return redirect(url_for("dashboard"))
    record = row.iloc[0].to_dict()
    features = {
        k: record[k]
        for k in df.columns
        if k not in LOG_HEADER[:4] + ["probability","risk","override_applied","override_reasons","email_status","summary"]
    }
    pdf_bytes = generate_patient_report(
        patient_id=record["patient_id"],
        doctor_name=record["doctor_name"],
        doctor_email=record["doctor_email"],
        features_dict=features,
        prob=float(record["probability"]),
        risk=record["risk"]
    )
    return send_file(
        io.BytesIO(pdf_bytes),
        download_name=f"Patient_{record['patient_id']}_{record['timestamp']}_report.pdf",
        as_attachment=True
    )

@app.route("/delete_record/<patient_id>/<timestamp>")
def delete_record(patient_id, timestamp):
    if not os.path.exists(LOG_FILE):
        flash("No logs found.", "danger")
        return redirect(url_for("patients"))
    df = pd.read_csv(LOG_FILE)
    def normalize(ts: str) -> str:
        return ts.replace(" ", "_").replace(":", "-")
    safe_ts = normalize(timestamp)
    df["ts_normalized"] = df["timestamp"].astype(str).apply(normalize)
    df = df[~((df["patient_id"] == patient_id) & (df["ts_normalized"] == safe_ts))]
    df.drop(columns=["ts_normalized"], errors="ignore", inplace=True)
    df.to_csv(LOG_FILE, index=False)
    flash(f"Record for Patient {patient_id} deleted successfully.", "success")
    return redirect(url_for("patients"))

@app.route("/export_data")
def export_data():
    if not os.path.exists(LOG_FILE):
        flash("No data to export.", "warning")
        return redirect(url_for("dashboard"))
    return send_file(
        LOG_FILE,
        download_name=f"hospital_predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        as_attachment=True
    )

# =====================
# Run Flask
# =====================
if __name__ == "__main__":
    app.run(host="0.0.0.0",port=int(os.getenv("PORT",8000)))
