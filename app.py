import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import pickle
import plotly.graph_objects as go
import os
import io
from datetime import datetime
import json
import uuid
from pathlib import Path

# =====================================================
# Page Configuration
# =====================================================
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# =====================================================
# Load Model with Error Handling
# =====================================================
@st.cache_resource
def load_model():
    try:
        with open("diabetes_model.pkl", "rb") as f:
            model = pickle.load(f)
        
        with open("imputer.pkl", "rb") as f:
            medians = pickle.load(f)
        
        # Handle different imputer formats
        if isinstance(medians, dict):
            median_dict = medians
        else:
            # If it's a SimpleImputer object
            try:
                columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
                median_dict = dict(zip(columns, medians.statistics_))
            except:
                median_dict = {}
        
        return model, median_dict
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model, medians = load_model()

# =====================================================
# Custom CSS
# =====================================================
st.markdown("""
<style>
.stApp {
    background: #f5f7fa;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 30px;
}

.stTabs [data-baseweb="tab"] {
    font-size: 18px;
    font-weight: bold;
    color: #1a237e;
}

/* Main Title */
.main-title {
    text-align: center;
    color: #1A237E;
    font-size: 48px;
    font-weight: 800;
}

.sub-title {
    text-align: center;
    font-size: 20px;
    color: #555;
    margin-bottom: 40px;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 5px 20px rgba(0,0,0,.08);
}

.section {
    font-size: 30px;
    font-weight: 700;
    margin-bottom: 15px;
}

.info {
    background: #dbeafe;
    padding: 18px;
    border-radius: 12px;
    color: #0f172a;
    font-size: 17px;
}

/* Normal Button Styles */
div.stButton > button {
    width: 100%;
    background: #f0f2f6;
    color: #1a1a1a;
    border: 1px solid #d0d5dd;
    border-radius: 8px;
    height: 50px;
    font-size: 16px;
    font-weight: 500;
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    background: #e4e7ec;
    border-color: #b0b5bd;
    color: #1a1a1a;
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

div.stButton > button:active {
    transform: translateY(0px);
}

/* Form Button */
div.stForm button {
    background: #f0f2f6;
    color: #1a1a1a;
    border: 1px solid #d0d5dd;
    border-radius: 8px;
    height: 50px;
    font-size: 16px;
    font-weight: 500;
    transition: all 0.2s ease;
}

div.stForm button:hover {
    background: #e4e7ec;
    border-color: #b0b5bd;
    color: #1a1a1a;
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

/* ============================================= */
/* BMI Calculator - Bigger Inputs */
/* ============================================= */
.bmi-input-container {
    background: transparent;
    border-radius: 16px;
    padding: 0px;
    box-shadow: none;
    margin-bottom: 20px;
}

.bmi-input-container .stNumberInput {
    width: 100%;
}

.bmi-input-container .stNumberInput input {
    font-size: 24px !important;
    padding: 20px 15px !important;
    height: 70px !important;
    border-radius: 12px !important;
    border: 2px solid #e0e0e0 !important;
    transition: all 0.3s ease;
}

.bmi-input-container .stNumberInput input:focus {
    border-color: #1A237E !important;
    box-shadow: 0 0 0 3px rgba(26, 35, 126, 0.1) !important;
}

.bmi-input-container .stNumberInput label {
    font-size: 16px !important;
    font-weight: 600 !important;
    color: #333 !important;
    margin-bottom: 8px !important;
}

.bmi-calculate-btn {
    margin-top: 20px;
}

.bmi-calculate-btn button {
    width: 100% !important;
    height: 60px !important;
    font-size: 20px !important;
    font-weight: 700 !important;
    background: #1A237E !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
}

.bmi-calculate-btn button:hover {
    background: #283593 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 15px rgba(26, 35, 126, 0.3) !important;
}

.bmi-calculate-btn button:active {
    transform: translateY(0px) !important;
}

/* ============================================= */
/* Slider Styling - Dark Blue */
/* ============================================= */
div[data-baseweb="slider"] {
    margin-top: 5px;
}

div[data-baseweb="slider"] div[role="slider"] {
    background: #1A237E !important;
    width: 18px !important;
    height: 18px !important;
    border: 2px solid white !important;
    box-shadow: 0 2px 6px rgba(26, 35, 126, 0.3) !important;
}

div[data-baseweb="slider"] div[data-testid="stSliderTrack"] {
    background: #e0e0e0 !important;
    height: 6px !important;
    border-radius: 3px !important;
}

div[data-baseweb="slider"] div[data-testid="stSliderTrack"] > div {
    background: #1A237E !important;
}

/* ============================================= */
/* Number Input - No Box Around +/- Buttons */
/* ============================================= */
div[data-testid="stNumberInput"] {
    position: relative;
}

div[data-testid="stNumberInput"] button {
    background: transparent !important;
    color: #1A237E !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 4px 8px !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    min-width: 30px !important;
    min-height: 30px !important;
    box-shadow: none !important;
    transition: all 0.2s ease;
}

div[data-testid="stNumberInput"] button:hover {
    background: rgba(26, 35, 126, 0.08) !important;
    color: #1A237E !important;
    border: none !important;
    box-shadow: none !important;
}

div[data-testid="stNumberInput"] button:active {
    background: rgba(26, 35, 126, 0.15) !important;
    transform: scale(0.95);
}

div[data-testid="stNumberInput"] button:focus {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}

div[data-testid="stNumberInput"] div[data-baseweb="input"] {
    border: 2px solid #d0d5dd !important;
    border-radius: 8px !important;
    background: white !important;
}

div[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within {
    border-color: #1A237E !important;
    box-shadow: 0 0 0 2px rgba(26, 35, 126, 0.1) !important;
}

/* ============================================= */
/* BMI Result Styles */
/* ============================================= */
.bmi-result-box {
    background: white;
    border-radius: 16px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}

.bmi-value-large {
    font-size: 56px;
    font-weight: 800;
    color: #1a1a1a;
    line-height: 1;
    margin: 10px 0 5px 0;
}

.bmi-category {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 10px;
}

.bmi-message {
    font-size: 16px;
    color: #555;
}

.bmi-scale-container {
    background: white;
    border-radius: 16px;
    padding: 25px 30px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    margin: 20px 0;
}

.bmi-scale-bar {
    position: relative;
    height: 30px;
    border-radius: 15px;
    background: linear-gradient(to right, #4fc3f7, #81c784, #fff176, #ff8a65, #ef5350);
    margin: 20px 0 30px 0;
    overflow: visible;
}

.bmi-marker {
    position: absolute;
    top: -12px;
    transform: translateX(-50%);
    width: 28px;
    height: 28px;
    background: #1a237e;
    border: 3px solid white;
    border-radius: 50%;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    z-index: 10;
    transition: left 0.5s ease;
}

.bmi-labels {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    color: #555;
    padding: 0 5px;
    margin-top: 5px;
}

.bmi-labels span {
    text-align: center;
    flex: 1;
}

.bmi-info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 15px;
    margin: 20px 0;
}

.bmi-info-item {
    background: white;
    padding: 18px 20px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    border-left: 4px solid #1a237e;
}

.bmi-info-item .label {
    font-size: 13px;
    color: #888;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.bmi-info-item .value {
    font-size: 20px;
    font-weight: 700;
    color: #1a1a1a;
    margin-top: 4px;
}

.bmi-category-item {
    display: flex;
    justify-content: space-between;
    padding: 10px 15px;
    border-radius: 8px;
    margin: 5px 0;
    font-size: 14px;
}

.bmi-category-item .range {
    color: #666;
    font-size: 13px;
}

.bmi-category-item.active {
    background: #e8eaf6;
    font-weight: 600;
    border-left: 4px solid #1a237e;
}

.bmi-category-item.underweight { border-left: 4px solid #4fc3f7; }
.bmi-category-item.normal { border-left: 4px solid #81c784; }
.bmi-category-item.overweight { border-left: 4px solid #fff176; }
.bmi-category-item.obese { border-left: 4px solid #ef5350; }

.bmi-note {
    background: #f8f9fa;
    padding: 15px 20px;
    border-radius: 10px;
    font-size: 14px;
    color: #666;
    margin-top: 20px;
    border-left: 4px solid #1a237e;
}

/* File Uploader */
div[data-testid="stFileUploader"] button {
    background: #f0f2f6 !important;
    color: #1a1a1a !important;
    border: 1px solid #d0d5dd !important;
    border-radius: 8px !important;
}

div[data-testid="stFileUploader"] button:hover {
    background: #e4e7ec !important;
    border-color: #b0b5bd !important;
}

/* Download Button */
div[data-testid="stDownloadButton"] button {
    background: #f0f2f6 !important;
    color: #1a1a1a !important;
    border: 1px solid #d0d5dd !important;
    border-radius: 8px !important;
}

div[data-testid="stDownloadButton"] button:hover {
    background: #e4e7ec !important;
    border-color: #b0b5bd !important;
}

/* Home Page Styles */
.feature-card {
    background: white;
    padding: 30px;
    border-radius: 16px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    text-align: center;
    transition: all 0.3s ease;
    height: 100%;
}

.feature-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
}

.feature-icon {
    font-size: 48px;
    margin-bottom: 15px;
}

.feature-title {
    font-size: 20px;
    font-weight: 700;
    color: #1A237E;
    margin-bottom: 10px;
}

.feature-desc {
    color: #666;
    font-size: 14px;
    line-height: 1.6;
}

/* History Page Styles */
.history-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 15px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    border-left: 4px solid #1A237E;
}

.history-date {
    font-size: 12px;
    color: #888;
}

.history-result {
    font-size: 16px;
    font-weight: 600;
}

/* Error Box */
.error-box {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 12px;
    padding: 20px;
    margin: 15px 0;
}

.error-box .error-title {
    color: #dc2626;
    font-weight: 700;
    font-size: 18px;
    margin-bottom: 10px;
}

.error-box .error-message {
    color: #991b1b;
    margin-bottom: 15px;
    white-space: pre-line;
}

.error-box .error-solution {
    background: white;
    padding: 15px;
    border-radius: 8px;
    margin-top: 10px;
}

/* Responsive */
@media (max-width: 768px) {
    .bmi-info-grid {
        grid-template-columns: 1fr;
    }
    .bmi-value-large {
        font-size: 40px;
    }
    .bmi-input-container .stNumberInput input {
        font-size: 18px !important;
        height: 55px !important;
        padding: 15px !important;
    }
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# Helper Functions
# =====================================================
def validate_required_fields(glucose, blood_pressure, bmi, age, dpf, skin, insulin, pregnancies):
    """Validate required fields for prediction"""
    errors = []
    
    if pregnancies < 0 or pregnancies > 20:
        errors.append("Pregnancies must be between 0 and 20.")
    
    if glucose <= 0 or glucose > 300:  # Reject 0
        errors.append("Glucose must be between 1 and 300 mg/dL.")
    
    if blood_pressure <= 0 or blood_pressure > 200:  # Reject 0
        errors.append("Blood Pressure must be between 1 and 200 mmHg.")
    
    if skin <= 0 or skin > 99:  # Reject 0
        errors.append("Skin Thickness must be between 1 and 99 mm.")
    
    if insulin < 0 or insulin > 900:
        errors.append("Insulin must be between 0 and 900 mu U/ml.")
    
    if bmi <= 0 or bmi > 100:
        errors.append("BMI must be between 0.1 and 100 kg/m².")
    
    if age < 1 or age > 120:
        errors.append("Age must be between 1 and 120 years.")
    
    if dpf <= 0 or dpf > 3:
        errors.append("Diabetes Pedigree Function must be between 0.01 and 3.0.")
    
    return errors

def replace_zero_values(df, columns):
    """Replace zero values with median values (for manual input only)"""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].replace(0, medians.get(col, 0))
    return df

def create_gauge_chart(diabetes_prob):
    """Create a gauge chart for diabetes risk"""
    if diabetes_prob is None:
        return None
    
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=diabetes_prob,
            number={"suffix": "%"},
            title={"text": "Diabetes Risk"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1A237E"},
                "steps": [
                    {"range": [0, 20], "color": "#4CAF50"},
                    {"range": [20, 40], "color": "#8BC34A"},
                    {"range": [40, 60], "color": "#FFC107"},
                    {"range": [60, 80], "color": "#FF9800"},
                    {"range": [80, 100], "color": "#F44336"}
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.8,
                    "value": diabetes_prob
                }
            }
        )
    )
    
    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def display_recommendation(prediction):
    """Display health recommendations based on prediction"""
    if prediction == 1:
        st.warning("""
        ### ⚠️ High Risk Detected
        
        The machine learning model predicts an elevated likelihood of diabetes.
        
        ### 📋 Recommended Actions
        - Consult a healthcare professional immediately
        - Schedule comprehensive laboratory testing
        - Monitor blood glucose regularly
        - Follow a balanced, low-sugar diet
        - Exercise for at least 30 minutes daily
        - Maintain a healthy weight
        - Attend regular medical checkups
        """)
    else:
        st.success("""
        ### ✅ Low Risk Detected
        
        The model predicts a lower likelihood of diabetes.
        
        ### 📋 Recommended Actions
        - Continue a balanced and nutritious diet
        - Exercise regularly (30+ minutes/day)
        - Stay hydrated
        - Maintain a healthy weight
        - Get annual health checkups
        - Practice healthy lifestyle habits
        """)

def add_to_history(patient_data, prediction, diabetes_prob):
    """Add prediction to history"""
    if "history" not in st.session_state:
        st.session_state.history = []
    
    # Convert patient_data to dict if it's a DataFrame
    if isinstance(patient_data, pd.DataFrame):
        patient_dict = patient_data.to_dict('records')[0] if not patient_data.empty else {}
    else:
        patient_dict = patient_data
    
    history_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "patient_data": patient_dict,
        "prediction": int(prediction),
        "diabetes_probability": round(diabetes_prob, 2) if diabetes_prob is not None else None,
        "risk_level": get_risk_level(diabetes_prob) if diabetes_prob is not None else "Unknown"
    }
    
    st.session_state.history.insert(0, history_entry)
    
    # Keep only last 100 entries
    if len(st.session_state.history) > 100:
        st.session_state.history = st.session_state.history[:100]
    
    # Save to CSV for persistence
    save_history_to_csv()

def save_history_to_csv():
    """Save history to CSV file for persistence"""
    try:
        if "history" in st.session_state and st.session_state.history:
            export_data = []
            for entry in st.session_state.history:
                row = {
                    "Timestamp": entry["timestamp"],
                    "Prediction": "Diabetes" if entry["prediction"] == 1 else "No Diabetes",
                    "Diabetes_Probability": entry.get("diabetes_probability", 0),
                    "Risk_Level": entry.get("risk_level", "Unknown")
                }
                patient_data = entry.get("patient_data", {})
                for key, value in patient_data.items():
                    row[key] = value
                export_data.append(row)
            
            df_export = pd.DataFrame(export_data)
            df_export.to_csv("history.csv", index=False)
    except:
        pass

def load_history_from_csv():
    """Load history from CSV file"""
    try:
        if Path("history.csv").exists():
            df = pd.read_csv("history.csv")
            history = []
            for _, row in df.iterrows():
                entry = {
                    "timestamp": row["Timestamp"],
                    "prediction": 1 if row["Prediction"] == "Diabetes" else 0,
                    "diabetes_probability": row["Diabetes_Probability"],
                    "risk_level": row["Risk_Level"],
                    "patient_data": {}
                }
                exclude_cols = ["Timestamp", "Prediction", "Diabetes_Probability", "Risk_Level"]
                for col in df.columns:
                    if col not in exclude_cols:
                        entry["patient_data"][col] = row[col]
                history.append(entry)
            return history
    except:
        pass
    return []

def get_risk_level(diabetes_prob):
    """Get risk level based on probability"""
    if diabetes_prob is None:
        return "Unknown"
    if diabetes_prob < 20:
        return "Low"
    elif diabetes_prob < 40:
        return "Mild"
    elif diabetes_prob < 60:
        return "Moderate"
    elif diabetes_prob < 80:
        return "High"
    else:
        return "Very High"

def get_risk_color(risk_level):
    """Get color for risk level"""
    colors = {
        "Low": "#4CAF50",
        "Mild": "#8BC34A",
        "Moderate": "#FFC107",
        "High": "#FF9800",
        "Very High": "#F44336",
        "Unknown": "#666"
    }
    return colors.get(risk_level, "#666")

def validate_uploaded_data(df):
    """Validate uploaded data - strict: no zero values allowed"""
    errors = []
    
    # Check for required columns
    required_columns = [
        "Pregnancies", "Glucose", "BloodPressure", 
        "SkinThickness", "Insulin", "BMI", 
        "DiabetesPedigreeFunction", "Age"
    ]
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing columns: {', '.join(missing_cols)}")
        return errors
    
    # Check for empty dataframe
    if df.empty:
        errors.append("The uploaded file is empty.")
        return errors
    
    # Validate each column - STRICT: No zeros allowed for critical values
    # Pregnancies: 0-20 (0 is allowed — a patient can genuinely have had none)
    invalid_preg = df[(df['Pregnancies'] < 0) | (df['Pregnancies'] > 20)]
    if not invalid_preg.empty:
        errors.append(f"⚠️ Pregnancies must be between 0 and 20. Found {len(invalid_preg)} invalid rows.")
    
    # Glucose: 1-300 (0 is NOT allowed)
    invalid_glucose = df[(df['Glucose'] <= 0) | (df['Glucose'] > 300)]
    if not invalid_glucose.empty:
        errors.append(f"⚠️ Glucose must be between 1 and 300 mg/dL. Found {len(invalid_glucose)} invalid rows with 0 or negative values.")
    
    # BloodPressure: 1-200 (0 is NOT allowed)
    invalid_bp = df[(df['BloodPressure'] <= 0) | (df['BloodPressure'] > 200)]
    if not invalid_bp.empty:
        errors.append(f"⚠️ Blood Pressure must be between 1 and 200 mmHg. Found {len(invalid_bp)} invalid rows with 0 or negative values.")
    
    # SkinThickness: 1-99 (0 is NOT allowed)
    invalid_skin = df[(df['SkinThickness'] <= 0) | (df['SkinThickness'] > 99)]
    if not invalid_skin.empty:
        errors.append(f"⚠️ Skin Thickness must be between 1 and 99 mm. Found {len(invalid_skin)} invalid rows with 0 or negative values.")
    
    # Insulin: 1-900 (0 is NOT allowed — treated as missing, same as Glucose/BP/Skin/BMI)
    invalid_insulin = df[(df['Insulin'] <= 0) | (df['Insulin'] > 900)]
    if not invalid_insulin.empty:
        errors.append(f"⚠️ Insulin must be between 1 and 900 mu U/ml. Found {len(invalid_insulin)} invalid rows with 0 or negative values.")
    
    # BMI: 0.1-100
    invalid_bmi = df[(df['BMI'] <= 0) | (df['BMI'] > 100)]
    if not invalid_bmi.empty:
        errors.append(f"⚠️ BMI must be between 0.1 and 100 kg/m². Found {len(invalid_bmi)} invalid rows with 0 or negative values.")
    
    # DiabetesPedigreeFunction: 0.01-3.0
    invalid_dpf = df[(df['DiabetesPedigreeFunction'] <= 0) | (df['DiabetesPedigreeFunction'] > 3)]
    if not invalid_dpf.empty:
        errors.append(f"⚠️ Diabetes Pedigree Function must be between 0.01 and 3.0. Found {len(invalid_dpf)} invalid rows with 0 or negative values.")
    
    # Age: 1-120
    invalid_age = df[(df['Age'] < 1) | (df['Age'] > 120)]
    if not invalid_age.empty:
        errors.append(f"⚠️ Age must be between 1 and 120 years. Found {len(invalid_age)} invalid rows with 0 or negative values.")
    
    # Check for null values
    null_counts = df[required_columns].isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if not null_cols.empty:
        null_messages = [f"{col}: {count} null values" for col, count in null_cols.items()]
        errors.append(f"⚠️ Null values found: {', '.join(null_messages)}")
    
    return errors

def predict_patient_manual(patient_data):
    """Make prediction for manual input with zero replacement.

    NOTE: The RandomForestClassifier in diabetes_model.pkl was trained on
    RAW (unscaled) feature values -- tree splits use thresholds on the
    original scale (e.g. Glucose > 154.5), not standardized values. Do NOT
    apply StandardScaler here; doing so compresses every input into a
    narrow range the model's thresholds were never trained against, which
    silently collapses nearly all predictions into "No Diabetes".
    """
    try:
        # Replace zero values with medians (for manual input)
        zero_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        patient_processed = replace_zero_values(patient_data.copy(), zero_columns)
        
        # Feed raw (unscaled) values directly -- matches training
        prediction = model.predict(patient_processed)[0]
        
        # Get probability if available
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(patient_processed)[0]
            diabetes_prob = probability[1] * 100
            healthy_prob = probability[0] * 100
        else:
            diabetes_prob = None
            healthy_prob = None
        
        return prediction, diabetes_prob, healthy_prob, patient_processed
    
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None, None, None

def predict_patient_upload(patient_data):
    """Make prediction for upload data - NO zero replacement.

    NOTE: See predict_patient_manual -- the model expects raw, unscaled
    values. Scaling here would reintroduce the model/scaler mismatch.
    """
    try:
        # Feed raw (unscaled) values directly -- matches training
        patient_values = patient_data.values if hasattr(patient_data, "values") else patient_data
        prediction = model.predict(patient_values)[0]
        
        # Get probability if available
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(patient_values)[0]
            diabetes_prob = probability[1] * 100
            healthy_prob = probability[0] * 100
        else:
            diabetes_prob = None
            healthy_prob = None
        
        # Flatten raw values for debug display
        raw_values = list(patient_values[0]) if hasattr(patient_values, "__getitem__") else None
        
        return prediction, diabetes_prob, healthy_prob, raw_values
    
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None, None, None

# =====================================================
# BMI Calculator Component
# =====================================================
def bmi_calculator():
    """BMI Calculator with bigger inputs and full-width layout"""
    
    st.markdown("<h1 style='text-align: center; color: #1A237E;'>⚖️ BMI Calculator</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #555; margin-bottom: 30px;'>Calculate your Body Mass Index and assess your health status</p>", unsafe_allow_html=True)
    
    # Input Container - Full Width with bigger inputs (removed white box)
    st.markdown('<div class="bmi-input-container">', unsafe_allow_html=True)
    
    # Weight and Height in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        weight = st.number_input(
            "Weight (kg)",
            min_value=20.0,
            max_value=250.0,
            value=70.0,
            step=0.5,
            help="Enter your weight in kilograms",
            key="bmi_weight"
        )
    
    with col2:
        height = st.number_input(
            "Height (m)",
            min_value=0.50,
            max_value=2.50,
            value=1.70,
            step=0.01,
            help="Enter your height in meters",
            key="bmi_height"
        )
    
    # Calculate Button - Full width below inputs
    st.markdown('<div class="bmi-calculate-btn">', unsafe_allow_html=True)
    calculate_clicked = st.button("📊 Calculate BMI", use_container_width=True, key="bmi_calculate")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display results if calculated
    if calculate_clicked:
        bmi = weight / (height ** 2)
        
        # Determine category
        if bmi < 18.5:
            category = "Underweight"
            color = "#4fc3f7"
            emoji = "📉"
            message = "Consider consulting a nutritionist for a healthy weight gain plan."
            position = (bmi / 40) * 100
        elif bmi < 25:
            category = "Normal Weight"
            color = "#66bb6a"
            emoji = "✅"
            message = "Great job! Maintain your healthy lifestyle."
            position = ((bmi - 18.5) / (24.9 - 18.5)) * 25 + 25
        elif bmi < 30:
            category = "Overweight"
            color = "#ffca28"
            emoji = "⚠️"
            message = "Consider lifestyle changes to reach a healthy weight."
            position = ((bmi - 25) / (29.9 - 25)) * 25 + 50
        else:
            category = "Obese"
            color = "#ef5350"
            emoji = "❌"
            message = "Please consult a healthcare professional for guidance."
            position = min(((bmi - 30) / 10) * 25 + 75, 95)
        
        position = max(2, min(98, position))
        
        # BMI Result Display
        st.markdown(f"""
        <div class="bmi-result-box">
            <div style="font-size: 16px; color: #888; font-weight: 500;">Your BMI</div>
            <div class="bmi-value-large">{bmi:.1f}</div>
            <div class="bmi-category" style="color: {color};">{emoji} {category}</div>
            <div class="bmi-message">{message}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # BMI Scale Bar
        st.markdown("""
        <div class="bmi-scale-container">
            <div style="text-align: center; font-weight: 600; font-size: 18px; margin-bottom: 10px;">
                BMI Scale
            </div>
            <div class="bmi-scale-bar">
                <div class="bmi-marker" style="left: {:.1f}%;"></div>
            </div>
            <div class="bmi-labels">
                <span style="color: #4fc3f7;">Underweight</span>
                <span style="color: #81c784;">Normal</span>
                <span style="color: #fff176;">Overweight</span>
                <span style="color: #ef5350;">Obese</span>
            </div>
        </div>
        """.format(position), unsafe_allow_html=True)
        
        # Detailed Information
        st.markdown("### 📋 Detailed BMI Information")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"""
            <div class="bmi-info-grid">
                <div class="bmi-info-item">
                    <div class="label">Your BMI</div>
                    <div class="value">{bmi:.1f}</div>
                </div>
                <div class="bmi-info-item" style="border-left-color: {color};">
                    <div class="label">Category</div>
                    <div class="value" style="color: {color};">{category}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            categories = [
                ("Underweight", "< 18.5", bmi < 18.5, "underweight"),
                ("Normal", "18.5 - 24.9", 18.5 <= bmi < 25, "normal"),
                ("Overweight", "25 - 29.9", 25 <= bmi < 30, "overweight"),
                ("Obese", ">= 30", bmi >= 30, "obese")
            ]
            
            for name, range_text, active, class_name in categories:
                active_class = "active" if active else ""
                st.markdown(f"""
                <div class="bmi-category-item {class_name} {active_class}">
                    <span>{name}</span>
                    <span class="range">{range_text}</span>
                </div>
                """, unsafe_allow_html=True)
        
        # Health Implications
        st.markdown("### 💡 Health Implications")
        
        if bmi < 18.5:
            st.info("""
            **Underweight (< 18.5):** May indicate malnutrition, eating disorders, 
            or other health issues. Consider consulting a healthcare provider.
            """)
        elif bmi < 25:
            st.success("""
            **Normal (18.5 - 24.9):** Healthy weight range for most adults. 
            Keep up the good work with a balanced diet and regular exercise.
            """)
        elif bmi < 30:
            st.warning("""
            **Overweight (25 - 29.9):** Increased risk of health problems. 
            Consider adopting healthier eating habits and increasing physical activity.
            """)
        else:
            st.error("""
            **Obese (>= 30):** High risk of health problems including diabetes, 
            heart disease, and more. Please consult a healthcare professional.
            """)
        
        st.markdown("""
        <div class="bmi-note">
            <strong>📌 Note:</strong> BMI is a screening tool and doesn't account for 
            muscle mass, bone density, or overall body composition. It should be used 
            as a general guideline, not a definitive diagnostic tool.
        </div>
        """, unsafe_allow_html=True)

# =====================================================
# Home Page
# =====================================================
def home_page():
    st.markdown(
        "<h1 class='main-title'>🏥 Diabetes Prediction System</h1>",
        unsafe_allow_html=True
    )
    
    st.markdown(
        "<p class='sub-title'>An AI-powered tool for early diabetes risk assessment and health monitoring</p>",
        unsafe_allow_html=True
    )
    
    # Features Section
    st.markdown("### 🚀 Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🩺</div>
            <div class="feature-title">Diabetes Prediction</div>
            <div class="feature-desc">
                AI-powered prediction using 8 health parameters.
                Get instant risk assessment and personalized recommendations.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚖️</div>
            <div class="feature-title">BMI Calculator</div>
            <div class="feature-desc">
                Calculate your Body Mass Index and get detailed health insights.
                Track your weight status and receive lifestyle recommendations.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">History Tracking</div>
            <div class="feature-desc">
                View your prediction history and track health trends over time.
                Monitor changes and make informed health decisions.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # How it works
    st.markdown("### 📋 How It Works")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **1️⃣ Enter Data**
        - Manual input or file upload
        - 8 health parameters required
        - Age, Glucose, BMI, etc.
        """)
    
    with col2:
        st.markdown("""
        **2️⃣ AI Analysis**
        - Machine learning prediction
        - Instant risk assessment
        - Probability scoring
        """)
    
    with col3:
        st.markdown("""
        **3️⃣ Get Results**
        - Risk probability score
        - Visual gauge chart
        - Personalized recommendations
        """)

# =====================================================
# History Page
# =====================================================
def history_page():
    st.markdown(
        "<h1 class='main-title'>📊 Prediction History</h1>",
        unsafe_allow_html=True
    )
    
    st.markdown(
        "<p class='sub-title'>View your past predictions and track health trends</p>",
        unsafe_allow_html=True
    )
    
    if "history" not in st.session_state or not st.session_state.history:
        st.info("📭 No predictions in history yet. Start by making a prediction!")
        return
    
    # Summary statistics
    total = len(st.session_state.history)
    high_risk = sum(1 for h in st.session_state.history if h.get("risk_level") in ["High", "Very High"])
    diabetic = sum(1 for h in st.session_state.history if h.get("prediction") == 1)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", total)
    with col2:
        st.metric("High Risk", high_risk, delta=f"{high_risk/total*100:.1f}%" if total > 0 else "0%")
    with col3:
        st.metric("Diabetes Detected", diabetic, delta=f"{diabetic/total*100:.1f}%" if total > 0 else "0%")
    with col4:
        st.metric("Low Risk", total - high_risk, delta=f"{(total-high_risk)/total*100:.1f}%" if total > 0 else "0%")
    
    st.markdown("---")
    
    # History list
    st.markdown("### 📋 Prediction Records")
    
    # Filter options
    filter_col1, filter_col2 = st.columns(2)
    
    with filter_col1:
        risk_filter = st.selectbox(
            "Filter by Risk Level",
            ["All", "Low", "Mild", "Moderate", "High", "Very High"]
        )
    
    with filter_col2:
        sort_order = st.selectbox(
            "Sort by",
            ["Newest First", "Oldest First", "Highest Risk", "Lowest Risk"]
        )
    
    # Filter and sort history with proper datetime handling
    filtered_history = st.session_state.history.copy()
    
    if risk_filter != "All":
        filtered_history = [h for h in filtered_history if h.get("risk_level") == risk_filter]
    
    if sort_order == "Newest First":
        filtered_history.sort(key=lambda x: datetime.strptime(x["timestamp"], "%Y-%m-%d %H:%M:%S"), reverse=True)
    elif sort_order == "Oldest First":
        filtered_history.sort(key=lambda x: datetime.strptime(x["timestamp"], "%Y-%m-%d %H:%M:%S"))
    elif sort_order == "Highest Risk":
        filtered_history.sort(key=lambda x: x.get("diabetes_probability", 0) if x.get("diabetes_probability") is not None else -1, reverse=True)
    elif sort_order == "Lowest Risk":
        filtered_history.sort(key=lambda x: x.get("diabetes_probability", 0) if x.get("diabetes_probability") is not None else 999)
    
    # =====================================================
    # Diabetic vs Non-Diabetic Pie Chart + Risk Level Breakdown
    # (dynamic - both follow the filter above)
    # =====================================================
    st.markdown("### 📊 Diabetic vs Non-Diabetic Overview")
    
    diabetic_count = sum(1 for h in filtered_history if h.get("prediction") == 1)
    non_diabetic_count = sum(1 for h in filtered_history if h.get("prediction") == 0)
    
    if diabetic_count + non_diabetic_count > 0:
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            pie_fig = go.Figure(
                go.Pie(
                    labels=["Non-Diabetic", "Diabetic"],
                    values=[non_diabetic_count, diabetic_count],
                    marker_colors=["#4CAF50", "#F44336"],
                    hole=0.45,
                    textinfo="label+percent",
                    textfont=dict(size=13)
                )
            )
            pie_fig.update_layout(
                height=380,
                margin=dict(l=20, r=20, t=40, b=20),
                title=dict(text="Prediction Outcome", x=0.5, xanchor="center"),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
                paper_bgcolor="white"
            )
            st.plotly_chart(pie_fig, use_container_width=True, key="history_pie_chart")
        
        with chart_col2:
            # Risk Level Breakdown - shows severity distribution, not just binary outcome
            risk_order = ["Low", "Mild", "Moderate", "High", "Very High"]
            risk_colors_map = {
                "Low": "#4CAF50", "Mild": "#8BC34A", "Moderate": "#FFC107",
                "High": "#FF9800", "Very High": "#F44336"
            }
            risk_counts = {r: sum(1 for h in filtered_history if h.get("risk_level") == r) for r in risk_order}
            risk_counts = {r: c for r, c in risk_counts.items() if c > 0}
            
            if risk_counts:
                risk_fig = go.Figure(
                    go.Pie(
                        labels=list(risk_counts.keys()),
                        values=list(risk_counts.values()),
                        marker_colors=[risk_colors_map[r] for r in risk_counts.keys()],
                        hole=0.45,
                        textinfo="label+percent",
                        textfont=dict(size=13)
                    )
                )
                risk_fig.update_layout(
                    height=380,
                    margin=dict(l=20, r=20, t=40, b=20),
                    title=dict(text="Risk Level Breakdown", x=0.5, xanchor="center"),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
                    paper_bgcolor="white"
                )
                st.plotly_chart(risk_fig, use_container_width=True, key="history_risk_pie_chart")
        
        st.caption(
            f"Showing {diabetic_count + non_diabetic_count} record(s) "
            f"matching the current filter ({risk_filter})."
        )
        
        # =====================================================
        # Diabetes Probability Trend Over Time
        # Shows whether risk is climbing, falling, or stable across
        # successive predictions -- useful for tracking a single
        # patient over repeated checks, or spotting drift in a batch.
        # =====================================================
        st.markdown("### 📈 Diabetes Probability Trend")
        
        trend_entries = [
            h for h in filtered_history if h.get("diabetes_probability") is not None
        ]
        trend_entries = sorted(
            trend_entries,
            key=lambda x: datetime.strptime(x["timestamp"], "%Y-%m-%d %H:%M:%S")
        )
        
        if len(trend_entries) >= 2:
            timestamps = [e["timestamp"] for e in trend_entries]
            probs = [e["diabetes_probability"] for e in trend_entries]
            point_colors = [get_risk_color(e.get("risk_level", "Unknown")) for e in trend_entries]
            
            trend_fig = go.Figure(
                go.Scatter(
                    x=timestamps,
                    y=probs,
                    mode="lines+markers",
                    line=dict(color="#1A237E", width=2),
                    marker=dict(size=9, color=point_colors, line=dict(width=1, color="white")),
                    hovertemplate="%{x}<br>Diabetes Probability: %{y:.1f}%<extra></extra>"
                )
            )
            trend_fig.add_hline(
                y=50, line_dash="dot", line_color="#999",
                annotation_text="50% threshold", annotation_position="top left"
            )
            trend_fig.update_layout(
                height=380,
                margin=dict(l=20, r=20, t=30, b=60),
                yaxis_title="Diabetes Probability (%)",
                xaxis_title="Prediction Timestamp",
                yaxis=dict(range=[0, 100], gridcolor="#eee"),
                xaxis=dict(showgrid=False, tickangle=-30),
                plot_bgcolor="white",
                paper_bgcolor="white",
                showlegend=False
            )
            st.plotly_chart(trend_fig, use_container_width=True, key="history_trend_chart")
            st.caption(
                "Marker color reflects risk level at that prediction. "
                "Useful for spotting whether risk is rising, falling, or stable across repeated checks."
            )
        else:
            st.info("Need at least 2 matching predictions to plot a trend.")
    else:
        st.info("No records match the current filter to display in the chart.")
    
    st.markdown("---")
    
    # Display history entries
    for entry in filtered_history:
        risk_level = entry.get("risk_level", "Unknown")
        color = get_risk_color(risk_level)
        diabetes_prob = entry.get("diabetes_probability")
        
        with st.container():
            st.markdown(f"""
            <div class="history-card" style="border-left-color: {color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span class="history-date">🕐 {entry['timestamp']}</span>
                        <br>
                        <span class="history-result" style="color: {color};">
                            {entry['prediction'] == 1 and '🔴' or '🟢'} 
                            {entry['prediction'] == 1 and 'Diabetes Detected' or 'No Diabetes'}
                        </span>
                        <br>
                        <span style="font-size: 14px; color: #666;">
                            Risk Level: <strong style="color: {color};">{risk_level}</strong>
                            {f'| Probability: <strong>{diabetes_prob:.1f}%</strong>' if diabetes_prob is not None else ''}
                        </span>
                    </div>
                    <div style="text-align: right; font-size: 12px; color: #888;">
                        <span>Pregnancies: {entry.get('patient_data', {}).get('Pregnancies', 'N/A')}</span><br>
                        <span>Glucose: {entry.get('patient_data', {}).get('Glucose', 'N/A')}</span><br>
                        <span>BMI: {entry.get('patient_data', {}).get('BMI', 'N/A')}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Export functionality
    st.markdown("---")
    st.markdown("### 💾 Export History")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            if Path("history.csv").exists():
                Path("history.csv").unlink()
            st.rerun()
    
    with col2:
        if st.session_state.history:
            # Convert history to DataFrame for export
            export_data = []
            for entry in st.session_state.history:
                row = {
                    "Timestamp": entry["timestamp"],
                    "Prediction": "Diabetes" if entry["prediction"] == 1 else "No Diabetes",
                    "Diabetes_Probability": entry.get("diabetes_probability", 0),
                    "Risk_Level": entry.get("risk_level", "Unknown")
                }
                # Add patient data
                patient_data = entry.get("patient_data", {})
                for key, value in patient_data.items():
                    row[key] = value
                export_data.append(row)
            
            df_export = pd.DataFrame(export_data)
            csv = df_export.to_csv(index=False)
            
            st.download_button(
                "📥 Download History as CSV",
                csv,
                "prediction_history.csv",
                "text/csv",
                use_container_width=True
            )


# =====================================================
# =====================================================
# Model Insights Page — REDESIGNED FROM NEW TRAINING NOTEBOOK
# =====================================================
st.markdown("""
<style>
.insight-hero {
    background: linear-gradient(135deg, #eef2ff 0%, #ffffff 100%);
    border: 1px solid #d9def5;
    border-radius: 20px;
    padding: 24px 26px;
    margin: 8px 0 22px 0;
}
.insight-hero-title {
    color: #1A237E;
    font-size: 28px;
    font-weight: 800;
    margin-bottom: 6px;
}
.insight-hero-text {
    color: #4b5563;
    font-size: 15.5px;
    line-height: 1.7;
}
.insight-section-header {
    font-size: 26px;
    font-weight: 800;
    color: #1A237E;
    margin: 10px 0 5px 0;
}
.insight-section-sub {
    color: #4b5563;
    font-size: 16px;
    line-height: 1.7;
    margin-bottom: 20px;
}
.insight-card {
    background: white;
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 2px 12px rgba(0,0,0,.06);
    border: 1px solid #edf0f7;
    height: 100%;
}
.insight-card-title {
    color: #1A237E;
    font-size: 17px;
    font-weight: 750;
    margin-bottom: 8px;
}
.insight-card-text {
    color: #555;
    font-size: 14px;
    line-height: 1.7;
}
.insight-kpi {
    background: white;
    border-radius: 16px;
    padding: 18px 10px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,.06);
    border-top: 4px solid #1A237E;
}
.insight-kpi-value {
    font-size: 23px;
    font-weight: 800;
    color: #1A237E;
}
.insight-kpi-label {
    font-size: 11px;
    color: #777;
    text-transform: uppercase;
    letter-spacing: .4px;
    margin-top: 4px;
}
.insight-divider {
    border: none;
    border-top: 2px dotted #b0b8e0;
    margin: 28px 0;
}
.insight-note {
    background: #f8f9fc;
    border-left: 4px solid #1A237E;
    border-radius: 10px;
    padding: 14px 16px;
    color: #4b5563;
    font-size: 14px;
    line-height: 1.7;
}
.insight-warning {
    background: #fff8e1;
    border-left: 4px solid #e0a800;
    border-radius: 10px;
    padding: 14px 16px;
    color: #6b5200;
    font-size: 14px;
    line-height: 1.7;
}
.insight-nav-step {
    text-align: center;
    color: #8891bb;
    font-size: 13px;
    font-weight: 700;
    padding-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------
# Numbers below are taken from the NEW DiabetesPredictor notebook.
# The notebook uses:
#   - 768 original rows / 8 input features
#   - 80:20 stratified split = 614 train / 154 test
#   - median imputation fitted on training data only
#   - LOF (20 neighbours, 5% contamination) on TRAINING data only
#   - 583 clean training rows after removing 31 outliers
#   - 5-fold Stratified CV
#   - GridSearchCV optimising RECALL
#   - final selection based on Recall + F1 + ROC-AUC ranking
# -----------------------------------------------------------------

INSIGHT_TUNED_RESULTS = pd.DataFrame([
    {
        "Model": "Tuned KNN",
        "Accuracy": 0.7468,
        "Precision": 0.6471,
        "Recall": 0.6111,
        "F1-score": 0.6286,
        "ROC-AUC": 0.7812,
        "False Negative": 21,
        "True Positive": 33,
        "False Positive": 18,
        "True Negative": 82,
    },
    {
        "Model": "Tuned SVM",
        "Accuracy": 0.6623,
        "Precision": 0.5122,
        "Recall": 0.7778,
        "F1-score": 0.6176,
        "ROC-AUC": 0.7859,
        "False Negative": 12,
        "True Positive": 42,
        "False Positive": 40,
        "True Negative": 60,
    },
    {
        "Model": "Tuned Random Forest",
        "Accuracy": 0.7468,
        "Precision": 0.6119,
        "Recall": 0.7593,
        "F1-score": 0.6777,
        "ROC-AUC": 0.8213,
        "False Negative": 13,
        "True Positive": 41,
        "False Positive": 26,
        "True Negative": 74,
    },
])

INSIGHT_BASELINE_RESULTS = pd.DataFrame([
    {"Model": "Baseline KNN", "Accuracy": 0.7403, "Precision": 0.6346, "Recall": 0.6111, "F1-score": 0.6226, "ROC-AUC": 0.7931},
    {"Model": "Baseline SVM", "Accuracy": 0.7273, "Precision": 0.6429, "Recall": 0.5000, "F1-score": 0.5625, "ROC-AUC": 0.8010},
    {"Model": "Baseline Random Forest", "Accuracy": 0.7338, "Precision": 0.6383, "Recall": 0.5556, "F1-score": 0.5941, "ROC-AUC": 0.8065},
])

INSIGHT_FEATURE_IMPORTANCE = pd.DataFrame([
    ("Glucose", 0.379880),
    ("BMI", 0.206123),
    ("Age", 0.173983),
    ("Insulin", 0.086861),
    ("DiabetesPedigreeFunction", 0.058203),
    ("SkinThickness", 0.042808),
    ("Pregnancies", 0.029772),
    ("BloodPressure", 0.022370),
], columns=["Feature", "Importance"])

INSIGHT_OVERFITTING = pd.DataFrame([
    ("KNN", 0.7715, 0.5912, 0.1803),
    ("SVM", 0.8244, 0.7982, 0.0262),
    ("Random Forest", 0.8459, 0.7828, 0.0631),
], columns=["Model", "Training Recall", "Validation Recall", "Train-Validation Gap"])

INSIGHT_ACCURACY_GAPS = pd.DataFrame([
    ("Baseline KNN", 0.8370, 0.7403, 0.0968),
    ("Baseline SVM", 0.8405, 0.7273, 0.1132),
    ("Baseline Random Forest", 1.0000, 0.7338, 0.2662),
    ("Tuned KNN", 0.8542, 0.7468, 0.1074),
    ("Tuned SVM", 0.7547, 0.6623, 0.0924),
    ("Tuned Random Forest", 0.8096, 0.7468, 0.0629),
], columns=["Model", "Train Accuracy", "Test Accuracy", "Train-Test Gap"])


def insight_header(title, subtitle):
    st.markdown(
        f"""
        <div class="insight-section-header">{title}</div>
        <div class="insight-section-sub">{subtitle}</div>
        """,
        unsafe_allow_html=True
    )


def insight_card(title, text):
    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-card-title">{title}</div>
            <div class="insight-card-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def insight_nav_callback(section_name):
    st.session_state["insights_section_v2"] = section_name


def model_insights_page():
    """Model Insights redesigned to match the latest training notebook."""

    sections = [
        "📊 Overview",
        "🤖 Model Performance",
        "🎯 Error & Generalization",
        "🌟 Feature Importance",
    ]

    if "insights_section_v2" not in st.session_state:
        st.session_state["insights_section_v2"] = sections[0]

    st.markdown(
        """
        <div class="insight-hero">
            <div class="insight-hero-title">📈 Model Insights</div>
            <div class="insight-hero-text">
                Explore how the latest training pipeline prepared the diabetes dataset,
                compared tuned models, analysed prediction errors, and selected the final
                Random Forest model. All figures on this page come from the new training notebook.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # KPI strip — values from the new notebook
    kpis = [
        ("🗂️", "768", "Original patients"),
        ("🧪", "583", "Clean training rows"),
        ("🧍", "154", "Untouched test rows"),
        ("🌳", "RF", "Selected model"),
        ("🎯", "82.13%", "RF ROC-AUC"),
    ]

    cols = st.columns(5)
    for col, (icon, value, label) in zip(cols, kpis):
        with col:
            st.markdown(
                f"""
                <div class="insight-kpi">
                    <div style="font-size:24px;">{icon}</div>
                    <div class="insight-kpi-value">{value}</div>
                    <div class="insight-kpi-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    section = st.radio(
        "Model insight section",
        sections,
        horizontal=True,
        label_visibility="collapsed",
        key="insights_section_v2"
    )

    # =============================================================
    # 1. OVERVIEW
    # =============================================================
    if section == "📊 Overview":
        insight_header(
            "📊 Training Pipeline Overview",
            "A compact view of the exact data-processing and model-selection workflow used by the new notebook."
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            insight_card(
                "1. Data preparation",
                "<b>768</b> patient records and <b>8</b> clinical input features were used. "
                "The target contains <b>500 non-diabetic</b> and <b>268 diabetic</b> cases "
                "(34.90% diabetic)."
            )
        with c2:
            insight_card(
                "2. Missing-value treatment",
                "Zeros in Glucose, Blood Pressure, Skin Thickness, Insulin and BMI were treated "
                "as missing. Median imputation was fitted on the training set only, then applied to the test set."
            )
        with c3:
            insight_card(
                "3. Outlier treatment",
                "Local Outlier Factor used 20 neighbours and 5% contamination. "
                "<b>31 training rows</b> were flagged, leaving <b>583 clean training rows</b>. "
                "The 154-row test set was not outlier-filtered."
            )

        st.markdown('<hr class="insight-divider">', unsafe_allow_html=True)

        st.markdown("### 🧹 Missing Values Recovered from Zero Codes")

        zero_df = pd.DataFrame({
            "Feature": ["Insulin", "SkinThickness", "BloodPressure", "BMI", "Glucose"],
            "Zero values treated as missing": [374, 227, 35, 11, 5]
        })

        fig = go.Figure(
            go.Bar(
                x=zero_df["Zero values treated as missing"],
                y=zero_df["Feature"],
                orientation="h",
                text=zero_df["Zero values treated as missing"],
                textposition="auto"
            )
        )
        fig.update_layout(
            height=340,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Number of records",
            yaxis_title="",
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
            <div class="insight-note">
                <b>Why this matters:</b> the notebook does not treat these zeros as real physiological
                measurements. They are converted to missing values before imputation, preventing impossible
                values such as zero blood pressure from being learned by the models.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<hr class="insight-divider">', unsafe_allow_html=True)

        st.markdown("### 🔬 Training / Test Split and Tuning")

        split_col, tune_col = st.columns(2)
        with split_col:
            st.metric("Training set", "614 rows")
            st.metric("After LOF", "583 rows")
            st.metric("Test set", "154 rows")
        with tune_col:
            st.markdown(
                """
                <div class="insight-card-text">
                <b>KNN:</b> 28 configurations, 5-fold CV<br>
                <b>SVM:</b> 32 configurations, 5-fold CV<br>
                <b>Random Forest:</b> 432 configurations, 5-fold CV<br><br>
                GridSearchCV optimised <b>recall</b>, which is especially relevant here because
                missing a diabetic case is represented by a false negative.
                </div>
                """,
                unsafe_allow_html=True
            )

    # =============================================================
    # 2. MODEL PERFORMANCE
    # =============================================================
    elif section == "🤖 Model Performance":
        insight_header(
            "🤖 Model Performance",
            "The three tuned models were evaluated on the same untouched 154-row test set."
        )

        display_df = INSIGHT_TUNED_RESULTS[
            ["Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
        ].copy()

        styled = display_df.style.format({
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1-score": "{:.2%}",
            "ROC-AUC": "{:.2%}",
        }).background_gradient(
            cmap="Blues",
            subset=["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.markdown("### 📊 Tuned Model Comparison")

        metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
        fig = go.Figure()

        for metric in metrics:
            fig.add_trace(
                go.Bar(
                    name=metric,
                    x=INSIGHT_TUNED_RESULTS["Model"],
                    y=INSIGHT_TUNED_RESULTS[metric],
                    text=[f"{v:.3f}" for v in INSIGHT_TUNED_RESULTS[metric]],
                    textposition="auto"
                )
            )

        fig.update_layout(
            barmode="group",
            height=450,
            yaxis=dict(range=[0, 1], title="Score"),
            xaxis_title="Model",
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
            <div class="insight-note">
                <b>Why Random Forest was selected:</b> it ties KNN on accuracy (74.68%) but has a much stronger
                F1-score (67.77%) and the highest ROC-AUC (82.13%). Its recall is also high at 75.93%,
                identifying 41 of the 54 diabetic test cases.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<hr class="insight-divider">', unsafe_allow_html=True)

        st.markdown("### 📈 ROC-AUC Comparison")

        auc_df = INSIGHT_TUNED_RESULTS[["Model", "ROC-AUC"]]
        auc_fig = go.Figure(
            go.Bar(
                x=auc_df["Model"],
                y=auc_df["ROC-AUC"],
                text=[f"{v:.3f}" for v in auc_df["ROC-AUC"]],
                textposition="auto"
            )
        )
        auc_fig.update_layout(
            height=330,
            yaxis=dict(range=[0.5, 0.9], title="ROC-AUC"),
            xaxis_title="",
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        st.plotly_chart(auc_fig, use_container_width=True)

        st.caption(
            "ROC-AUC summarises how well each model separates diabetic and non-diabetic patients "
            "across probability thresholds."
        )

    # =============================================================
    # 3. ERROR & GENERALIZATION
    # =============================================================
    elif section == "🎯 Error & Generalization":
        insight_header(
            "🎯 Error Analysis & Generalization",
            "For a health-risk screening model, false negatives and generalization are more informative than accuracy alone."
        )

        # Confusion matrices
        cm_cols = st.columns(3)
        for col, row in zip(cm_cols, INSIGHT_TUNED_RESULTS.to_dict("records")):
            with col:
                cm = [
                    [row["True Negative"], row["False Positive"]],
                    [row["False Negative"], row["True Positive"]]
                ]

                fig = go.Figure(
                    go.Heatmap(
                        z=cm,
                        x=["Predicted 0", "Predicted 1"],
                        y=["Actual 0", "Actual 1"],
                        text=cm,
                        texttemplate="%{text}",
                        colorscale="Blues",
                        showscale=False
                    )
                )
                fig.update_layout(
                    title=row["Model"],
                    height=300,
                    margin=dict(l=30, r=20, t=50, b=35),
                    xaxis_title="Prediction",
                    yaxis_title="Actual"
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🚨 False-Negative Comparison")

        fn_fig = go.Figure(
            go.Bar(
                x=INSIGHT_TUNED_RESULTS["Model"],
                y=INSIGHT_TUNED_RESULTS["False Negative"],
                text=INSIGHT_TUNED_RESULTS["False Negative"],
                textposition="auto"
            )
        )
        fn_fig.update_layout(
            height=330,
            yaxis_title="False negatives",
            xaxis_title="",
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        st.plotly_chart(fn_fig, use_container_width=True)

        st.markdown(
            """
            <div class="insight-note">
                <b>Tuned Random Forest:</b> 13 false negatives out of 54 actual diabetic cases,
                corresponding to 75.93% recall. Tuned SVM has fewer false negatives (12), but it also
                produces 40 false positives and has much lower overall accuracy and F1-score.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<hr class="insight-divider">', unsafe_allow_html=True)

        st.markdown("### 🧠 5-Fold Cross-Validation Generalization")

        gap_fig = go.Figure()
        gap_fig.add_trace(
            go.Bar(
                name="Training Recall",
                x=INSIGHT_OVERFITTING["Model"],
                y=INSIGHT_OVERFITTING["Training Recall"]
            )
        )
        gap_fig.add_trace(
            go.Bar(
                name="Validation Recall",
                x=INSIGHT_OVERFITTING["Model"],
                y=INSIGHT_OVERFITTING["Validation Recall"]
            )
        )
        gap_fig.update_layout(
            barmode="group",
            height=360,
            yaxis=dict(range=[0, 1], title="Recall"),
            xaxis_title="Model",
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        st.plotly_chart(gap_fig, use_container_width=True)

        st.dataframe(
            INSIGHT_OVERFITTING.style.format({
                "Training Recall": "{:.2%}",
                "Validation Recall": "{:.2%}",
                "Train-Validation Gap": "{:.2%}",
            }),
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            """
            <div class="insight-warning">
                <b>Important:</b> KNN has the largest train-validation recall gap (18.03%).
                Random Forest's gap is 6.31%, while SVM's is 2.62%. This means Random Forest does not
                show the strongest overfitting signal in the 5-fold recall analysis.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### 📉 Train vs Test Accuracy")

        acc_fig = go.Figure()
        acc_fig.add_trace(go.Bar(
            name="Train Accuracy",
            x=INSIGHT_ACCURACY_GAPS["Model"],
            y=INSIGHT_ACCURACY_GAPS["Train Accuracy"]
        ))
        acc_fig.add_trace(go.Bar(
            name="Test Accuracy",
            x=INSIGHT_ACCURACY_GAPS["Model"],
            y=INSIGHT_ACCURACY_GAPS["Test Accuracy"]
        ))
        acc_fig.update_layout(
            barmode="group",
            height=420,
            yaxis=dict(range=[0, 1.05], title="Accuracy"),
            xaxis_title="Model",
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        st.plotly_chart(acc_fig, use_container_width=True)

        st.caption(
            "The tuned Random Forest has a 6.29-point train-test accuracy gap, down substantially "
            "from the baseline Random Forest's 26.62-point gap."
        )

    # =============================================================
    # 4. FEATURE IMPORTANCE
    # =============================================================
    elif section == "🌟 Feature Importance":
        insight_header(
            "🌟 Feature Importance & Final Model",
            "The final Random Forest uses all eight clinical inputs, but their contribution is not equal."
        )

        fi = INSIGHT_FEATURE_IMPORTANCE.sort_values("Importance", ascending=True)

        fig = go.Figure(
            go.Bar(
                x=fi["Importance"],
                y=fi["Feature"],
                orientation="h",
                text=[f"{v:.1%}" for v in fi["Importance"]],
                textposition="auto"
            )
        )
        fig.update_layout(
            height=450,
            xaxis=dict(range=[0, 0.42], title="Random Forest feature importance"),
            yaxis_title="",
            margin=dict(l=20, r=30, t=20, b=20),
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            INSIGHT_FEATURE_IMPORTANCE.style.format({
                "Importance": "{:.2%}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            """
            <div class="insight-note">
                <b>Main finding:</b> Glucose is the strongest feature (37.99%), followed by BMI (20.61%)
                and Age (17.40%). Together, these three account for about 76% of the Random Forest's
                total feature importance. Diabetes Pedigree Function contributes 5.82%.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<hr class="insight-divider">', unsafe_allow_html=True)

        st.markdown("### 🌳 Selected Random Forest Configuration")

        p1, p2 = st.columns(2)
        with p1:
            st.metric("Test Accuracy", "74.68%")
            st.metric("Recall", "75.93%")
            st.metric("F1-score", "67.77%")
            st.metric("ROC-AUC", "82.13%")

        with p2:
            st.markdown(
                """
                <div class="insight-card">
                    <div class="insight-card-title">Best parameters from GridSearchCV</div>
                    <div class="insight-card-text">
                        <b>n_estimators:</b> 30<br>
                        <b>max_depth:</b> 4<br>
                        <b>min_samples_split:</b> 2<br>
                        <b>min_samples_leaf:</b> 5<br>
                        <b>max_features:</b> log2<br>
                        <b>class_weight:</b> balanced<br><br>
                        <b>5-fold CV recall:</b> 78.28%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown('<hr class="insight-divider">', unsafe_allow_html=True)

        st.markdown("### 🏆 Why the Notebook Selected Random Forest")

        st.markdown(
            """
            <div class="insight-card-text">
                The notebook ranks the tuned models using <b>Recall</b>, <b>F1-score</b> and
                <b>ROC-AUC</b>. Random Forest ranks first with an overall rank of <b>1.333</b>,
                ahead of Tuned SVM (2.000) and Tuned KNN (2.667). It therefore becomes the final
                model saved by the notebook.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="insight-warning" style="margin-top:16px;">
                <b>Deployment note:</b> the new notebook saves the final model as
                <code>final_model.pkl</code> and the median imputer as <code>median_imputer.pkl</code>.
                The current prediction UI loads <code>diabetes_model.pkl</code> and <code>imputer.pkl</code>.
                If you want the live prediction tab to use exactly the new notebook's final artifacts,
                those filenames/load paths should be aligned separately.
            </div>
            """,
            unsafe_allow_html=True
        )

    # Section navigation
    st.markdown('<hr class="insight-divider" style="margin:26px 0 10px 0;">', unsafe_allow_html=True)

    current_idx = sections.index(section)
    nav_back, nav_step, nav_next = st.columns([1, 1, 1])

    with nav_back:
        if current_idx > 0:
            st.button(
                f"⬅️ {sections[current_idx - 1]}",
                use_container_width=True,
                key="insights_v2_back",
                on_click=insight_nav_callback,
                args=(sections[current_idx - 1],)
            )

    with nav_step:
        st.markdown(
            f'<div class="insight-nav-step">SECTION {current_idx + 1} OF {len(sections)}</div>',
            unsafe_allow_html=True
        )

    with nav_next:
        if current_idx < len(sections) - 1:
            st.button(
                f"{sections[current_idx + 1]} ➡️",
                use_container_width=True,
                key="insights_v2_next",
                on_click=insight_nav_callback,
                args=(sections[current_idx + 1],)
            )

# Navigation - Tabs
# =====================================================
# Load history from CSV on first load
if "history" not in st.session_state:
    history = load_history_from_csv()
    st.session_state.history = history if history else []

# Create tabs
tab_home, tab_diabetes, tab_bmi, tab_history, tab_insights = st.tabs([
    "🏠 Home",
    "🩺 Diabetes Prediction",
    "⚖️ BMI Calculator",
    "📊 History",
    "📈 Model Insights"
])

# =====================================================
# Home Tab
# =====================================================
with tab_home:
    home_page()

# =====================================================
# BMI Calculator Tab
# =====================================================
with tab_bmi:
    bmi_calculator()

# =====================================================
# History Tab
# =====================================================
with tab_history:
    history_page()

# =====================================================
# Model Insights Tab (NEW)
# =====================================================
with tab_insights:
    model_insights_page()

# =====================================================
# Diabetes Prediction Tab
# =====================================================
with tab_diabetes:
    st.markdown(
        "<h1 class='main-title'>🩺 Diabetes Prediction</h1>",
        unsafe_allow_html=True
    )
    
    st.markdown(
        "<p class='sub-title'>Enter patient details below for risk assessment.</p>",
        unsafe_allow_html=True
    )
    
    st.markdown(
        "<div class='section'>📋 Select Input Method</div>",
        unsafe_allow_html=True
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        manual = st.button("✏️ Manual Input", use_container_width=True)
    
    with col2:
        upload = st.button("📁 Upload File", use_container_width=True)
    
    if "mode" not in st.session_state:
        st.session_state.mode = "manual"
    
    if manual:
        st.session_state.mode = "manual"
        # Clear any previous prediction results when switching modes
        if "prediction" in st.session_state:
            del st.session_state.prediction
            del st.session_state.patient
            del st.session_state.diabetes_prob
            del st.session_state.healthy_prob
        # Clear upload error if exists
        if "upload_error" in st.session_state:
            del st.session_state.upload_error
        if "upload_data_error" in st.session_state:
            del st.session_state.upload_data_error
        # Reset uploader key
        if "uploader_key" in st.session_state:
            st.session_state.uploader_key = str(uuid.uuid4())
        # Clear uploaded data results
        if "upload_prediction_done" in st.session_state:
            del st.session_state.upload_prediction_done
    
    if upload:
        st.session_state.mode = "upload"
        # Clear any previous prediction results when switching modes
        if "prediction" in st.session_state:
            del st.session_state.prediction
            del st.session_state.patient
            del st.session_state.diabetes_prob
            del st.session_state.healthy_prob
        # Clear upload error if exists
        if "upload_error" in st.session_state:
            del st.session_state.upload_error
        if "upload_data_error" in st.session_state:
            del st.session_state.upload_data_error
        # Set uploader key
        if "uploader_key" not in st.session_state:
            st.session_state.uploader_key = str(uuid.uuid4())
        # Clear uploaded data results
        if "upload_prediction_done" in st.session_state:
            del st.session_state.upload_prediction_done
    
    # Display mode indicator
    if st.session_state.mode == "manual":
        st.markdown("""
        <div class="info">
        📋 Currently using <b>Manual Input</b> mode. Enter patient details below.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info">
        📁 Currently using <b>File Upload</b> mode. Upload a CSV or Excel file with patient data.
        </div>
        """, unsafe_allow_html=True)
    
    # =====================================================
    # MANUAL INPUT
    # =====================================================
    if st.session_state.mode == "manual":
        # Initialize session state for form values if not exists
        if "form_values" not in st.session_state:
            st.session_state.form_values = {
                "pregnancies": 0,
                "glucose": 0,
                "blood_pressure": 0,
                "skin": 0,
                "insulin": 0,
                "bmi": 25.0,
                "dpf": 0.471,
                "age": 21
            }
        
        with st.form("prediction_form"):
            left, right = st.columns(2)
            
            with left:
                pregnancies = st.slider(
                    "👶 Pregnancies",
                    0,
                    20,
                    value=st.session_state.form_values["pregnancies"],
                    help="Number of pregnancies"
                )
                
                glucose = st.number_input(
                    "🩸 Glucose (mg/dL)",
                    min_value=0,
                    max_value=300,
                    value=st.session_state.form_values["glucose"],
                    help="Glucose level in blood (1-300 mg/dL)"
                )
                
                blood_pressure = st.number_input(
                    "❤️ Blood Pressure (mmHg)",
                    min_value=0,
                    max_value=200,
                    value=st.session_state.form_values["blood_pressure"],
                    help="Diastolic blood pressure (1-200 mmHg)"
                )
                
                skin = st.number_input(
                    "📏 Skin Thickness (mm)",
                    min_value=0,
                    max_value=99,
                    value=st.session_state.form_values["skin"],
                    step=1,
                    help="Triceps skin fold thickness (1-99 mm)"
                )
            
            with right:
                insulin = st.number_input(
                    "💉 Insulin (mu U/ml)",
                    min_value=0,
                    max_value=900,
                    value=st.session_state.form_values["insulin"],
                    help="2-Hour serum insulin (0-900)"
                )
                
                bmi = st.number_input(
                    "⚖️ BMI",
                    min_value=0.0,
                    max_value=100.0,
                    value=st.session_state.form_values["bmi"],
                    step=0.1,
                    help="Body Mass Index (0.1-100)"
                )
                
                dpf = st.number_input(
                    "📊 Diabetes Pedigree Function",
                    min_value=0.0,
                    max_value=3.0,
                    value=st.session_state.form_values["dpf"],
                    step=0.01,
                    help="Diabetes pedigree function (0.01-3.0)"
                )
                
                age = st.number_input(
                    "🎂 Age",
                    min_value=0,
                    max_value=120,
                    value=st.session_state.form_values["age"],
                    help="Age in years (1-120)"
                )
            
            col1, col2 = st.columns(2)
            
            with col1:
                predict = st.form_submit_button("🔍 Predict Diabetes", use_container_width=True)
            
            with col2:
                reset = st.form_submit_button("🔄 Reset Form", use_container_width=True)
        
        # Handle Reset
        if reset:
            # Reset form values in session state
            st.session_state.form_values = {
                "pregnancies": 0,
                "glucose": 0,
                "blood_pressure": 0,
                "skin": 0,
                "insulin": 0,
                "bmi": 25.0,
                "dpf": 0.471,
                "age": 21
            }
            # Clear prediction results
            if "prediction" in st.session_state:
                del st.session_state.prediction
                del st.session_state.patient
                del st.session_state.diabetes_prob
                del st.session_state.healthy_prob
            st.rerun()
        
        if predict:
            # Validate all inputs
            errors = validate_required_fields(
                glucose, blood_pressure, bmi, age, dpf, skin, insulin, pregnancies
            )
            
            if errors:
                for error in errors:
                    st.error(f"❌ {error}")
                st.stop()
            
            # Check for zero values and show warnings
            zero_warnings = []
            if glucose == 0:
                zero_warnings.append("Glucose is 0. Will be replaced with median value.")
            if blood_pressure == 0:
                zero_warnings.append("Blood Pressure is 0. Will be replaced with median value.")
            if skin == 0:
                zero_warnings.append("Skin Thickness is 0. Will be replaced with median value.")
            if insulin == 0:
                zero_warnings.append("Insulin is 0. Will be replaced with median value.")
            if bmi == 0:
                zero_warnings.append("BMI is 0. Will be replaced with median value.")
            
            if zero_warnings:
                st.warning("⚠️ **Zero Values Detected**")
                for warning in zero_warnings:
                    st.warning(warning)
                st.info("ℹ️ Zero values will be replaced with median values from the dataset for prediction.")
            
            # Create patient dataframe
            patient = pd.DataFrame(
                [[pregnancies, glucose, blood_pressure, skin, insulin, bmi, dpf, age]],
                columns=[
                    "Pregnancies", "Glucose", "BloodPressure", 
                    "SkinThickness", "Insulin", "BMI", 
                    "DiabetesPedigreeFunction", "Age"
                ]
            )
            
            # Make prediction
            prediction, diabetes_prob, healthy_prob, patient_processed = predict_patient_manual(patient)
            
            if prediction is not None:
                st.session_state.prediction = prediction
                st.session_state.patient = patient
                st.session_state.diabetes_prob = diabetes_prob
                st.session_state.healthy_prob = healthy_prob
                
                # Store current values for persistence
                st.session_state.form_values = {
                    "pregnancies": pregnancies,
                    "glucose": glucose,
                    "blood_pressure": blood_pressure,
                    "skin": skin,
                    "insulin": insulin,
                    "bmi": bmi,
                    "dpf": dpf,
                    "age": age
                }
                
                # Add to history
                add_to_history(patient_processed, prediction, diabetes_prob)
    
    # =====================================================
    # FILE UPLOAD (Strict: No zero replacement)
    # =====================================================
    if st.session_state.mode == "upload":
        # Check if there's an upload error in session state
        if "upload_error" in st.session_state and st.session_state.upload_error:
            # Display error with solution options
            st.markdown(f"""
            <div class="error-box">
                <div class="error-title">❌ File Upload Error</div>
                <div class="error-message">{st.session_state.upload_error}</div>
                <div class="error-solution">
                    <strong>💡 How to fix this:</strong><br>
                    • Make sure your file is in CSV or Excel format (.csv, .xlsx, .xls)<br>
                    • Check that your file contains the required columns:<br>
                    <code>Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age</code><br>
                    • Make sure the file is not empty or corrupted
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Options to resolve the error
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📁 Upload Again", use_container_width=True):
                    del st.session_state.upload_error
                    st.session_state.uploader_key = str(uuid.uuid4())
                    st.rerun()
            
            with col2:
                if st.button("✏️ Switch to Manual Input", use_container_width=True):
                    del st.session_state.upload_error
                    st.session_state.mode = "manual"
                    st.rerun()
            
            st.stop()
        
        # Check if there's a data validation error
        if "upload_data_error" in st.session_state and st.session_state.upload_data_error:
            # Display error with solution options
            st.markdown(f"""
            <div class="error-box">
                <div class="error-title">❌ Data Validation Error</div>
                <div class="error-message">{st.session_state.upload_data_error}</div>
                <div class="error-solution">
                    <strong>💡 How to fix this:</strong><br>
                    • Make sure all values are within valid ranges (no zero values allowed)<br>
                    • Check for missing or null values in your data<br>
                    • Ensure all required fields are filled correctly<br>
                    • <strong>Important:</strong> Zero values are not accepted. Please provide valid measurements.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Options to resolve the error
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📁 Upload New File", use_container_width=True, key="upload_new_file"):
                    del st.session_state.upload_data_error
                    st.session_state.uploader_key = str(uuid.uuid4())
                    st.rerun()
            
            with col2:
                if st.button("✏️ Switch to Manual Input", use_container_width=True, key="switch_to_manual_error"):
                    del st.session_state.upload_data_error
                    st.session_state.mode = "manual"
                    st.rerun()
            
            st.stop()
        
        # File uploader with unique key
        uploader_key = st.session_state.get("uploader_key", str(uuid.uuid4()))
        
        uploaded_file = st.file_uploader(
            "📤 Upload CSV or Excel File",
            type=["csv", "xlsx", "xls"],
            help="Upload a CSV or Excel file with the required columns. Zero values are not accepted.",
            key=uploader_key
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.lower().endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                required_columns = [
                    "Pregnancies",
                    "Glucose",
                    "BloodPressure",
                    "SkinThickness",
                    "Insulin",
                    "BMI",
                    "DiabetesPedigreeFunction",
                    "Age"
                ]
                
                missing_columns = [
                    col for col in required_columns
                    if col not in df.columns
                ]
                
                if missing_columns:
                    st.session_state.upload_error = f"Missing required columns: {', '.join(missing_columns)}"
                    st.rerun()
                else:
                    # Validate data values - STRICT (no zeros)
                    validation_errors = validate_uploaded_data(df)
                    
                    if validation_errors:
                        # Join errors with line breaks for better display
                        error_message = "\n".join(validation_errors)
                        st.session_state.upload_data_error = error_message
                        st.rerun()
                    
                    # If we get here, data is valid
                    st.subheader("📊 Uploaded Data")
                    st.dataframe(df, use_container_width=True)
                    
                    if st.button("🚀 Predict Uploaded Data", use_container_width=True):
                        with st.spinner("Making predictions..."):
                            # Process each row - NO zero replacement
                            results = []
                            df_processed = df.copy()
                            
                            debug_rows = []
                            for idx, row in df.iterrows():
                                patient = pd.DataFrame([row[required_columns]])
                                prediction, diabetes_prob, healthy_prob, raw_values = predict_patient_upload(patient)
                                
                                if prediction is not None:
                                    results.append({
                                        "Prediction": prediction,
                                        "Diabetes_Probability": diabetes_prob if diabetes_prob is not None else 0,
                                        "Healthy_Probability": healthy_prob if healthy_prob is not None else 0,
                                        "Risk_Level": get_risk_level(diabetes_prob) if diabetes_prob is not None else "Unknown"
                                    })
                                    
                                    # Add to history
                                    add_to_history(row.to_dict(), prediction, diabetes_prob)
                                    
                                    # Capture debug info: raw feature values actually fed to the model
                                    debug_entry = {"Row": idx + 1}
                                    for col in required_columns:
                                        debug_entry[f"raw_{col}"] = row[col]
                                    debug_entry["Diabetes_Prob_%"] = round(diabetes_prob, 2) if diabetes_prob is not None else None
                                    debug_rows.append(debug_entry)
                                else:
                                    results.append({
                                        "Prediction": None,
                                        "Diabetes_Probability": None,
                                        "Healthy_Probability": None,
                                        "Risk_Level": "Error"
                                    })
                            
                            # Add results to dataframe
                            result_df = pd.DataFrame(results)
                            df["Prediction"] = result_df["Prediction"].apply(lambda x: "Diabetes" if x == 1 else "No Diabetes" if x == 0 else "Error")
                            df["Diabetes_Probability"] = result_df["Diabetes_Probability"]
                            df["Risk_Level"] = result_df["Risk_Level"]
                            
                            # Store in session state for display
                            st.session_state.upload_prediction_done = True
                            st.session_state.upload_results_df = df
                            st.session_state.upload_results = results
                            st.session_state.upload_original_df = df_processed
                            st.session_state.upload_debug_rows = debug_rows
                            
                            st.rerun()
            
            except pd.errors.EmptyDataError:
                st.session_state.upload_error = "The uploaded file is empty. Please upload a valid file."
                st.rerun()
            except Exception as e:
                st.session_state.upload_error = f"Error reading file: {str(e)}"
                st.rerun()
        
        # Display upload prediction results
        if "upload_prediction_done" in st.session_state and st.session_state.upload_prediction_done:
            st.markdown("---")
            st.success("✅ Prediction completed!")
            
            # Display results with gauge charts
            results_df = st.session_state.upload_results_df
            results = st.session_state.upload_results
            
            # Show results table
            st.dataframe(results_df, use_container_width=True)
            
            # =====================================================
            # Debug panel - shows exactly what values the model saw
            # =====================================================
            with st.expander("🔍 Debug: Raw values fed to the model"):
                st.success(
                    "**Model input mode:** Raw (unscaled) values — matches how "
                    "`diabetes_model.pkl` (RandomForestClassifier) was trained. "
                    "`scaler.pkl` is intentionally not applied here; it was fit "
                    "for the KNN/SVM experiments during model comparison, not "
                    "for this tree-based model."
                )
                
                debug_rows = st.session_state.get("upload_debug_rows", [])
                if debug_rows:
                    st.markdown("**Raw values per row** (what the model actually saw):")
                    st.dataframe(pd.DataFrame(debug_rows), use_container_width=True)
                    st.caption(
                        "If predictions still look off, sanity-check individual "
                        "rows against known clinical expectations (e.g. very high "
                        "Glucose + high BMI should trend toward higher probability)."
                    )
            
            # Show gauge chart for each prediction (show first 3 or all if less)
            st.subheader("📊 Risk Visualization")
            
            num_to_show = min(len(results), 5)  # Show up to 5 charts
            cols = st.columns(min(num_to_show, 3))
            
            for i in range(num_to_show):
                col_idx = i % 3
                with cols[col_idx]:
                    if results[i]["Diabetes_Probability"] is not None:
                        st.markdown(f"**Patient {i+1}**")
                        fig = create_gauge_chart(results[i]["Diabetes_Probability"])
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, key=f"gauge_upload_{i}")
                        st.caption(f"Risk Level: {results[i]['Risk_Level']}")
            
            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                "💾 Download Results",
                csv,
                "predictions.csv",
                "text/csv",
                use_container_width=True
            )
            
            # Reset button under download
            if st.button("🔄 Reset Upload & Start Over", use_container_width=True, key="reset_upload_after_results"):
                if "prediction" in st.session_state:
                    del st.session_state.prediction
                if "upload_prediction_done" in st.session_state:
                    del st.session_state.upload_prediction_done
                if "upload_results_df" in st.session_state:
                    del st.session_state.upload_results_df
                if "upload_results" in st.session_state:
                    del st.session_state.upload_results
                if "upload_original_df" in st.session_state:
                    del st.session_state.upload_original_df
                if "upload_debug_rows" in st.session_state:
                    del st.session_state.upload_debug_rows
                st.session_state.uploader_key = str(uuid.uuid4())
                st.rerun()
    
    # =====================================================
    # SHOW MANUAL PREDICTION RESULT
    # =====================================================
    if "prediction" in st.session_state and st.session_state.mode == "manual":
        st.markdown("---")
        
        prediction = st.session_state.prediction
        patient = st.session_state.patient
        diabetes_prob = st.session_state.diabetes_prob
        healthy_prob = st.session_state.healthy_prob
        
        col1, col2 = st.columns([1, 1])
        
        # Prediction Summary
        with col1:
            st.subheader("📊 Prediction Result")
            
            if prediction == 1:
                st.error("🔴 **Diabetes Detected**")
            else:
                st.success("🟢 **No Diabetes Detected**")
            
            if diabetes_prob is not None:
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.metric("Diabetes Probability", f"{diabetes_prob:.2f}%")
                with col_b:
                    st.metric("Healthy Probability", f"{healthy_prob:.2f}%")
                
                # Risk Level
                risk_level = get_risk_level(diabetes_prob)
                color = get_risk_color(risk_level)
                st.markdown(f"**Risk Level:** <span style='color: {color}; font-weight: bold;'>{risk_level}</span>", unsafe_allow_html=True)
            else:
                st.info("ℹ️ Probability scores not available for this model.")
        
        # Gauge Chart
        with col2:
            if diabetes_prob is not None:
                fig = create_gauge_chart(diabetes_prob)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ℹ️ Gauge chart not available for this model.")
        
        st.markdown("---")
        
        # Patient Information
        st.subheader("👤 Patient Information")
        st.dataframe(patient, use_container_width=True)
        
        st.markdown("---")
        
        # Recommendations
        display_recommendation(prediction)
