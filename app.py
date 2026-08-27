import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib  # Using joblib instead of pickle to fix the STACK_GLOBAL error
import plotly.graph_objects as go
import os
from datetime import datetime
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
        # Strictly use joblib to avoid UnpicklingError STACK_GLOBAL
        model = joblib.load("final_model.pkl")
        medians_obj = joblib.load("median_imputer.pkl")
        
        # Handle different imputer formats
        if isinstance(medians_obj, dict):
            median_dict = medians_obj
        else:
            # If it's a SimpleImputer object
            try:
                columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
                median_dict = dict(zip(columns, medians_obj.statistics_))
            except:
                median_dict = {}
        
        return model, median_dict
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {type(e).__name__} - {str(e)}")
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

/* =====================================================
   Model Insights Page Styles
   ===================================================== */
.insight-section-header {
    font-size: 26px;
    font-weight: 800;
    color: #1A237E;
    margin: 6px 0 4px 0;
}
.insight-section-sub {
    color: #444;
    font-size: 16.5px;
    line-height: 1.85;
    margin-bottom: 22px;
    text-align: justify;
}
.insight-stat-card {
    background: white;
    border-radius: 16px;
    padding: 18px 10px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    border-top: 4px solid #1A237E;
    height: 100%;
}
.insight-stat-icon { font-size: 26px; margin-bottom: 4px; }
.insight-stat-value { font-size: 20px; font-weight: 800; color: #1A237E; }
.insight-stat-label {
    font-size: 12px;
    color: #777;
    margin-top: 2px;
    text-transform: uppercase;
    letter-spacing: .4px;
}
.insight-plot-title {
    font-size: 20px;
    font-weight: 700;
    color: #1A237E;
    margin: 26px 0 12px 0;
    text-align: center; /* Center the title too */
}
.insight-plot-desc {
    font-size: 16.5px;
    color: #444;
    line-height: 1.85;
    margin-top: 14px;
    margin-bottom: 6px;
    text-align: justify;
}
.insight-missing-box {
    background: #fff8e1;
    border: 1px dashed #e0a800;
    border-radius: 12px;
    padding: 30px 18px;
    text-align: center;
    color: #7a5c00;
    font-size: 13.5px;
}
.insight-divider {
    border: none;
    border-top: 2px dotted #b0b8e0;
    margin: 30px 0;
}
.insight-nav-step {
    text-align: center;
    color: #8891bb;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: .3px;
    padding-top: 10px;
}
div[data-testid="stRadio"] > div {
    gap: 6px;
}
div[data-testid="stRadio"] label {
    background: #f0f2f6;
    padding: 8px 16px;
    border-radius: 20px;
    margin-right: 4px;
    font-weight: 600;
}

/* ============================================= */
/* Model Insight Images - Consistent Centering */
/* ============================================= */
.insight-image-wrap {
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 12px auto 20px auto;
}

.insight-image-wrap img {
    display: block;
    width: auto;
    max-width: 100%;
    height: auto;
    max-height: 620px;
    object-fit: contain;
    border-radius: 12px;
    box-shadow: 0 3px 14px rgba(0,0,0,0.07);
}

div[data-testid="stImage"] {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    text-align: center !important;
    margin: 10px auto 18px auto !important;
    width: 100% !important;
}

div[data-testid="stImage"] img {
    display: block !important;
    margin: 0 auto !important;
    max-width: 100% !important;
    height: auto !important;
    object-fit: contain !important;
    border-radius: 12px !important;
}

.insight-image-caption {
    text-align: center;
    color: #777;
    font-size: 13px;
    margin-top: -8px;
}

/* Preprocessing Box Border Adjustment */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 12px;
    background-color: #ffffff;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    padding: 15px 20px; /* Internal padding so contents breathe */
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
    div[data-testid="stImage"] > img {
        max-width: 100% !important; /* Restores image width on smaller screens */
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
    
    if glucose < 0 or glucose > 300:
        errors.append("Glucose must be between 0 and 300 mg/dL. A value of 0 will be treated as missing and replaced with the training median.")
    
    if blood_pressure < 0 or blood_pressure > 200:
        errors.append("Blood Pressure must be between 0 and 200 mmHg. A value of 0 will be treated as missing and replaced with the training median.")
    
    if skin < 0 or skin > 99:
        errors.append("Skin Thickness must be between 0 and 99 mm. A value of 0 will be treated as missing and replaced with the training median.")
    
    if insulin < 0 or insulin > 900:
        errors.append("Insulin must be between 0 and 900 mu U/ml.")
    
    if bmi < 0 or bmi > 100:
        errors.append("BMI must be between 0 and 100 kg/m². A value of 0 will be treated as missing and replaced with the training median.")
    
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
    
    # Glucose: 0-300 (0 is treated as missing and replaced by the training median)
    invalid_glucose = df[(df['Glucose'] < 0) | (df['Glucose'] > 300)]
    if not invalid_glucose.empty:
        errors.append(f"⚠️ Glucose must be between 0 and 300 mg/dL. Found {len(invalid_glucose)} invalid rows.")
    
    # BloodPressure: 0-200 (0 is treated as missing and replaced by the training median)
    invalid_bp = df[(df['BloodPressure'] < 0) | (df['BloodPressure'] > 200)]
    if not invalid_bp.empty:
        errors.append(f"⚠️ Blood Pressure must be between 0 and 200 mmHg. Found {len(invalid_bp)} invalid rows.")
    
    # SkinThickness: 0-99 (0 is treated as missing and replaced by the training median)
    invalid_skin = df[(df['SkinThickness'] < 0) | (df['SkinThickness'] > 99)]
    if not invalid_skin.empty:
        errors.append(f"⚠️ Skin Thickness must be between 0 and 99 mm. Found {len(invalid_skin)} invalid rows.")
    
    # Insulin: 0-900 (0 is treated as missing and replaced by the training median)
    invalid_insulin = df[(df['Insulin'] < 0) | (df['Insulin'] > 900)]
    if not invalid_insulin.empty:
        errors.append(f"⚠️ Insulin must be between 0 and 900 mu U/ml. Found {len(invalid_insulin)} invalid rows.")
    
    # BMI: 0-100 (0 is treated as missing and replaced by the training median)
    invalid_bmi = df[(df['BMI'] < 0) | (df['BMI'] > 100)]
    if not invalid_bmi.empty:
        errors.append(f"⚠️ BMI must be between 0 and 100 kg/m². Found {len(invalid_bmi)} invalid rows.")
    
    # DiabetesPedigreeFunction: 0-3.0
    invalid_dpf = df[(df['DiabetesPedigreeFunction'] < 0) | (df['DiabetesPedigreeFunction'] > 3)]
    if not invalid_dpf.empty:
        errors.append(f"⚠️ Diabetes Pedigree Function must be between 0 and 3.0. Found {len(invalid_dpf)} invalid rows.")
    
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
    """Make prediction for manual input with zero replacement."""
    try:
        # Replace zero values with medians (for manual input)
        zero_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        patient_processed = replace_zero_values(patient_data.copy(), zero_columns)
        
        # Feed raw (unscaled) values directly -- matches training logic provided
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
    """Make prediction for uploaded data with the same zero-to-median treatment."""
    try:
        patient_df = patient_data.copy()

        zero_columns = [
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI"
        ]

        patient_processed = replace_zero_values(patient_df, zero_columns)

        # Feed the processed values to the deployed model.
        prediction = model.predict(patient_processed)[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(patient_processed)[0]
            diabetes_prob = probability[1] * 100
            healthy_prob = probability[0] * 100
        else:
            diabetes_prob = None
            healthy_prob = None

        raw_values = list(patient_processed.iloc[0].values)

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
# Model Insights Helper Functions
# =====================================================

# The training notebook is the source of truth for the figures and values
# shown on this page.
PLOTS_DIR = ""

# Exact values produced by the supplied training notebook
NOTEBOOK_ZERO_COUNTS = {
    "Glucose": 5,
    "BloodPressure": 35,
    "SkinThickness": 227,
    "Insulin": 374,
    "BMI": 11
}

NOTEBOOK_DATASET_ROWS = 768
NOTEBOOK_DATASET_COLUMNS = 9
NOTEBOOK_TEST_SIZE = 154

NOTEBOOK_METRICS = {
    "Tuned KNN": {
        "Accuracy": 0.7208,
        "Precision": 0.6170,
        "Recall": 0.5370,
        "F1-score": 0.5743,
        "ROC-AUC": 0.7943
    },
    "Tuned SVM": {
        "Accuracy": 0.7208,
        "Precision": 0.5821,
        "Recall": 0.7222,
        "F1-score": 0.6446,
        "ROC-AUC": 0.8105
    },
    "Tuned Random Forest": {
        "Accuracy": 0.7857,
        "Precision": 0.6615,
        "Recall": 0.7963,
        "F1-score": 0.7227,
        "ROC-AUC": 0.8250
    }
}

NOTEBOOK_CV = {
    "Tuned Random Forest": {
        "CV Recall": 0.7474,
        "CV F1-score": 0.7018,
        "CV ROC-AUC": 0.8365,
        "Average Rank": 1.3333,
        "Overall Rank": 1
    },
    "Tuned SVM": {
        "CV Recall": 0.7850,
        "CV F1-score": 0.6982,
        "CV ROC-AUC": 0.8321,
        "Average Rank": 1.6667,
        "Overall Rank": 2
    },
    "Tuned KNN": {
        "CV Recall": 0.5986,
        "CV F1-score": 0.6495,
        "CV ROC-AUC": 0.8279,
        "Average Rank": 3.0000,
        "Overall Rank": 3
    }
}

NOTEBOOK_FEATURE_IMPORTANCE = {
    "Glucose": 0.4083,
    "BMI": 0.1697,
    "Age": 0.1356,
    "Insulin": 0.0833,
    "DiabetesPedigreeFunction": 0.0792,
    "SkinThickness": 0.0443,
    "BloodPressure": 0.0414,
    "Pregnancies": 0.0382
}


def show_insight_plot(filename, icon, title, explanation, caption=None, max_width=900):
    """Render one insight image centered at a consistent, readable size."""
    st.markdown(
        f'<div class="insight-plot-title">{icon} {title}</div>',
        unsafe_allow_html=True
    )

    path = os.path.join(PLOTS_DIR, filename)

    if os.path.exists(path):
        # A centered 3-column layout prevents the image from sticking to the left.
        left, center, right = st.columns([1, 3, 1])
        with center:
            st.image(path, width=max_width)
            if caption:
                st.markdown(
                    f'<div class="insight-image-caption">{caption}</div>',
                    unsafe_allow_html=True
                )
    else:
        st.markdown(
            f"""
            <div class="insight-missing-box">
                📁 <b>{filename}</b> wasn't found.<br>
                <span style="font-size:12.5px;">
                    Please ensure the image file is placed next to <code>app.py</code>.
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        f'<div class="insight-plot-desc">{explanation}</div>',
        unsafe_allow_html=True
    )


def show_insight_plot_slot(filename, caption=None, max_width=330):
    """Render a centered image inside a multi-column layout."""
    path = os.path.join(PLOTS_DIR, filename)

    if os.path.exists(path):
        # Keep each confusion matrix visually balanced inside its column.
        st.markdown('<div class="insight-image-wrap">', unsafe_allow_html=True)
        st.image(path, width=max_width, caption=caption)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="insight-missing-box">📁 {filename}<br>not found</div>',
            unsafe_allow_html=True
        )


def insight_divider(margin=None):
    """A simple dotted-line separator between insight blocks."""
    style = f' style="margin:{margin};"' if margin else ""
    st.markdown(
        f'<hr class="insight-divider"{style}>',
        unsafe_allow_html=True
    )


def insight_group_header(title, subtitle):
    st.markdown(
        f"""
        <div class="insight-section-header">{title}</div>
        <div class="insight-section-sub">{subtitle}</div>
        """,
        unsafe_allow_html=True
    )


def _set_insights_section(section_name):
    """Callback for the Back/Next navigation buttons."""
    st.session_state["insights_section"] = section_name
    st.session_state["insights_scroll_top"] = True


def _scroll_insights_to_top():
    """Force the browser back to the top when navigating sections."""
    if st.session_state.get("insights_scroll_top"):
        st.session_state["insights_scroll_top"] = False
        nonce = uuid.uuid4().hex
        components.html(
            f"""
            <script>
                // nonce: {nonce}
                function scrollAppToTop() {{
                    try {{
                        var w = window.parent;
                        var doc = w.document;
                        w.scrollTo(0, 0);
                        doc.documentElement.scrollTop = 0;
                        doc.body.scrollTop = 0;

                        var selectors = [
                            'section.main',
                            '[data-testid="stMain"]',
                            '[data-testid="stAppViewContainer"]',
                            '[data-testid="stAppViewContainer"] > div',
                            '.main .block-container'
                        ];

                        selectors.forEach(function(sel) {{
                            var el = doc.querySelector(sel);
                            if (el) {{
                                el.scrollTop = 0;
                            }}
                        }});
                    }} catch (e) {{}}
                }}

                scrollAppToTop();
                setTimeout(scrollAppToTop, 50);
                setTimeout(scrollAppToTop, 150);
                setTimeout(scrollAppToTop, 350);
                setTimeout(scrollAppToTop, 600);
            </script>
            """,
            height=0
        )


def model_insights_page():
    """Display the training notebook results in a clear, presentation-ready layout."""

    st.markdown(
        "<h1 class='main-title'>📈 Model Insights & Visualizations</h1>",
        unsafe_allow_html=True
    )

    _scroll_insights_to_top()

    st.markdown(
        "<p class='sub-title'>A behind-the-scenes look at the data, preprocessing, "
        "model evaluation and feature importance from the training notebook.</p>",
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # Quick statistics
    # -----------------------------------------------------
    quick_stats = [
        ("🗂️", "768", "Patient records"),
        ("⚖️", "34.9%", "Diabetic cases"),
        ("🌳", "RF (Tuned)", "Selected model"),
        ("🎯", "78.57%", "Test accuracy"),
        ("📈", "0.8250", "Test ROC-AUC")
    ]

    cols = st.columns(5)

    for col, (icon, value, label) in zip(cols, quick_stats):
        with col:
            st.markdown(
                f"""
                <div class="insight-stat-card">
                    <div class="insight-stat-icon">{icon}</div>
                    <div class="insight-stat-value">{value}</div>
                    <div class="insight-stat-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)

    insights_sections = [
        "🔍 EDA",
        "🧹 Preprocessing",
        "🤖 Model Performance",
        "🌟 Feature Importance",
        "🏆 Final Comparison"
    ]

    section = st.radio(
        "Section",
        insights_sections,
        horizontal=True,
        label_visibility="collapsed",
        key="insights_section"
    )

    st.markdown("<div style='margin-top:6px;'></div>", unsafe_allow_html=True)

    # =====================================================
    # SECTION 1 — EDA
    # =====================================================
    if section == "🔍 EDA":

        insight_group_header(
            "🔍 Exploratory Data Analysis",
            "Understanding the original dataset before preprocessing."
        )

        with st.container(border=True):
            show_insight_plot(
                "01_target_distribution.png",
                "⚖️",
                "Target Distribution",
                "The dataset contains <b>768 patient records</b>. "
                "There are <b>500 non-diabetic cases (65.1%)</b> and "
                "<b>268 diabetic cases (34.9%)</b>. "
                "The outcome is therefore imbalanced toward the non-diabetic class, "
                "which is why the training pipeline uses class balancing."
            )

        insight_divider()

        with st.container(border=True):
            show_insight_plot(
                "04_correlation_analysis.png",
                "🔗",
                "Correlation Analysis After Zero-Value Treatment",
                "After treating the physiologically impossible zero values as missing, "
                "the notebook shows that <b>Glucose</b> has the strongest correlation "
                "with Outcome at approximately <b>0.49</b>, followed by "
                "<b>BMI (0.31)</b>, <b>Insulin (0.30)</b>, and <b>Age (0.24)</b>. "
                "Correlation describes association with the target; it does not prove causation."
            )

    # =====================================================
    # SECTION 2 — PREPROCESSING
    # =====================================================
    elif section == "🧹 Preprocessing":

        insight_group_header(
            "🧹 Data Cleaning & Preprocessing",
            "The notebook treats selected zero values as missing observations, "
            "then applies median imputation, standardisation and LOF-based outlier filtering."
        )

        # -------------------------------------------------
        # 1. ZERO VALUE ANALYSIS — SEPARATE BLOCK
        # -------------------------------------------------
        with st.container(border=True):

            st.markdown(
                '<div class="insight-plot-title">🕳️ Zero Value Analysis</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="insight-plot-desc">'
                "The original dataset contains zeros in five clinical measurements "
                "that are treated as missing rather than genuine measurements. "
                "The notebook reports the following counts:"
                "</div>",
                unsafe_allow_html=True
            )

            z1, z2, z3, z4, z5 = st.columns(5)

            zero_cards = [
                (z1, "Glucose", "5"),
                (z2, "Blood Pressure", "35"),
                (z3, "Skin Thickness", "227"),
                (z4, "Insulin", "374"),
                (z5, "BMI", "11")
            ]

            for col, name, value in zero_cards:
                with col:
                    st.markdown(
                        f"""
                        <div class="insight-stat-card">
                            <div class="insight-stat-icon">0</div>
                            <div class="insight-stat-value">{value}</div>
                            <div class="insight-stat-label">{name} zeros</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

            show_insight_plot(
                "02_zero_value_analysis.png",
                "",
                "Zero Counts by Feature",
                "The exact zero counts from the notebook are "
                "<b>Glucose = 5</b>, <b>Blood Pressure = 35</b>, "
                "<b>Skin Thickness = 227</b>, <b>Insulin = 374</b>, and "
                "<b>BMI = 11</b>. Pregnancies and Outcome are not included in "
                "this missing-value treatment."
            )

        insight_divider()

        # -------------------------------------------------
        # 2. REPLACING ZEROS WITH NaN — SEPARATE BLOCK
        # -------------------------------------------------
        with st.container(border=True):

            st.markdown(
                '<div class="insight-plot-title">🔄 Replacing Zeros with NaN</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="insight-plot-desc">'
                "The notebook replaces the selected zero values with "
                "<b>NaN (Not a Number)</b>. This makes the hidden missing observations "
                "explicit before imputation. Importantly, the NaN conversion does "
                "not delete the records; it changes only the affected cell values."
                "</div>",
                unsafe_allow_html=True
            )

            show_insight_plot(
                "03_missing_values.png",
                "NaN",
                "Missing Values After Zero-to-NaN Conversion",
                "After conversion, the missing-value counts are "
                "<b>Glucose = 5</b>, <b>Blood Pressure = 35</b>, "
                "<b>Skin Thickness = 227</b>, <b>Insulin = 374</b>, and "
                "<b>BMI = 11</b>. The remaining columns have zero missing values "
                "at this stage."
            )

        insight_divider()

        # -------------------------------------------------
        # 3. MEDIAN IMPUTATION
        # -------------------------------------------------
        with st.container(border=True):
            show_insight_plot(
                "image_d55835.jpg",
                "🧮",
                "Median Imputation",
                "The missing values are filled using the <b>median</b> of the "
                "corresponding feature. Median imputation is less affected by extreme "
                "values than mean imputation. The notebook's machine-learning "
                "pipelines perform imputation within the training workflow."
            )

        insight_divider()

        # -------------------------------------------------
        # 4. STRATIFIED TRAIN/TEST SPLIT
        # -------------------------------------------------
        with st.container(border=True):
            show_insight_plot(
                "12_train_test_split.png",
                "📊",
                "Stratified Training and Testing Sets",
                "The notebook uses a stratified train/test split so the class proportions "
                "are preserved as closely as possible between training and testing data. "
                "The final test set contains <b>154 observations</b>, consisting of "
                "<b>100 non-diabetic</b> and <b>54 diabetic</b> cases."
            )

        insight_divider()

        # -------------------------------------------------
        # 5. OUTLIER DETECTION
        # -------------------------------------------------
        with st.container(border=True):
            show_insight_plot(
                "06_outlier_detection_treatment.png",
                "🎯",
                "Outlier Detection with Local Outlier Factor",
                "The notebook applies <b>Local Outlier Factor (LOF)</b> after imputation "
                "and standardisation. LOF identifies observations whose local density "
                "differs substantially from their neighbours. The notebook reports that "
                "the diabetic proportion changes from <b>34.9%</b> before filtering to "
                "approximately <b>34.0%</b> after the outlier-treatment step."
            )

    # =====================================================
    # SECTION 3 — MODEL PERFORMANCE
    # =====================================================
    elif section == "🤖 Model Performance":

        insight_group_header(
            "🤖 Model Performance",
            "Final test-set performance of the three tuned classifiers from the notebook."
        )

        with st.container(border=True):

            st.markdown(
                '<div class="insight-plot-title">📋 Tuned Model Performance Comparison</div>',
                unsafe_allow_html=True
            )

            comparison_display = pd.DataFrame([
                {
                    "Model": "Tuned KNN",
                    "Accuracy": "72.08%",
                    "Precision": "61.70%",
                    "Recall": "53.70%",
                    "F1-score": "57.43%",
                    "ROC-AUC": "0.7943"
                },
                {
                    "Model": "Tuned SVM",
                    "Accuracy": "72.08%",
                    "Precision": "58.21%",
                    "Recall": "72.22%",
                    "F1-score": "64.46%",
                    "ROC-AUC": "0.8105"
                },
                {
                    "Model": "Tuned Random Forest",
                    "Accuracy": "78.57%",
                    "Precision": "66.15%",
                    "Recall": "79.63%",
                    "F1-score": "72.27%",
                    "ROC-AUC": "0.8250"
                }
            ])

            st.dataframe(
                comparison_display,
                use_container_width=True,
                hide_index=True
            )

            st.markdown(
                '<div class="insight-plot-desc">'
                "The tuned Random Forest has the strongest overall test performance "
                "among the three tuned models: <b>78.57% accuracy</b>, "
                "<b>79.63% recall</b>, <b>72.27% F1-score</b>, and "
                "<b>0.8250 ROC-AUC</b>. Recall is especially important here because "
                "it measures how many actual diabetic cases were correctly identified."
                "</div>",
                unsafe_allow_html=True
            )

        insight_divider()

        # Confusion matrices
        with st.container(border=True):

            st.markdown(
                '<div class="insight-plot-title">🧩 Confusion Matrices</div>',
                unsafe_allow_html=True
            )

            cm1, cm2, cm3 = st.columns(3)

            with cm1:
                show_insight_plot_slot(
                    "10_confusion_matrix_knn.png",
                    max_width=290
                )
                st.caption(
                    "Tuned KNN — TN 82, FP 18, FN 25, TP 29"
                )

            with cm2:
                show_insight_plot_slot(
                    "10_confusion_matrix_svm.png",
                    max_width=290
                )
                st.caption(
                    "Tuned SVM — TN 72, FP 28, FN 15, TP 39"
                )

            with cm3:
                show_insight_plot_slot(
                    "10_confusion_matrix_random_forest.png",
                    max_width=290
                )
                st.caption(
                    "Tuned Random Forest — TN 78, FP 22, FN 11, TP 43"
                )

            st.markdown(
                '<div class="insight-plot-desc">'
                "All three matrices are evaluated on the same <b>154-patient test set</b>. "
                "For the tuned Random Forest, there are <b>78 true negatives</b>, "
                "<b>22 false positives</b>, <b>11 false negatives</b>, and "
                "<b>43 true positives</b>. The relatively low false-negative count "
                "corresponds to its high diabetic recall of 79.63%."
                "</div>",
                unsafe_allow_html=True
            )

        insight_divider()

        # ROC curve
        with st.container(border=True):
            roc_filename = (
                "11_roc_curve_2.png"
                if os.path.exists("11_roc_curve_2.png")
                else "11_roc_curve.png"
            )

            show_insight_plot(
                roc_filename,
                "📈",
                "ROC Curves for Tuned Models",
                "The notebook reports test ROC-AUC values of "
                "<b>0.7943 for Tuned KNN</b>, <b>0.8105 for Tuned SVM</b>, and "
                "<b>0.8250 for Tuned Random Forest</b>. A higher ROC-AUC indicates "
                "better overall separation between diabetic and non-diabetic cases "
                "across classification thresholds."
            )

    # =====================================================
    # SECTION 4 — FEATURE IMPORTANCE
    # =====================================================
    elif section == "🌟 Feature Importance":

        insight_group_header(
            "🌟 Feature Importance",
            "Which of the eight input features contribute most to the final Random Forest?"
        )

        with st.container(border=True):

            show_insight_plot(
                "13_feature_importance.png",
                "🏅",
                "Random Forest Feature Importance",
                "The notebook reports the following importance values: "
                "<b>Glucose = 0.4083</b>, <b>BMI = 0.1697</b>, "
                "<b>Age = 0.1356</b>, <b>Insulin = 0.0833</b>, "
                "<b>Diabetes Pedigree Function = 0.0792</b>, "
                "<b>Skin Thickness = 0.0443</b>, "
                "<b>Blood Pressure = 0.0414</b>, and "
                "<b>Pregnancies = 0.0382</b>."
            )

            st.markdown(
                '<div class="insight-plot-desc">'
                "<b>Glucose is the most important feature</b> in this Random Forest, "
                "with an importance of 0.4083. BMI and Age follow at 0.1697 and 0.1356. "
                "Feature importance indicates predictive usefulness within this model; "
                "it does <b>not</b> mean that a feature causes diabetes."
                "</div>",
                unsafe_allow_html=True
            )

    # =====================================================
    # SECTION 5 — FINAL COMPARISON
    # =====================================================
    elif section == "🏆 Final Comparison":

        insight_group_header(
            "🏆 Final Model Comparison",
            "The final model was selected using 5-fold stratified cross-validation, "
            "then evaluated once on the held-out test set."
        )

        with st.container(border=True):

            show_insight_plot(
                "17_final_comparison.png",
                "🎯",
                "All Models, All Metrics",
                "This visualization compares Accuracy, Precision, Recall, F1-score "
                "and ROC-AUC across the candidate models. The final tuned Random Forest "
                "achieves the strongest overall test-set results among the tuned models."
            )

        insight_divider()

        st.markdown(
            '<div class="insight-plot-title">🏆 Why Random Forest Was Selected</div>',
            unsafe_allow_html=True
        )

        cv_table = pd.DataFrame([
            {
                "Model": "Tuned Random Forest",
                "CV Recall": "0.7474",
                "CV F1-score": "0.7018",
                "CV ROC-AUC": "0.8365",
                "Average Rank": "1.3333",
                "Overall Rank": "1"
            },
            {
                "Model": "Tuned SVM",
                "CV Recall": "0.7850",
                "CV F1-score": "0.6982",
                "CV ROC-AUC": "0.8321",
                "Average Rank": "1.6667",
                "Overall Rank": "2"
            },
            {
                "Model": "Tuned KNN",
                "CV Recall": "0.5986",
                "CV F1-score": "0.6495",
                "CV ROC-AUC": "0.8279",
                "Average Rank": "3.0000",
                "Overall Rank": "3"
            }
        ])

        st.dataframe(
            cv_table,
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            '<div class="insight-plot-desc">'
            "The notebook selected <b>Tuned Random Forest</b> using the lowest average "
            "rank across CV Recall, CV F1-score and CV ROC-AUC. "
            "Random Forest ranked first overall with an average rank of "
            "<b>1.3333</b>. The test set was not used to choose the model."
            "</div>",
            unsafe_allow_html=True
        )

        insight_divider()

        st.markdown(
            '<div class="insight-plot-title">🌳 Deployed Model: Tuned Random Forest</div>',
            unsafe_allow_html=True
        )

        d1, d2, d3, d4 = st.columns(4)

        with d1:
            st.metric("Test Accuracy", "78.57%")

        with d2:
            st.metric("Precision", "66.15%")

        with d3:
            st.metric("Recall", "79.63%")

        with d4:
            st.metric("ROC-AUC", "0.8250")

        st.markdown(
            '<div class="insight-plot-desc">'
            "The final Random Forest uses the notebook-selected parameters: "
            "<code>class_weight='balanced'</code>, <code>max_depth=6</code>, "
            "<code>max_features='log2'</code>, <code>min_samples_leaf=5</code>, "
            "<code>min_samples_split=2</code>, and <code>n_estimators=80</code>. "
            "The notebook then fits the final model using the full training dataset "
            "while keeping the test set separate."
            "</div>",
            unsafe_allow_html=True
        )

    # =====================================================
    # SECTION NAVIGATION
    # =====================================================
    insight_divider(margin="26px 0 10px 0")

    current_idx = insights_sections.index(section)
    is_first = current_idx == 0
    is_last = current_idx == len(insights_sections) - 1

    nav_back, nav_step, nav_next = st.columns([1, 1, 1])

    with nav_back:
        if not is_first:
            st.button(
                f"⬅️  {insights_sections[current_idx - 1]}",
                use_container_width=True,
                key="insights_nav_back",
                on_click=_set_insights_section,
                args=(insights_sections[current_idx - 1],)
            )

    with nav_step:
        st.markdown(
            f'<div class="insight-nav-step">'
            f'SECTION {current_idx + 1} OF {len(insights_sections)}'
            f'</div>',
            unsafe_allow_html=True
        )

    with nav_next:
        if not is_last:
            st.button(
                f"{insights_sections[current_idx + 1]}  ➡️",
                use_container_width=True,
                key="insights_nav_next",
                on_click=_set_insights_section,
                args=(insights_sections[current_idx + 1],)
            )

# =====================================================
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
                    help="Glucose level in blood (0-300 mg/dL). A value of 0 is treated as missing and replaced by the training median."
                )
                
                blood_pressure = st.number_input(
                    "❤️ Blood Pressure (mmHg)",
                    min_value=0,
                    max_value=200,
                    value=st.session_state.form_values["blood_pressure"],
                    help="Diastolic blood pressure (0-200 mmHg). A value of 0 is treated as missing and replaced by the training median."
                )
                
                skin = st.number_input(
                    "📏 Skin Thickness (mm)",
                    min_value=0,
                    max_value=99,
                    value=st.session_state.form_values["skin"],
                    step=1,
                    help="Triceps skin fold thickness (0-99 mm). A value of 0 is treated as missing and replaced by the training median."
                )
            
            with right:
                insulin = st.number_input(
                    "💉 Insulin (mu U/ml)",
                    min_value=0,
                    max_value=900,
                    value=st.session_state.form_values["insulin"],
                    help="2-Hour serum insulin (0-900). A value of 0 is treated as missing and replaced by the training median."
                )
                
                bmi = st.number_input(
                    "⚖️ BMI",
                    min_value=0.0,
                    max_value=100.0,
                    value=st.session_state.form_values["bmi"],
                    step=0.1,
                    help="Body Mass Index (0-100). A value of 0 is treated as missing and replaced by the training median."
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
                    "`final_model.pkl` (RandomForestClassifier) was trained. "
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
