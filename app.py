import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix
import pickle
import os
 
# =====================================================
# PAGE CONFIG & STYLING
# =====================================================
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)
 
# Your existing CSS styling
st.markdown("""
<style>
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
 
.section {
    font-size: 30px;
    font-weight: 700;
    margin-bottom: 15px;
}
 
.insight-section-header {
    font-size: 32px;
    font-weight: 800;
    color: #1A237E;
    margin: 10px 0 8px 0;
}
 
.insight-section-sub {
    color: #555;
    font-size: 17px;
    line-height: 1.9;
    margin-bottom: 24px;
}
 
.insight-stat-card {
    background: white;
    border-radius: 16px;
    padding: 20px 12px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    border-top: 5px solid #1A237E;
}
 
.insight-plot-title {
    font-size: 22px;
    font-weight: 700;
    color: #1A237E;
    margin: 28px 0 14px 0;
}
 
.insight-plot-desc {
    font-size: 17px;
    color: #555;
    line-height: 1.9;
    margin: 14px 0 8px 0;
}
</style>
""", unsafe_allow_html=True)
 
# =====================================================
# HELPER FUNCTIONS FOR MODEL INSIGHTS
# =====================================================
 
def load_model_data(model_path="final_model.pkl", results_path="model_comparison_results.csv"):
    """Load model and results data"""
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        if os.path.exists(results_path):
            results_df = pd.read_csv(results_path)
        else:
            results_df = None
        
        return model, results_df
    except Exception as e:
        st.error(f"Error loading model data: {e}")
        return None, None
 
def create_model_comparison_chart(results_df):
    """Create interactive model comparison bar chart"""
    if results_df is None:
        return None
    
    metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    
    fig = go.Figure()
    
    for model in results_df["Model"]:
        values = results_df[results_df["Model"] == model][metrics].values[0]
        fig.add_trace(go.Bar(
            name=model,
            x=metrics,
            y=values,
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
        ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        barmode="group",
        height=450,
        yaxis=dict(range=[0, 1.05]),
        template="plotly_white",
    )
    
    return fig
 
def create_confusion_matrices(results_df):
    """Create confusion matrix visualizations"""
    if results_df is None:
        return None
    
    matrices = {}
    for _, row in results_df.iterrows():
        model_name = row["Model"]
        tn = row["True Negative"]
        fp = row["False Positive"]
        fn = row["False Negative"]
        tp = row["True Positive"]
        
        cm = np.array([[tn, fp], [fn, tp]])
        matrices[model_name] = cm
    
    return matrices
 
def plot_confusion_matrix(cm, model_name):
    """Plot single confusion matrix"""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        cbar=False,
        xticklabels=['Non-Diabetes', 'Diabetes'],
        yticklabels=['Non-Diabetes', 'Diabetes'],
        ax=ax
    )
    
    ax.set_title(f"{model_name} - Confusion Matrix", fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig
 
def create_error_analysis(results_df):
    """Create error analysis visualization"""
    if results_df is None:
        return None
    
    errors_data = []
    for _, row in results_df.iterrows():
        errors_data.append({
            "Model": row["Model"],
            "False Negatives": row["False Negative"],
            "False Positives": row["False Positive"],
        })
    
    errors_df = pd.DataFrame(errors_data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name="False Negatives",
        x=errors_df["Model"],
        y=errors_df["False Negatives"],
        marker=dict(color="#ef5350"),
    ))
    
    fig.add_trace(go.Bar(
        name="False Positives",
        x=errors_df["Model"],
        y=errors_df["False Positives"],
        marker=dict(color="#ffa726"),
    ))
    
    fig.update_layout(
        title="Error Analysis",
        barmode="group",
        height=450,
        template="plotly_white",
    )
    
    return fig
 
def create_best_model_analysis(results_df):
    """Identify and analyze the best model"""
    if results_df is None:
        return None, None
    
    results_df["score"] = (
        results_df["Accuracy"] * 0.25 +
        results_df["Precision"] * 0.25 +
        results_df["Recall"] * 0.25 +
        results_df["ROC-AUC"] * 0.25
    )
    
    best_idx = results_df["score"].idxmax()
    best_model = results_df.iloc[best_idx]
    
    return best_model, results_df
 
# =====================================================
# MODEL INSIGHTS PAGE
# =====================================================
 
def model_insights_page():
    """Display comprehensive model insights"""
    
    st.markdown(
        "<h1 style='text-align: center; color: #1A237E;'>"
        "📈 Model Insights & Performance</h1>",
        unsafe_allow_html=True
    )
    
    st.markdown(
        "<p style='text-align: center; color: #666; font-size: 18px;'>"
        "Behind-the-scenes analysis of model training and evaluation</p>",
        unsafe_allow_html=True
    )
    
    # Load data
    model, results_df = load_model_data()
    
    if results_df is None or len(results_df) == 0:
        st.error("⚠️ Model comparison results not found.")
        return
    
    # Quick stats
    best_model, _ = create_best_model_analysis(results_df)
    
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    
    with stats_col1:
        st.metric("Best Accuracy", f"{best_model['Accuracy']:.1%}")
    
    with stats_col2:
        st.metric("Best Precision", f"{best_model['Precision']:.1%}")
    
    with stats_col3:
        st.metric("Best Recall", f"{best_model['Recall']:.1%}")
    
    with stats_col4:
        st.metric("Best ROC-AUC", f"{best_model['ROC-AUC']:.3f}")
    
    st.markdown("---")
    
    # Section navigation
    section = st.radio(
        "Select Section",
        ["📊 Overview", "🔥 Performance", "⚠️ Errors", "🎯 Ranking"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # =====================================================
    # SECTION 1: OVERVIEW
    # =====================================================
    if section == "📊 Overview":
        st.markdown(
            "<div class='insight-section-header'>📊 Model Comparison Overview</div>",
            unsafe_allow_html=True
        )
        
        st.markdown(
            "<div class='insight-section-sub'>"
            "All three tuned models evaluated on the same test set. Here's how they compare across all metrics."
            "</div>",
            unsafe_allow_html=True
        )
        
        # Performance table
        st.markdown("<div class='insight-plot-title'>📋 Performance Metrics</div>", unsafe_allow_html=True)
        
        display_df = results_df[["Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]].copy()
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        st.markdown("""
        <div class="insight-plot-desc">
        <b>What these metrics mean:</b><br>
        • <b>Accuracy</b>: Overall correctness<br>
        • <b>Precision</b>: When model says "Diabetes", how often is it right?<br>
        • <b>Recall</b>: Of all actual diabetic cases, how many does the model catch?<br>
        • <b>F1-Score</b>: Balance between Precision and Recall<br>
        • <b>ROC-AUC</b>: Ability to discriminate between classes
        </div>
        """, unsafe_allow_html=True)
        
        # Comparison chart
        st.markdown("<div class='insight-plot-title'>📈 Metrics Comparison</div>", unsafe_allow_html=True)
        fig = create_model_comparison_chart(results_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    # =====================================================
    # SECTION 2: PERFORMANCE
    # =====================================================
    elif section == "🔥 Performance":
        st.markdown(
            "<div class='insight-section-header'>🔥 Detailed Performance Analysis</div>",
            unsafe_allow_html=True
        )
        
        st.markdown(
            "<div class='insight-section-sub'>"
            "Confusion matrices and detailed metrics for each model."
            "</div>",
            unsafe_allow_html=True
        )
        
        # Confusion matrices
        st.markdown("<div class='insight-plot-title'>🔲 Confusion Matrices</div>", unsafe_allow_html=True)
        
        matrices = create_confusion_matrices(results_df)
        cols = st.columns(len(matrices))
        
        for col, (model_name, cm) in zip(cols, matrices.items()):
            with col:
                fig = plot_confusion_matrix(cm, model_name)
                st.pyplot(fig, use_container_width=True)
                
                # Calculate rates
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                st.metric("Sensitivity", f"{sensitivity:.1%}")
    
    # =====================================================
    # SECTION 3: ERRORS
    # =====================================================
    elif section == "⚠️ Errors":
        st.markdown(
            "<div class='insight-section-header'>⚠️ Error Analysis</div>",
            unsafe_allow_html=True
        )
        
        st.markdown(
            "<div class='insight-section-sub'>"
            "False Negatives (missed cases) are more harmful than False Positives."
            "</div>",
            unsafe_allow_html=True
        )
        
        st.markdown("<div class='insight-plot-title'>📊 False Negatives vs False Positives</div>", unsafe_allow_html=True)
        fig = create_error_analysis(results_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    # =====================================================
    # SECTION 4: RANKING
    # =====================================================
    elif section == "🎯 Ranking":
        st.markdown(
            "<div class='insight-section-header'>🎯 Final Model Ranking</div>",
            unsafe_allow_html=True
        )
        
        best_model, ranked_df = create_best_model_analysis(results_df)
        ranked_df = ranked_df.sort_values("score", ascending=False).reset_index(drop=True)
        
        st.markdown("<div class='insight-plot-title'>🏆 Models Ranked</div>", unsafe_allow_html=True)
        
        for idx, (_, row) in enumerate(ranked_df.iterrows()):
            medal = ["🥇", "🥈", "🥉"][idx] if idx < 3 else "  "
            
            col1, col2 = st.columns([0.5, 2])
            
            with col1:
                st.markdown(f"<div style='font-size: 24px;'>{medal}</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style='padding: 10px; background: #f5f7fa; border-radius: 8px;'>
                    <b>{row['Model']}</b> - Score: {row['score']:.4f}
                </div>
                """, unsafe_allow_html=True)
 
# =====================================================
# HOME PAGE (Your existing code)
# =====================================================
 
def home_page():
    """Home page"""
    st.markdown(
        "<h1 class='main-title'>🏥 Diabetes Prediction System</h1>",
        unsafe_allow_html=True
    )
    
    st.markdown(
        "<p class='sub-title'>An AI-powered tool for early diabetes risk assessment</p>",
        unsafe_allow_html=True
    )
    
    st.markdown("""
    ### 🚀 Features
    
    - 🩺 **Diabetes Prediction**: AI-powered prediction using 8 health parameters
    - ⚖️ **BMI Calculator**: Calculate your Body Mass Index
    - 📊 **History Tracking**: View your prediction history
    - 📈 **Model Insights**: Understand how the model works
    """)
 
# =====================================================
# MAIN APP NAVIGATION
# =====================================================
 
# Create tabs
tab_home, tab_prediction, tab_insights = st.tabs([
    "🏠 Home",
    "🩺 Diabetes Prediction",
    "📈 Model Insights"
])
 
with tab_home:
    home_page()
 
with tab_prediction:
    st.markdown("<h1 class='main-title'>🩺 Diabetes Prediction</h1>", unsafe_allow_html=True)
    st.info("Your diabetes prediction form would go here...")
 
with tab_insights:
    model_insights_page()
 
# =====================================================
# Footer
# =====================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 12px; padding: 20px;'>
    <p>🔒 Data Privacy Notice: All predictions are computed locally. No data is stored.</p>
    <p>⚕️ Disclaimer: This tool is for educational purposes only. Always consult healthcare professionals.</p>
</div>
""", unsafe_allow_html=True)
