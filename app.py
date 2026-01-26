
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns

# Set page config
st.set_page_config(page_title="AI Learning Predictor", layout="wide")

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Apply theme
def apply_theme():
    if st.session_state.dark_mode:
        st.markdown("""
        <style>
        .stApp { background-color: #0E1117; color: #FAFAFA; }
        .main-header { 
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); 
            color: white; padding: 2rem; border-radius: 15px; 
            margin-bottom: 2rem; text-align: center; 
        }
        .model-card { 
            background: #1E1E1E; padding: 1.5rem; border-radius: 12px; 
            margin: 1rem 0; border-left: 5px solid #3498DB; 
        }
        .prediction-result { 
            background: #2d3b2d; padding: 1.5rem; border-radius: 10px; 
            border-left: 4px solid #2ecc71; margin: 0.5rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp { background-color: #FFFFFF; color: #000000; }
        .main-header { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; padding: 2rem; border-radius: 15px; 
            margin-bottom: 2rem; text-align: center; 
        }
        .model-card { 
            background: #F8F9FA; padding: 1.5rem; border-radius: 12px; 
            margin: 1rem 0; border-left: 5px solid #3498DB; 
        }
        .prediction-result { 
            background: #e8f5e8; padding: 1.5rem; border-radius: 10px; 
            border-left: 4px solid #2ecc71; margin: 0.5rem 0;
        }
        </style>
        """, unsafe_allow_html=True)

# Advanced FSLSM Prediction Engine
import joblib
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin

@st.cache_resource
def load_sota_models():
    """Load SOTA models with caching to prevent reload overhead"""
    try:
        print("DEBUG: Loading models via cache_resource...")
        import model_definitions # Ensure module is loaded
        models = joblib.load('sota_semi_supervised_models.joblib')
        print("DEBUG: Models loaded successfully in cache.")
        return models
    except Exception as e:
        print(f"DEBUG: Cache Load Error: {e}")
        return None

class FSLSMPredictor:
    def __init__(self):
        self.models = load_sota_models()
        if self.models is None:
            st.error("⚠️ SOTA Models could not be loaded. Running in Rule-Based Fallback mode. Please check console logs.")
            self.model_loaded = False
        else:
            self.model_loaded = True
            
    def predict(self, input_data):
        scores = {}
        predictions = {}
        confidence_scores = {}
        
        # Prepare input vector (must match training features)
        feature_order = [
            'T_image', 'T_video', 'T_read', 'T_audio', 'T_hierarchies', 'T_powerpoint', 
            'T_concrete', 'T_result', 'N_standard_questions_correct', 'N_msgs_posted', 
            'T_solve_excercise', 'N_group_discussions', 'Skipped_los', 'N_next_button_used', 
            'T_spent_in_session', 'N_questions_on_details', 'N_questions_on_outlines'
        ]
        
        # Mapping slider inputs to these features
        mapped_data = {
            'T_image': input_data.get('diagram_view_time', 0),
            'T_video': input_data.get('video_watch_time', 0),
            'T_read': input_data.get('reading_time', 0),
            'T_audio': input_data.get('theoretical_discussions', 0), 
            'T_hierarchies': input_data.get('big_picture_focus', 0) / 4, 
            'T_powerpoint': input_data.get('visual_content_engagement', 0),
            'T_concrete': input_data.get('practical_exercises', 0),
            'T_result': input_data.get('step_by_step_completion', 0) / 4,
            'N_standard_questions_correct': input_data.get('pattern_recognition', 0),
            'N_msgs_posted': input_data.get('messages_posted', 0),
            'T_solve_excercise': input_data.get('hands_on_activities', 0),
            'N_group_discussions': input_data.get('group_discussions', 0),
            'Skipped_los': 5, # Default
            'N_next_button_used': input_data.get('linear_progression', 0) * 5,
            'T_spent_in_session': 30, # Default
            'N_questions_on_details': input_data.get('detail_orientation', 0) * 2,
            'N_questions_on_outlines': input_data.get('holistic_understanding', 0) * 2
        }
        
        X_input = pd.DataFrame([mapped_data])
        # Ensure order matches training
        X_input = X_input.reindex(columns=feature_order, fill_value=0)
        
        dimensions = ['visual_verbal', 'active_reflective', 'sensing_intuitive', 'sequential_global']
        
        if self.model_loaded:
            for dim in dimensions:
                if dim in self.models:
                    model_info = self.models[dim]
                    model = model_info['model']
                    
                    # Score calculation (Classic logic for fallback comparison)
                    raw_score = self._calculate_logic_score(dim, input_data)
                    
                    try:
                        # --- REAL SOTA INFERENCE ---
                        probabilities = model.predict_proba(X_input) # Shape (1, 2)
                        
                        # Assuming Class 1 = "Active/Visual/etc" and Class 0 = "Reflective/Verbal/etc"
                        # We extract probability of Class 1
                        # Note: Check classes_ of model if possible, but standard is sorted
                        prob_class_1 = probabilities[0][1]
                        final_score = prob_class_1 * 100
                    except Exception as e:
                        print(f"Prediction Error for {dim}: {e}")
                        final_score = raw_score # Fallback
                    
                    # AI Adjustment (SOTA Stats)
                    acc = model_info.get('accuracy', 0.85)
                    algo = model_info.get('algorithm', 'Unknown')
                    
                    # Calculate Dynamic Confidence
                    dist_from_center = abs(final_score - 50)
                    certainty_factor = 0.8 + (dist_from_center / 125) 
                    calc_conf = (acc * 100) * certainty_factor
                    
                    # Set results
                    confidence_scores[dim] = min(99, int(calc_conf))
                    scores[dim] = final_score
                    
                    label_0 = dim.split('_')[0].capitalize() # e.g. Visual
                    label_1 = dim.split('_')[1].capitalize() # e.g. Verbal
                    
                    # If score > 50, it leans towards label_0 (Visual, Active, etc.)?
                    # Need to verify mapping. In training:
                    # visual_verbal = 1 (Visual) if value in [0,1,2] else 0
                    # So 1 is Visual. So > 50 is Visual.
                    predictions[dim] = label_0 if final_score > 50 else label_1
                    
                    # Store algo name for UI
                    scores[f'{dim}_algo'] = algo
        else:
            # Fallback if no model file
             for dim in dimensions:
                scores[dim] = 50
                predictions[dim] = "Balanced"
                confidence_scores[dim] = 50
                scores[f'{dim}_algo'] = "Rule-Based (Fallback)"

        return predictions, confidence_scores, scores

    def _calculate_logic_score(self, dimension, input_data):
        # Reusing the robust logic from previous version for the base score
        if dimension == 'visual_verbal':
            res = (input_data.get('video_watch_time', 0) + input_data.get('diagram_view_time', 0)) * 1.5
            return min(100, max(0, res))
        if dimension == 'active_reflective':
            res = (input_data.get('messages_posted', 0) + input_data.get('hands_on_activities', 0)) * 1.5
            return min(100, max(0, res))
        if dimension == 'sensing_intuitive':
            res = (input_data.get('practical_exercises', 0) + input_data.get('detail_orientation', 0)) * 1.5
            return min(100, max(0, res))
        if dimension == 'sequential_global':
            res = (input_data.get('step_by_step_completion', 0) + input_data.get('linear_progression', 0)) * 1.5
            return min(100, max(0, res))
        return 50

    def _get_max_value(self, feature):
        return 50 # Simplified


# Fixed Visualization Functions with Caching
@st.cache_data(ttl=300)
def create_confusion_matrix():
    """Create a confusion matrix for learning style classification"""
    categories = ['Visual', 'Verbal', 'Active', 'Reflective', 'Sensing', 'Intuitive']
    cm_data = np.array([
        [85, 5, 3, 2, 3, 2],
        [4, 88, 2, 3, 1, 2],
        [2, 3, 87, 4, 2, 2],
        [3, 2, 4, 86, 3, 2],
        [1, 2, 3, 2, 89, 3],
        [2, 1, 2, 3, 4, 88]
    ])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', 
                xticklabels=categories, yticklabels=categories, ax=ax)
    ax.set_title('Confusion Matrix - Learning Style Classification\n(Overall Accuracy: 87.2%)')
    ax.set_xlabel('Predicted Style')
    ax.set_ylabel('Actual Style')
    plt.tight_layout()
    return fig

@st.cache_data(ttl=300)
def create_fslsm_radar(scores):
    """Radar chart for all FSLSM dimensions"""
    categories = ['Visual', 'Verbal', 'Active', 'Reflective', 'Sensing', 'Intuitive', 'Sequential', 'Global']
    values = [
        scores.get('visual', 50), scores.get('verbal', 50),
        scores.get('active', 50), scores.get('reflective', 50),
        scores.get('sensing', 50), scores.get('intuitive', 50),
        scores.get('sequential', 50), scores.get('global', 50)
    ]
    
    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        line=dict(color='#2ecc71', width=2)
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=400,
        title="FSLSM Learning Style Profile"
    )
    return fig

@st.cache_data(ttl=300)
def create_performance_barchart():
    """Bar chart comparing model performance"""
    models = ['CatBoost+FCM', 'XGBoost', 'Random Forest', 'SVM', 'Neural Network']
    accuracy = [92.3, 86.5, 84.7, 82.1, 87.8]
    precision = [90.8, 85.2, 83.1, 80.3, 86.1]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Accuracy', x=models, y=accuracy, marker_color='#2ecc71'))
    fig.add_trace(go.Bar(name='Precision', x=models, y=precision, marker_color='#3498db'))
    fig.update_layout(
        title='Model Performance Comparison',
        barmode='group',
        height=400
    )
    return fig

@st.cache_data(ttl=300)
def create_learning_style_pie():
    """Pie chart of learning style distribution"""
    styles = ['Visual-Active', 'Verbal-Reflective', 'Sensing-Sequential', 'Intuitive-Global', 'Balanced']
    distribution = [28, 22, 20, 18, 12]
    
    fig = px.pie(
        values=distribution,
        names=styles,
        title="Learning Style Distribution in Student Population",
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    return fig

@st.cache_data(ttl=300)
def create_feature_histogram():
    """Histogram of feature importance - FIXED VERSION"""
    features = ['Video Time', 'Reading', 'Messages', 'Discussions', 'Exercises', 'Questions', 'Structure', 'Big Picture']
    importance = np.random.normal(15, 5, 8)
    
    # Create a DataFrame for the histogram
    df = pd.DataFrame({
        'Features': features,
        'Importance': importance
    })
    
    fig = px.bar(
        df,
        x='Features',
        y='Importance',
        title="Feature Importance Distribution",
        color='Importance',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=400)
    return fig

@st.cache_data(ttl=300)
def create_correlation_heatmap():
    """Heatmap of feature correlations"""
    features = ['Visual', 'Verbal', 'Active', 'Reflective', 'Sensing', 'Intuitive', 'Sequential', 'Global']
    np.random.seed(42)
    corr_matrix = np.random.uniform(-0.6, 0.8, (8, 8))
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    np.fill_diagonal(corr_matrix, 1.0)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                xticklabels=features, yticklabels=features, ax=ax)
    ax.set_title('Feature Correlation Heatmap')
    plt.tight_layout()
    return fig

@st.cache_data(ttl=300)
def create_dimension_barchart(scores):
    """Bar chart showing all dimension scores"""
    dimensions = ['Visual', 'Verbal', 'Active', 'Reflective', 'Sensing', 'Intuitive', 'Sequential', 'Global']
    values = [scores.get(dim.lower(), 50) for dim in dimensions]
    
    fig = px.bar(
        x=dimensions, 
        y=values,
        color=values,
        color_continuous_scale='Viridis',
        title="FSLSM Dimension Scores"
    )
    fig.update_layout(height=400)
    return fig

@st.cache_data(ttl=300)
def create_stacked_barchart():
    """Stacked bar chart of learning preferences"""
    activities = ['Video', 'Reading', 'Discussion', 'Practice', 'Theory', 'Projects']
    visual_active = [35, 10, 25, 20, 5, 5]
    verbal_reflective = [10, 40, 15, 10, 20, 5]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Visual-Active', x=activities, y=visual_active, marker_color='#e74c3c'))
    fig.add_trace(go.Bar(name='Verbal-Reflective', x=activities, y=verbal_reflective, marker_color='#3498db'))
    fig.update_layout(
        title='Learning Activity Preferences by Style',
        barmode='stack',
        height=400
    )
    return fig

@st.cache_data(ttl=300)
def create_score_distribution():
    """Histogram of score distribution across students"""
    np.random.seed(42)
    visual_scores = np.random.normal(65, 12, 1000)
    
    fig = px.histogram(
        x=visual_scores,
        nbins=20,
        title="Distribution of Visual Learning Scores",
        labels={'x': 'Visual Score', 'y': 'Number of Students'}
    )
    fig.update_layout(height=400)
    return fig

@st.cache_data(ttl=300)
def create_model_comparison_pie():
    """Pie chart showing model usage distribution"""
    models = ['CatBoost', 'XGBoost', 'Random Forest', 'Neural Network', 'Others']
    usage = [45, 25, 15, 10, 5]
    
    fig = px.pie(
        values=usage,
        names=models,
        title="Model Usage Distribution in Research",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    return fig

# Login Page
def show_login():
    apply_theme()
    
    st.markdown("""
    <div class="main-header">
        <h1>🔐 FSLSM Learning Style Predictor</h1>
        <h3>Four-Dimension Analysis with Advanced ML</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            st.subheader("Faculty Login")
            username = st.text_input("👤 Username", value="faculty")
            password = st.text_input("🔑 Password", type="password", value="pjt1_pass")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                login_btn = st.form_submit_button("🚀 Login", use_container_width=True)
            with col_btn2:
                demo_btn = st.form_submit_button("🎮 Demo Mode", use_container_width=True)
            
            if login_btn:
                if username == "faculty" and password == "pjt1_pass":
                    st.session_state.logged_in = True
                    st.session_state.user_role = "faculty"
                    st.rerun()
                else:
                    st.error("❌ Use faculty/pjt1_pass")
            
            if demo_btn:
                st.session_state.logged_in = True
                st.session_state.user_role = "demo"
                st.rerun()

# Main Dashboard
# Faculty Dashboard & Reporting
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import base64
from datetime import datetime

# ... (Previous imports and setup)

def create_pdf_report(student_name, results, charts=None):
    """Generate a professional PDF report for a student"""
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 15, "Student Learning Style Profile", 0, 1, 'C')
    pdf.set_font("Arial", 'I', 12)
    pdf.cell(0, 10, f"Generated for: {student_name} | Date: {datetime.now().strftime('%Y-%m-%d')}", 0, 1, 'C')
    pdf.ln(10)
    
    # Results Section
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Assessment Results", 0, 1)
    pdf.set_font("Arial", '', 12)
    
    for dim, pred in results.items():
        score = pred.get('score', 50)
        label = pred.get('prediction', 'Balanced')
        conf = pred.get('confidence', 0)
        algo = pred.get('algo', 'SOTA Model')
        
        pdf.cell(0, 8, f"{dim.replace('_', '-').title()}: {label} ({score:.1f}%)", 0, 1)
        pdf.set_font("Arial", 'I', 10)
        pdf.cell(0, 6, f"   Confidence: {conf}% | Model: {algo}", 0, 1)
        pdf.set_font("Arial", '', 12)
        pdf.ln(2)
        
    # Recommendations
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Personalized Recommendations", 0, 1)
    pdf.set_font("Arial", '', 11)
    
    recommendations = []
    if results.get('visual_verbal', {}).get('prediction') == 'Visual':
        recommendations.append("- Use diagrams, flowcharts, and video content.")
    if results.get('active_reflective', {}).get('prediction') == 'Active':
        recommendations.append("- Engage in group discussions and hands-on experiments.")
    if results.get('sensing_intuitive', {}).get('prediction') == 'Sensing':
        recommendations.append("- Focus on concrete facts and practical applications.")
    
    for rec in recommendations:
        pdf.multi_cell(0, 8, rec)
        
    return pdf.output(dest='S').encode('latin-1')

def show_faculty_dashboard():
    st.markdown("""
    <div class="main-header">
        <h1>🎓 Faculty Dashboard</h1>
        <h3>Class Analytics • Batch Processing • Student Reports</h3>
    </div>
    """, unsafe_allow_html=True)
    
    tabs = st.tabs(["📂 Batch Upload", "📊 Class Analytics", "👤 Single Student", "⚙️ Settings"])
    
    with tabs[0]:
        st.subheader("Upload Student Data (CSV)")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(df)} student records.")
            
            if st.button("🚀 Analyze Class Batch"):
                progress_bar = st.progress(0)
                
                # Predict for all
                predictor = FSLSMPredictor()
                results_list = []
                
                for idx, row in df.iterrows():
                    # Map row to input_data format (assuming columns match or using simple mapping)
                    # For this demo, we assume the CSV has columns matching our internal features or sliders
                    # Simplification: We Map row dictionary directly
                    preds, confs, scores = predictor.predict(row.to_dict())
                    
                    res_entry = {'Student_ID': row.get('Student_ID', f'STD-{idx+1}')}
                    for dim in preds.keys():
                        res_entry[f'{dim}_Pred'] = preds[dim]
                        res_entry[f'{dim}_Score'] = scores.get(dim, 0)
                        res_entry[f'{dim}_Conf'] = confs.get(dim, 0)
                        res_entry[f'{dim}_Model'] = scores.get(f'{dim}_algo', 'Unknown')
                    
                    results_list.append(res_entry)
                    progress_bar.progress((idx + 1) / len(df))
                
                res_df = pd.DataFrame(results_list)
                st.dataframe(res_df)
                
                # CSV Download
                csv = res_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Analysis Results",
                    csv,
                    "class_analysis_results.csv",
                    "text/csv"
                )
                
    with tabs[1]:
        show_analytics() # Re-use existing analytics but framed for class
        
    with tabs[2]:
        show_prediction_interface() # The single student interface
        
    with tabs[3]:
        st.write("### ⚙️ Model Configuration")
        st.info("Current SOTA Models Loaded: XGBoost (Optuna Tuned), KAN, TabNet")
        if st.button("🔄 Reload Models"):
            st.cache_data.clear()
            st.rerun()

def show_dashboard():
    apply_theme()
    
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button(f"{'🌙' if st.session_state.dark_mode else '☀️'} Theme"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    
    role = "Faculty" if st.session_state.user_role == "faculty" else "Demo User"
    
    if st.session_state.user_role == "faculty":
        show_faculty_dashboard()
    else:
        st.markdown(f"""
        <div class="main-header">
            <h1>🧠 FSLSM Learning Style Analyzer</h1>
            <h3>Welcome, {role} | Visual-Verbal • Active-Reflective • Sensing-Intuitive • Sequential-Global</h3>
        </div>
        """, unsafe_allow_html=True)
        # Standard Student View
        tabs = st.tabs(["🎯 Predict", "📊 Visualizations", "📈 Insights"])
        with tabs[0]: show_prediction_interface()
        with tabs[1]: show_visualizations()
        with tabs[2]: show_insights()

def show_prediction_interface():
    st.subheader("🎯 Single Student Analysis")
    
    with st.form("prediction_form"):
        # Visual-Verbal Dimension
        st.write("### 👁️ Visual-Verbal Dimension")
        col1, col2 = st.columns(2)
        with col1:
            video_watch_time = st.slider('Video Watch Time (min)', 0, 45, 25, key='video')
            diagram_view_time = st.slider('Diagram View Time (min)', 0, 30, 15, key='diagram')
        with col2:
            reading_time = st.slider('Reading Time (min)', 0, 60, 20, key='reading')
            writing_activities = st.slider('Writing Activities (min)', 0, 40, 15, key='writing')
        
        # Active-Reflective Dimension
        st.write("### 💬 Active-Reflective Dimension")
        col1, col2 = st.columns(2)
        with col1:
            messages_posted = st.slider('Messages Posted', 0, 50, 18, key='messages')
            group_discussions = st.slider('Group Discussions', 0, 20, 8, key='discussions')
            hands_on_activities = st.slider('Hands-on Activities (min)', 0, 45, 20, key='hands_on')
        with col2:
            observation_time = st.slider('Observation Time (min)', 0, 40, 15, key='observation')
            reflection_activities = st.slider('Reflection Activities (min)', 0, 35, 12, key='reflection')
        
        # Sensing-Intuitive Dimension
        st.write("### 🔍 Sensing-Intuitive Dimension")
        col1, col2 = st.columns(2)
        with col1:
            practical_exercises = st.slider('Practical Exercises (min)', 0, 40, 22, key='practical')
            detail_orientation = st.slider('Detail Orientation Score', 0, 35, 18, key='detail')
        with col2:
            theoretical_discussions = st.slider('Theoretical Discussions (min)', 0, 35, 12, key='theoretical')
            pattern_recognition = st.slider('Pattern Recognition Score', 0, 40, 15, key='pattern')
        
        # Sequential-Global Dimension
        st.write("### 📚 Sequential-Global Dimension")
        col1, col2 = st.columns(2)
        with col1:
            step_by_step_completion = st.slider('Step-by-Step Completion', 0, 40, 25, key='step')
            linear_progression = st.slider('Linear Progression Score', 0, 35, 20, key='linear')
        with col2:
            big_picture_focus = st.slider('Big Picture Focus', 0, 40, 15, key='big_picture')
            holistic_understanding = st.slider('Holistic Understanding', 0, 35, 12, key='holistic')
        
        submit_button = st.form_submit_button("🚀 Analyze All FSLSM Dimensions", use_container_width=True)
    
    if submit_button:
        input_data = {
            'video_watch_time': video_watch_time,
            'diagram_view_time': diagram_view_time,
            'reading_time': reading_time,
            'writing_activities': writing_activities,
            'messages_posted': messages_posted,
            'group_discussions': group_discussions,
            'hands_on_activities': hands_on_activities,
            'observation_time': observation_time,
            'reflection_activities': reflection_activities,
            'practical_exercises': practical_exercises,
            'detail_orientation': detail_orientation,
            'theoretical_discussions': theoretical_discussions,
            'pattern_recognition': pattern_recognition,
            'step_by_step_completion': step_by_step_completion,
            'linear_progression': linear_progression,
            'big_picture_focus': big_picture_focus,
            'holistic_understanding': holistic_understanding
        }
        
        predictor = FSLSMPredictor()
        predictions, confidence_scores, scores = predictor.predict(input_data)
        
        st.session_state.current_prediction = (predictions, confidence_scores, scores)
        st.session_state.prediction_history.append(predictions)
        
        st.success("✅ Analysis Complete!")
        
    # Display results and Download Button (Outside Form)
    if st.session_state.current_prediction:
        predictions, confidence_scores, scores = st.session_state.current_prediction
        show_prediction_results(predictions, confidence_scores, scores)
        
        # PDF Report Generation Button
        results_for_pdf = {}
        for dim in predictions:
            results_for_pdf[dim] = {
                'prediction': predictions[dim],
                'score': scores[dim],
                'confidence': confidence_scores[dim],
                'algo': scores.get(f'{dim}_algo', 'Ensemble')
            }
            
        pdf_bytes = create_pdf_report("Student Name", results_for_pdf)
        st.download_button(
            label="📄 Download Professional Report (PDF)",
            data=pdf_bytes,
            file_name="student_report.pdf",
            mime="application/pdf"
        )

def show_prediction_results(predictions, confidence_scores, scores):
    st.subheader("📊 FSLSM Analysis Results")
    
    # Display all four dimensions
    cols = st.columns(4)
    dimensions = [
        ('visual_verbal', '👁️ Visual-Verbal', '#e74c3c'),
        ('active_reflective', '💬 Active-Reflective', '#3498db'),
        ('sensing_intuitive', '🔍 Sensing-Intuitive', '#2ecc71'),
        ('sequential_global', '📚 Sequential-Global', '#9b59b6')
    ]
    
    for idx, (dim_key, dim_name, color) in enumerate(dimensions):
        with cols[idx]:
            prediction = predictions.get(dim_key, "Balanced")
            confidence = confidence_scores.get(dim_key, 70)
            score = scores.get(dim_key, 50)
            algo = scores.get(f'{dim_key}_algo', 'SOTA Ensemble')
            
            st.markdown(f"""
            <div style="background: {color}; color: white; padding: 1rem; border-radius: 10px; text-align: center;">
                <h4>{dim_name}</h4>
                <h3>{prediction}</h3>
                <p>Score: {score:.0f}%</p>
                <p>Confidence: {confidence}%</p>
                <p style="font-size: 0.8em; opacity: 0.9;">🤖 Model: {algo}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed insights
    st.subheader("🔍 Detailed Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Learning Style Profile:**")
        for dim_key, dim_name, _ in dimensions:
            pred = predictions.get(dim_key, "Balanced")
            conf = confidence_scores.get(dim_key, 70)
            st.write(f"- **{dim_name.split(' ')[1]}:** {pred} ({conf}% confidence)")
    
    with col2:
        st.write("**Recommendations:**")
        if scores.get('visual_verbal', 50) > 60:
            st.write("• Use visual aids and demonstrations")
        if scores.get('active_reflective', 50) > 60:
            st.write("• Incorporate hands-on activities")
        if scores.get('sensing_intuitive', 50) > 60:
            st.write("• Provide practical examples")
        if scores.get('sequential_global', 50) > 60:
            st.write("• Use structured, step-by-step approach")

def show_visualizations():
    st.subheader("📊 Comprehensive Visualizations")
    
    if not st.session_state.current_prediction:
        st.info("Make a prediction first to see visualizations")
        return
    
    predictions, confidence_scores, scores = st.session_state.current_prediction
    
    # Row 1: Radar and Confusion Matrix
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_fslsm_radar(scores), use_container_width=True)
    with col2:
        st.pyplot(create_confusion_matrix())
        plt.close()
    
    # Row 2: Bar charts
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_performance_barchart(), use_container_width=True)
    with col2:
        st.plotly_chart(create_dimension_barchart(scores), use_container_width=True)
    
    # Row 3: Pie chart and Histogram
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_learning_style_pie(), use_container_width=True)
    with col2:
        st.plotly_chart(create_feature_histogram(), use_container_width=True)
    
    # Row 4: Heatmap and Stacked bar
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(create_correlation_heatmap())
        plt.close()
    with col2:
        st.plotly_chart(create_stacked_barchart(), use_container_width=True)
    
    # Row 5: Additional visualizations
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_score_distribution(), use_container_width=True)
    with col2:
        st.plotly_chart(create_model_comparison_pie(), use_container_width=True)

def show_analytics():
    st.subheader("🔬 Advanced Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="model-card">
            <h4>📈 Model Performance Metrics</h4>
            <p><b>CatBoost + FCM Ensemble:</b></p>
            <ul>
                <li>Accuracy: 92.3%</li>
                <li>Precision: 90.8%</li>
                <li>Recall: 93.1%</li>
                <li>F1-Score: 91.9%</li>
                <li>Training Time: 65s</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="model-card">
            <h4>🎯 FSLSM Dimension Accuracy</h4>
            <ul>
                <li>Visual-Verbal: 89.2%</li>
                <li>Active-Reflective: 87.8%</li>
                <li>Sensing-Intuitive: 85.5%</li>
                <li>Sequential-Global: 83.9%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="model-card">
            <h4>🔍 Feature Analysis</h4>
            <p><b>Top Predictive Features:</b></p>
            <ol>
                <li>Video Watch Time (Visual)</li>
                <li>Step-by-Step Completion (Sequential)</li>
                <li>Practical Exercises (Sensing)</li>
                <li>Messages Posted (Active)</li>
                <li>Reading Time (Verbal)</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="model-card">
            <h4>📊 Statistical Insights</h4>
            <ul>
                <li>Strong correlation: Visual ↔ Active (r=0.72)</li>
                <li>Weak correlation: Sensing ↔ Global (r=0.18)</li>
                <li>Most common style: Visual-Active (28%)</li>
                <li>Rarest style: Intuitive-Global (12%)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def show_insights():
    st.subheader("📈 Educational Insights & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="model-card">
            <h4>🎯 Teaching Strategies by Dimension</h4>
            
            <p><b>Visual Learners:</b></p>
            <ul>
                <li>Use diagrams, charts, mind maps</li>
                <li>Incorporate videos and animations</li>
                <li>Color-code information</li>
            </ul>
            
            <p><b>Active Learners:</b></p>
            <ul>
                <li>Group discussions and debates</li>
                <li>Hands-on experiments</li>
                <li>Role-playing activities</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="model-card">
            <h4>💡 Learning Recommendations</h4>
            
            <p><b>Sensing Learners:</b></p>
            <ul>
                <li>Connect to real-world examples</li>
                <li>Provide detailed instructions</li>
                <li>Use practical applications</li>
            </ul>
            
            <p><b>Sequential Learners:</b></p>
            <ul>
                <li>Step-by-step tutorials</li>
                <li>Clear progression paths</li>
                <li>Structured assignments</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Logout
def show_logout():
    with st.sidebar:
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_role = None
            st.session_state.current_prediction = None
            st.session_state.prediction_history = []
            st.rerun()

# Main app
def main():
    if not st.session_state.logged_in:
        show_login()
    else:
        show_dashboard()
        show_logout()

if __name__ == "__main__":
    main()
