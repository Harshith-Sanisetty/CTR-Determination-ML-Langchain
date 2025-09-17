import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyB8AN8PFVFDHhcZ3_Nvx8kOFBFGsA-IWqc")


MODEL_PATH = "GradientBoosting_champion.pkl"
PREPROCESSOR_PATH = "preprocessor.pkl"


st.set_page_config(
    page_title="Ad CTR Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Force override Streamlit's theme */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%) !important;
        font-family: 'Inter', sans-serif !important;
        color: white !important;
    }
    
    .main .block-container {
        background: transparent !important;
        padding-top: 2rem !important;
    }
    
    /* Header Styles */
    .hero-header {
        background: linear-gradient(145deg, rgba(255,255,255,0.15), rgba(255,255,255,0.05)) !important;
        backdrop-filter: blur(20px) !important;
        border-radius: 20px !important;
        padding: 2.5rem !important;
        margin-bottom: 2rem !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1) !important;
        text-align: center !important;
        color: white !important;
    }
    
    .hero-title {
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
        color: white !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
    }
    
    .hero-subtitle {
        font-size: 1.3rem !important;
        opacity: 0.9 !important;
        margin-bottom: 0 !important;
        color: rgba(255,255,255,0.9) !important;
    }
    
    /* Card Styles */
    .glass-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.15), rgba(255,255,255,0.05)) !important;
        backdrop-filter: blur(20px) !important;
        border-radius: 20px !important;
        padding: 2rem !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1) !important;
        margin-bottom: 1.5rem !important;
        transition: all 0.3s ease !important;
    }
    
    .glass-card:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Form Input Overrides */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.95) !important;
        border: 2px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 12px !important;
        color: #2d3748 !important;
        font-weight: 500 !important;
        padding: 0.75rem 1rem !important;
        font-size: 1rem !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    .stSelectbox > div > div > div {
        background: rgba(255, 255, 255, 0.95) !important;
        border: 2px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 12px !important;
        color: #2d3748 !important;
    }
    
    .stDateInput > div > div > input,
    .stTimeInput > div > div > input {
        background: rgba(255, 255, 255, 0.95) !important;
        border: 2px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 12px !important;
        color: #2d3748 !important;
    }
    
    /* Labels */
    .stTextInput > label,
    .stSelectbox > label,
    .stDateInput > label,
    .stTimeInput > label {
        color: white !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 1rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.6) !important;
        background: linear-gradient(45deg, #764ba2, #667eea) !important;
    }
    
    /* Section Headers */
    .section-header {
        color: white !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        margin-bottom: 1.5rem !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.75rem !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-block !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 25px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        margin-bottom: 1rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    .status-success {
        background: linear-gradient(45deg, #22c55e, #16a34a) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(34, 197, 94, 0.4) !important;
    }
    
    .status-warning {
        background: linear-gradient(45deg, #f59e0b, #d97706) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.4) !important;
    }
    
    .status-error {
        background: linear-gradient(45deg, #ef4444, #dc2626) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4) !important;
    }
    
    /* Result Cards */
    .result-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.2), rgba(255,255,255,0.1)) !important;
        backdrop-filter: blur(20px) !important;
        border-radius: 20px !important;
        padding: 2rem !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1) !important;
        text-align: center !important;
        color: white !important;
        margin: 0.5rem 0 !important;
        transition: all 0.3s ease !important;
    }
    
    .result-card:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 20px 45px rgba(0, 0, 0, 0.15) !important;
    }
    
    .result-value {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
    }
    
    .result-label {
        font-size: 1.1rem !important;
        opacity: 0.9 !important;
        font-weight: 500 !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #22c55e, #16a34a) !important;
        border-radius: 10px !important;
    }
    
    /* Expander Styles */
    .streamlit-expanderHeader {
        background: linear-gradient(145deg, rgba(255,255,255,0.15), rgba(255,255,255,0.05)) !important;
        border-radius: 15px !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 1rem 1.5rem !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    .streamlit-expanderContent {
        background: linear-gradient(145deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05)) !important;
        border-radius: 0 0 15px 15px !important;
        padding: 1.5rem !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-top: none !important;
    }
    
    /* Metrics Override */
    [data-testid="metric-container"] {
        background: linear-gradient(145deg, rgba(255,255,255,0.15), rgba(255,255,255,0.05)) !important;
        border-radius: 15px !important;
        padding: 1.5rem !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    [data-testid="metric-container"] > div {
        color: white !important;
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: white !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="metric-container"] [data-testid="metric-label"] {
        color: rgba(255,255,255,0.8) !important;
        font-weight: 600 !important;
    }
    
    /* Text colors */
    .stMarkdown, .stMarkdown p, .stMarkdown li {
        color: white !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    
    /* Loading spinner */
    .stSpinner > div {
        border-top-color: white !important;
    }
    
    /* Success/Error/Warning message boxes */
    .stSuccess {
        background: rgba(34, 197, 94, 0.15) !important;
        border: 1px solid rgba(34, 197, 94, 0.3) !important;
        border-radius: 10px !important;
        color: #22c55e !important;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.15) !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        border-radius: 10px !important;
        color: #ef4444 !important;
    }
    
    .stWarning {
        background: rgba(251, 191, 36, 0.15) !important;
        border: 1px solid rgba(251, 191, 36, 0.3) !important;
        border-radius: 10px !important;
        color: #fbbf24 !important;
    }
    
    .stInfo {
        background: rgba(59, 130, 246, 0.15) !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        border-radius: 10px !important;
        color: #3b82f6 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts(model_path, preprocessor_path):
    try:
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        return model, preprocessor
    except FileNotFoundError:
        return None, None

@st.cache_resource
def setup_langchain_chain(_prompt_template, api_key):
    if not LANGCHAIN_AVAILABLE: 
        return None
    if not api_key or api_key == "PASTE_YOUR_GEMINI_API_KEY_HERE":
        return None
    
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            api_key=api_key,
            temperature=0
        )
        prompt = PromptTemplate(
            template=_prompt_template, 
            input_variables=["input_text", "category_list"]
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"AI initialization failed: {str(e)}")
        return None

def process_timestamp(timestamp_obj):
    day_of_week = timestamp_obj.strftime('%A')
    hour = timestamp_obj.hour
    if 5 <= hour < 12: 
        time_of_day = 'Morning'
    elif 12 <= hour < 17: 
        time_of_day = 'Afternoon'
    elif 17 <= hour < 21: 
        time_of_day = 'Evening'
    else: 
        time_of_day = 'Night'
    return day_of_week, time_of_day

def classify_with_fallback(chain, input_text, categories, fallback="Other"):
    if not chain:
        return fallback
    
    try:
        result = chain.run(input_text=input_text, category_list=str(categories))
        result = result.strip()
        if result in categories:
            return result
        else:
            return fallback
    except Exception:
        return fallback

def create_gauge_chart(probability):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Click Probability %", 'font': {'color': 'white', 'size': 16}},
        delta = {'reference': 50, 'increasing': {'color': "#22c55e"}, 'decreasing': {'color': "#ef4444"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickcolor': "white", 'tickfont': {'color': 'white'}},
            'bar': {'color': "#667eea"},
            'bgcolor': "rgba(255,255,255,0.1)",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, 25], 'color': "rgba(239, 68, 68, 0.3)"},
                {'range': [25, 50], 'color': "rgba(251, 191, 36, 0.3)"},
                {'range': [50, 75], 'color': "rgba(34, 197, 94, 0.3)"},
                {'range': [75, 100], 'color': "rgba(34, 197, 94, 0.5)"}],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white", 'family': "Inter"},
        height=300
    )
    return fig

def create_feature_radar_chart(input_data):
    
    features = ['Age', 'Area Income', 'Daily Time Spent on Site', 'Daily Internet Usage']
    values = []
    
    
    age_norm = min(input_data['Age'].iloc[0] / 80, 1) * 100
    income_norm = min(input_data['Area Income'].iloc[0] / 100000, 1) * 100
    time_norm = min(input_data['Daily Time Spent on Site'].iloc[0] / 120, 1) * 100
    usage_norm = min(input_data['Daily Internet Usage'].iloc[0] / 300, 1) * 100
    
    values = [age_norm, income_norm, time_norm, usage_norm]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=features,
        fill='toself',
        name='User Profile',
        line=dict(color='#667eea'),
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickcolor='white',
                gridcolor='rgba(255,255,255,0.3)',
                tickfont=dict(color='white')
            ),
            angularaxis=dict(
                tickcolor='white',
                gridcolor='rgba(255,255,255,0.3)',
                tickfont=dict(color='white')
            )
        ),
        showlegend=True,
        legend=dict(font=dict(color='white')),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color='white', family='Inter'),
        height=400
    )
    return fig

model, preprocessor = load_artifacts(MODEL_PATH, PREPROCESSOR_PATH)


st.markdown("""
<div class="hero-header">
    <div class="hero-title">🎯 Ad CTR Predictor</div>
    <div class="hero-subtitle">Click-Through Rate Prediction System : Harshith Sanisetty Version</div>
</div>
""", unsafe_allow_html=True)


col1, col2, col3 = st.columns(3)

with col1:
    if model is not None and preprocessor is not None:
        st.markdown('<div class="status-badge status-success">✅ Model Ready</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge status-error">❌ Model Missing</div>', unsafe_allow_html=True)

with col2:
    if API_KEY != "PASTE_YOUR_GEMINI_API_KEY_HERE" and API_KEY:
        st.markdown('<div class="status-badge status-success">🤖 Used Gemini API for Ad topic line to category classification</div>', unsafe_allow_html=True)
        ai_enabled = True
    else:
        st.markdown('<div class="status-badge status-warning">⚠️ AI Disabled</div>', unsafe_allow_html=True)
        ai_enabled = False

with col3:
    if LANGCHAIN_AVAILABLE:
        st.markdown('<div class="status-badge status-success">🔗 Dependencies OK</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge status-warning">📦 Install LangChain</div>', unsafe_allow_html=True)

if model is None or preprocessor is None:
    st.error("🚨 **Critical Error:** Model files not found. Please ensure model files are in the correct directory.")
    st.stop()


st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">📊 Input Parameters</div>', unsafe_allow_html=True)

with st.form("prediction_form", clear_on_submit=False):
    
    st.markdown('<div class="section-header">👤 User Demographics</div>', unsafe_allow_html=True)
    demo_col1, demo_col2, demo_col3 = st.columns(3)
    
    with demo_col1:
        age = st.text_input("Age (years)", "35", help="Enter user age between 18-80")
    with demo_col2:
        area_income = st.text_input("Area Income ($)", "55000", help="Annual household income in USD")
    with demo_col3:
        gender = st.selectbox("Gender", ["Male", "Female"], help="User gender")

    st.markdown("---")
    
   
    st.markdown('<div class="section-header">🌐 User Behavior</div>', unsafe_allow_html=True)
    behavior_col1, behavior_col2 = st.columns(2)
    
    with behavior_col1:
        daily_time_spent = st.text_input("Daily Site Time (minutes)", "60.0", 
                                       help="Average minutes spent on site per day")
    with behavior_col2:
        daily_internet_usage = st.text_input("Daily Internet Usage (MB)", "180.0",
                                            help="Daily internet consumption in MB")

    st.markdown("---")
    
   
    st.markdown('<div class="section-header">📢 Advertisement Details</div>', unsafe_allow_html=True)
    ad_col1, ad_col2 = st.columns(2)
    
    with ad_col1:
        ad_topic_line = st.text_input("Ad Topic Line", "Cloned 5thgeneration orchestration",
                                    help="The headline or topic of the advertisement")
        country = st.text_input("Country", "United States", help="User's country")
    
    with ad_col2:
        date_col, time_col = st.columns(2)
        with date_col:
            click_date = st.date_input("Display Date", datetime.now())
        with time_col:
            click_time = st.time_input("Display Time", datetime.now().time())

    st.markdown("---")
    
    
    submitted = st.form_submit_button("🔮 Predict Click-Through Rate", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)


if submitted:
    try:
        
        age = int(age)
        area_income = float(area_income)
        daily_time_spent = float(daily_time_spent)
        daily_internet_usage = float(daily_internet_usage)
        
        
        if not (18 <= age <= 100):
            st.error("⚠️ Please enter a valid age between 18-100 years")
            st.stop()
        if area_income <= 0:
            st.error("⚠️ Please enter a positive income value")
            st.stop()
        if daily_time_spent < 0:
            st.error("⚠️ Please enter a positive time value")
            st.stop()
        if daily_internet_usage < 0:
            st.error("⚠️ Please enter a positive internet usage value")
            st.stop()
        
        with st.spinner("🔮 Analyzing user profile and predicting..."):
           
            timestamp = datetime.combine(click_date, click_time)
            day_of_week, time_of_day = process_timestamp(timestamp)
            
          
            ad_categories = [
                'Infrastructure', 'Other', 'Generational Technology', 
                'Business Performance', 'Human Resources', 'Software & Applications', 
                'Organizational Structure', 'Process Improvement', 
                'Data Analysis & Prediction', 'Quality Assurance', 
                'Customer Service', 'Security', 'Project Management', 'Hardware'
            ]
            
            if ai_enabled:
                ad_prompt = "Classify the following ad headline: '{input_text}' into one of these categories: {category_list}. Return only the single, most appropriate category name."
                ad_chain = setup_langchain_chain(ad_prompt, API_KEY)
                ad_category = classify_with_fallback(ad_chain, ad_topic_line, ad_categories, "Other")
            else:
                ad_category = "Other"

            
            continents = ['Unknown', 'Asia', 'North America', 'Africa', 'South America', 'Oceania', 'Europe', 'Antarctica']
            
            if ai_enabled:
                continent_prompt = "What continent is the country '{input_text}' in? Choose from: {category_list}. Return only the continent name."
                continent_chain = setup_langchain_chain(continent_prompt, API_KEY)
                continent = classify_with_fallback(continent_chain, country, continents, "Unknown")
            else:
                country_mapping = {
                    "united states": "North America", "usa": "North America", "us": "North America",
                    "canada": "North America", "mexico": "North America",
                    "china": "Asia", "japan": "Asia", "india": "Asia", "south korea": "Asia",
                    "germany": "Europe", "france": "Europe", "uk": "Europe", "united kingdom": "Europe",
                    "spain": "Europe", "italy": "Europe", "russia": "Europe",
                    "brazil": "South America", "argentina": "South America", "chile": "South America",
                    "australia": "Oceania", "new zealand": "Oceania",
                    "south africa": "Africa", "nigeria": "Africa", "egypt": "Africa"
                }
                continent = country_mapping.get(country.lower(), "Unknown")

           
            input_data = pd.DataFrame({
                'Daily Time Spent on Site': [daily_time_spent],
                'Age': [age],
                'Area Income': [area_income],
                'Daily Internet Usage': [daily_internet_usage],
                'Gender': [gender],
                'Ad_Category': [ad_category.strip()],
                'day_of_week': [day_of_week],
                'time_of_day': [time_of_day],
                'Continent': [continent.strip()]
            })
            
            
            transformed_data = preprocessor.transform(input_data)
            prediction_proba = model.predict_proba(transformed_data)[0][1]
            prediction = model.predict(transformed_data)[0]

           
            st.markdown("---")
            st.markdown('<div class="section-header pulse">🎯 Prediction Results</div>', unsafe_allow_html=True)
            
            
            result_col1, result_col2, result_col3 = st.columns(3)
            
            with result_col1:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                if prediction == 1:
                    st.markdown('<div class="result-value" style="color: #22c55e;">✅ WILL CLICK</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-value" style="color: #ef4444;">❌ WON\'T CLICK</div>', unsafe_allow_html=True)
                st.markdown('<div class="result-label">Model Prediction</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with result_col2:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="result-value">{prediction_proba:.1%}</div>', unsafe_allow_html=True)
                st.markdown('<div class="result-label">Click Probability</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with result_col3:
                confidence_level = "High" if prediction_proba >= 0.7 or prediction_proba <= 0.3 else "Medium"
                confidence_color = "#22c55e" if confidence_level == "High" else "#fbbf24"
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="result-value" style="color: {confidence_color};">{confidence_level}</div>', unsafe_allow_html=True)
                st.markdown('<div class="result-label">Confidence Level</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

           
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("**📊 Probability Gauge**")
                gauge_fig = create_gauge_chart(prediction_proba)
                st.plotly_chart(gauge_fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with viz_col2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("**🎯 User Profile Radar**")
                radar_fig = create_feature_radar_chart(input_data)
                st.plotly_chart(radar_fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

           
            
           
            

    except ValueError as e:
        st.error(f"❌ **Input Error:** Please ensure all fields contain valid values. {str(e)}")
    except Exception as e:
        st.error(f"❌ **System Error:** {str(e)}")
        st.info("Please try again or contact support if the issue persists.")

