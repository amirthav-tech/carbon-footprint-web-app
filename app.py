import streamlit as st
import base64

import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ------------------ Page config & CSS ------------------
st.set_page_config(
    page_title="🌍 CarbonCurve - ML Carbon Footprint Predictor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# -------- BACKGROUND IMAGE FUNCTION --------
# ✅ AUTO DETECT BACKGROUND (works from any folder)

# ------------------ BACKGROUND FUNCTION ------------------
def add_bg_from_local(image_file):
    """Add background image from local path"""
    abs_path = os.path.abspath(image_file)
    

    if not os.path.exists(abs_path):
        st.error(f"❌ Background image not found at: {abs_path}")
        return

    # Read and encode image
    with open(abs_path, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()

    # Apply as CSS background
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: 
                linear-gradient(rgba(0,0,0,0.25), rgba(0,0,0,0.25)), 
                url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ USE THE EXACT PATH YOU GAVE
add_bg_from_local("app/background_min.jpg")




st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
   .stApp { 
    font-family: 'Poppins', sans-serif; 
    /* background removed to allow image from add_bg_from_local() */
} background: linear-gradient(135deg, #e8f5e8 0%, #dcf5dc 50%, #f0fdf4 100%); }
  div.eco-header h1, .stApp h1, h1 {
    color: #00ffcc !important;              /* 💚 change color here */
    text-shadow: 2px 2px 8px rgba(0,0,0,0.8) !important;
    font-weight: 800 !important;
    font-size: 2.6rem !important;
}

/* Make sure Streamlit default header color doesn't override */
.stMarkdown h1, .block-container h1, [data-testid="stMarkdownContainer"] h1 {
    color: #00ffcc !important;
}
.eco-header p {
    color: #e6ffee !important;
    text-shadow: 1px 1px 5px rgba(0,0,0,0.6) !important;
}
    .form-container { background: rgba(255,255,255,0.95); 
            border-radius: 20px; 
            border: 2px solid rgba(34,197,94,0.12);
             padding: 20px; }
    .section-header { background: linear-gradient(135deg, #059669 0%, #22c55e 100%); 
            color: white; 
            padding: 10px 12px; 
            border-radius: 12px;
             margin: 14px 0; 
            text-align: center; 
            font-weight: 600; }
    .results-container { background: linear-gradient(135deg,#e8f5e8,#f1f8e9);
             border-radius: 12px; 
            padding: 18px; 
            margin-top: 18px; }
    .stButton>button { background: linear-gradient(45deg,#4caf50,#66bb6a); 
            color: white; 
            border-radius: 50px; 
            padding: 12px 20px; 
            font-weight:600; 
            border: none; }
    .stButton>button:hover { transform: translateY(-2px); 
            box-shadow: 0 5px 15px rgba(76,175,80,0.3); }
    .reset-button button { background: linear-gradient(45deg,#f39c12,#e67e22) !important; 
            color: white !important;
             border-radius: 50px !important; 
            padding: 8px 16px !important; 
            font-weight:500 !important; }
    .success-banner { background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
            color: white; 
            padding: 15px; 
            border-radius: 12px; 
            text-align: center;
             margin: 20px 0; 
            animation: fadeIn 0.5s; }
    div[data-testid="stMarkdownContainer"] ul li,
div[data-testid="stMarkdownContainer"] li,
section.main ul li,
section.main li {
    color: white !important;            /* ✅ Force text color to white */
    font-weight: 500 !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.6);
    list-style-type: disc !important;
    margin-left: 20px !important;
}

/* Optional hover effect for visibility */
div[data-testid="stMarkdownContainer"] ul li:hover {
    color: #d1fae5 !important;          /* light green hover */
    transform: scale(1.01);
    transition: all 0.2s ease-in-out;
}
 
    @keyframes fadeIn { from { opacity: 0; transform: translateY(-10px); } to { opacity: 1; transform: translateY(0); } }
    #MainMenu{visibility:hidden;} footer{visibility:hidden;} header{visibility:hidden}
            /* --- Force white text for results content --- */
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] strong {
    color: white !important;
    text-shadow: 0 2px 6px rgba(0,0,0,0.6);
}

/* Keep the highlighted CO₂ value green */
[data-testid="stMarkdownContainer"] strong b,
[data-testid="stMarkdownContainer"] b {
    color: #22c55e !important;
}

</style>
""", unsafe_allow_html=True)

# ------------------ Session State Initialization ------------------
def initialize_session_state():
    """Initialize session state with default values"""
    if 'form_submitted' not in st.session_state:
        st.session_state.form_submitted = False
    if 'reset_form' not in st.session_state:
        st.session_state.reset_form = False

def reset_form():
    """Reset all form values to defaults"""
    st.session_state.form_submitted = False
    st.session_state.reset_form = True
    # Clear any cached results
    if 'last_footprint' in st.session_state:
        del st.session_state.last_footprint

# Initialize session state
initialize_session_state()

# ------------------ Model loading ------------------
@st.cache_resource
def load_models():
    """Try to load saved model and scaler. If not found, return (None, None, False)."""
    model_paths = ["models/PolynomialRegression.joblib", "../models/PolynomialRegression.joblib"]
    scaler_paths = ["scaler.pkl", "../scaler.pkl"]
    mlbcooking_paths=["mlb_cooking.pkl", "../mlb_cooking.pkl"]
    mlbrecycle_paths=["mlb_recycle.pkl", "../mlb_recycle.pkl"]
    trainingcols_paths=["models/training_columns.joblib", "../models/training_columns.joblib"]

    for mpath in model_paths:
        for spath in scaler_paths:
            for mlbcpath in mlbcooking_paths:
                for mlbrpath in mlbrecycle_paths:
                    for trainpath in trainingcols_paths:
                        try:
                            if os.path.exists(mpath) and os.path.exists(spath):
                                model = joblib.load(mpath)
                                scaler = joblib.load(spath)
                                mlbcooking = joblib.load(mlbcpath)
                                mlbrecycling = joblib.load(mlbrpath)
                                training_columns = joblib.load(trainpath)
                                return model, scaler, mlbcooking, mlbrecycling, training_columns, True
                        except Exception:
                            continue

                    return None, None, False

# Feature mapping dictionaries (used when creating model input)
FEATURE_MAPPINGS = {
    'Body Type': {'underweight': 0, 'normal': 1, 'overweight': 2, 'obese': 3},
    'Diet': {'omnivore': 0, 'pescatarian': 1, 'vegetarian': 2, 'vegan': 3},
    'How Often Shower': {'less frequently': 0, 'daily': 1, 'more frequently': 2, 'twice a day': 3},
    'Heating Energy Source': {'coal': 0, 'wood': 1, 'natural gas': 2, 'electricity': 3},
    'Transport': {'walk/bicycle': 0, 'public': 1, 'private': 2},
    'Vehicle Type': {'none': 0, 'petrol': 1, 'lpg': 2, 'diesel': 3, 'electric': 4, 'hybrid': 5},
    'Waste Bag Size': {'small': 0, 'medium': 1, 'large': 2, 'extra large': 3},
    'Frequency of Traveling by Air': {'never': 0, 'rarely': 1, 'occasionally': 2, 'frequently': 3, 'very frequently': 4}
}

# ------------------ Helper functions ------------------

def prepare_data_for_model(form):
    """Turn the form dictionary into a DataFrame suitable for the model.
    This function attempts to reproduce the preprocessing used at training time:
    - numeric columns with the exact names expected by the scaler
    - categorical columns mapped to integers using FEATURE_MAPPINGS (when available)
    - cooking appliance flags (binary)
    - simple binary encoding for sex/social/efficiency

    If you have the original encoders, replace this logic with the real encoders to guarantee exact column ordering.
    """
    try:
        # Base numeric fields (the names match what the rest of the app expects when scaling)
        row = {
            "Monthly Grocery Bill": float(form.get('grocery', 0.0)),
            "Vehicle Monthly Distance Km": float(form.get('vehicle_km', 0.0)),
            "Waste Bag Weekly Count": float(form.get('waste_count', 0.0)),
            "How Long TV PC Daily Hour": float(form.get('tv_hours', 0.0)),
            "How Many New Clothes Monthly": float(form.get('clothes', 0.0)),
            "How Long Internet Daily Hour": float(form.get('internet_hours', 0.0))
        }

        # Map categorical features to integers when we have a mapping
        # Use `.get` with defaults to avoid KeyErrors
        row['Body Type'] = FEATURE_MAPPINGS['Body Type'].get(form.get('body_type', '').lower(), 1) if form.get('body_type') else 1
        row['Diet'] = FEATURE_MAPPINGS['Diet'].get(form.get('diet', '').lower(), 1) if form.get('diet') else 1
        row['How Often Shower'] = FEATURE_MAPPINGS['How Often Shower'].get(form.get('shower', '').lower(), 1) if form.get('shower') else 1
        row['Heating Energy Source'] = FEATURE_MAPPINGS['Heating Energy Source'].get(form.get('heating', '').lower(), 2) if form.get('heating') else 2
        row['Transport'] = FEATURE_MAPPINGS['Transport'].get(form.get('transport', '').lower(), 1) if form.get('transport') else 1
        row['Vehicle Type'] = FEATURE_MAPPINGS['Vehicle Type'].get(form.get('vehicle_type', '').lower(), 0) if form.get('vehicle_type') else 0
        row['Waste Bag Size'] = FEATURE_MAPPINGS['Waste Bag Size'].get(form.get('waste_size', '').lower(), 1) if form.get('waste_size') else 1
        row['Frequency of Traveling by Air'] = FEATURE_MAPPINGS['Frequency of Traveling by Air'].get(form.get('air_travel', '').lower(), 0) if form.get('air_travel') else 0

        # Binary-like fields
        sex = form.get('sex', '').lower()
        row['Sex_Male'] = 1 if sex == 'male' else 0
        social = form.get('social', '').lower()
        row['Social_SometimesOrOften'] = 1 if social in ['sometimes', 'often'] else 0
        efficiency = form.get('efficiency', '')
        row['Efficiency_Yes'] = 1 if efficiency == 'Yes' else (0 if efficiency == 'No' else 0)

        # Cooking appliances -> one-hot style flags
        cooking = form.get('cooking', []) or []
        all_cooking = ['Stove', 'Oven', 'Microwave', 'Grill', 'Airfryer']
        for item in all_cooking:
            row[f'cooking_{item.lower()}'] = 1 if item in cooking else 0

        # Recycling count (how many types they recycle)
        recycling = form.get('recycling', []) or []
        row['Recycling_Count'] = len(recycling)

        # Turn into DataFrame with stable column order
        df = pd.DataFrame([row])

        return df
    except Exception as e:
        st.error(f"Error preparing data for model: {e}")
        return None


def calculate_demo_footprint(form_data):
    """Fallback heuristic calculator used when the trained model is not available."""
    footprint = 3.0  # base

    diet_impact = {'vegan': 0.5, 'vegetarian': 1.0, 'pescatarian': 1.5, 'omnivore': 2.5}
    footprint += diet_impact.get(form_data.get('diet', '').lower(), 1.0)

    transport_impact = {'walk/bicycle': 0.2, 'public': 1.0, 'private': 2.5}
    footprint += transport_impact.get(form_data.get('transport', '').lower(), 1.0)

    vehicle_impact = {'none': 0, 'electric': 0.5, 'hybrid': 1.0, 'lpg': 1.5, 'petrol': 2.0, 'diesel': 2.5}
    footprint += vehicle_impact.get(form_data.get('vehicle_type', '').lower(), 0)

    footprint += float(form_data.get('vehicle_km', 0)) / 1000 * 0.15

    air_impact = {'never': 0, 'rarely': 0.5, 'frequently': 2.0, 'very frequently': 4.0}
    footprint += air_impact.get(form_data.get('air_travel', '').lower(), 0)

    heating_impact = {'electricity': 0.5, 'natural gas': 1.0, 'wood': 1.5, 'coal': 2.5}
    footprint += heating_impact.get(form_data.get('heating', '').lower(), 1.0)

    waste_impact = {'small': 0.1, 'medium': 0.3, 'large': 0.6, 'extra large': 1.0}
    footprint += waste_impact.get(form_data.get('waste_size', '').lower(), 0.3)
    footprint += float(form_data.get('waste_count', 0)) * 0.05

    footprint += float(form_data.get('tv_hours', 0)) * 0.02
    footprint += float(form_data.get('internet_hours', 0)) * 0.01

    footprint += float(form_data.get('clothes', 0)) * 0.08
    footprint += float(form_data.get('grocery', 0)) / 100 * 0.1

    efficiency_reduction = {'Yes': 0.8, 'Sometimes': 0.9, 'No': 1.0}
    footprint *= efficiency_reduction.get(form_data.get('efficiency', ''), 1.0)

    if form_data.get('recycling'):
        reduction = len(form_data['recycling']) * 0.05
        footprint *= (1 - min(reduction, 0.15))

    return max(footprint, 0.5)


# ------------------ App UI ------------------
model, scaler, mlbcooking, mlbrecycling, training_columns, models_loaded = load_models()


# Header
st.markdown("""
<div class="eco-header">
    <h1>🌍 CarbonCurve - ML Carbon Footprint Predictor</h1>
    <p>Calculate your environmental impact and discover ways to reduce your carbon footprint</p>
</div>
""", unsafe_allow_html=True)

# Show success message after form submission
if st.session_state.form_submitted and 'last_footprint' in st.session_state:
    st.markdown(f"""
    <div class="success-banner">
        <h3>✅ Calculation Complete!</h3>
        <p>Your carbon footprint: <strong>{st.session_state.last_footprint:.2f} tons CO₂/year</strong></p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="form-container">', unsafe_allow_html=True)

# Set default values based on reset state
def get_default_index(options, default_value, reset_override=0):
    if st.session_state.reset_form:
        return reset_override
    try:
        return options.index(default_value) if default_value in options else reset_override
    except:
        return reset_override

def get_default_numeric(default_value, reset_override=0):
    if st.session_state.reset_form:
        return reset_override
    return default_value

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "👤 Personal Info",
    "🚗 Transportation", 
    "🗑️ Waste Management",
    "⚡ Energy Usage",
    "🛒 Consumption"
])

with tab1:
    st.markdown('<div class="section-header">Personal Information</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        body_type_options = ["Select...", "underweight", "normal", "overweight", "obese"]
        body_type = st.selectbox("🏃 Body Type", body_type_options, 
                                index=get_default_index(body_type_options, "normal", 0))
        
        sex_options = ["Select...", "female", "male"]
        sex = st.selectbox("👥 Sex", sex_options, 
                          index=get_default_index(sex_options, "female", 0))
        
        diet_options = ["Select...", "vegan", "vegetarian", "pescatarian", "omnivore"]
        diet = st.selectbox("🥗 Diet Type", diet_options, 
                           index=get_default_index(diet_options, "omnivore", 0))
    with col2:
        social_options = ["Select...", "never", "sometimes", "often"]
        social = st.selectbox("🎭 Social Activity Frequency", social_options, 
                             index=get_default_index(social_options, "sometimes", 0))
        
        shower_options = ["Select...", "less frequently", "daily", "more frequently", "twice a day"]
        shower = st.selectbox("🚿 Shower Frequency", shower_options, 
                             index=get_default_index(shower_options, "daily", 0))
        
        grocery = st.number_input("🛒 Monthly Grocery Bill ($)", min_value=0.0, 
                                 value=get_default_numeric(200.0, 0.0), step=10.0)

with tab2:
    st.markdown('<div class="section-header">Transportation Details</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        transport_options = ["Select...", "walk/bicycle", "public", "private"]
        transport = st.selectbox("🚌 Primary Transportation", transport_options, 
                                index=get_default_index(transport_options, "public", 0))
        
        vehicle_options = ["Select...", "none", "petrol", "diesel", "hybrid", "lpg", "electric"]
        vehicle_type = st.selectbox("🚙 Vehicle Type", vehicle_options, 
                                   index=get_default_index(vehicle_options, "none", 0))
    with col2:
        vehicle_km = st.number_input("📏 Monthly Vehicle Distance (km)", min_value=0, 
                                    value=get_default_numeric(500, 0), step=50)
        
        air_options = ["Select...", "never", "rarely", "frequently", "very frequently"]
        air_travel = st.selectbox("✈️ Air Travel Frequency", air_options, 
                                 index=get_default_index(air_options, "never", 0))

with tab3:
    st.markdown('<div class="section-header">Waste Management</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        waste_options = ["Select...", "small", "medium", "large", "extra large"]
        waste_size = st.selectbox("🗑️ Waste Bag Size", waste_options, 
                                 index=get_default_index(waste_options, "medium", 0))
        
        waste_count = st.number_input("📦 Weekly Waste Bag Count", min_value=0, 
                                     value=get_default_numeric(2, 0), step=1)
    with col2:
        recycling_options = ["Glass", "Metal", "Paper", "Plastic"]
        recycling = st.multiselect("♻️ Recycling Items", recycling_options, 
                                  default=[] if st.session_state.reset_form else ["Paper", "Plastic"])

with tab4:
    st.markdown('<div class="section-header">Energy Usage</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        heating_options = ["Select...", "coal", "natural gas", "wood", "electricity"]
        heating = st.selectbox("🔥 Heating Energy Source", heating_options, 
                              index=get_default_index(heating_options, "electricity", 0))
        
        efficiency_options = ["Select...", "No", "Sometimes", "Yes"]
        efficiency = st.selectbox("💡 Energy Efficiency Measures", efficiency_options, 
                                 index=get_default_index(efficiency_options, "Sometimes", 0))
    with col2:
        tv_hours = st.number_input("📺 Daily TV/PC Hours", min_value=0.0, 
                                  value=get_default_numeric(3.0, 0.0), step=0.5)
        
        internet_hours = st.number_input("🌐 Daily Internet Hours", min_value=0.0, 
                                        value=get_default_numeric(4.0, 0.0), step=0.5)

with tab5:
    st.markdown('<div class="section-header">Consumption Habits</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        cooking_options = ["Stove", "Oven", "Microwave", "Grill", "Airfryer"]
        cooking = st.multiselect("🍳 Cooking Appliances", cooking_options, 
                                default=[] if st.session_state.reset_form else ["Stove", "Microwave"])
    with col2:
        clothes = st.number_input("👕 New Clothes per Month", min_value=0, 
                                 value=get_default_numeric(2, 0), step=1)

st.markdown('</div>', unsafe_allow_html=True)

# Reset the reset flag after form is rendered
if st.session_state.reset_form:
    st.session_state.reset_form = False

# Calculate button with reset option
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    if st.button("🌱 Calculate My Carbon Footprint", use_container_width=True):
        # form_data = {  
        form_data = {
            'body_type': body_type if body_type != "Select..." else "",
            'sex': sex if sex != "Select..." else "",
            'diet': diet if diet != "Select..." else "",
            'social': social if social != "Select..." else "",
            'shower': shower if shower != "Select..." else "",
            'grocery': grocery,
            'transport': transport if transport != "Select..." else "",
            'vehicle_type': vehicle_type if vehicle_type != "Select..." else "",
            'vehicle_km': vehicle_km,
            'air_travel': air_travel if air_travel != "Select..." else "",
            'waste_size': waste_size if waste_size != "Select..." else "",
            'waste_count': waste_count,
            'recycling': recycling,
            'heating': heating if heating != "Select..." else "",
            'efficiency': efficiency if efficiency != "Select..." else "",
            'tv_hours': tv_hours,
            'internet_hours': internet_hours,
            'cooking': cooking,
            'clothes': clothes
        }

        required_fields = ['body_type', 'sex', 'diet', 'transport']
        missing_fields = [field for field in required_fields if not form_data[field]]

        if missing_fields:
            st.error(f"⚠️ Please fill in the following required fields: {', '.join([f.replace('_', ' ').title() for f in missing_fields])}")
        else:
            # Prediction flow
            if models_loaded and model is not None and scaler is not None:
                prepared = prepare_data_for_model(form_data)
                if prepared is not None:
                    try:
                        # Ensure columns match what model was trained on
                        prepared = prepared.reindex(columns=training_columns, fill_value=0)

                        # Find numeric columns we expect to scale
                        numeric_cols = [
                            "Monthly Grocery Bill",
                            "Vehicle Monthly Distance Km",
                            "Waste Bag Weekly Count",
                            "How Long TV PC Daily Hour",
                            "How Many New Clothes Monthly",
                            "How Long Internet Daily Hour"
                        ]

                        # Ensure all numeric cols exist in prepared
                        for c in numeric_cols:
                            if c not in prepared.columns:
                                prepared[c] = 0.0

                        # Scale numeric columns (scaler expects same ordering as training)
                        try:
                            prepared_scaled = prepared.copy()
                            prepared_scaled[numeric_cols] = scaler.transform(prepared[numeric_cols])
                        except Exception:
                            # If scaler fails (mismatch), fall back to unscaled prepared
                            prepared_scaled = prepared.copy()

                        preds = model.predict(prepared_scaled)
                        footprint = float(preds[0])
                    except Exception as e:
                        st.warning(f"Model prediction failed, using demo heuristic. ({e})")
                        footprint = calculate_demo_footprint(form_data)
                else:
                    footprint = calculate_demo_footprint(form_data)
            else:
                footprint = calculate_demo_footprint(form_data)

            footprint = max(0.5, min(footprint, 50.0))
            
            # Store results in session state
            st.session_state.form_submitted = True
            st.session_state.last_footprint = footprint

            # Display results
            st.markdown('<div class="results-container">', unsafe_allow_html=True)
            st.markdown(f"""
                <h2>🌍 Your Carbon Footprint Results</h2>
                <h3 style='color:#059669;'>🎯 Your estimated annual carbon footprint: <strong>{footprint:.2f} tons CO₂</strong></h3>
                <p style='font-size:1.05em;'>This is <strong>{'above' if footprint > 7 else 'below'}</strong> the global average of 7 tons per person per year.</p>
            """, unsafe_allow_html=True)

            # Recommendations
            recommendations = []
            if form_data['diet'].lower() == "omnivore":
                recommendations.append("🥗 Consider reducing meat consumption and trying more plant-based meals")
            if form_data['vehicle_type'].lower() in ["petrol", "diesel"]:
                recommendations.append("🚌 Use public transportation or consider switching to electric/hybrid vehicles")
            if form_data['air_travel'].lower() in ["frequently", "very frequently"]:
                recommendations.append("✈️ Reduce air travel frequency and consider carbon offset programs")
            if form_data['efficiency'] == "No":
                recommendations.append("💡 Invest in energy-efficient appliances and LED lighting")
            if form_data['heating'].lower() in ["coal", "wood"]:
                recommendations.append("🔥 Consider switching to cleaner heating sources like natural gas or electricity")
            if not form_data['recycling']:
                recommendations.append("♻️ Start recycling paper, plastic, glass, and metal waste")
            if form_data['vehicle_km'] > 1000:
                recommendations.append("🚗 Try to reduce monthly driving distance through trip planning")
            if form_data['tv_hours'] + form_data['internet_hours'] > 8:
                recommendations.append("⚡ Consider reducing screen time to save energy")

            if len(recommendations) < 3:
                recommendations.extend([
                    "🌱 Use renewable energy sources when possible",
                    "🏠 Improve home insulation to reduce energy consumption",
                    "🚲 Walk, bike, or use public transport for short trips",
                    "💧 Fix water leaks and use water-efficient fixtures"
                ])

            st.markdown("### 📋 Personalized Recommendations")
            st.markdown('<ul>', unsafe_allow_html=True)
            for rec in recommendations[:5]:
                st.markdown(f"<li>{rec}</li>", unsafe_allow_html=True)
            st.markdown('</ul>', unsafe_allow_html=True)

            # Pie breakdown
            transport_impact = 0.3 if form_data['transport'].lower() == 'private' else 0.15
            energy_impact = 0.3 if form_data['heating'].lower() in ['coal', 'wood'] else 0.2
            food_impact = 0.3 if form_data['diet'].lower() == 'omnivore' else 0.15
            waste_impact = 0.1 + (form_data['waste_count'] * 0.01)
            consumption_impact = 0.1 + (form_data['clothes'] * 0.01)
            total_impact = transport_impact + energy_impact + food_impact + waste_impact + consumption_impact

            categories = ['Transportation', 'Energy', 'Food', 'Waste', 'Consumption']
            values = [
                footprint * transport_impact / total_impact,
                footprint * energy_impact / total_impact,
                footprint * food_impact / total_impact,
                footprint * waste_impact / total_impact,
                footprint * consumption_impact / total_impact
            ]

            fig = px.pie(values=values, names=categories, title="🔍 Carbon Footprint Breakdown")
            fig.update_layout(title_font_size=18, font_size=12, height=420)
            st.plotly_chart(fig, use_container_width=True)

            # Comparison bar
            comp_df = pd.DataFrame({
                'Category': ['Your Footprint', 'Global Average', 'Target (Paris Agreement)'],
                'CO2 Emissions (tons)': [footprint, 7.0, 2.3]
            })
            fig_bar = px.bar(comp_df, x='Category', y='CO2 Emissions (tons)', title='📊 How You Compare')
            fig_bar.update_layout(title_font_size=18, font_size=12, height=380)
            st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)
            
            # Auto-scroll to results
            st.markdown("""
            <script>
                window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'});
            </script>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Reset button positioned after calculate button
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    if st.button("🔄 Start Over", key="reset_btn", help="Clear all form inputs and start fresh", use_container_width=True):
        reset_form()
        st.rerun()

