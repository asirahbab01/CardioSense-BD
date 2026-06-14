import time

import streamlit as st 
import pandas as pd
import numpy as np
import joblib 
import plotly.graph_objects as go 

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction (হৃদরোগ পূর্বাভাস ব্যবস্থা)",
    page_icon="🫀",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 0rem 1rem;}
    .stAlert {padding: 1rem; border-radius: 0.5rem;}
    h1 {color: #1f77b4; padding-bottom: 1rem;}
    </style>
    """, unsafe_allow_html=True)


# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('heart_disease_model.pkl')
        scaler = joblib.load('heart_scaler.pkl')
        # Handle case where model is saved as dictionary
        if isinstance(model, dict):
            model = model.get('model', model)
        return model, scaler
    except FileNotFoundError:
        return None, None


# Header
st.title("🫀 Heart Disease Risk Assessment System for Bangladesh (বাংলাদেশের জন্য হৃদরোগ ঝুঁকি নির্ণয় ব্যবস্থা)")
st.markdown("### Predictive Health Assessment Tool (স্বাস্থ্য ঝুঁকি পূর্বাভাস ব্যবস্থা)")

# Load model
model, scaler = load_model_and_scaler()

if model is None or scaler is None:
    st.error("❌ **Model files not found!**")
    st.info("""
    Please ensure the following model files exist:
    - heart_disease_model.pkl
    - heart_scaler.pkl
    
    Run the Jupyter notebook to train and save the model files.
    """)
    st.stop()

# Function to convert English digits to Bangla digits
def to_bangla_digits(number):
    eng = "0123456789"
    ban = "০১২৩৪৫৬৭৮৯"
    return str(number).translate(str.maketrans(eng, ban))

# Sidebar inputs
st.sidebar.title("⚕️ Patient Information (রোগীর তথ্য)")

st.sidebar.subheader("Demographics (জনসংখ্যাগত তথ্য)")
st.sidebar.markdown("\n")

# Age Category (Strong Predictor #1: Correlation 0.2393)
age_category_options = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older']
age_category = st.sidebar.selectbox('Age Category (বয়সের পরিসীমা)', age_category_options, index=5)
st.sidebar.markdown(
    f' <p style="font-size: 13px;"> নির্ধারিত বয়স: {age_category}</p>', unsafe_allow_html=True
)
st.sidebar.markdown("\n\n")

# General Health (Strong Predictor #2: Correlation -0.2379)
genhealth_options = ['Poor', 'Fair', 'Good', 'Very good', 'Excellent']
genhealth = st.sidebar.selectbox('General Health (সাধারণ স্বাস্থ্য অবস্থা)', genhealth_options, index=2)
st.sidebar.markdown(
    f' <p style="font-size: 13px;"> নির্ধারিত স্বাস্থ্য: {genhealth}</p>', unsafe_allow_html=True
)
st.sidebar.markdown("\n\n")

st.sidebar.subheader("Medical History (চিকিৎসা ইতিহাস)")
st.sidebar.markdown("\n")

# Difficulty Walking (Strong Predictor #3: Correlation 0.1964)
diff_walking = st.sidebar.radio('Difficulty Walking (হাঁটতে অসুবিধা)', ['No', 'Yes'], index=0)
st.sidebar.markdown(
    f' <p style="font-size: 13px;"> নির্ধারিত: {diff_walking}</p>', unsafe_allow_html=True
)
st.sidebar.markdown("\n\n")

# Stroke History (Strong Predictor #4: Correlation 0.1947)
stroke = st.sidebar.radio('Stroke History (স্ট্রোকের ইতিহাস)', ['No', 'Yes'], index=0)
st.sidebar.markdown(
    f' <p style="font-size: 13px;"> নির্ধারিত: {stroke}</p>', unsafe_allow_html=True
)
st.sidebar.markdown("\n\n")

st.sidebar.subheader("Health Measurements (স্বাস্থ্য পরিমাপ)")
st.sidebar.markdown("\n")

# Physical Health (Strong Predictor #5: Correlation 0.1652)
physical_health = st.sidebar.slider('Physical Health Days (শারীরিক স্বাস্থ্য দিন)', 0, 30, 0)
st.sidebar.markdown(
    f' <p style="font-size: 13px;"> নির্ধারিত দিন: {to_bangla_digits(physical_health)}</p>', unsafe_allow_html=True
)
st.sidebar.markdown("\n\n")

st.sidebar.markdown("---\n")

predict_btn = st.sidebar.button("🔮 Predict (পূর্বাভাস দিন)", type="primary", width='stretch')
loader_placeholder = st.empty()

st.sidebar.markdown("\n\n"
    "<div style='color:#27ae60; font-size:12px;'>"
    "📱 Close the sidebar (on mobile) or click outside to view results<br>"
    "(মোবাইলে সাইডবার বন্ধ করুন অথবা সাইডবারের বাইরে ক্লিক করে ফলাফল দেখুন)"
    "</div>",
    unsafe_allow_html=True
)

invalid_input = False


# Main content
if predict_btn and not invalid_input:
    loader_placeholder.markdown(
        """
        <div style="display:flex; justify-content:center; align-items:center; height:38px;">
            <div class="spinner"></div>
            <span style="
                color:#e74c3c;
                font-size:11px;
                white-space:nowrap;
            ">
                Analyzing...বিশ্লেষণ চলছে...
            </span>
        </div>

        <style>
        .spinner {
            width: 14px;
            height: 14px;
            border: 2px solid #fadbd8;
            border-top: 1px solid #e74c3c;
            border-radius: 30%;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    time.sleep(1.5)

    # Create encoding for categorical inputs
    age_category_idx = age_category_options.index(age_category)
    genhealth_idx = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Very good': 3, 'Excellent': 4}[genhealth]
    diff_walking_val = 1 if diff_walking == 'Yes' else 0
    stroke_val = 1 if stroke == 'Yes' else 0
    
    # Prepare input - Using only the strong predictors
    # These should match the order used in model training
    feature_names = ['AgeCategory', 'GenHealth', 'DiffWalking', 'Stroke', 'PhysicalHealth']
    input_data = pd.DataFrame([[
        age_category_idx, genhealth_idx, diff_walking_val, stroke_val, physical_health
        ]], columns=feature_names)
        
    # Standardize
    input_std = scaler.transform(input_data)
        
    # Predict
    prediction = model.predict(input_std)[0]

    # Optional tiny delay
    time.sleep(0.2)
    
    # REMOVE loader after prediction
    loader_placeholder.empty()

    # Get probability if available
    try:
        probability = model.predict_proba(input_std)[0]
        prob_healthy = probability[0] * 100
        prob_disease = probability[1] * 100
    except:
        prob_disease = 100 if prediction == 1 else 0
        prob_healthy = 100 - prob_disease
    
    # Display results
    st.markdown("---")
    st.header("🎯 Prediction Results (পূর্বাভাস ফলাফল)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Prediction box
        if prediction == 0:
            if prob_disease < 33:
                st.success("#### ✅ LOW RISK - Heart Healthy (কম ঝুঁকি-হৃদয় সুস্থ)")
            else:
                st.warning("#### ⚠️ MODERATE RISK - Precaution Advised (মাঝারি ঝুঁকি-সতর্ক থাকুন)")
        else:
            if prob_disease > 65:
                st.error("#### 🔴 HIGH RISK - Heart Disease (উচ্চ ঝুঁকি-হৃদরোগের সম্ভাবনা)")
            else:
                st.warning("#### ⚠️ MODERATE RISK - Medical Consultation Recommended (মাঝারি ঝুঁকি-চিকিৎসকের পরামর্শ নিন)")
        
        # Probabilities
        st.subheader("Probability Breakdown (সম্ভাব্যতা বিশ্লেষণ)")
        pcol1, pcol2 = st.columns(2)
        formatted_disease = f"{prob_disease:.1f}"
        formatted_healthy = f"{prob_healthy:.1f}"
        pcol1.metric("Heart Healthy (সুস্থ)", f"{formatted_healthy}% ({to_bangla_digits(formatted_healthy)}%)")
        pcol2.metric("Heart Disease (হৃদরোগ)", f"{formatted_disease}% ({to_bangla_digits(formatted_disease)}%)")
    
    with col2:
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_disease,
            title={'text': "Heart Disease Risk (ঝুঁকি স্তর)"},
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#4cd137"},
                'steps': [
                    {'range': [0, 33], 'color': "#A3B4AA"},
                    {'range': [33, 65], 'color': "yellow"},
                    {'range': [65, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.6,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=320, margin=dict(l=50, r=60, t=50, b=20))
        st.plotly_chart(fig, width='stretch')
    
    # Risk factors
    st.markdown("---")
    st.subheader("⚠️ Risk Factor Analysis (ঝুঁকি কারণ বিশ্লেষণ)")
    
    risk_factors = []
    positive_factors = []
    
    # Age factor
    age_numeric = int(age_category.split('-')[0])
    if age_numeric >= 65:
        risk_factors.append("🔴 Age 65+: Higher Risk Group (বয়স ৬৫+: উচ্চ ঝুঁকির গ্রুপ)")
    elif age_numeric >= 50:
        risk_factors.append("🟡 Age 50+: Moderate Risk Group (বয়স ৫০+: মাঝারি ঝুঁকির গ্রুপ)")
    else:
        positive_factors.append("🟢 Younger Age Group: Lower Risk (তরুণ বয়স: কম ঝুঁকি)")
    
    # General health
    if genhealth == 'Poor':
        risk_factors.append("🔴 Poor General Health (দুর্বল সাধারণ স্বাস্থ্য)")
    elif genhealth == 'Fair':
        risk_factors.append("🟡 Fair General Health (মোটামুটি সাধারণ স্বাস্থ্য)")
    elif genhealth in ['Very good', 'Excellent']:
        positive_factors.append("🟢 Excellent General Health (চমৎকার সাধারণ স্বাস্থ্য)")
    
    # Difficulty walking
    if diff_walking == 'Yes':
        risk_factors.append("🔴 Difficulty Walking: Mobility Issue (হাঁটতে অসুবিধা: গতিশীলতার সমস্যা)")
    else:
        positive_factors.append("🟢 No Difficulty Walking: Good Mobility (হাঁটতে কোনো অসুবিধা নেই: ভালো গতিশীলতা)")
    
    # Stroke history
    if stroke == 'Yes':
        risk_factors.append("🔴 Stroke History: Major Risk Factor (স্ট্রোকের ইতিহাস: প্রধান ঝুঁকি কারণ)")
    
    # Physical health
    if physical_health > 15:
        risk_factors.append(f"🔴 Poor Physical Health: {physical_health} days (দুর্বল শারীরিক স্বাস্থ্য: {physical_health} দিন)")
    elif physical_health > 0:
        risk_factors.append(f"🟡 Moderate Physical Health Issues: {physical_health} days (মাঝারি স্বাস্থ্য সমস্যা: {physical_health} দিন)")
    else:
        positive_factors.append("🟢 Excellent Physical Health (চমৎকার শারীরিক স্বাস্থ্য)")

    
    if risk_factors:
        st.warning("**Identified Risk Factors (ঝুঁকি কারণ):**")
        for factor in risk_factors:
            st.markdown(f"- {factor}")
    
    if positive_factors:
        st.success("**Positive Health Indicators (ভাল স্বাস্থ্য সূচক):**")
        for factor in positive_factors:
            st.markdown(f"- {factor}")
    
    # Recommendations
    st.markdown("---")
    st.subheader("💡 Recommendations (পরামর্শ)")
    
    if prediction == 1:
        st.error("""
        **Important Actions (গুরুত্বপূর্ণ পদক্ষেপ):**
        - Consult a Cardiologist immediately 
                 (অবিলম্বে একজন কার্ডিওলজিস্ট বা হৃদরোগ বিশেষজ্ঞের পরামর্শ নিন।)
        - Get comprehensive heart screening (ECG, Echocardiogram) 
                 (সম্পূর্ণ হৃদয় পরীক্ষা করান।)
        - Monitor blood pressure regularly 
                 (নিয়মিত রক্তচাপ পরীক্ষা করুন।)
        - Reduce stress and increase physical activity gradually 
                 (চাপ কমান এবং ধীরে ধীরে শারীরিক কার্যকলাপ বাড়ান।)
        - Avoid smoking and excessive salt intake 
                 (ধূমপান এবং অতিরিক্ত লবণ এড়ান।)
        """)
    else:
        st.success("""
        **Maintain Healthy Practices (স্বাস্থ্যকর অভ্যাস বজায় রাখুন):**
        - Regular heart health check-ups (annually) 
                   (নিয়মিত হৃদয় স্বাস্থ্য পরীক্ষা করান)
        - Heart-healthy diet (low sodium, low saturated fat) 
                   (হৃদয়-স্বাস্থ্যকর খাদ্য)
        - Exercise regularly (30+ min daily) 
                   (নিয়মিত ব্যায়াম করুন (প্রতিদিন ৩০ মিনিট বা তার বেশি))
        - Manage stress through meditation or yoga 
                   (মেডিটেশন বা যোগের মাধ্যমে চাপ পরিচালনা করুন)
        - Avoid smoking and limit alcohol intake 
                   (ধূমপান এড়ান এবং অ্যালকোহল সীমিত করুন)
        """)
    
    # Disclaimer
    st.markdown("---")
    st.warning("""
    **⚠️ MEDICAL DISCLAIMER (চিকিৎসা সংক্রান্ত সতর্কতা)**
    
    This prediction is for educational purposes only. It should NOT replace 
    professional medical advice. Always consult qualified healthcare professionals 
    for medical concerns. The developers are not responsible for any health decisions made based on this tool.       
        এই পূর্বাভাস শুধুমাত্র শিক্ষামূলক উদ্দেশ্যে তৈরি করা হয়েছে। এটি পেশাদার চিকিৎসা পরামর্শের বিকল্প নয়। স্বাস্থ্য সংক্রান্ত যেকোনো উদ্বেগের জন্য সর্বদা যোগ্য চিকিৎসা পেশাদারদের সাথে পরামর্শ করুন। এই টুলের উপর ভিত্তি করে নেওয়া কোনো স্বাস্থ্য সিদ্ধান্তের জন্য ডেভেলপাররা দায়ী নয়।
    """)
    
else:
    # Initial page
    st.markdown("---")
    st.info("👈 Enter patient information in the left sidebar and click **Predict** (বাম পাশে থাকা সাইডবারে প্রয়োজনীয় তথ্য দিন এবং **পূর্বাভাস দিন** বাটনে ক্লিক করুন)")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Type (মডেলের ধরন)", "Gaussian Naive Bayes")
    col2.metric("Accuracy (সঠিকতা)", "~83% (প্রায় ৮৩%)")
    col3.metric("Dataset (ডেটাসেট)", "319K samples")