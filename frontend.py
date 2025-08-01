import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict" 
API_URL_LIVE = "https://insurance-premium-prediction-5udv.onrender.com/predict" 


st.title("Insurance Premium Category Predictor")
st.markdown("Enter your details below:")

# Input fields
age = st.number_input("Age", min_value=1, max_value=119, value=30)
weight = st.number_input("Weight (kg)", min_value=1.0, value=65.0)
height = st.number_input("Height (m)", min_value=0.5, max_value=2.5, value=1.7)
income_lpa = st.number_input("Annual Income (LPA)", min_value=0.1, value=10.0)
smoker = st.selectbox("Are you a smoker?", options=[True, False])
city = st.text_input("City", value="Mumbai")
occupation = st.selectbox(
    "Occupation",
    ['retired', 'freelancer', 'student', 'government_job', 'business_owner', 'unemployed', 'private_job']
)

if st.button("Predict Premium Category"):
    input_data = {
        "age": age,
        "weight": weight,
        "height": height,
        "income_lpa": income_lpa,
        "smoker": smoker,
        "city": city,
        "occupation": occupation
    }

    try:
        response = requests.post(API_URL_LIVE, json=input_data)
        result = response.json()

        if response.status_code == 200 and "response" in result:
            prediction = result["response"]
            category = prediction["predicted_category"]
            if category == 'High':
                st.error(f"Premium Category:  **{category}**")
            elif category == 'Medium':
                st.warning(f"Premium Category:  **{category}**")
            else:
                st.success(f"Premium Category:  **{category}**")
            st.info(f"""🔍 Confidence: **{prediction["confidence"]}**""")
            st.info(
                f"""
                📊 **Class Probabilities:**
                - High: {prediction['class_probabilities']['High']}
                - Medium: {prediction['class_probabilities']['Medium']}
                - Low: {prediction['class_probabilities']['Low']}
                """
            )

        else:
            st.error(f"API Error: {response.status_code}")
            st.write(result)

    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to the FastAPI server. Make sure it's running.")
