import streamlit as st
import requests

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("Customer Churn Dashboard")
st.write("Machine Learning Inference UI")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.header("Demographics")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen?", ["No", "Yes"])
    partner = st.selectbox("Has Partner?", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
    
    st.header("Account Information")
    tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=1)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

with col2:
    st.header("Active Services")
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    
    st.header("Billing Metrics")
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=120.0, value=29.85)

st.markdown("---")
if st.button("Predict Churn Probability", use_container_width=True):
    
    # Preprocess inputs
    senior_int = 1 if senior_citizen == "Yes" else 0
    total_charges = tenure * monthly_charges

    # Construct feature payload
    customer_data = {
        "gender": gender,
        "SeniorCitizen": senior_int,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": "No",
        "Contract": contract,
        "PaperlessBilling": "Yes",
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges)
    }

    try:
        # Execute API call 
        response = requests.post("http://127.0.0.1:8000/predict", json=customer_data)
        
        if response.status_code == 200:
            result = response.json()
            is_churning = result["churn_prediction"]
            probability = result["churn_probability"]

            # Output results
            if is_churning:
                st.error(f"HIGH RISK: Churn probability evaluated at {probability:.1%}")
            else:
                st.success(f"RETENTION LIKELY: Churn probability evaluated at {probability:.1%}")
        else:
            st.error(f"Inference failure: Status code {response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("Connection Error: FastAPI inference backend is unresponsive.")
