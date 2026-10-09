"""
Customer Churn Predictor - Streamlit app
Run locally:  streamlit run app.py
Requires churn_outputs/churn_model.pkl (created by churn_analysis.py)
"""

import os
import pickle

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "churn_outputs", "churn_model.pkl")


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    return saved["model"], saved["threshold"]


def risk_level(prob: float) -> str:
    if prob >= 0.6:
        return "HIGH"
    if prob >= 0.3:
        return "MEDIUM"
    return "LOW"


def recommendations(c: dict, prob: float) -> list:
    tips = []
    if c["Contract"] == "Month-to-month":
        tips.append("Offer a discount to move from a month-to-month plan to a 1- or 2-year contract.")
    if c["tenure"] < 12:
        tips.append("New customer (under 12 months): schedule an onboarding call and a 90-day check-in.")
    if c["PaymentMethod"] == "Electronic check":
        tips.append("Encourage automatic payment (bank transfer or credit card) with a small bill credit.")
    if c["InternetService"] != "No" and c["TechSupport"] == "No":
        tips.append("Bundle free or discounted Tech Support for 3 months.")
    if c["InternetService"] != "No" and c["OnlineSecurity"] == "No":
        tips.append("Offer Online Security as a trial add-on to increase service stickiness.")
    if c["MonthlyCharges"] > 80:
        tips.append("High monthly bill: review the plan and propose a better-value package.")
    if not tips:
        tips.append("No specific red flags. Keep the customer engaged with regular service updates.")
    return tips


st.title("📉 Customer Churn Predictor")
st.caption("Enter a telecom customer's details to estimate the probability that they will leave.")

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Run `python churn_analysis.py` first to create `churn_outputs/churn_model.pkl`.")
    st.stop()

model, threshold = load_model()

# ----------------------------- Inputs ---------------------------------------
st.sidebar.header("Customer details")

st.sidebar.subheader("Profile")
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
senior = st.sidebar.selectbox("Senior citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Has partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Has dependents", ["No", "Yes"])

st.sidebar.subheader("Account")
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 3)
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.sidebar.selectbox("Paperless billing", ["Yes", "No"])
payment = st.sidebar.selectbox(
    "Payment method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
)
monthly = st.sidebar.number_input("Monthly charges ($)", min_value=18.0, max_value=130.0, value=89.5, step=0.5)
total = round(tenure * monthly, 2)
st.sidebar.caption(f"Total charges (estimated): ${total:,.2f}")

st.sidebar.subheader("Services")
phone = st.sidebar.selectbox("Phone service", ["Yes", "No"])
if phone == "Yes":
    multiple = st.sidebar.selectbox("Multiple lines", ["No", "Yes"])
else:
    multiple = "No phone service"

internet = st.sidebar.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
addon_options = ["No", "Yes"] if internet != "No" else ["No internet service"]
online_security = st.sidebar.selectbox("Online security", addon_options)
online_backup = st.sidebar.selectbox("Online backup", addon_options)
device_protection = st.sidebar.selectbox("Device protection", addon_options)
tech_support = st.sidebar.selectbox("Tech support", addon_options)
streaming_tv = st.sidebar.selectbox("Streaming TV", addon_options)
streaming_movies = st.sidebar.selectbox("Streaming movies", addon_options)

customer = {
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple,
    "InternetService": internet,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": float(monthly),
    "TotalCharges": float(total),
}

# ----------------------------- Prediction -----------------------------------
if st.sidebar.button("Predict churn risk", type="primary", use_container_width=True):
    prob = float(model.predict_proba(pd.DataFrame([customer]))[0, 1])
    level = risk_level(prob)
    will_churn = prob >= threshold

    col1, col2, col3 = st.columns(3)
    col1.metric("Churn probability", f"{prob:.1%}")
    col2.metric("Risk level", level)
    col3.metric("Prediction", "Likely to churn" if will_churn else "Likely to stay")

    st.progress(min(max(prob, 0.0), 1.0))

    if level == "HIGH":
        st.error("High risk: contact this customer with a retention offer as soon as possible.")
    elif level == "MEDIUM":
        st.warning("Medium risk: monitor this customer and consider a proactive offer.")
    else:
        st.success("Low risk: this customer is likely to stay.")

    st.subheader("Recommended retention actions")
    for tip in recommendations(customer, prob):
        st.markdown(f"- {tip}")

    with st.expander("Customer details used for this prediction"):
        st.dataframe(pd.DataFrame([customer]).T.rename(columns={0: "Value"}), use_container_width=True)
else:
    st.info("Fill in the customer details in the sidebar, then click **Predict churn risk**.")

st.divider()
st.caption(
    "Model trained on historical telecom customer data (Python, scikit-learn). "
    "Predictions are estimates intended to support, not replace, business judgment."
)
