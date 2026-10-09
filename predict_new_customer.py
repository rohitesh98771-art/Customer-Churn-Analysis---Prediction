"""
Check churn risk for a NEW customer using the model saved by churn_analysis.py

1. Run churn_analysis.py first (it creates churn_outputs/churn_model.pkl)
2. Edit the customer details below
3. Run:  python predict_new_customer.py
"""

import os
import pickle

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "churn_outputs", "churn_model.pkl")

# ---------------------------------------------------------------------------
# EDIT THESE VALUES FOR THE CUSTOMER YOU WANT TO CHECK
# ---------------------------------------------------------------------------
customer = {
    "gender": "Female",                    # Male / Female
    "SeniorCitizen": "No",                 # Yes / No
    "Partner": "No",                       # Yes / No
    "Dependents": "No",                    # Yes / No
    "tenure": 3,                           # months with the company
    "PhoneService": "Yes",                 # Yes / No
    "MultipleLines": "No",                 # Yes / No / No phone service
    "InternetService": "Fiber optic",      # DSL / Fiber optic / No
    "OnlineSecurity": "No",                # Yes / No / No internet service
    "OnlineBackup": "No",                  # Yes / No / No internet service
    "DeviceProtection": "No",              # Yes / No / No internet service
    "TechSupport": "No",                   # Yes / No / No internet service
    "StreamingTV": "Yes",                  # Yes / No / No internet service
    "StreamingMovies": "Yes",              # Yes / No / No internet service
    "Contract": "Month-to-month",          # Month-to-month / One year / Two year
    "PaperlessBilling": "Yes",             # Yes / No
    "PaymentMethod": "Electronic check",   # Electronic check / Mailed check /
                                           # Bank transfer (automatic) / Credit card (automatic)
    "MonthlyCharges": 89.5,                # current monthly bill
    "TotalCharges": 268.5,                 # total billed so far (about tenure x monthly)
}
# ---------------------------------------------------------------------------

if not os.path.exists(MODEL_PATH):
    raise SystemExit("Model not found. Run churn_analysis.py first.")

with open(MODEL_PATH, "rb") as f:
    saved = pickle.load(f)

model, threshold = saved["model"], saved["threshold"]
prob = model.predict_proba(pd.DataFrame([customer]))[0, 1]

if prob >= 0.6:
    risk = "HIGH"
elif prob >= 0.3:
    risk = "MEDIUM"
else:
    risk = "LOW"

print(f"\nChurn probability : {prob:.1%}")
print(f"Risk level        : {risk}")
print(f"Prediction        : {'LIKELY TO CHURN' if prob >= threshold else 'likely to stay'}")
if prob >= threshold:
    print("Suggested action  : offer a discount, a yearly contract or extra support.")
