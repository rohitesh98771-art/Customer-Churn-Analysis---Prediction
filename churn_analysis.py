"""
Customer Churn Analysis & Prediction
------------------------------------
Run in VS Code:  python churn_analysis.py

Dataset: Telco Customer Churn (Kaggle, IBM sample data)
  File name expected: WA_Fn-UseC_-Telco-Customer-Churn.csv  (place next to this script)
  If the file is missing, a synthetic dataset with the same columns is generated
  so the script still runs end to end.

Install requirements:
  pip install pandas numpy matplotlib seaborn scikit-learn
"""

import os
import pickle
import warnings
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# Set to True to open plot windows; False saves PNGs only (works on any machine)
SHOW_PLOTS = False
if not SHOW_PLOTS:
    matplotlib.use("Agg")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # works no matter where VS Code runs from
DATA_FILE = os.path.join(BASE_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "churn_outputs")
RANDOM_STATE = 42
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_or_show(name: str) -> None:
    path = os.path.join(OUTPUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    print(f"  saved -> {path}")


# ----------------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------------
def make_synthetic_data(n: int = 5000) -> pd.DataFrame:
    """Synthetic data shaped like the Telco churn dataset (fallback only)."""
    rng = np.random.default_rng(RANDOM_STATE)
    tenure = rng.integers(0, 73, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    monthly = np.round(
        20 + (internet == "Fiber optic") * 45 + (internet == "DSL") * 25 + rng.normal(0, 8, n), 2
    ).clip(18, 120)
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n
    )
    yes_no = lambda p=0.5: rng.choice(["Yes", "No"], n, p=[p, 1 - p])

    logit = (
        -0.6
        - 0.04 * tenure
        + 1.1 * (contract == "Month-to-month")
        - 0.9 * (contract == "Two year")
        + 0.6 * (internet == "Fiber optic")
        + 0.5 * (payment == "Electronic check")
        + 0.012 * (monthly - 65)
        + rng.normal(0, 0.6, n)
    )
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))

    df = pd.DataFrame(
        {
            "customerID": [f"C{i:05d}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
            "Partner": yes_no(0.48),
            "Dependents": yes_no(0.3),
            "tenure": tenure,
            "PhoneService": yes_no(0.9),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
            "InternetService": internet,
            "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
            "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n),
            "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n),
            "TechSupport": rng.choice(["Yes", "No", "No internet service"], n),
            "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n),
            "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n),
            "Contract": contract,
            "PaperlessBilling": yes_no(0.59),
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "Churn": np.where(churn, "Yes", "No"),
        }
    )
    df["TotalCharges"] = (df["tenure"] * df["MonthlyCharges"]).round(2).astype(str)
    df.loc[df["tenure"] == 0, "TotalCharges"] = " "  # mimic the real dataset's blanks
    return df


if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {DATA_FILE}")
else:
    print(f"'{DATA_FILE}' not found -> using SYNTHETIC data (download the real file from Kaggle for real results).")
    df = make_synthetic_data()

print(f"Shape: {df.shape}")
print(df.head())

# ----------------------------------------------------------------------------
# 2. CLEAN DATA
# ----------------------------------------------------------------------------
print("\n=== DATA CLEANING ===")
df = df.drop(columns=["customerID"])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("Missing values after conversion:\n", df.isna().sum()[df.isna().sum() > 0])
df["TotalCharges"] = df["TotalCharges"].fillna(0)  # new customers (tenure 0) have no charges yet
df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
print(f"Duplicates removed: {df.duplicated().sum()}")
df = df.drop_duplicates()

# ----------------------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS
# ----------------------------------------------------------------------------
print("\n=== EXPLORATORY ANALYSIS ===")
churn_rate = df["Churn"].mean()
print(f"Overall churn rate: {churn_rate:.2%}")

# 3a. Churn distribution
plt.figure(figsize=(5, 4))
ax = sns.countplot(x="Churn", data=df, palette=["#4C9F70", "#D1495B"])
ax.set_xticklabels(["Stayed", "Churned"])
ax.set_title("Churn Distribution")
save_or_show("01_churn_distribution.png")

# 3b. Churn rate by key categorical features
cat_cols = df.select_dtypes(include="object").columns.tolist()
focus = [c for c in ["Contract", "InternetService", "PaymentMethod", "TechSupport", "SeniorCitizen", "PaperlessBilling"] if c in cat_cols]
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, col in zip(axes.ravel(), focus):
    rate = df.groupby(col)["Churn"].mean().sort_values(ascending=False) * 100
    sns.barplot(x=rate.values, y=rate.index, ax=ax, palette="rocket")
    ax.set_title(f"Churn % by {col}")
    ax.set_xlabel("Churn rate (%)")
    ax.set_ylabel("")
save_or_show("02_churn_by_category.png")

# 3c. Numeric feature distributions by churn
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, col in zip(axes, num_cols):
    sns.kdeplot(data=df, x=col, hue="Churn", fill=True, common_norm=False, ax=ax, palette=["#4C9F70", "#D1495B"])
    ax.set_title(f"{col} by Churn")
save_or_show("03_numeric_distributions.png")

# 3d. Correlation heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(df[num_cols + ["Churn"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation (numeric features)")
save_or_show("04_correlation.png")

# 3e. Tenure bucket analysis
df["TenureGroup"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"])
tenure_rate = df.groupby("TenureGroup", observed=True)["Churn"].mean() * 100
print("\nChurn % by tenure group (months):\n", tenure_rate.round(1))
df = df.drop(columns=["TenureGroup"])

# ----------------------------------------------------------------------------
# 4. PREPROCESSING + MODEL TRAINING
# ----------------------------------------------------------------------------
print("\n=== MODEL TRAINING ===")
X = df.drop(columns=["Churn"])
y = df["Churn"]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include="object").columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=10, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results, fitted = [], {}

for name, clf in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", clf)])
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc").mean()
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    results.append(
        {
            "Model": name,
            "CV ROC-AUC": cv_auc,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "Test ROC-AUC": roc_auc_score(y_test, proba),
        }
    )
    fitted[name] = pipe

results_df = pd.DataFrame(results).set_index("Model").round(3)
print(results_df)
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"))

best_name = results_df["Test ROC-AUC"].idxmax()
best_model = fitted[best_name]
print(f"\nBest model (by test ROC-AUC): {best_name}")

# ----------------------------------------------------------------------------
# 5. EVALUATE BEST MODEL (with a tuned decision threshold)
# ----------------------------------------------------------------------------
# The default 0.5 cut-off often misses many churners. Pick the threshold that
# maximises F1 using cross-validated predictions on the TRAINING data only
# (so the test set stays untouched), then apply it to the test set.
cv_proba = cross_val_predict(best_model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
prec, rec, thr = precision_recall_curve(y_train, cv_proba)
f1_scores = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
THRESHOLD = float(thr[np.argmax(f1_scores)])
print(f"\nTuned decision threshold: {THRESHOLD:.2f} (default would be 0.50)")

test_proba = best_model.predict_proba(X_test)[:, 1]
y_pred = (test_proba >= THRESHOLD).astype(int)
print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=["Stayed", "Churned"]))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Stayed", "Churned"], cmap="Blues", ax=axes[0])
axes[0].set_title(f"Confusion Matrix - {best_name}")
for name, pipe in fitted.items():
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=axes[1])
axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
axes[1].set_title("ROC Curves")
save_or_show("05_model_evaluation.png")

# ----------------------------------------------------------------------------
# 6. FEATURE IMPORTANCE (what drives churn?)
# ----------------------------------------------------------------------------
feature_names = best_model.named_steps["prep"].get_feature_names_out()
final_estimator = best_model.named_steps["model"]

if hasattr(final_estimator, "feature_importances_"):
    importance = final_estimator.feature_importances_
elif hasattr(final_estimator, "coef_"):
    importance = np.abs(final_estimator.coef_[0])
else:
    importance = None

if importance is not None:
    imp = (
        pd.Series(importance, index=feature_names)
        .sort_values(ascending=False)
        .head(15)
        .rename(lambda s: s.replace("num__", "").replace("cat__", ""))
    )
    plt.figure(figsize=(8, 6))
    sns.barplot(x=imp.values, y=imp.index, palette="viridis")
    plt.title(f"Top 15 Churn Drivers - {best_name}")
    plt.xlabel("Importance")
    save_or_show("06_feature_importance.png")
    print("\nTop 10 churn drivers:\n", imp.head(10).round(4))

# ----------------------------------------------------------------------------
# 7. SCORE CUSTOMERS BY CHURN RISK
# ----------------------------------------------------------------------------
scored = X_test.copy()
scored["ChurnProbability"] = test_proba
scored["ActualChurn"] = y_test.values
scored["RiskSegment"] = pd.cut(
    scored["ChurnProbability"], bins=[-0.01, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"]
)
scored.sort_values("ChurnProbability", ascending=False).to_csv(
    os.path.join(OUTPUT_DIR, "customer_risk_scores.csv"), index=False
)
print("\nRisk segment counts:\n", scored["RiskSegment"].value_counts())
print("Top 5 highest-risk customers:")
print(scored.sort_values("ChurnProbability", ascending=False).head()[["tenure", "Contract", "MonthlyCharges", "ChurnProbability"]])

# ----------------------------------------------------------------------------
# 8. SAVE MODEL + PREDICT FOR A NEW CUSTOMER
# ----------------------------------------------------------------------------
model_path = os.path.join(OUTPUT_DIR, "churn_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump({"model": best_model, "threshold": THRESHOLD}, f)
print(f"\nModel saved -> {model_path} ({os.path.getsize(model_path) / 1024:.0f} KB)")

new_customer = pd.DataFrame(
    [
        {
            "gender": "Female",
            "SeniorCitizen": "No",
            "Partner": "No",
            "Dependents": "No",
            "tenure": 3,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 89.5,
            "TotalCharges": 268.5,
        }
    ]
)
with open(model_path, "rb") as f:
    saved = pickle.load(f)
loaded_model, loaded_threshold = saved["model"], saved["threshold"]
prob = loaded_model.predict_proba(new_customer)[0, 1]
print(f"\nNew customer churn probability: {prob:.1%} -> "
      f"{'LIKELY TO CHURN' if prob >= loaded_threshold else 'likely to stay'}")
