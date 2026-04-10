import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import pickle

# ── 1. Load Data ──────────────────────────────────────────────
df = pd.read_csv('churn.csv')

# ── 2. Clean Data ─────────────────────────────────────────────
# Fix TotalCharges column (has spaces instead of nulls)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Drop customerID (not useful for prediction)
df.drop('customerID', axis=1, inplace=True)

# ── 3. Encode Categorical Columns ─────────────────────────────
le = LabelEncoder()
categorical_cols = df.select_dtypes(include='object').columns.tolist()

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# ── 4. Split Features & Target ────────────────────────────────
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 5. Train XGBoost Model ────────────────────────────────────
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# ── 6. Evaluate ───────────────────────────────────────────────
y_pred = model.predict(X_test)

print("=== Model Performance ===")
print(f"Accuracy: {round(accuracy_score(y_test, y_pred) * 100, 2)}%")
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# ── 7. Save Model & Feature Names ─────────────────────────────
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(X.columns.tolist(), open('features.pkl', 'wb'))

print("\n✅ Model saved as model.pkl")