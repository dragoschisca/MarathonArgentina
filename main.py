# loan_data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('loan_data.csv')
print(f"Original dataset shape: {df.shape}")

# -------------------------
# FIX 1: Remove duplicate records
df = df.drop_duplicates()
print(f"Shape after removing duplicates: {df.shape}")

# -------------------------
# FIX 2: Remove features causing data leakage
# 'last_audit_team_id' gives away target -> remove it
# Optionally, if monthly_payment_capacity is derived from other features + target info, consider dropping it
leak_cols = ['last_audit_team_id']  # you can add 'monthly_payment_capacity' if necessary
df = df.drop(columns=leak_cols)
print(f"Shape after removing leakage columns: {df.shape}")

# -------------------------
# FIX 3: Feature engineering without target leakage
X = df.drop(columns=['loan_defaulted']).copy()
y = df['loan_defaulted'].copy()

# Safe engineered features
X['debt_to_income_ratio'] = (X['loan_amount'] / X['annual_income']).round(4)
X['loan_term_risk'] = (X['loan_term_months'] / 12 * X['interest_rate']).round(2)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# Save preprocessed dataset
train_df_preprocessed = X_train.copy()
test_df_preprocessed = X_test.copy()

train_df_preprocessed['loan_defaulted'] = y_train.values
test_df_preprocessed['loan_defaulted'] = y_test.values

preprocessed_df = pd.concat([train_df_preprocessed, test_df_preprocessed], ignore_index=True)
preprocessed_df.to_csv('loan_data_preprocessed.csv', index=False)

print("Preprocessed data saved successfully.")
print(f"Shape of the saved data: {preprocessed_df.shape}")
