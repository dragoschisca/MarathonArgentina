import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('loan_data.csv')
print(f"Original dataset shape: {df.shape}")

# Remove duplicate records
# ===========================
df = df.drop_duplicates()
print(f"Shape after removing duplicates: {df.shape}")

# Remove features causing data leakage
# ===========================
leak_cols = ['last_audit_team_id']
df = df.drop(columns=leak_cols)
print(f"Shape after removing leakage columns: {df.shape}")

# Feature engineering (BEFORE scaling and split)
# ===========================
X = df.drop(columns=['loan_defaulted']).copy()
y = df['loan_defaulted'].copy()

# Safe engineered features
X['debt_to_income_ratio'] = (X['loan_amount'] / X['annual_income']).round(4)
X['loan_term_risk'] = (X['loan_term_months'] / 12 * X['interest_rate']).round(2)

print(f"Feature matrix shape: {X.shape}")

# Proper train/test split BEFORE scaling
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Scale features correctly (fit on train only)
# ===========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame
X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

print(f"StandardScaler applied correctly (fit on train only)")

# Save preprocessed dataset
# ===========================
train_df_preprocessed = X_train.copy()
test_df_preprocessed = X_test.copy()

train_df_preprocessed['loan_defaulted'] = y_train.values
test_df_preprocessed['loan_defaulted'] = y_test.values

preprocessed_df = pd.concat([train_df_preprocessed, test_df_preprocessed], ignore_index=True)
preprocessed_df.to_csv('loan_data_preprocessed.csv', index=False)

print("\n" + "="*60)
print("✅ ALL THREE DATA LEAKAGE ISSUES FIXED")
print("="*60)
print("Fix 1: Removed duplicate records")
print("Fix 2: Removed 'last_audit_team_id' (temporal leakage)")
print("Fix 3: Applied StandardScaler AFTER split (fit on train only)")
print("="*60)
print(f"\nPreprocessed data saved successfully.")
print(f"Shape of the saved data: {preprocessed_df.shape}")