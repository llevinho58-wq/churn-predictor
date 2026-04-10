import pandas as pd

# Load data
df = pd.read_csv('churn.csv')

# Basic info
print("=== Shape ===")
print(df.shape)

print("\n=== First 5 Rows ===")
print(df.head())

print("\n=== Column Names ===")
print(df.columns.tolist())

print("\n=== Missing Values ===")
print(df.isnull().sum())

print("\n=== Churn Distribution ===")
print(df['Churn'].value_counts())
print(f"\nChurn Rate: {round(df['Churn'].value_counts(normalize=True)['Yes'] * 100, 2)}%")