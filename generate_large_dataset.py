"""
Generate Large Professional Dataset - 5000 Records
Lung Cancer Risk Prediction System
"""

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of records
n_records = 5000

print("=" * 80)
print("🔄 GENERATING PROFESSIONAL DATASET - 5000 RECORDS")
print("=" * 80)

# Generate realistic data
data = {
    'Age': np.random.randint(18, 85, n_records),
    'Smoking': np.random.randint(1, 9, n_records),
    'Air Pollution': np.random.randint(1, 9, n_records),
    'Fatigue': np.random.randint(1, 10, n_records),
    'Coughing of Blood': np.random.randint(1, 10, n_records)
}

# Create DataFrame
df = pd.DataFrame(data)

# Generate Risk Score based on features with realistic correlations
# Coughing of Blood has highest impact
# Air Pollution and Fatigue have medium impact
# Smoking has moderate impact
# Age has minimal impact

risk_score = (
    df['Age'] * 0.05 +
    df['Smoking'] * 1.2 +
    df['Air Pollution'] * 3.5 +
    df['Fatigue'] * 4.2 +
    df['Coughing of Blood'] * 5.8 +
    np.random.normal(10, 8, n_records)  # Add some randomness
)

# Normalize to 0-100 range
risk_score = np.clip(risk_score, 0, 100)
df['Risk_Score'] = risk_score.astype(int)

# Save dataset
output_path = '01_Dataset/lung_cancer_data.csv'
df.to_csv(output_path, index=False)

print(f"\n✅ Dataset generated successfully!")
print(f"   File: {output_path}")
print(f"   Records: {len(df)}")
print(f"   Features: {len(df.columns) - 1}")
print(f"   Target: Risk_Score")

print("\n📊 Dataset Statistics:")
print("-" * 80)
print(df.describe())

print("\n📈 Risk Score Distribution:")
print("-" * 80)
low_risk = len(df[df['Risk_Score'] <= 40])
medium_risk = len(df[(df['Risk_Score'] > 40) & (df['Risk_Score'] <= 70)])
high_risk = len(df[df['Risk_Score'] > 70])

print(f"   Low Risk (0-40):     {low_risk} ({low_risk/len(df)*100:.1f}%)")
print(f"   Medium Risk (41-70): {medium_risk} ({medium_risk/len(df)*100:.1f}%)")
print(f"   High Risk (71-100):  {high_risk} ({high_risk/len(df)*100:.1f}%)")

print("\n📋 Sample Data (First 10 rows):")
print("-" * 80)
print(df.head(10))

print("\n" + "=" * 80)
print("✅ DATASET GENERATION COMPLETE!")
print("=" * 80)
print("\nNext Steps:")
print("1. Run: python RUN_COMPLETE_PROJECT.py")
print("2. Run: python web_app.py")
print("=" * 80)
