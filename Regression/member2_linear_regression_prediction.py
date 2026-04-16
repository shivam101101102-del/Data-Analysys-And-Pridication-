"""
================================================================================
MEMBER 2: LINEAR REGRESSION + MODEL TRAINING + PREDICTION
================================================================================
TYBCA SEM 6 - Data Analytics using Python (DAP) Project
Subject: 602 - Data Analytics Using Python

Student Name: [Maurya Shivam Hanumanprasad]
Roll Number: [3567]

Module Focus:
- Feature Selection
- Train-Test Split
- Linear Regression Algorithm
- Model Training
- Prediction Generation
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

print("=" * 80)
print("MODULE 2: LINEAR REGRESSION + MODEL TRAINING + PREDICTION")
print("=" * 80)
print(f"Member: [Member 2 Name]")
print(f"Focus: Linear Regression, Training, Prediction")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD CLEANED DATASET (FROM MEMBER 1)
# ============================================================================
print("\n📊 STEP 1: LOADING CLEANED DATASET")
print("-" * 80)

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(BASE_DIR, "Dataset", "cleaned_data.csv")

# Load cleaned dataset from Member 1
df = pd.read_csv(dataset_path)

print(f"✅ Cleaned dataset loaded successfully!")
print(f"   File: cleaned_data.csv (from Member 1)")
print(f"   Records: {len(df)}")
print(f"   Features: {len(df.columns) - 1}")

print("\n[+] Dataset Preview:")
print(df.head())

# ============================================================================
# STEP 2: FEATURE SELECTION (X) AND TARGET (y)
# ============================================================================
print("\n📊 STEP 2: FEATURE SELECTION")
print("-" * 80)

# Separate features (X) and target (y)
X = df.drop('Risk_Score', axis=1)  # Independent variables
y = df['Risk_Score']                # Dependent variable

print(f"✅ Features (X) - Independent Variables:")
for i, col in enumerate(X.columns, 1):
    print(f"   {i}. {col}")

print(f"\n✅ Target (y) - Dependent Variable:")
print(f"   Risk_Score (Numeric: 0-100)")

print(f"\n[+] Data Shapes:")
print(f"   X shape: {X.shape} (rows, features)")
print(f"   y shape: {y.shape} (rows,)")

# ============================================================================
# STEP 3: TRAIN-TEST SPLIT (80% - 20%)
# ============================================================================
print("\n📊 STEP 3: TRAIN-TEST SPLIT")
print("-" * 80)

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✅ Data split completed!")
print(f"\n[+] Training Set:")
print(f"   X_train: {X_train.shape}")
print(f"   y_train: {y_train.shape}")
print(f"   Total: {len(X_train)} samples (80%)")

print(f"\n[+] Testing Set:")
print(f"   X_test: {X_test.shape}")
print(f"   y_test: {y_test.shape}")
print(f"   Total: {len(X_test)} samples (20%)")

# ============================================================================
# STEP 4: LINEAR REGRESSION MODEL TRAINING
# ============================================================================
print("\n📊 STEP 4: LINEAR REGRESSION MODEL TRAINING")
print("-" * 80)

# Create Linear Regression model
model = LinearRegression()

# Train the model
print("[+] Training Linear Regression model...")
model.fit(X_train, y_train)
print("✅ Model trained successfully!")

# Display model parameters
print(f"\n[+] Model Parameters:")
print(f"   Intercept (β₀): {model.intercept_:.4f}")
print(f"\n   Coefficients (β₁ to β₅):")
for feature, coef in zip(X.columns, model.coef_):
    print(f"   {feature:25s}: {coef:8.4f}")

# Mathematical equation
print(f"\n[+] Linear Regression Equation:")
print(f"   Risk_Score = {model.intercept_:.2f}", end="")
for feature, coef in zip(X.columns, model.coef_):
    sign = "+" if coef >= 0 else ""
    print(f" {sign} {coef:.2f}×{feature}", end="")
print()

# ============================================================================
# STEP 5: PREDICTION ON TEST DATA
# ============================================================================
print("\n📊 STEP 5: PREDICTION ON TEST DATA")
print("-" * 80)

# Make predictions
y_pred = model.predict(X_test)

print(f"✅ Predictions generated!")
print(f"   Total predictions: {len(y_pred)}")

# Display sample predictions
print(f"\n[+] Sample Predictions (First 10):")
comparison_df = pd.DataFrame({
    'Actual': y_test.values[:10],
    'Predicted': y_pred[:10],
    'Difference': np.abs(y_test.values[:10] - y_pred[:10])
})
comparison_df.index = range(1, 11)
print(comparison_df.to_string())

# ============================================================================
# STEP 6: SAVE PREDICTIONS FOR MEMBER 3
# ============================================================================
print("\n📊 STEP 6: SAVING PREDICTIONS FOR MEMBER 3")
print("-" * 80)

# Create predictions dataframe
predictions_df = pd.DataFrame({
    'Actual_Risk_Score': y_test.values,
    'Predicted_Risk_Score': y_pred,
    'Absolute_Error': np.abs(y_test.values - y_pred)
})

# Save predictions
predictions_path = os.path.join(BASE_DIR, "Dataset", "predictions.csv")
predictions_df.to_csv(predictions_path, index=False)
print(f"✅ Predictions saved: predictions.csv")

# Save model
model_path = os.path.join(BASE_DIR, "Dataset", "trained_model.pkl")
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"✅ Trained model saved: trained_model.pkl")

# Save train-test data for Member 3
train_test_data = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test,
    'y_pred': y_pred
}
data_path = os.path.join(BASE_DIR, "Dataset", "train_test_data.pkl")
with open(data_path, 'wb') as f:
    pickle.dump(train_test_data, f)
print(f"✅ Train-test data saved: train_test_data.pkl")

# ============================================================================
# STEP 7: VISUALIZATION - REGRESSION ANALYSIS
# ============================================================================
print("\n📊 STEP 7: GENERATING VISUALIZATIONS")
print("-" * 80)

# Create output directory
output_dir = os.path.join(BASE_DIR, "05_Final_Output", "Member2_Graphs")
os.makedirs(output_dir, exist_ok=True)

# ─── 7.1: ACTUAL VS PREDICTED (SCATTER PLOT) ───
print("\n[+] 7.1: Actual vs Predicted Scatter Plot")

plt.figure(figsize=(10, 7))
plt.scatter(y_test, y_pred, alpha=0.6, color='purple', edgecolors='black', 
            linewidth=0.5, s=80)

# Perfect prediction line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=3, 
         label='Perfect Prediction Line')

plt.xlabel('Actual Risk Score', fontsize=13, fontweight='bold')
plt.ylabel('Predicted Risk Score', fontsize=13, fontweight='bold')
plt.title('Linear Regression: Actual vs Predicted Risk Scores\n(Model Performance)', 
          fontsize=15, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '1_Actual_vs_Predicted_Scatter.png'), 
            dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: 1_Actual_vs_Predicted_Scatter.png")
plt.show()

# ─── 7.2: ACTUAL VS PREDICTED (BAR CHART) ───
print("\n[+] 7.2: Actual vs Predicted Bar Chart")

plt.figure(figsize=(14, 6))
indices = range(min(20, len(y_test)))
x_pos = np.arange(len(indices))

plt.bar(x_pos - 0.2, y_test.values[indices], width=0.4, label='Actual', 
        color='steelblue', alpha=0.8, edgecolor='black')
plt.bar(x_pos + 0.2, y_pred[indices], width=0.4, label='Predicted', 
        color='coral', alpha=0.8, edgecolor='black')

plt.xlabel('Sample Index', fontsize=12, fontweight='bold')
plt.ylabel('Risk Score', fontsize=12, fontweight='bold')
plt.title('Actual vs Predicted Risk Scores (First 20 Samples)\n(Comparison)', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '2_Actual_vs_Predicted_Bar.png'), 
            dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: 2_Actual_vs_Predicted_Bar.png")
plt.show()

# ─── 7.3: RESIDUAL PLOT ───
print("\n[+] 7.3: Residual Plot (Error Analysis)")

residuals = y_test.values - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.6, color='green', edgecolors='black', 
            linewidth=0.5, s=60)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Error Line')
plt.xlabel('Predicted Risk Score', fontsize=12, fontweight='bold')
plt.ylabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')
plt.title('Residual Plot - Error Distribution\n(Model Bias Check)', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '3_Residual_Plot.png'), 
            dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: 3_Residual_Plot.png")
plt.show()

# ─── 7.4: FEATURE IMPORTANCE (COEFFICIENTS) ───
print("\n[+] 7.4: Feature Importance (Model Coefficients)")

plt.figure(figsize=(10, 6))
features = X.columns
coefficients = model.coef_
colors = ['#e74c3c' if c < 0 else '#2ecc71' for c in coefficients]

plt.barh(features, coefficients, color=colors, edgecolor='black', linewidth=1.5)
plt.xlabel('Coefficient Value', fontsize=12, fontweight='bold')
plt.ylabel('Features', fontsize=12, fontweight='bold')
plt.title('Feature Importance - Linear Regression Coefficients\n(Impact on Risk Score)', 
          fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
plt.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (feature, coef) in enumerate(zip(features, coefficients)):
    plt.text(coef, i, f'  {coef:.3f}', va='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '4_Feature_Importance.png'), 
            dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: 4_Feature_Importance.png")
plt.show()

# ============================================================================
# STEP 8: SUMMARY REPORT
# ============================================================================
print("\n📊 STEP 8: GENERATING SUMMARY REPORT")
print("-" * 80)

report = f"""
================================================================================
MEMBER 2: LINEAR REGRESSION + TRAINING + PREDICTION - SUMMARY REPORT
================================================================================

Student: [Member 2 Name]
Module: Linear Regression, Model Training, Prediction

DATASET INFORMATION:
--------------------
Total Records: {len(df)}
Training Samples: {len(X_train)} (80%)
Testing Samples: {len(X_test)} (20%)
Features: {len(X.columns)}

FEATURES (INDEPENDENT VARIABLES):
----------------------------------
{chr(10).join([f"{i}. {col}" for i, col in enumerate(X.columns, 1)])}

TARGET VARIABLE:
----------------
Risk_Score (Numeric: 0-100)

LINEAR REGRESSION MODEL:
------------------------
Algorithm: Linear Regression (Supervised Learning)
Library: Scikit-learn (sklearn.linear_model.LinearRegression)

MODEL PARAMETERS:
-----------------
Intercept (β₀): {model.intercept_:.4f}

Coefficients:
{chr(10).join([f"  {feature:25s}: {coef:8.4f}" for feature, coef in zip(X.columns, model.coef_)])}

MATHEMATICAL EQUATION:
----------------------
Risk_Score = {model.intercept_:.2f}{''.join([f" {'+' if coef >= 0 else ''} {coef:.2f}×{feature}" for feature, coef in zip(X.columns, model.coef_)])}

INTERPRETATION:
---------------
• Highest Impact: {X.columns[np.argmax(np.abs(model.coef_))]} (|coef| = {np.max(np.abs(model.coef_)):.4f})
• Lowest Impact: {X.columns[np.argmin(np.abs(model.coef_))]} (|coef| = {np.min(np.abs(model.coef_)):.4f})

PREDICTIONS GENERATED:
----------------------
Total Predictions: {len(y_pred)}
Sample Predictions (First 5):
{comparison_df.head().to_string()}

VISUALIZATIONS GENERATED:
-------------------------
1. Actual vs Predicted Scatter Plot (Model fit)
2. Actual vs Predicted Bar Chart (Comparison)
3. Residual Plot (Error distribution)
4. Feature Importance (Coefficient values)

FILES SAVED FOR MEMBER 3:
--------------------------
1. predictions.csv - All predictions with actual values
2. trained_model.pkl - Trained Linear Regression model
3. train_test_data.pkl - Train-test split data

CONCLUSION:
-----------
✅ Linear Regression model trained successfully
✅ Predictions generated on test data
✅ Model parameters calculated and interpreted
✅ 4 comprehensive visualizations created
✅ All data saved for Member 3 evaluation

NEXT STEPS:
-----------
→ Member 3: Use predictions.csv for model evaluation
→ Member 3: Calculate MSE, MAE, R² Score
→ Member 3: Generate evaluation visualizations

================================================================================
Generated by: Member 2 - Linear Regression Module
Date: April 2026
================================================================================
"""

# Save report
report_path = os.path.join(BASE_DIR, "05_Final_Output", "Member2_Report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(report)
print(f"\n✅ Summary report saved: Member2_Report.txt")

# ============================================================================
# MODULE 2 COMPLETE
# ============================================================================
print("\n" + "=" * 80)
print("✅ MODULE 2 COMPLETED SUCCESSFULLY!")
print("=" * 80)
print("\nGenerated Files:")
print("  📊 Graphs (4):")
print("     1. 1_Actual_vs_Predicted_Scatter.png")
print("     2. 2_Actual_vs_Predicted_Bar.png")
print("     3. 3_Residual_Plot.png")
print("     4. 4_Feature_Importance.png")
print("\n  📄 Data:")
print("     5. predictions.csv (for Member 3)")
print("     6. trained_model.pkl (for Member 3)")
print("     7. train_test_data.pkl (for Member 3)")
print("\n  📝 Report:")
print("     8. Member2_Report.txt")
print("\n" + "=" * 80)
print("Next: Member 3 will use predictions.csv for Model Evaluation")
print("=" * 80)
