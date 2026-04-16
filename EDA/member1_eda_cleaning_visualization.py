"""
================================================================================
MEMBER 1: EXPLORATORY DATA ANALYSIS (EDA) + DATA CLEANING + VISUALIZATION
================================================================================
TYBCA SEM 6 - Data Analytics using Python (DAP) Project
Subject: 602 - Data Analytics Using Python

Student Name: [Maurya Chandan Shankar]
Roll Number: [3570]

Module Focus:
- Data Understanding
- Exploratory Data Analysis (EDA)
- Data Cleaning
- Data Visualization (Charts, Graphs)
- Finding Patterns and Insights
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("=" * 80)
print("MODULE 1: EXPLORATORY DATA ANALYSIS + DATA CLEANING + VISUALIZATION")
print("=" * 80)
print(f"Member: [Member 1 Name]")
print(f"Focus: EDA, Data Cleaning, Visualization")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD DATASET
# ============================================================================
print("\n📊 STEP 1: LOADING DATASET")
print("-" * 80)

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(BASE_DIR, "Dataset", "lung_cancer_data.csv")

# Load dataset
df = pd.read_csv(dataset_path)

print(f"✅ Dataset loaded successfully!")
print(f"   File: lung_cancer_data.csv")
print(f"   Records: {len(df)}")
print(f"   Columns: {len(df.columns)}")

# ============================================================================
# STEP 2: DATA UNDERSTANDING
# ============================================================================
print("\n📊 STEP 2: DATA UNDERSTANDING")
print("-" * 80)

print("\n[+] First 10 rows of dataset:")
print(df.head(10))

print(f"\n[+] Dataset Shape: {df.shape}")
print(f"    Rows: {df.shape[0]}")
print(f"    Columns: {df.shape[1]}")

print(f"\n[+] Column Names:")
for i, col in enumerate(df.columns, 1):
    print(f"    {i}. {col}")

print("\n[+] Data Types:")
print(df.dtypes)

print("\n[+] Statistical Summary:")
print(df.describe())

# ============================================================================
# STEP 3: DATA QUALITY CHECK
# ============================================================================
print("\n📊 STEP 3: DATA QUALITY CHECK")
print("-" * 80)

# Check missing values
print("\n[+] Missing Values Check:")
missing_values = df.isnull().sum()
print(missing_values)

if missing_values.sum() == 0:
    print("\n✅ No missing values found! Dataset is clean.")
else:
    print(f"\n⚠️  Found {missing_values.sum()} missing values")
    print("   Handling missing values...")
    # Fill numeric columns with mean
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64'] and df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mean(), inplace=True)
    print("✅ Missing values filled with mean")

# Check duplicates
duplicates = df.duplicated().sum()
print(f"\n[+] Duplicate Rows: {duplicates}")
if duplicates > 0:
    df.drop_duplicates(inplace=True)
    print(f"✅ Removed {duplicates} duplicate rows")

# ============================================================================
# STEP 4: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n📊 STEP 4: EXPLORATORY DATA ANALYSIS (EDA)")
print("-" * 80)

# Create output directory
output_dir = os.path.join(BASE_DIR, "05_Final_Output", "Member1_Graphs")
os.makedirs(output_dir, exist_ok=True)

# ─── 4.1: UNIVARIATE ANALYSIS ───
print("\n[+] 4.1: Univariate Analysis - Target Variable Distribution")

plt.figure(figsize=(14, 5))

# Histogram
plt.subplot(1, 3, 1)
plt.hist(df['Risk_Score'], bins=25, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Risk Score', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.title('Distribution of Risk Score\n(Target Variable)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Boxplot
plt.subplot(1, 3, 2)
plt.boxplot(df['Risk_Score'], vert=True, patch_artist=True,
            boxprops=dict(facecolor='lightcoral', alpha=0.7),
            medianprops=dict(color='red', linewidth=2))
plt.ylabel('Risk Score', fontsize=12, fontweight='bold')
plt.title('Boxplot - Risk Score\n(Outlier Detection)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Density Plot
plt.subplot(1, 3, 3)
df['Risk_Score'].plot(kind='density', color='green', linewidth=2)
plt.xlabel('Risk Score', fontsize=12, fontweight='bold')
plt.ylabel('Density', fontsize=12, fontweight='bold')
plt.title('Density Plot - Risk Score\n(Distribution Shape)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '1_Univariate_Analysis.png'), dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: 1_Univariate_Analysis.png")
plt.show()

# ─── 4.2: BIVARIATE ANALYSIS ───
print("\n[+] 4.2: Bivariate Analysis - Features vs Target")

features = ['Age', 'Smoking', 'Air Pollution', 'Fatigue', 'Coughing of Blood']
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, feature in enumerate(features):
    axes[idx].scatter(df[feature], df['Risk_Score'], alpha=0.5, 
                     color=plt.cm.Set2(idx), edgecolors='black', linewidth=0.5)
    axes[idx].set_xlabel(feature, fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('Risk Score', fontsize=11, fontweight='bold')
    axes[idx].set_title(f'{feature} vs Risk Score', fontsize=12, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(df[feature], df['Risk_Score'], 1)
    p = np.poly1d(z)
    axes[idx].plot(df[feature], p(df[feature]), "r--", linewidth=2, alpha=0.8)

# Remove empty subplot
fig.delaxes(axes[5])

plt.suptitle('Bivariate Analysis: Features vs Risk Score', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '2_Bivariate_Analysis.png'), dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: 2_Bivariate_Analysis.png")
plt.show()

# ─── 4.3: CORRELATION HEATMAP ───
print("\n[+] 4.3: Multivariate Analysis - Correlation Heatmap")

plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
            square=True, linewidths=2, cbar_kws={"shrink": 0.8},
            annot_kws={'size': 10, 'weight': 'bold'})
plt.title('Correlation Heatmap - All Features\n(Finding Relationships)', 
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '3_Correlation_Heatmap.png'), dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: 3_Correlation_Heatmap.png")
plt.show()

# ─── 4.4: FEATURE DISTRIBUTIONS ───
print("\n[+] 4.4: Feature Distributions - All Variables")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, col in enumerate(df.columns):
    axes[idx].hist(df[col], bins=20, color=plt.cm.Paired(idx), 
                   edgecolor='black', alpha=0.7)
    axes[idx].set_xlabel(col, fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[idx].set_title(f'Distribution: {col}', fontsize=12, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('Feature Distributions - All Variables', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '4_Feature_Distributions.png'), dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: 4_Feature_Distributions.png")
plt.show()

# ============================================================================
# STEP 5: FINDING PATTERNS AND INSIGHTS
# ============================================================================
print("\n📊 STEP 5: FINDING PATTERNS AND INSIGHTS")
print("-" * 80)

# Calculate correlations with target
correlations = df.corr()['Risk_Score'].sort_values(ascending=False)
print("\n[+] Feature Correlations with Risk Score:")
print(correlations)

print("\n[+] Key Insights:")
print(f"   1. Strongest Predictor: {correlations.index[1]} (correlation: {correlations.iloc[1]:.3f})")
print(f"   2. Second Strongest: {correlations.index[2]} (correlation: {correlations.iloc[2]:.3f})")
print(f"   3. Weakest Predictor: {correlations.index[-1]} (correlation: {correlations.iloc[-1]:.3f})")

# Risk Score Categories
print("\n[+] Risk Score Categories:")
df['Risk_Category'] = pd.cut(df['Risk_Score'], bins=[0, 40, 70, 100], 
                              labels=['Low Risk', 'Medium Risk', 'High Risk'])
print(df['Risk_Category'].value_counts())

# Visualization of Risk Categories
plt.figure(figsize=(10, 6))
risk_counts = df['Risk_Category'].value_counts()
colors = ['#2ecc71', '#f39c12', '#e74c3c']
plt.bar(risk_counts.index, risk_counts.values, color=colors, edgecolor='black', linewidth=2)
plt.xlabel('Risk Category', fontsize=12, fontweight='bold')
plt.ylabel('Number of Patients', fontsize=12, fontweight='bold')
plt.title('Patient Distribution by Risk Category\n(Pattern Analysis)', 
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, v in enumerate(risk_counts.values):
    plt.text(i, v + 5, str(v), ha='center', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '5_Risk_Categories.png'), dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: 5_Risk_Categories.png")
plt.show()

# ============================================================================
# STEP 6: DATA CLEANING - OUTLIER HANDLING
# ============================================================================
print("\n📊 STEP 6: DATA CLEANING - OUTLIER HANDLING")
print("-" * 80)

print(f"\n[+] Original Dataset Shape: {df.shape}")

# Detect outliers using IQR method
Q1 = df[features].quantile(0.25)
Q3 = df[features].quantile(0.75)
IQR = Q3 - Q1

# Remove outliers
outlier_mask = ~((df[features] < (Q1 - 1.5 * IQR)) | (df[features] > (Q3 + 1.5 * IQR))).any(axis=1)
df_cleaned = df[outlier_mask].copy()

print(f"[+] Cleaned Dataset Shape: {df_cleaned.shape}")
print(f"[+] Outliers Removed: {df.shape[0] - df_cleaned.shape[0]} rows")

# Save cleaned dataset
cleaned_path = os.path.join(BASE_DIR, "Dataset", "cleaned_data.csv")
df_cleaned.drop('Risk_Category', axis=1, inplace=True)  # Remove temporary column
df_cleaned.to_csv(cleaned_path, index=False)
print(f"\n✅ Cleaned dataset saved: cleaned_data.csv")

# ============================================================================
# STEP 7: SUMMARY REPORT
# ============================================================================
print("\n📊 STEP 7: GENERATING SUMMARY REPORT")
print("-" * 80)

report = f"""
================================================================================
MEMBER 1: EDA + DATA CLEANING + VISUALIZATION - SUMMARY REPORT
================================================================================

Student: [Member 1 Name]
Module: Exploratory Data Analysis, Data Cleaning, Visualization

DATASET INFORMATION:
--------------------
Original Records: {df.shape[0]}
Cleaned Records: {df_cleaned.shape[0]}
Features: {len(df.columns) - 1}
Target Variable: Risk_Score

DATA QUALITY:
-------------
Missing Values: {missing_values.sum()} (Handled ✓)
Duplicate Rows: {duplicates} (Removed ✓)
Outliers Removed: {df.shape[0] - df_cleaned.shape[0]} rows

FEATURE CORRELATIONS WITH RISK SCORE:
--------------------------------------
{correlations.to_string()}

KEY INSIGHTS:
-------------
1. Strongest Predictor: {correlations.index[1]} (r = {correlations.iloc[1]:.3f})
2. Second Strongest: {correlations.index[2]} (r = {correlations.iloc[2]:.3f})
3. Third Strongest: {correlations.index[3]} (r = {correlations.iloc[3]:.3f})

RISK DISTRIBUTION:
------------------
{df['Risk_Category'].value_counts().to_string()}

VISUALIZATIONS GENERATED:
-------------------------
1. Univariate Analysis (Histogram, Boxplot, Density)
2. Bivariate Analysis (Scatter plots with trend lines)
3. Correlation Heatmap (Feature relationships)
4. Feature Distributions (All variables)
5. Risk Categories (Pattern analysis)

CONCLUSION:
-----------
✅ Data successfully explored and analyzed
✅ Patterns and insights identified
✅ Data cleaned and ready for modeling
✅ 5 comprehensive visualizations generated
✅ Cleaned dataset saved for Member 2 & 3

NEXT STEPS:
-----------
→ Member 2: Use cleaned_data.csv for Linear Regression
→ Member 3: Use predictions for model evaluation

================================================================================
Generated by: Member 1 - EDA Module
Date: April 2026
================================================================================
"""

# Save report
report_path = os.path.join(BASE_DIR, "05_Final_Output", "Member1_Report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(report)
print(f"\n✅ Summary report saved: Member1_Report.txt")

# ============================================================================
# MODULE 1 COMPLETE
# ============================================================================
print("\n" + "=" * 80)
print("✅ MODULE 1 COMPLETED SUCCESSFULLY!")
print("=" * 80)
print("\nGenerated Files:")
print("  📊 Graphs (5):")
print("     1. 1_Univariate_Analysis.png")
print("     2. 2_Bivariate_Analysis.png")
print("     3. 3_Correlation_Heatmap.png")
print("     4. 4_Feature_Distributions.png")
print("     5. 5_Risk_Categories.png")
print("\n  📄 Data:")
print("     6. cleaned_data.csv (for Member 2 & 3)")
print("\n  📝 Report:")
print("     7. Member1_Report.txt")
print("\n" + "=" * 80)
print("Next: Member 2 will use cleaned_data.csv for Linear Regression")
print("=" * 80)
