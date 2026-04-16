"""
Quick Test Script - Verify Project Setup
"""
import os
import sys

print("=" * 80)
print("🔍 TESTING PROJECT SETUP")
print("=" * 80)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Check folders
folders = [
    "01_Dataset",
    "02_Member1_EDA",
    "03_Member2_Regression",
    "04_Member3_Evaluation",
    "05_Final_Output",
    "06_Documentation"
]

print("\n✅ Checking Folder Structure:")
for folder in folders:
    path = os.path.join(BASE_DIR, folder)
    status = "✓" if os.path.exists(path) else "✗"
    print(f"   {status} {folder}/")

# Check dataset
print("\n✅ Checking Dataset:")
dataset_path = os.path.join(BASE_DIR, "01_Dataset", "lung_cancer_data.csv")
if os.path.exists(dataset_path):
    import pandas as pd
    df = pd.read_csv(dataset_path)
    print(f"   ✓ lung_cancer_data.csv")
    print(f"     Records: {len(df)}")
    print(f"     Columns: {len(df.columns)}")
else:
    print(f"   ✗ lung_cancer_data.csv NOT FOUND!")

# Check Python files
print("\n✅ Checking Python Files:")
files = [
    ("02_Member1_EDA", "member1_eda_cleaning_visualization.py"),
    ("03_Member2_Regression", "member2_linear_regression_prediction.py"),
    ("04_Member3_Evaluation", "member3_evaluation_comparison.py")
]

for folder, file in files:
    path = os.path.join(BASE_DIR, folder, file)
    status = "✓" if os.path.exists(path) else "✗"
    print(f"   {status} {folder}/{file}")

# Check libraries
print("\n✅ Checking Required Libraries:")
libraries = ["pandas", "numpy", "matplotlib", "seaborn", "sklearn"]
for lib in libraries:
    try:
        __import__(lib)
        print(f"   ✓ {lib}")
    except ImportError:
        print(f"   ✗ {lib} - NOT INSTALLED!")

print("\n" + "=" * 80)
print("✅ PROJECT SETUP TEST COMPLETE!")
print("=" * 80)
print("\nIf all items show ✓, you're ready to run the project!")
print("Run: python RUN_COMPLETE_PROJECT.py")
print("=" * 80)
