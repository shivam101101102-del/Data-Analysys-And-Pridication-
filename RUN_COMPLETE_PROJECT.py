"""
================================================================================
COMPLETE PROJECT RUNNER - ALL 3 MODULES
================================================================================
TYBCA SEM 6 - Data Analytics using Python (DAP) Project
Subject: 602 - Data Analytics Using Python

This script runs all 3 member modules in sequence:
- Member 1: EDA + Data Cleaning + Visualization
- Member 2: Linear Regression + Training + Prediction
- Member 3: Model Evaluation + Comparison + Results

Total Execution Time: 3-5 minutes
Total Outputs: 14 graphs + 3 reports + 4 data files
================================================================================
"""

import os
import sys
import time

print("=" * 80)
print("🚀 LUNG CANCER RISK PREDICTION - COMPLETE PROJECT")
print("=" * 80)
print("TYBCA SEM 6 - Data Analytics using Python (DAP)")
print("Subject: 602 - Data Analytics Using Python")
print("=" * 80)
print("\nProject Structure:")
print("  Module 1: EDA + Data Cleaning + Visualization")
print("  Module 2: Linear Regression + Training + Prediction")
print("  Module 3: Model Evaluation + Comparison + Results")
print("=" * 80)

input("\nPress ENTER to start the complete project execution...")

# Get base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# MODULE 1: MEMBER 1 - EDA + DATA CLEANING + VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("📊 EXECUTING MODULE 1: EDA + DATA CLEANING + VISUALIZATION")
print("=" * 80)
print("Member: [Member 1 Name]")
print("Expected Time: 1-2 minutes")
print("Expected Outputs: 5 graphs + 1 report + 1 cleaned dataset")
print("=" * 80)

input("\nPress ENTER to run Module 1...")

start_time = time.time()

# Run Member 1 script
member1_script = os.path.join(BASE_DIR, "EDA", "member1_eda_cleaning_visualization.py")
os.system(f'python "{member1_script}"')

elapsed = time.time() - start_time
print(f"\n✅ Module 1 completed in {elapsed:.2f} seconds")

input("\nPress ENTER to continue to Module 2...")

# ============================================================================
# MODULE 2: MEMBER 2 - LINEAR REGRESSION + TRAINING + PREDICTION
# ============================================================================
print("\n" + "=" * 80)
print("🤖 EXECUTING MODULE 2: LINEAR REGRESSION + TRAINING + PREDICTION")
print("=" * 80)
print("Member: [Member 2 Name]")
print("Expected Time: 1-2 minutes")
print("Expected Outputs: 4 graphs + 1 report + 3 data files")
print("=" * 80)

input("\nPress ENTER to run Module 2...")

start_time = time.time()

# Run Member 2 script
member2_script = os.path.join(BASE_DIR, "Regression", "member2_linear_regression_prediction.py")
os.system(f'python "{member2_script}"')

elapsed = time.time() - start_time
print(f"\n✅ Module 2 completed in {elapsed:.2f} seconds")

input("\nPress ENTER to continue to Module 3...")

# ============================================================================
# MODULE 3: MEMBER 3 - MODEL EVALUATION + COMPARISON + RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("📈 EXECUTING MODULE 3: MODEL EVALUATION + COMPARISON + RESULTS")
print("=" * 80)
print("Member: [Member 3 Name]")
print("Expected Time: 1-2 minutes")
print("Expected Outputs: 5 graphs + 1 report + 1 metrics file")
print("=" * 80)

input("\nPress ENTER to run Module 3...")

start_time = time.time()

# Run Member 3 script
member3_script = os.path.join(BASE_DIR, "Evaluation", "member3_evaluation_comparison.py")
os.system(f'python "{member3_script}"')

elapsed = time.time() - start_time
print(f"\n✅ Module 3 completed in {elapsed:.2f} seconds")

# ============================================================================
# PROJECT COMPLETION SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("🎉 ALL 3 MODULES COMPLETED SUCCESSFULLY!")
print("=" * 80)

print("\n📊 TOTAL OUTPUTS GENERATED:")
print("-" * 80)
print("\nMember 1 (EDA Module):")
print("  ✓ 5 Visualization Graphs")
print("  ✓ 1 Summary Report")
print("  ✓ 1 Cleaned Dataset")

print("\nMember 2 (Regression Module):")
print("  ✓ 4 Visualization Graphs")
print("  ✓ 1 Summary Report")
print("  ✓ 3 Data Files (predictions, model, train-test data)")

print("\nMember 3 (Evaluation Module):")
print("  ✓ 5 Visualization Graphs")
print("  ✓ 1 Summary Report")
print("  ✓ 1 Metrics File")

print("\n" + "=" * 80)
print("📁 OUTPUT LOCATIONS:")
print("=" * 80)
print(f"\nGraphs:")
print(f"  • 05_Final_Output/Member1_Graphs/ (5 files)")
print(f"  • 05_Final_Output/Member2_Graphs/ (4 files)")
print(f"  • 05_Final_Output/Member3_Graphs/ (5 files)")

print(f"\nReports:")
print(f"  • 05_Final_Output/Member1_Report.txt")
print(f"  • 05_Final_Output/Member2_Report.txt")
print(f"  • 05_Final_Output/Member3_Report.txt")

print(f"\nData Files:")
print(f"  • 01_Dataset/cleaned_data.csv")
print(f"  • 01_Dataset/predictions.csv")
print(f"  • 01_Dataset/trained_model.pkl")
print(f"  • 01_Dataset/train_test_data.pkl")
print(f"  • 05_Final_Output/Evaluation_Metrics.csv")

print("\n" + "=" * 80)
print("✅ PROJECT READY FOR SUBMISSION!")
print("=" * 80)
print("\nNext Steps:")
print("  1. Review all generated graphs in 05_Final_Output/")
print("  2. Read all 3 member reports")
print("  3. Check Evaluation_Metrics.csv for final results")
print("  4. Prepare presentation using reports")
print("  5. Submit all files to college")

print("\n" + "=" * 80)
print("Thank you for using the Lung Cancer Risk Prediction System!")
print("=" * 80)

input("\nPress ENTER to exit...")
