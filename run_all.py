"""
Run all 3 modules non-interactively
"""
import os
import sys
import time
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("LUNG CANCER RISK PREDICTION - COMPLETE PROJECT")
print("=" * 60)

# Module 1 - EDA
print("\n[1/3] Running Module 1: EDA + Data Cleaning...")
m1 = os.path.join(BASE_DIR, "EDA", "member1_eda_cleaning_visualization.py")
start = time.time()
ret = os.system(f'python "{m1}"')
print(f"Module 1 done in {time.time()-start:.1f}s (exit code: {ret})")

# Module 2 - Regression
print("\n[2/3] Running Module 2: Linear Regression...")
m2 = os.path.join(BASE_DIR, "Regression", "member2_linear_regression_prediction.py")
start = time.time()
ret = os.system(f'python "{m2}"')
print(f"Module 2 done in {time.time()-start:.1f}s (exit code: {ret})")

# Module 3 - Evaluation
print("\n[3/3] Running Module 3: Model Evaluation...")
m3 = os.path.join(BASE_DIR, "Evaluation", "member3_evaluation_comparison.py")
start = time.time()
ret = os.system(f'python "{m3}"')
print(f"Module 3 done in {time.time()-start:.1f}s (exit code: {ret})")

print("\n" + "=" * 60)
print("ALL 3 MODULES COMPLETED!")
print("=" * 60)
print("\nOutputs saved in: 05_Final_Output/")
print("Now run: python web_app.py")
