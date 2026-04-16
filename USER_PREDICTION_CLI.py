"""
================================================================================
COMMAND-LINE USER PREDICTION INTERFACE
================================================================================
TYBCA SEM 6 - Data Analytics using Python (DAP) Project
Subject: 602 - Data Analytics Using Python

Simple command-line interface for users to:
- Enter their own data
- Get instant predictions
- View risk assessment
================================================================================
"""

import pickle
import os
import pandas as pd
from datetime import datetime

class LungCancerPredictor:
    def __init__(self):
        self.load_model()
        self.history = []
    
    def load_model(self):
        """Load the trained Linear Regression model"""
        try:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(BASE_DIR, "01_Dataset", "trained_model.pkl")
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            print("✅ Model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("⚠️  Please run Module 2 first to train the model.")
            return False
    
    def get_user_input(self):
        """Get input from user"""
        print("\n" + "=" * 80)
        print("📝 ENTER PATIENT INFORMATION")
        print("=" * 80)
        
        try:
            # Age
            while True:
                age = float(input("\n1. Age (years, e.g., 45): "))
                if 0 <= age <= 120:
                    break
                print("   ⚠️  Please enter age between 0 and 120")
            
            # Smoking
            while True:
                smoking = int(input("\n2. Smoking Level (1-8, where 1=Low, 8=High): "))
                if 1 <= smoking <= 8:
                    break
                print("   ⚠️  Please enter value between 1 and 8")
            
            # Air Pollution
            while True:
                pollution = int(input("\n3. Air Pollution Exposure (1-8, where 1=Low, 8=High): "))
                if 1 <= pollution <= 8:
                    break
                print("   ⚠️  Please enter value between 1 and 8")
            
            # Fatigue
            while True:
                fatigue = int(input("\n4. Fatigue Level (1-9, where 1=Low, 9=High): "))
                if 1 <= fatigue <= 9:
                    break
                print("   ⚠️  Please enter value between 1 and 9")
            
            # Coughing of Blood
            while True:
                coughing = int(input("\n5. Coughing of Blood (1-9, where 1=Low, 9=High): "))
                if 1 <= coughing <= 9:
                    break
                print("   ⚠️  Please enter value between 1 and 9")
            
            return [age, smoking, pollution, fatigue, coughing]
        
        except ValueError:
            print("\n❌ Invalid input! Please enter numeric values only.")
            return None
        except KeyboardInterrupt:
            print("\n\n⚠️  Input cancelled by user.")
            return None
    
    def predict_risk(self, inputs):
        """Make prediction"""
        try:
            prediction = self.model.predict([inputs])[0]
            return prediction
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None
    
    def get_risk_category(self, score):
        """Determine risk category"""
        if score <= 40:
            return "LOW RISK 🟢", "green"
        elif score <= 70:
            return "MEDIUM RISK 🟡", "orange"
        else:
            return "HIGH RISK 🔴", "red"
    
    def get_recommendation(self, category):
        """Get recommendation based on category"""
        if "LOW" in category:
            return """
• Continue maintaining a healthy lifestyle
• Regular annual checkups recommended
• Monitor any changes in symptoms
• Avoid smoking and polluted environments
"""
        elif "MEDIUM" in category:
            return """
• Consult a doctor for detailed evaluation
• Consider lifestyle modifications immediately
• Regular monitoring (every 3-6 months)
• Reduce exposure to risk factors
• Consider smoking cessation programs
"""
        else:
            return """
• ⚠️  URGENT: Consult a specialist IMMEDIATELY
• Further diagnostic tests required (CT scan, biopsy)
• Do not delay medical consultation
• Discuss treatment options with oncologist
• Immediate lifestyle changes necessary
"""
    
    def display_results(self, inputs, prediction):
        """Display prediction results"""
        category, color = self.get_risk_category(prediction)
        recommendation = self.get_recommendation(category)
        
        print("\n" + "=" * 80)
        print("📊 PREDICTION RESULTS")
        print("=" * 80)
        
        print("\n📋 INPUT SUMMARY:")
        print("-" * 80)
        print(f"  Age:                    {inputs[0]:.0f} years")
        print(f"  Smoking Level:          {inputs[1]}/8")
        print(f"  Air Pollution:          {inputs[2]}/8")
        print(f"  Fatigue Level:          {inputs[3]}/9")
        print(f"  Coughing of Blood:      {inputs[4]}/9")
        
        print("\n🎯 PREDICTION:")
        print("-" * 80)
        print(f"  Risk Score:             {prediction:.2f} / 100")
        print(f"  Risk Category:          {category}")
        
        print("\n💡 RECOMMENDATIONS:")
        print("-" * 80)
        print(recommendation)
        
        print("\n⚠️  IMPORTANT DISCLAIMER:")
        print("-" * 80)
        print("  • This is a screening tool for preliminary assessment only")
        print("  • NOT a substitute for professional medical diagnosis")
        print("  • Always consult qualified healthcare professionals")
        print("  • Model accuracy: ~69% (R² Score)")
        
        print("\n" + "=" * 80)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Add to history
        self.history.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'age': inputs[0],
            'smoking': inputs[1],
            'pollution': inputs[2],
            'fatigue': inputs[3],
            'coughing': inputs[4],
            'risk_score': prediction,
            'category': category
        })
    
    def save_history(self):
        """Save prediction history to CSV"""
        if not self.history:
            print("\n⚠️  No predictions to save.")
            return
        
        try:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            history_path = os.path.join(BASE_DIR, "01_Dataset", "user_prediction_history.csv")
            
            df = pd.DataFrame(self.history)
            
            # Append to existing file if it exists
            if os.path.exists(history_path):
                existing_df = pd.read_csv(history_path)
                df = pd.concat([existing_df, df], ignore_index=True)
            
            df.to_csv(history_path, index=False)
            print(f"\n✅ Prediction history saved to: user_prediction_history.csv")
            print(f"   Total predictions: {len(df)}")
        
        except Exception as e:
            print(f"\n❌ Error saving history: {e}")
    
    def show_menu(self):
        """Display main menu"""
        print("\n" + "=" * 80)
        print("🫁 LUNG CANCER RISK PREDICTION SYSTEM - MAIN MENU")
        print("=" * 80)
        print("\n1. 🔮 Make New Prediction")
        print("2. 📜 View Prediction History (Current Session)")
        print("3. 💾 Save History to File")
        print("4. ℹ️  About This System")
        print("5. 🚪 Exit")
        print("\n" + "=" * 80)
    
    def show_history(self):
        """Display prediction history"""
        if not self.history:
            print("\n⚠️  No predictions made yet in this session.")
            return
        
        print("\n" + "=" * 80)
        print("📜 PREDICTION HISTORY (CURRENT SESSION)")
        print("=" * 80)
        
        for i, record in enumerate(self.history, 1):
            print(f"\n--- Prediction #{i} ---")
            print(f"Time:        {record['timestamp']}")
            print(f"Age:         {record['age']:.0f} years")
            print(f"Smoking:     {record['smoking']}/8")
            print(f"Pollution:   {record['pollution']}/8")
            print(f"Fatigue:     {record['fatigue']}/9")
            print(f"Coughing:    {record['coughing']}/9")
            print(f"Risk Score:  {record['risk_score']:.2f}")
            print(f"Category:    {record['category']}")
        
        print("\n" + "=" * 80)
    
    def show_about(self):
        """Display about information"""
        print("\n" + "=" * 80)
        print("ℹ️  ABOUT THIS SYSTEM")
        print("=" * 80)
        print("""
PROJECT: Lung Cancer Risk Prediction System
COURSE:  TYBCA SEM 6 - Data Analytics using Python (DAP)
SUBJECT: 602 - Data Analytics Using Python

ALGORITHM: Linear Regression (Supervised Learning)
FEATURES:  5 input features (Age, Smoking, Pollution, Fatigue, Coughing)
TARGET:    Risk Score (0-100)

MODEL PERFORMANCE:
• R² Score: ~69.45% (Good performance)
• MAE: ~9.90 (Average error: ±10 points)
• Training Data: 500 patient records

RISK CATEGORIES:
• LOW RISK (0-40):     Regular monitoring
• MEDIUM RISK (41-70): Medical consultation recommended
• HIGH RISK (71-100):  Urgent specialist consultation

DISCLAIMER:
This system is designed for preliminary screening and educational purposes.
It should NOT be used as the sole basis for medical decisions. Always
consult qualified healthcare professionals for proper diagnosis and treatment.

MODEL LIMITATIONS:
• Limited to 5 features (more comprehensive models use 20+ features)
• Dataset size: 500 records (larger datasets improve accuracy)
• Does not consider genetic factors, family history, or detailed medical history
• Should be used alongside clinical examination and diagnostic tests
        """)
        print("=" * 80)
    
    def run(self):
        """Main application loop"""
        print("\n" + "=" * 80)
        print("🚀 LUNG CANCER RISK PREDICTION SYSTEM")
        print("=" * 80)
        print("TYBCA SEM 6 - Data Analytics using Python (DAP)")
        print("Subject: 602 - Data Analytics Using Python")
        print("=" * 80)
        
        if not hasattr(self, 'model'):
            print("\n❌ Cannot start: Model not loaded.")
            return
        
        while True:
            self.show_menu()
            
            try:
                choice = input("\nEnter your choice (1-5): ").strip()
                
                if choice == '1':
                    # Make prediction
                    inputs = self.get_user_input()
                    if inputs:
                        prediction = self.predict_risk(inputs)
                        if prediction is not None:
                            self.display_results(inputs, prediction)
                
                elif choice == '2':
                    # View history
                    self.show_history()
                
                elif choice == '3':
                    # Save history
                    self.save_history()
                
                elif choice == '4':
                    # About
                    self.show_about()
                
                elif choice == '5':
                    # Exit
                    print("\n" + "=" * 80)
                    print("👋 Thank you for using the Lung Cancer Risk Prediction System!")
                    print("=" * 80)
                    
                    if self.history:
                        save = input("\nDo you want to save prediction history before exiting? (y/n): ").strip().lower()
                        if save == 'y':
                            self.save_history()
                    
                    print("\n✅ Goodbye!\n")
                    break
                
                else:
                    print("\n❌ Invalid choice! Please enter 1-5.")
            
            except KeyboardInterrupt:
                print("\n\n⚠️  Program interrupted by user.")
                print("👋 Goodbye!\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

def main():
    """Main function"""
    predictor = LungCancerPredictor()
    predictor.run()

if __name__ == "__main__":
    main()
