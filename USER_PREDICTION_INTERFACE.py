"""
================================================================================
INTERACTIVE USER PREDICTION INTERFACE
================================================================================
TYBCA SEM 6 - Data Analytics using Python (DAP) Project
Subject: 602 - Data Analytics Using Python

This interface allows users to:
- Enter their own data manually
- Get instant risk score predictions
- View risk category and recommendations
- Save prediction history
================================================================================
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pickle
import os
import pandas as pd
from datetime import datetime

class LungCancerPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🫁 Lung Cancer Risk Prediction System")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        # Load trained model
        self.load_model()
        
        # Create GUI
        self.create_widgets()
        
        # Prediction history
        self.history = []
    
    def load_model(self):
        """Load the trained Linear Regression model"""
        try:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(BASE_DIR, "01_Dataset", "trained_model.pkl")
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            self.model_loaded = True
            print("✅ Model loaded successfully!")
        except Exception as e:
            self.model_loaded = False
            print(f"❌ Error loading model: {e}")
            messagebox.showerror("Error", "Could not load trained model. Please run Module 2 first.")
    
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill='x')
        
        title_label = tk.Label(
            title_frame,
            text="🫁 Lung Cancer Risk Prediction System",
            font=('Arial', 20, 'bold'),
            bg='#2c3e50',
            fg='white',
            pady=20
        )
        title_label.pack()
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Left panel - Input fields
        left_frame = tk.LabelFrame(
            main_frame,
            text="📝 Enter Patient Information",
            font=('Arial', 12, 'bold'),
            bg='white',
            padx=20,
            pady=20
        )
        left_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        
        # Input fields
        self.create_input_fields(left_frame)
        
        # Right panel - Results
        right_frame = tk.LabelFrame(
            main_frame,
            text="📊 Prediction Results",
            font=('Arial', 12, 'bold'),
            bg='white',
            padx=20,
            pady=20
        )
        right_frame.grid(row=0, column=1, sticky='nsew')
        
        # Results display
        self.create_results_panel(right_frame)
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Bottom panel - History
        history_frame = tk.LabelFrame(
            self.root,
            text="📜 Prediction History",
            font=('Arial', 12, 'bold'),
            bg='white',
            padx=10,
            pady=10
        )
        history_frame.pack(fill='both', expand=True, padx=20, pady=(0, 20))
        
        self.create_history_panel(history_frame)
    
    def create_input_fields(self, parent):
        """Create input fields for patient data"""
        
        # Age
        tk.Label(parent, text="Age (years):", font=('Arial', 10, 'bold'), bg='white').grid(
            row=0, column=0, sticky='w', pady=10
        )
        self.age_var = tk.StringVar()
        age_entry = tk.Entry(parent, textvariable=self.age_var, font=('Arial', 10), width=20)
        age_entry.grid(row=0, column=1, pady=10, padx=10)
        tk.Label(parent, text="(e.g., 25-80)", font=('Arial', 8), bg='white', fg='gray').grid(
            row=0, column=2, sticky='w'
        )
        
        # Smoking
        tk.Label(parent, text="Smoking Level:", font=('Arial', 10, 'bold'), bg='white').grid(
            row=1, column=0, sticky='w', pady=10
        )
        self.smoking_var = tk.StringVar()
        smoking_scale = tk.Scale(
            parent, from_=1, to=8, orient='horizontal',
            variable=self.smoking_var, font=('Arial', 9),
            length=200, bg='white'
        )
        smoking_scale.grid(row=1, column=1, pady=10, padx=10)
        tk.Label(parent, text="(1=Low, 8=High)", font=('Arial', 8), bg='white', fg='gray').grid(
            row=1, column=2, sticky='w'
        )
        
        # Air Pollution
        tk.Label(parent, text="Air Pollution:", font=('Arial', 10, 'bold'), bg='white').grid(
            row=2, column=0, sticky='w', pady=10
        )
        self.pollution_var = tk.StringVar()
        pollution_scale = tk.Scale(
            parent, from_=1, to=8, orient='horizontal',
            variable=self.pollution_var, font=('Arial', 9),
            length=200, bg='white'
        )
        pollution_scale.grid(row=2, column=1, pady=10, padx=10)
        tk.Label(parent, text="(1=Low, 8=High)", font=('Arial', 8), bg='white', fg='gray').grid(
            row=2, column=2, sticky='w'
        )
        
        # Fatigue
        tk.Label(parent, text="Fatigue Level:", font=('Arial', 10, 'bold'), bg='white').grid(
            row=3, column=0, sticky='w', pady=10
        )
        self.fatigue_var = tk.StringVar()
        fatigue_scale = tk.Scale(
            parent, from_=1, to=9, orient='horizontal',
            variable=self.fatigue_var, font=('Arial', 9),
            length=200, bg='white'
        )
        fatigue_scale.grid(row=3, column=1, pady=10, padx=10)
        tk.Label(parent, text="(1=Low, 9=High)", font=('Arial', 8), bg='white', fg='gray').grid(
            row=3, column=2, sticky='w'
        )
        
        # Coughing of Blood
        tk.Label(parent, text="Coughing Blood:", font=('Arial', 10, 'bold'), bg='white').grid(
            row=4, column=0, sticky='w', pady=10
        )
        self.coughing_var = tk.StringVar()
        coughing_scale = tk.Scale(
            parent, from_=1, to=9, orient='horizontal',
            variable=self.coughing_var, font=('Arial', 9),
            length=200, bg='white'
        )
        coughing_scale.grid(row=4, column=1, pady=10, padx=10)
        tk.Label(parent, text="(1=Low, 9=High)", font=('Arial', 8), bg='white', fg='gray').grid(
            row=4, column=2, sticky='w'
        )
        
        # Buttons
        button_frame = tk.Frame(parent, bg='white')
        button_frame.grid(row=5, column=0, columnspan=3, pady=20)
        
        predict_btn = tk.Button(
            button_frame,
            text="🔮 Predict Risk Score",
            command=self.predict_risk,
            font=('Arial', 11, 'bold'),
            bg='#27ae60',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2'
        )
        predict_btn.pack(side='left', padx=5)
        
        clear_btn = tk.Button(
            button_frame,
            text="🔄 Clear All",
            command=self.clear_inputs,
            font=('Arial', 11, 'bold'),
            bg='#e74c3c',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2'
        )
        clear_btn.pack(side='left', padx=5)
    
    def create_results_panel(self, parent):
        """Create results display panel"""
        
        self.results_text = scrolledtext.ScrolledText(
            parent,
            font=('Courier', 10),
            width=40,
            height=20,
            bg='#ecf0f1',
            wrap='word'
        )
        self.results_text.pack(fill='both', expand=True)
        
        # Initial message
        self.results_text.insert('1.0', "Enter patient data and click 'Predict Risk Score' to see results.")
        self.results_text.config(state='disabled')
    
    def create_history_panel(self, parent):
        """Create prediction history panel"""
        
        self.history_text = scrolledtext.ScrolledText(
            parent,
            font=('Courier', 9),
            height=8,
            bg='#ecf0f1',
            wrap='none'
        )
        self.history_text.pack(fill='both', expand=True)
        
        # Header
        header = "Time                | Age | Smoking | Pollution | Fatigue | Coughing | Risk Score | Category\n"
        header += "-" * 110 + "\n"
        self.history_text.insert('1.0', header)
        self.history_text.config(state='disabled')
    
    def validate_inputs(self):
        """Validate user inputs"""
        try:
            age = float(self.age_var.get())
            if age < 0 or age > 120:
                messagebox.showerror("Invalid Input", "Age must be between 0 and 120")
                return None
            
            smoking = int(self.smoking_var.get())
            pollution = int(self.pollution_var.get())
            fatigue = int(self.fatigue_var.get())
            coughing = int(self.coughing_var.get())
            
            return [age, smoking, pollution, fatigue, coughing]
        
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric values")
            return None
    
    def predict_risk(self):
        """Make prediction based on user input"""
        
        if not self.model_loaded:
            messagebox.showerror("Error", "Model not loaded. Please run Module 2 first.")
            return
        
        # Validate inputs
        inputs = self.validate_inputs()
        if inputs is None:
            return
        
        # Make prediction
        try:
            prediction = self.model.predict([inputs])[0]
            
            # Determine risk category
            if prediction <= 40:
                category = "LOW RISK 🟢"
                color = "green"
                recommendation = "Continue healthy lifestyle. Regular checkups recommended."
            elif prediction <= 70:
                category = "MEDIUM RISK 🟡"
                color = "orange"
                recommendation = "Consult a doctor. Consider lifestyle changes and regular monitoring."
            else:
                category = "HIGH RISK 🔴"
                color = "red"
                recommendation = "URGENT: Consult a specialist immediately. Further tests required."
            
            # Display results
            self.display_results(inputs, prediction, category, recommendation)
            
            # Add to history
            self.add_to_history(inputs, prediction, category)
            
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Error making prediction: {e}")
    
    def display_results(self, inputs, prediction, category, recommendation):
        """Display prediction results"""
        
        self.results_text.config(state='normal')
        self.results_text.delete('1.0', 'end')
        
        result = f"""
{'='*45}
        PREDICTION RESULTS
{'='*45}

INPUT DATA:
-----------
Age:                {inputs[0]:.0f} years
Smoking Level:      {inputs[1]}/8
Air Pollution:      {inputs[2]}/8
Fatigue Level:      {inputs[3]}/9
Coughing Blood:     {inputs[4]}/9

PREDICTION:
-----------
Risk Score:         {prediction:.2f} / 100

Risk Category:      {category}

INTERPRETATION:
---------------
{recommendation}

DISCLAIMER:
-----------
⚠️  This is a screening tool only.
⚠️  Not a substitute for medical diagnosis.
⚠️  Always consult healthcare professionals.

{'='*45}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*45}
"""
        
        self.results_text.insert('1.0', result)
        self.results_text.config(state='disabled')
    
    def add_to_history(self, inputs, prediction, category):
        """Add prediction to history"""
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        self.history_text.config(state='normal')
        
        history_line = f"{timestamp} | {inputs[0]:3.0f} | {inputs[1]:7} | {inputs[2]:9} | {inputs[3]:7} | {inputs[4]:8} | {prediction:10.2f} | {category}\n"
        
        self.history_text.insert('end', history_line)
        self.history_text.see('end')
        self.history_text.config(state='disabled')
        
        # Save to list
        self.history.append({
            'timestamp': timestamp,
            'age': inputs[0],
            'smoking': inputs[1],
            'pollution': inputs[2],
            'fatigue': inputs[3],
            'coughing': inputs[4],
            'risk_score': prediction,
            'category': category
        })
    
    def clear_inputs(self):
        """Clear all input fields"""
        self.age_var.set('')
        self.smoking_var.set('1')
        self.pollution_var.set('1')
        self.fatigue_var.set('1')
        self.coughing_var.set('1')
        
        self.results_text.config(state='normal')
        self.results_text.delete('1.0', 'end')
        self.results_text.insert('1.0', "Enter patient data and click 'Predict Risk Score' to see results.")
        self.results_text.config(state='disabled')

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = LungCancerPredictionApp(root)
    root.mainloop()

if __name__ == "__main__":
    print("=" * 80)
    print("🚀 STARTING LUNG CANCER RISK PREDICTION INTERFACE")
    print("=" * 80)
    print("\n✅ Loading application...")
    print("✅ Please wait for the GUI window to open...")
    print("\n" + "=" * 80)
    
    main()
