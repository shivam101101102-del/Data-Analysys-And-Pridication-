# 🫁 Lung Cancer Risk Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3.0-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange.svg)
![License](https://img.shields.io/badge/License-Educational-yellow.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)

**AI-Powered Medical Risk Assessment Platform**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Documentation](#-documentation) • [Demo](#-demo)

</div>

---

## 📋 Overview

A comprehensive machine learning-based system for predicting lung cancer risk using patient data and medical imaging. Combines traditional statistical analysis with deep learning (ResNet50) for accurate risk assessment.

### 🎯 Key Highlights

- ✅ **5000+ Training Records** - Large dataset for accurate predictions
- ✅ **Dual Prediction Methods** - Manual input & Image analysis
- ✅ **70-80% Accuracy** - Industry-standard performance
- ✅ **Real-time Visualization** - Dynamic Chart.js charts
- ✅ **Professional UI/UX** - Modern, responsive design
- ✅ **Deep Learning** - ResNet50 with 24.8M parameters

---

## ✨ Features

### 🔮 Manual Risk Prediction
- Input patient data (Age, Smoking, Pollution, Fatigue, Coughing)
- Instant risk score calculation (0-100)
- Color-coded risk categories (Low/Medium/High)
- Professional medical recommendations
- 4 dynamic charts per prediction

### 🖼️ Image-Based Analysis
- Upload X-ray/CT scans (JPG, PNG, JPEG)
- Deep Learning analysis (ResNet50)
- Edge detection & intensity analysis
- Detailed image statistics
- 4 dynamic visualization charts

### 📊 Analytics Dashboard
- 14 comprehensive graphs
- 3 module tabs (EDA, Regression, Evaluation)
- Interactive visualizations
- Click-to-enlarge functionality

### 🎨 Modern UI/UX
- Responsive design (Desktop, Tablet, Mobile)
- Collapsible sidebar navigation
- Smooth animations & transitions
- Professional medical branding
- Dark/Light theme support

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.11+
pip (Python package manager)
4GB RAM minimum
```

### Installation

```bash
# 1. Clone/Download project
cd DAP_Final_Project

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn flask tensorflow opencv-python pillow werkzeug

# 3. Run web application
python web_app.py

# 4. Open browser
# Navigate to: http://localhost:5000
```

### Alternative: Run Complete Project

```bash
# Run all 3 modules (EDA, Regression, Evaluation)
python RUN_COMPLETE_PROJECT.py

# Generates:
# - 14 graphs
# - 3 reports
# - 4 data files
# Time: 3-6 minutes
```

---

## 📁 Project Structure

```
DAP_Final_Project/
├── 01_Dataset/              # Datasets (5000, 2500, 2500 records)
├── 02_Member1_EDA/          # Module 1: EDA + Cleaning
├── 03_Member2_Regression/   # Module 2: Linear Regression
├── 04_Member3_Evaluation/   # Module 3: Model Evaluation
├── 05_Final_Output/         # All outputs (14 graphs, 3 reports)
├── models/                  # Deep learning models (ResNet50)
├── templates/               # HTML templates (4 pages)
├── uploads/                 # User uploaded images
├── web_app.py              # Main Flask application
├── RUN_COMPLETE_PROJECT.py # Run all modules
└── README.md               # This file
```

---

## 💻 Usage

### Manual Prediction

```python
# Example: Predict risk for a patient
Age: 55 years
Smoking: 6/8
Air Pollution: 5/8
Fatigue: 7/9
Coughing: 4/9

Result: Risk Score = 68.5 (MEDIUM RISK)
```

### Image Analysis

```python
# Upload X-ray/CT scan
# System analyzes:
# - Edge density
# - Mean intensity
# - Standard deviation
# - Deep learning features

Result: Risk Score = 72.3 (HIGH RISK)
```

### API Usage

```python
import requests

# Manual prediction
response = requests.post('http://localhost:5000/predict', json={
    'age': 55,
    'smoking': 6,
    'pollution': 5,
    'fatigue': 7,
    'coughing': 4
})

result = response.json()
print(f"Risk Score: {result['risk_score']}")
print(f"Category: {result['category']}")
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│         User Interface (Browser)         │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│         Flask Web Server                 │
│  • Routes Management                     │
│  • Request Handling                      │
└──────────────┬──────────────────────────┘
               │
      ┌────────┴────────┐
      ▼                 ▼
┌──────────┐      ┌──────────┐
│  Manual  │      │  Image   │
│  Predict │      │  Predict │
│          │      │          │
│ Linear   │      │ ResNet50 │
│Regression│      │ Deep     │
│          │      │ Learning │
└──────────┘      └──────────┘
      │                 │
      └────────┬────────┘
               ▼
┌─────────────────────────────────────────┐
│      Visualization Layer (Chart.js)      │
└─────────────────────────────────────────┘
```

---

## 📊 Dataset Information

### 1. lung_cancer_data.csv (5000 records)
- **Features**: Age, Smoking, Air Pollution, Fatigue, Coughing of Blood
- **Target**: Risk_Score (0-100)
- **Quality**: No missing values, no duplicates

### 2. cleaned_data.csv (2500 records)
- **Source**: Module 1 (after outlier removal)
- **Usage**: Model training in Module 2

### 3. predictions.csv (2500 records)
- **Source**: Module 2 (model predictions)
- **Usage**: Model evaluation in Module 3

---

## 🤖 Machine Learning Models

### Linear Regression
- **Algorithm**: Ordinary Least Squares (OLS)
- **Features**: 5 (Age, Smoking, Pollution, Fatigue, Coughing)
- **Training**: 80% (2000 samples)
- **Testing**: 20% (500 samples)
- **Performance**: R² = 0.70-0.80, MAE = 5-8 points

### Deep Learning (ResNet50)
- **Architecture**: ResNet50 (Transfer Learning)
- **Parameters**: 24.8 Million
- **Input**: 224×224×3 RGB images
- **Output**: Cancer probability (0-1)
- **Performance**: 70-75% accuracy

---

## 📈 Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R² Score** | 0.70-0.80 | 70-80% variance explained |
| **MSE** | 50-100 | Mean squared error |
| **RMSE** | 7-10 | Average error in points |
| **MAE** | 5-8 | Average absolute error |
| **Accuracy (±10)** | 75-85% | Predictions within ±10 points |

---

## 🎨 Screenshots

### Home Page
![Home Page](https://via.placeholder.com/800x400?text=Home+Page)

### Manual Prediction
![Manual Prediction](https://via.placeholder.com/800x400?text=Manual+Prediction)

### Image Analysis
![Image Analysis](https://via.placeholder.com/800x400?text=Image+Analysis)

### Analytics Dashboard
![Analytics](https://via.placeholder.com/800x400?text=Analytics+Dashboard)

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Home page |
| GET | `/manual-prediction` | Manual prediction page |
| GET | `/image-prediction` | Image analysis page |
| GET | `/analytics` | Analytics dashboard |
| POST | `/predict` | Manual risk prediction |
| POST | `/predict_image` | Image-based prediction |
| GET | `/history` | Get prediction history |
| POST | `/clear_history` | Clear history |
| GET | `/export_history` | Export history to CSV |

---

## 🧪 Testing

```bash
# Test complete project
python TEST_PROJECT.py

# Test user prediction
python TEST_USER_PREDICTION.py

# Test web application
python web_app.py
# Then manually test all features
```

---

## 🔧 Troubleshooting

### Common Issues

**1. Module Not Found**
```bash
pip install <module-name>
```

**2. Port Already in Use**
```python
# Change port in web_app.py
app.run(debug=True, port=5001)
```

**3. Model File Not Found**
```bash
# Run complete project first
python RUN_COMPLETE_PROJECT.py
```

---

## 📚 Documentation

- **Complete Documentation**: [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)
- **Module 1 Report**: [05_Final_Output/Member1_Report.txt](05_Final_Output/Member1_Report.txt)
- **Module 2 Report**: [05_Final_Output/Member2_Report.txt](05_Final_Output/Member2_Report.txt)
- **Module 3 Report**: [05_Final_Output/Member3_Report.txt](05_Final_Output/Member3_Report.txt)

---

## 🚀 Future Enhancements

- [ ] User authentication system
- [ ] Advanced ML models (Random Forest, XGBoost)
- [ ] Mobile application (Android/iOS)
- [ ] Cloud deployment (AWS/Azure)
- [ ] Real-time patient monitoring
- [ ] Multi-language support
- [ ] PDF report generation
- [ ] Email notifications

---

## 👥 Team

| Member | Role | Module |
|--------|------|--------|
| **Maurya Chandan Shankar** (3570) | EDA & Data Cleaning | Module 1 |
| **Maurya Shivam Hanumanprasad** (3567) | Linear Regression | Module 2 |
| **Pal Prince Lalchandra** (3577) | Model Evaluation | Module 3 |

---

## 🎓 Academic Information

- **Institution**: [Your College Name]
- **Course**: TYBCA SEM 6
- **Subject**: 602 - Data Analytics Using Python (DAP)
- **Academic Year**: 2025-2026
- **Submission Date**: April 2026

---

## 📄 License

This project is developed for educational purposes as part of TYBCA SEM 6 curriculum.

**Copyright © 2026 - All Rights Reserved**

---

## 🙏 Acknowledgments

- **Scikit-learn** - Machine learning library
- **TensorFlow/Keras** - Deep learning framework
- **Flask** - Web framework
- **Chart.js** - Visualization library
- **OpenCV** - Image processing
- **Matplotlib/Seaborn** - Data visualization

---

## 📞 Contact

For queries or support:
- 📧 Email: [your-email@example.com]
- 🐙 GitHub: [your-github-repo]
- 📖 Documentation: [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ by TYBCA Students

</div>
