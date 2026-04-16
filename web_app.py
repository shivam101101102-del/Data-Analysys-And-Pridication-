"""
================================================================================
LUNG CANCER RISK PREDICTION SYSTEM
================================================================================
Professional AI-Powered Medical Risk Assessment Platform
Advanced Machine Learning for Early Detection & Prevention
================================================================================
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import pickle
import os
import pandas as pd
from datetime import datetime
import json
from PIL import Image
import numpy as np
import cv2
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)
app.config['SECRET_KEY'] = 'lung-cancer-prediction-2026'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load deep learning model
dl_model = None
try:
    dl_model = keras.models.load_model('models/lung_cancer_detector.h5')
    print("✅ Deep Learning model loaded successfully!")
except:
    print("⚠️  Deep Learning model not found. Using basic analysis.")

# Global variables
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = None
history = []

def load_model():
    """Load the trained model"""
    global model
    try:
        model_path = os.path.join(BASE_DIR, "Dataset", "trained_model.pkl")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def get_available_graphs():
    """Get list of all available graphs from all modules"""
    graphs = {
        'member1': [],
        'member2': [],
        'member3': []
    }
    
    # Member 1 graphs
    member1_dir = os.path.join(BASE_DIR, "05_Final_Output", "Member1_Graphs")
    if os.path.exists(member1_dir):
        graphs['member1'] = [f for f in os.listdir(member1_dir) if f.endswith('.png')]
    
    # Member 2 graphs
    member2_dir = os.path.join(BASE_DIR, "05_Final_Output", "Member2_Graphs")
    if os.path.exists(member2_dir):
        graphs['member2'] = [f for f in os.listdir(member2_dir) if f.endswith('.png')]
    
    # Member 3 graphs
    member3_dir = os.path.join(BASE_DIR, "05_Final_Output", "Member3_Graphs")
    if os.path.exists(member3_dir):
        graphs['member3'] = [f for f in os.listdir(member3_dir) if f.endswith('.png')]
    
    return graphs

@app.route('/')
def home():
    """Home page"""
    return render_template('home.html')

@app.route('/manual-prediction')
def manual_prediction():
    """Manual prediction page"""
    graphs = get_available_graphs()
    return render_template('manual_prediction.html', graphs=graphs)

@app.route('/image-prediction')
def image_prediction():
    """Image prediction page"""
    return render_template('image_prediction.html')

@app.route('/analytics')
def analytics():
    """Analytics dashboard"""
    graphs = get_available_graphs()
    return render_template('analytics.html', graphs=graphs)

@app.route('/predict_image', methods=['POST'])
def predict_image():
    """Predict from medical image using Deep Learning"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image uploaded'})
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Save uploaded file
        filename = secure_filename(file.filename) if file.filename else 'uploaded_image.jpg'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load and preprocess image
            img = Image.open(filepath)
            img = img.convert('RGB')
            img_resized = img.resize((224, 224))
            img_array = np.array(img_resized)
            
            # Analyze with deep learning model
            if dl_model is not None:
                # Normalize image
                img_normalized = img_array / 255.0
                img_batch = np.expand_dims(img_normalized, axis=0)
                
                # Get prediction
                cancer_probability = dl_model.predict(img_batch, verbose=0)[0][0]
                
                # Calculate risk score (0-100)
                risk_score = cancer_probability * 100
                
                # Add some variation based on image features
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                mean_intensity = np.mean(gray)
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                
                # Adjust risk based on visual features
                if edge_density > 0.15:  # High edge density = more abnormalities
                    risk_score = min(100, risk_score + 15)
                if mean_intensity < 100:  # Darker areas = potential issues
                    risk_score = min(100, risk_score + 10)
                
            else:
                # Fallback to basic analysis
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                mean_intensity = np.mean(gray)
                std_intensity = np.std(gray)
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                
                # Calculate risk
                risk_score = (
                    (1 - mean_intensity / 255) * 50 +
                    edge_density * 100 * 0.5 +
                    (std_intensity / 128) * 30
                )
            
            risk_score = min(100, max(0, risk_score))
            
            # Determine category with realistic thresholds
            if risk_score <= 35:
                category = "LOW RISK"
                color = "#2ecc71"
                icon = "🟢"
                recommendation = "Image analysis shows normal lung patterns. No significant abnormalities detected. Continue regular health checkups."
            elif risk_score <= 65:
                category = "MEDIUM RISK"
                color = "#f39c12"
                icon = "🟡"
                recommendation = "Image shows some concerning patterns that require attention. Recommend consulting a pulmonologist for detailed examination and possible CT scan."
            else:
                category = "HIGH RISK"
                color = "#e74c3c"
                icon = "🔴"
                recommendation = "⚠️ URGENT: Image shows significant abnormalities consistent with potential lung pathology. IMMEDIATE specialist consultation and comprehensive diagnostic workup (CT scan, biopsy) strongly recommended."
            
            # Calculate detailed stats
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            mean_intensity = float(np.mean(gray))
            std_intensity = float(np.std(gray))
            edges = cv2.Canny(gray, 50, 150)
            edge_density = float(np.sum(edges > 0) / edges.size)
            risk_score = float(risk_score)
            
            # Add to history
            history_entry = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'type': 'Image',
                'age': 'N/A',
                'smoking': 'N/A',
                'pollution': 'N/A',
                'fatigue': 'N/A',
                'coughing': 'N/A',
                'risk_score': round(risk_score, 2),
                'category': category
            }
            history.append(history_entry)
            
            return jsonify({
                'success': True,
                'risk_score': round(risk_score, 2),
                'category': category,
                'color': color,
                'icon': icon,
                'recommendation': recommendation,
                'image_stats': {
                    'mean_intensity': round(mean_intensity, 2),
                    'std_intensity': round(std_intensity, 2),
                    'edge_density': round(edge_density * 100, 2),
                    'model_used': 'Deep Learning (ResNet50)' if dl_model else 'Basic Analysis'
                },
                'history': history[-10:],
                'chart_data': {
                    'image_features': {
                        'Mean Intensity': round(mean_intensity, 2),
                        'Std Intensity': round(std_intensity, 2),
                        'Edge Density': round(edge_density * 100, 2),
                        'Risk Score': round(risk_score, 2)
                    },
                    'normal_ranges': {
                        'Mean Intensity': 128,
                        'Std Intensity': 50,
                        'Edge Density': 10,
                        'Risk Score': 30
                    }
                }
            })
        
        except Exception as img_error:
            return jsonify({'success': False, 'error': f'Image processing error: {str(img_error)}'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': f'Upload error: {str(e)}'})

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction"""
    try:
        data = request.json
        
        # Extract features
        age = float(data['age'])
        smoking = int(data['smoking'])
        pollution = int(data['pollution'])
        fatigue = int(data['fatigue'])
        coughing = int(data['coughing'])
        
        # Make prediction
        features = [[age, smoking, pollution, fatigue, coughing]]
        prediction = model.predict(features)[0]
        
        # Calculate feature contributions (approximate)
        base_risk = 20
        age_contribution = (age / 100) * 25
        smoking_contribution = (smoking / 8) * 20
        pollution_contribution = (pollution / 8) * 15
        fatigue_contribution = (fatigue / 9) * 10
        coughing_contribution = (coughing / 9) * 10
        
        # Determine risk category
        if prediction <= 40:
            category = "LOW RISK"
            color = "#2ecc71"
            icon = "🟢"
            recommendation = "Continue maintaining a healthy lifestyle. Regular annual checkups recommended."
        elif prediction <= 70:
            category = "MEDIUM RISK"
            color = "#f39c12"
            icon = "🟡"
            recommendation = "Consult a doctor for detailed evaluation. Consider lifestyle modifications immediately."
        else:
            category = "HIGH RISK"
            color = "#e74c3c"
            icon = "🔴"
            recommendation = "⚠️ URGENT: Consult a specialist IMMEDIATELY. Further diagnostic tests required."
        
        # Add to history
        history_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'age': age,
            'smoking': smoking,
            'pollution': pollution,
            'fatigue': fatigue,
            'coughing': coughing,
            'risk_score': round(prediction, 2),
            'category': category
        }
        history.append(history_entry)
        
        return jsonify({
            'success': True,
            'risk_score': round(prediction, 2),
            'category': category,
            'color': color,
            'icon': icon,
            'recommendation': recommendation,
            'history': history[-10:],  # Last 10 predictions
            'chart_data': {
                'input_values': {
                    'age': age,
                    'smoking': smoking,
                    'pollution': pollution,
                    'fatigue': fatigue,
                    'coughing': coughing
                },
                'contributions': {
                    'Age': round(age_contribution, 2),
                    'Smoking': round(smoking_contribution, 2),
                    'Pollution': round(pollution_contribution, 2),
                    'Fatigue': round(fatigue_contribution, 2),
                    'Coughing': round(coughing_contribution, 2)
                },
                'normal_ranges': {
                    'age': 45,
                    'smoking': 2,
                    'pollution': 2,
                    'fatigue': 2,
                    'coughing': 2
                }
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/graphs/<member>/<filename>')
def serve_graph(member, filename):
    """Serve graph images"""
    graph_dir = os.path.join(BASE_DIR, "05_Final_Output", f"Member{member}_Graphs")
    return send_from_directory(graph_dir, filename)

@app.route('/history')
def get_history():
    """Get prediction history"""
    return jsonify(history)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear prediction history"""
    global history
    history = []
    return jsonify({'success': True})

@app.route('/export_history')
def export_history():
    """Export history to CSV"""
    if not history:
        return jsonify({'success': False, 'error': 'No history to export'})
    
    try:
        df = pd.DataFrame(history)
        csv_path = os.path.join(BASE_DIR, "Dataset", "web_prediction_history.csv")
        df.to_csv(csv_path, index=False)
        return jsonify({'success': True, 'message': 'History exported to web_prediction_history.csv'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("=" * 80)
    print("🚀 LUNG CANCER RISK PREDICTION - WEB APPLICATION")
    print("=" * 80)
    print("\n📊 Loading model...")
    
    if load_model():
        print("✅ Model loaded successfully!")
        print("\n🌐 Starting web server...")
        print("\n" + "=" * 80)
        print("🎉 Application is running!")
        print("=" * 80)
        print("\n📍 Open your browser and go to:")
        print("\n   👉 http://localhost:10000")
        print("\n" + "=" * 80)
        print("\nPress CTRL+C to stop the server")
        print("=" * 80 + "\n")
        
        # Use port from environment variable for Render
        port = int(os.environ.get("PORT", 10000))
        app.run(host='0.0.0.0', port=port)
    else:
        print("\n❌ Failed to load model!")
        print("⚠️  Please run: python RUN_COMPLETE_PROJECT.py first")
