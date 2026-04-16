"""
================================================================================
LUNG CANCER RISK PREDICTION SYSTEM
================================================================================
Professional AI-Powered Medical Risk Assessment Platform
Advanced Machine Learning for Early Detection & Prevention
================================================================================
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from datetime import datetime
import json
from werkzeug.utils import secure_filename
import gc

app = Flask(__name__)
app.config['SECRET_KEY'] = 'lung-cancer-prediction-2026'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Global variables
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = None
dl_model = None
history = []

def get_dl_model():
    """Lazy load deep learning model only when needed to save RAM"""
    global dl_model
    if dl_model is None:
        try:
            print("⏳ Loading TensorFlow and Keras (Dynamic Import)...")
            from tensorflow import keras
            import tensorflow as tf
            
            # Explicitly disable GPU and limit memory
            tf.config.set_visible_devices([], 'GPU')
            
            print("⏳ Loading deep learning model (ResNet50)...")
            model_path = os.path.join(BASE_DIR, 'models', 'lung_cancer_detector.h5')
            dl_model = keras.models.load_model(model_path)
            print("✅ Deep Learning model loaded successfully!")
        except Exception as e:
            print(f"⚠️  Deep Learning model error: {e}")
            dl_model = "FAILED"
        gc.collect()
    return dl_model

def load_ml_model():
    """Load the Linear Regression model only when needed"""
    global model
    if model is None:
        try:
            import pickle
            import sklearn # Required for unpickling sklearn models
            model_path = os.path.join(BASE_DIR, "Dataset", "trained_model.pkl")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print("✅ ML Model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    return True

def get_available_graphs():
    """Get list of all available graphs from all modules"""
    graphs = {'member1': [], 'member2': [], 'member3': []}
    
    for i in range(1, 4):
        folder = f"Member{i}_Graphs"
        dir_path = os.path.join(BASE_DIR, "05_Final_Output", folder)
        if os.path.exists(dir_path):
            graphs[f'member{i}'] = [f for f in os.listdir(dir_path) if f.endswith('.png')]
    return graphs

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/manual-prediction')
def manual_prediction():
    graphs = get_available_graphs()
    return render_template('manual_prediction.html', graphs=graphs)

@app.route('/image-prediction')
def image_prediction():
    return render_template('image_prediction.html')

@app.route('/analytics')
def analytics():
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
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Heavy imports inside function
        import numpy as np
        from PIL import Image
        import cv2

        # Load and preprocess image
        img = Image.open(filepath).convert('RGB')
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized)
        
        risk_score = 0.0
        current_dl_model = get_dl_model()
        
        if current_dl_model is not None and current_dl_model != "FAILED":
            # Normalize and predict
            img_normalized = img_array / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)
            cancer_probability = current_dl_model.predict(img_batch, verbose=0)[0][0]
            risk_score = float(cancer_probability * 100)
            
            # Refine risk with computer vision
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            mean_intensity = float(np.mean(gray))
            edges = cv2.Canny(gray, 50, 150)
            edge_density = float(np.sum(edges > 0) / edges.size)
            
            if edge_density > 0.15: risk_score = min(100.0, risk_score + 15.0)
            if mean_intensity < 100: risk_score = min(100.0, risk_score + 10.0)
        else:
            # Fallback to basic visual analysis if TF fails or is too heavy
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            mean_intensity = float(np.mean(gray))
            edge_density = float(np.sum(cv2.Canny(gray, 50, 150) > 0) / gray.size)
            risk_score = ((1 - mean_intensity/255)*50 + edge_density*100*0.5)*1.2

        risk_score = min(100.0, max(0.0, risk_score))
        
        # Determine category
        if risk_score <= 35:
            category, color, icon = "LOW RISK", "#2ecc71", "🟢"
            recommendation = "Normal patterns. No significant abnormalities. Regular checkups."
        elif risk_score <= 65:
            category, color, icon = "MEDIUM RISK", "#f39c12", "🟡"
            recommendation = "Concerning patterns detected. Consult a pulmonologist for CT scan."
        else:
            category, color, icon = "HIGH RISK", "#e74c3c", "🔴"
            recommendation = "⚠️ URGENT: Significant abnormalities. Immediate specialist consultation recommended."

        # Add to history
        history_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': 'Image', 'risk_score': round(risk_score, 2), 'category': category
        }
        history.append(history_entry)
        
        res = jsonify({
            'success': True, 'risk_score': round(risk_score, 2), 'category': category,
            'color': color, 'icon': icon, 'recommendation': recommendation,
            'image_stats': {'model_used': 'DL (ResNet50)' if dl_model and dl_model != 'FAILED' else 'Digital Analysis'},
            'history': history[-5:]
        })
        
        # Clear memory
        del img_array, img_resized
        gc.collect()
        return res
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        import sklearn # Ensure sklearn is available for model.predict
        if not load_ml_model():
            return jsonify({'success': False, 'error': 'Regression model failed to load'})
            
        import numpy as np
        data = request.json
        features = [[float(data['age']), int(data['smoking']), int(data['pollution']), int(data['fatigue']), int(data['coughing'])]]
        prediction = float(model.predict(features)[0])
        
        category = "LOW RISK" if prediction <= 40 else "MEDIUM RISK" if prediction <= 70 else "HIGH RISK"
        color = "#2ecc71" if prediction <= 40 else "#f39c12" if prediction <= 70 else "#e74c3c"
        
        history_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'age': data['age'], 'risk_score': round(prediction, 2), 'category': category
        }
        history.append(history_entry)
        
        return jsonify({
            'success': True, 'risk_score': round(prediction, 2), 'category': category,
            'color': color, 'history': history[-5:]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/graphs/<member>/<filename>')
def serve_graph(member, filename):
    graph_dir = os.path.join(BASE_DIR, "05_Final_Output", f"Member{member}_Graphs")
    return send_from_directory(graph_dir, filename)

@app.route('/export_history')
def export_history():
    if not history: return jsonify({'success': False, 'error': 'No history'})
    try:
        import pandas as pd
        df = pd.DataFrame(history)
        csv_path = os.path.join(BASE_DIR, "Dataset", "web_prediction_history.csv")
        df.to_csv(csv_path, index=False)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # For local testing, we still need to load the base model
    load_ml_model()
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
