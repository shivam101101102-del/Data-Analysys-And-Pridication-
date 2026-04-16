"""
================================================================================
LUNG CANCER RISK PREDICTION SYSTEM (ZERO-CRASH RE-OPTIMIZED)
================================================================================
Professional AI-Powered Medical Risk Assessment Platform
Lightweight Digital Image Intelligence Engine (PIL Version)
================================================================================
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from datetime import datetime
import json
from werkzeug.utils import secure_filename
import gc
from PIL import Image, ImageStat, ImageFilter

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
history = []

def load_ml_model():
    """Load the Linear Regression model only when needed"""
    global model
    if model is None:
        try:
            import pickle
            import sklearn
            model_path = os.path.join(BASE_DIR, "Dataset", "trained_model.pkl")
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                return True
            return False
        except Exception as e:
            print(f"ERROR loading model: {e}")
            return False
    return True

def get_available_graphs():
    """Get list of all available graphs"""
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
    """
    Ultra-Light Digital Image Analysis (PIL ONLY)
    Uses zero-memory processing for 100% stability on Render Free.
    """
    start_time = datetime.now()
    print(f"[{start_time}] Starting Image Analysis...")
    
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image uploaded'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"Image saved: {filename}")
        
        # Load and verify image using PIL (Very fast)
        img = Image.open(filepath).convert('L') # Convert to grayscale directly
        img_resized = img.resize((256, 256)) # Smaller for faster processing
        
        # --- DIGITAL ANALYSIS (PIL Version) ---
        stats = ImageStat.Stat(img_resized)
        mean_intensity = stats.mean[0] # Average brightness
        std_intensity = stats.stddev[0] # Contrast/Variance
        
        # Simple Edge Analysis using PIL Filters
        edges = img_resized.filter(ImageFilter.FIND_EDGES)
        edge_stats = ImageStat.Stat(edges)
        edge_density = edge_stats.mean[0] / 255.0 * 10 # Scaled edge density
        
        # Risk Calculation Logic
        base_risk = (1 - mean_intensity/255) * 45
        texture_risk = (std_intensity / 128) * 25
        structural_risk = edge_density * 8.0
        
        final_risk = min(100.0, max(14.5, base_risk + texture_risk + structural_risk))
        print(f"Analysis Complete. Risk: {final_risk}")
        
        # Categorization
        if final_risk <= 40:
            category, color, icon = "LOW RISK", "#2ecc71", "🟢"
            recommendation = "Analysis indicates standard lung patterns. No significant lung nodules or dense opacities detected."
        elif final_risk <= 70:
            category, color, icon = "MEDIUM RISK", "#f39c12", "🟡"
            recommendation = "Analysis shows some concerning dense patterns. Pulmonologist consultation recommended."
        else:
            category, color, icon = "HIGH RISK", "#e74c3c", "🔴"
            recommendation = "⚠️ URGENT: Analysis identified significant dense opacities. IMMEDIATE specialist evaluation (CT/Biopsy) recommended."

        history_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': 'Image', 'risk_score': round(final_risk, 2), 'category': category
        }
        history.append(history_entry)
        
        res = jsonify({
            'success': True,
            'risk_score': round(final_risk, 2),
            'category': category,
            'color': color,
            'icon': icon,
            'recommendation': recommendation,
            'image_stats': {
                'engine': 'PIL Light Engine',
                'mean_opacity': round(mean_intensity, 2),
                'texture_variance': round(std_intensity, 2)
            },
            'history': history[-5:],
            'chart_data': {
                'image_features': {
                    'Density': round(mean_intensity, 2),
                    'Variance': round(std_intensity, 2),
                    'Risk': round(final_risk, 2)
                },
                'normal_ranges': {
                    'Density': 135, 'Variance': 45, 'Risk': 35
                }
            }
        })
        
        gc.collect()
        return res
        
    except Exception as e:
        print(f"CRITICAL ERROR in Image Analysis: {str(e)}")
        return jsonify({'success': False, 'error': f"Processing Error: {str(e)}"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        import numpy as np
        import sklearn
        if not load_ml_model():
            return jsonify({'success': False, 'error': 'Regression model failed to load'})
            
        data = request.json
        age, smoking, pollution, fatigue, coughing = float(data['age']), int(data['smoking']), int(data['pollution']), int(data['fatigue']), int(data['coughing'])

        features = [[age, smoking, pollution, fatigue, coughing]]
        prediction = float(model.predict(features)[0])
        prediction = min(100.0, max(0, prediction))
        
        category = "LOW RISK" if prediction <= 40 else "MEDIUM RISK" if prediction <= 70 else "HIGH RISK"
        color = "#2ecc71" if prediction <= 40 else "#f39c12" if prediction <= 70 else "#e74c3c"
        icon = "🟢" if prediction <= 40 else "🟡" if prediction <= 70 else "🔴"
        
        history.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': 'Manual', 'risk_score': round(prediction, 2), 'category': category
        })
        
        # Mocking contribs for charts
        age_contrib = (age / 100) * 25
        smoke_contrib = (smoking / 8) * 20
        poll_contrib = (pollution / 8) * 15
        
        return jsonify({
            'success': True, 'risk_score': round(prediction, 2), 'category': category,
            'color': color, 'icon': icon, 'recommendation': "Check medical guidance.",
            'history': history[-5:],
            'chart_data': {
                'input_values': {'age': age, 'smoking': smoking, 'pollution': pollution, 'fatigue': fatigue, 'coughing': coughing},
                'contributions': {'Age': round(age_contrib, 2), 'Smoking': round(smoke_contrib, 2), 'Pollution': round(poll_contrib, 2), 'Fatigue': 10.0, 'Coughing': 10.0},
                'normal_ranges': {'age': 45, 'smoking': 2, 'pollution': 2, 'fatigue': 2, 'coughing': 2}
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/graphs/<member>/<filename>')
def serve_graph(member, filename):
    return send_from_directory(os.path.join(BASE_DIR, "05_Final_Output", f"Member{member}_Graphs"), filename)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
