from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import pickle

# Create Flask app
app = Flask(__name__)
CORS(app)

# In-memory storage (will reset on each deployment - use external DB for production)
cases_storage = []

# Load logistic regression model
model = None

def load_model():
    """Load the trained model from disk - embedded to avoid import issues"""
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'logistic_model.pkl')
        print(f"Attempting to load model from: {model_path}")
        print(f"File exists: {os.path.exists(model_path)}")
        
        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}")
            return None
            
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        print("Model loaded successfully!")
        return loaded_model
    except FileNotFoundError:
        print(f"Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

def predict_with_model(model, answers):
    """Make a prediction using the loaded model - embedded to avoid import issues"""
    try:
        import numpy as np
        # Reshape to 2D array as sklearn expects
        X = np.array(answers).reshape(1, -1)
        # Get probability of positive class (fair use)
        prob = model.predict_proba(X)[0][1]
        return float(prob)
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return None

# Try to load model, but don't crash if it fails
try:
    model = load_model()
    if model is not None:
        print("✓ Model loaded successfully")
    else:
        print("⚠ No model loaded - predictions will not be available")
except Exception as e:
    print(f"⚠ Model loading failed: {e}")
    import traceback
    traceback.print_exc()

@app.route("/", methods=["GET"])
@app.route("/api", methods=["GET"])
def index():
    return jsonify({"message": "Fair Use API is running", "status": "ok"})

@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded. Train a model first."}), 500
        
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        answers = data.get("answers")
        if not answers:
            return jsonify({"error": "No answers provided"}), 400
        
        # Validate we have 25 answers
        if len(answers) != 25:
            return jsonify({"error": "Must provide exactly 25 answers"}), 400
        
        # Make prediction
        score = predict_with_model(model, answers)
        
        if score is None:
            return jsonify({"error": "Prediction failed"}), 500

        return jsonify({"score": score})
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"Prediction failed: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@app.route("/api/add-case", methods=["POST"])
def add_case():
    """Save a labeled case to in-memory storage"""
    try:
        data = request.json
        case_name = data.get("case_name", "").strip()
        answers = data.get("answers", [])
        label = data.get("label")  # 1 for fair use, 0 for not fair use
        
        if not case_name:
            return jsonify({"error": "Case name is required"}), 400
        if len(answers) != 25:
            return jsonify({"error": "Must provide exactly 25 answers"}), 400
        if label not in [0, 1]:
            return jsonify({"error": "Label must be 0 (not fair use) or 1 (fair use)"}), 400
        
        case_id = len(cases_storage) + 1
        case = {
            "id": case_id,
            "case_name": case_name,
            "answers": answers,
            "label": label,
            "created_at": "now"
        }
        cases_storage.append(case)
        
        return jsonify({"success": True, "id": case_id}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/cases", methods=["GET"])
def get_cases():
    """Get all cases from storage"""
    try:
        return jsonify({"cases": cases_storage}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/cases/<int:case_id>", methods=["DELETE"])
def delete_case(case_id):
    """Delete a case by ID"""
    try:
        global cases_storage
        cases_storage = [c for c in cases_storage if c["id"] != case_id]
        return jsonify({"success": True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/export-cases", methods=["GET"])
def export_cases():
    """Export all cases as JSON"""
    try:
        return jsonify({"cases": cases_storage, "count": len(cases_storage)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/train-model", methods=["POST"])
def train_model_endpoint():
    """Training endpoint - not available in serverless environment"""
    return jsonify({
        "error": "Training is not available in the serverless environment. Use the local version for training.",
        "success": False
    }), 501

# This is the entry point for Vercel
if __name__ == "__main__":
    app.run(debug=False)
