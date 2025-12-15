from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os

# Create Flask app
app = Flask(__name__)
CORS(app)

# In-memory storage (will reset on each deployment - use external DB for production)
cases_storage = []

# Load logistic regression model
model = None

def reload_model():
    """Reload the model (useful after training)"""
    global model
    try:
        # Try different import methods (both should work)
        try:
            from model_def import load_model
        except ImportError:
            import sys
            sys.path.insert(0, os.path.dirname(__file__))
            from model_def import load_model
        
        model = load_model()
        if model is not None:
            print("✓ Logistic regression model loaded successfully")
        else:
            print("Warning: No trained model found.")
    except Exception as e:
        print(f"Warning: Could not load model ({str(e)}). Predictions will not be available.")

reload_model()

@app.route("/", methods=["GET"])
@app.route("/api", methods=["GET"])
def index():
    return jsonify({"message": "Fair Use API is running", "status": "ok"})

@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Train a model first."}), 500
    
    try:
        try:
            from model_def import predict as model_predict
        except ImportError:
            import sys
            sys.path.insert(0, os.path.dirname(__file__))
            from model_def import predict as model_predict
        
        data = request.json
        answers = data["answers"]  # list of numbers
        
        # Validate we have 25 answers
        if len(answers) != 25:
            return jsonify({"error": "Must provide exactly 25 answers"}), 400
        
        # Make prediction
        score = model_predict(model, answers)
        
        if score is None:
            return jsonify({"error": "Prediction failed"}), 500

        return jsonify({"score": score})
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

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
