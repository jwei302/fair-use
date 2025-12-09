from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import json
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)   # <-- REQUIRED so browser is allowed to POST from quiz.html

# Database setup
DB_PATH = "training_data.db"

def init_db():
    """Initialize the SQLite database with cases table"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_name TEXT NOT NULL,
            answers TEXT NOT NULL,
            label INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

# Load logistic regression model
model = None
def reload_model():
    """Reload the model (useful after training)"""
    global model
    try:
        from model_def import load_model
        model = load_model()
        if model is not None:
            print("✓ Logistic regression model loaded successfully")
        else:
            print("Warning: No trained model found. Train a model using the Admin tab or train_model.py")
    except Exception as e:
        print(f"Warning: Could not load model ({str(e)}). Data collection features will still work.")

reload_model()

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Train a model first using train_model.py"}), 500
    
    try:
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
    """Save a labeled case to the database"""
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
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            INSERT INTO cases (case_name, answers, label)
            VALUES (?, ?, ?)
        ''', (case_name, json.dumps(answers), label))
        conn.commit()
        case_id = c.lastrowid
        conn.close()
        
        return jsonify({"success": True, "id": case_id}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/cases", methods=["GET"])
def get_cases():
    """Get all cases from the database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            SELECT id, case_name, answers, label, created_at
            FROM cases
            ORDER BY created_at DESC
        ''')
        rows = c.fetchall()
        conn.close()
        
        cases = []
        for row in rows:
            cases.append({
                "id": row[0],
                "case_name": row[1],
                "answers": json.loads(row[2]),
                "label": row[3],
                "created_at": row[4]
            })
        
        return jsonify({"cases": cases}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/cases/<int:case_id>", methods=["DELETE"])
def delete_case(case_id):
    """Delete a case by ID"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('DELETE FROM cases WHERE id = ?', (case_id,))
        conn.commit()
        deleted = c.rowcount > 0
        conn.close()
        
        if deleted:
            return jsonify({"success": True}), 200
        else:
            return jsonify({"error": "Case not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/export-cases", methods=["GET"])
def export_cases():
    """Export all cases as JSON for training"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            SELECT case_name, answers, label
            FROM cases
            ORDER BY created_at
        ''')
        rows = c.fetchall()
        conn.close()
        
        cases = []
        for row in rows:
            cases.append({
                "case_name": row[0],
                "answers": json.loads(row[1]),
                "label": row[2]
            })
        
        return jsonify({"cases": cases, "count": len(cases)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/train-model", methods=["POST"])
def train_model_endpoint():
    """Train the logistic regression model on current database"""
    try:
        import io
        import sys
        from train_model import train_model as train_func
        
        # Capture stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        try:
            # Run training function directly
            success = train_func()
            
            # Get captured output
            output = sys.stdout.getvalue() + sys.stderr.getvalue()
        finally:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        if success:
            # Reload the model
            reload_model()
            
            return jsonify({
                "success": True,
                "message": "Model trained successfully",
                "output": output
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": "Training failed",
                "output": output
            }), 500
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        return jsonify({
            "error": error_msg,
            "traceback": traceback_str
        }), 500

if __name__ == "__main__":
    app.run(port=5000, debug=False)
