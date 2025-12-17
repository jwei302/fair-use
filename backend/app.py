from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sqlite3
import json
from datetime import datetime
import os
import tempfile
import traceback

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)   # <-- REQUIRED so browser is allowed to POST from quiz.html

# Database setup
DB_PATH = "training_data.db"

# Initialize cloud storage and video analysis modules
cloud_storage = None
def init_cloud_storage():
    """Initialize cloud storage on startup"""
    global cloud_storage
    try:
        from cloud_storage import get_storage
        cloud_storage = get_storage()
    except Exception as e:
        print(f"Warning: Could not initialize cloud storage ({str(e)}). Video analysis features will not work.")

init_cloud_storage()

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

# =====================================================
# VIDEO ANALYSIS ENDPOINTS
# =====================================================

@app.route("/api/get-upload-url", methods=["GET"])
def get_upload_url():
    """Generate a presigned URL for uploading video to S3"""
    if cloud_storage is None:
        return jsonify({"error": "Cloud storage not configured"}), 500
    
    try:
        # Generate unique video key
        video_key = cloud_storage.generate_video_key('mp4')
        
        # Get presigned URL
        result = cloud_storage.get_presigned_upload_url(video_key, expiration=600)
        
        return jsonify(result), 200
    except Exception as e:
        print(f"Error generating upload URL: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/analyze-video", methods=["POST"])
def analyze_video():
    """Analyze a video for fair-use risk assessment"""
    if cloud_storage is None:
        return jsonify({"error": "Cloud storage not configured"}), 500
    
    try:
        data = request.json
        video_key = data.get("video_key")
        
        if not video_key:
            return jsonify({"error": "video_key is required"}), 400
        
        # Check if video exists in S3
        if not cloud_storage.video_exists(video_key):
            return jsonify({"error": "Video not found in storage"}), 404
        
        print(f"\n{'='*60}")
        print(f"Starting video analysis: {video_key}")
        print(f"{'='*60}\n")
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, 'video.mp4')
            
            # Download video from S3
            print("1. Downloading video from S3...")
            if not cloud_storage.download_video(video_key, video_path):
                return jsonify({"error": "Failed to download video"}), 500
            
            # Process video (extract frames and audio)
            print("2. Processing video...")
            from video_processor import process_video_for_analysis
            frames, audio_path, video_info = process_video_for_analysis(video_path)
            
            if not frames or len(frames) == 0:
                return jsonify({"error": "No frames could be extracted from video"}), 500
            
            # Transcribe audio
            print("3. Transcribing audio...")
            from video_analyzer import transcribe_audio
            transcript = transcribe_audio(audio_path) if audio_path else None
            
            # Analyze video content
            print("4. Analyzing video content with GPT-4V...")
            from video_analyzer import analyze_video_complete
            analysis_result = analyze_video_complete(frames, transcript)
            
            # Clean up audio file
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
        
        # Optionally delete video from S3 (uncomment if you want auto-cleanup)
        # cloud_storage.delete_video(video_key)
        
        print(f"\n{'='*60}")
        print(f"Video analysis complete!")
        print(f"{'='*60}\n")
        
        # Return results
        return jsonify({
            "success": True,
            "video_info": video_info,
            "frames_analyzed": len(frames),
            "has_transcript": transcript is not None,
            "transcript_length": len(transcript) if transcript else 0,
            "analysis": analysis_result
        }), 200
        
    except Exception as e:
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f"Error analyzing video: {error_msg}")
        print(traceback_str)
        return jsonify({
            "error": error_msg,
            "traceback": traceback_str
        }), 500

# =====================================================
# SERVE STATIC HTML FILES
# =====================================================

@app.route("/")
def serve_index():
    """Serve the main index.html file"""
    # Go up one directory from backend/ to project root
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return send_from_directory(parent_dir, 'index.html')

@app.route("/<path:filename>")
def serve_static(filename):
    """Serve other static files from project root"""
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return send_from_directory(parent_dir, filename)

if __name__ == "__main__":
    app.run(port=5000, debug=False)
