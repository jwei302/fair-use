from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import pickle
import boto3
from datetime import datetime
import uuid

# Create Flask app
app = Flask(__name__)
CORS(app)

# Initialize S3 client for video uploads
s3_client = None
try:
    bucket_name = os.getenv("AWS_S3_BUCKET_NAME")
    region = os.getenv("AWS_REGION", "us-east-1")
    if bucket_name:
        s3_client = boto3.client(
            's3',
            region_name=region,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )
        print(f"✓ S3 client initialized for bucket: {bucket_name}")
    else:
        print("⚠ AWS_S3_BUCKET_NAME not set - video features disabled")
except Exception as e:
    print(f"⚠ S3 initialization failed: {e}")

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

@app.route("/api/get-upload-url", methods=["GET"])
def get_upload_url():
    """Generate a presigned URL for uploading video to S3"""
    try:
        if not s3_client:
            return jsonify({"error": "S3 not configured. Set AWS environment variables."}), 500
        
        bucket_name = os.getenv("AWS_S3_BUCKET_NAME")
        
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        unique_id = uuid.uuid4().hex[:8]
        object_key = f"videos/{timestamp}-{unique_id}.mp4"
        
        # Generate presigned URL for PUT operation (5 minutes expiry)
        upload_url = s3_client.generate_presigned_url(
            'put_object',
            Params={
                'Bucket': bucket_name,
                'Key': object_key,
                'ContentType': 'video/mp4'
            },
            ExpiresIn=300
        )
        
        return jsonify({
            "upload_url": upload_url,
            "video_key": object_key,
            "bucket": bucket_name
        }), 200
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"Failed to generate upload URL: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@app.route("/api/analyze-video", methods=["POST"])
def analyze_video():
    """Analyze video - placeholder for serverless environment"""
    return jsonify({
        "error": "Video analysis is not available in the Vercel serverless environment due to computational limitations. Please use the local version for video analysis.",
        "details": "Video analysis requires FFmpeg, OpenCV, and significant compute resources that exceed Vercel's serverless function limits."
    }), 501

@app.route("/api/analyze-video-client", methods=["POST"])
def analyze_video_client():
    """Analyze video using client-extracted frames and audio"""
    import openai
    import base64
    
    try:
        data = request.json
        frames = data.get('frames', [])
        audio_base64 = data.get('audio_base64')
        
        if not frames:
            return jsonify({"error": "No frames provided"}), 400
        
        # Initialize OpenAI
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            return jsonify({"error": "OpenAI API key not configured"}), 500
        
        # Step 1: Transcribe audio with Whisper
        transcript = ""
        if audio_base64:
            try:
                audio_data = base64.b64decode(audio_base64)
                # Note: Whisper API needs a file, so we'd need to save temporarily
                # For now, skip transcription in serverless
                transcript = "[Audio transcription skipped in serverless environment]"
            except Exception as e:
                print(f"Audio transcription error: {e}")
                transcript = "[Audio transcription failed]"
        
        # Step 2: Find similar content with GPT-4V
        similar_prompt = """You are analyzing video content to identify what it resembles or derives from.
Examine these frames. Identify:
- What copyrighted work(s) this video appears to use
- How much of the original is present
- The nature of the original work

Respond in JSON format:
{
  "summary": "brief description",
  "identified_works": [
    {"title": "work name", "creator": "creator", "confidence": "high/medium/low", "evidence": "what you observed"}
  ]
}"""
        
        # Take first 10 frames to stay within limits
        frame_messages = [{"type": "text", "text": similar_prompt}]
        for i, frame in enumerate(frames[:10]):
            frame_messages.append({
                "type": "image_url",
                "image_url": {"url": frame}
            })
        
        similar_response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[{"role": "user", "content": frame_messages}],
            max_tokens=1000
        )
        
        similar_content = json.loads(similar_response.choices[0].message.content)
        
        # Step 3: Evaluate fair use
        fair_use_prompt = f"""You are a fair-use assessment tool. Given video frames, transcript, and identified source material,
evaluate fair-use risk across four factors.

Transcript: {transcript}

Similar Content Found: {json.dumps(similar_content)}

For each factor, provide a score from 0-100 (0=strong fair use, 100=high infringement risk) and explanation.

Respond in JSON format:
{{
  "overall_risk_score": 0-100,
  "risk_level": "Low Risk" | "Moderate Risk" | "High Risk",
  "confidence_score": 0-100,
  "factors": {{
    "purpose_and_character": {{"score": 0-100, "explanation": "..."}},
    "nature_of_work": {{"score": 0-100, "explanation": "..."}},
    "amount_and_substantiality": {{"score": 0-100, "explanation": "..."}},
    "market_effect": {{"score": 0-100, "explanation": "..."}}
  }}
}}"""
        
        fair_use_messages = [{"type": "text", "text": fair_use_prompt}]
        for i, frame in enumerate(frames[:10]):
            fair_use_messages.append({
                "type": "image_url",
                "image_url": {"url": frame}
            })
        
        fair_use_response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[{"role": "user", "content": fair_use_messages}],
            max_tokens=2000
        )
        
        fair_use_eval = json.loads(fair_use_response.choices[0].message.content)
        
        return jsonify({
            "analysis": {
                "similar_content": similar_content,
                "fair_use_evaluation": fair_use_eval,
                "transcript": transcript
            }
        }), 200
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"Analysis failed: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

# This is the entry point for Vercel
if __name__ == "__main__":
    app.run(debug=False)
