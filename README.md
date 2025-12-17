# Fair Use Evaluation Quiz

This project is a simple website that asks 25 questions to help you think about whether a use of copyrighted material might qualify as **fair use**. It provides:

- An overall manual score (percentage)
- Scores and explanations for each of the four fair use factors
- A fifth “other considerations” score
- An optional machine-learning score (using PyTorch)

We also explore as a research question VLMs 

You **do not need programming experience** to run it.

---

## Files in the Project

```text
fair_use_calculator/
├── quiz.html              ← Main website (open in browser)
├── index.html             ← Alternative entry point
├── backend/
│   ├── app.py            ← Flask server (run this)
│   ├── train_model.py    ← Training script (can run manually)
│   ├── model_def.py      ← Logistic regression model code
│   ├── training_data.db  ← Database (created automatically)
│   ├── logistic_model.pkl ← Trained model (created after training)
│   └── requirements.txt  ← Dependencies
```

## Requirements
- Python 3.8 or later  
- A web browser (Chrome, Firefox, Safari, Edge, etc.)  
- Internet is **not** required after install  
- No hosting, no servers, no paid services  

---

## Quick Start Guide

### 1. Install Dependencies

Make sure you're in the project directory and run:

```bash
pip install -r backend/requirements.txt
```

Or manually install:
```bash
pip install flask flask-cors scikit-learn numpy
```

### 2. Start the Flask Backend

Navigate to the backend directory and run:

```bash
cd backend
python app.py
```

You should see:
```
Running on http://127.0.0.1:5000
```

**Keep this terminal window open** while using the data collection system.

### 3. Open the Quiz Website

- Locate `quiz.html` (or `index.html`) in your file explorer (Finder on Mac, File Explorer on Windows)
- Double-click the file to open it in your default web browser
- The quiz will load with tabs: Quiz, Training Data, Admin, Resources, Process

---

## How to Use the System

### Taking the Quiz
1. Click the **"Quiz"** tab (default view)
2. Answer all 25 questions using the sliders (1-9 scale)
3. Click **"Evaluate Fair Use"**
4. You'll see both:
   - **Manual score** (calculated from your answers)
   - **AI model score** (logistic regression prediction, if model is trained)

### Adding Training Data
1. Click the **"Training Data"** tab
2. Enter a court case name (e.g., "Campbell v. Acuff-Rose Music, 510 U.S. 569 (1994)")
3. Answer all 25 questions using the sliders (1-9 scale)
4. Select whether the case was ruled as **Fair Use** or **Not Fair Use**
5. Click **"Save Case"** - you'll see a success message

### Training the Model

The system uses **Logistic Regression** (simple, interpretable ML model).

**Option 1: Train from Web Interface (Easiest)**
1. Add at least 2-4 training cases using the Training Data tab
2. Go to the **"Admin"** tab
3. Click **"Train Model"** button
4. Wait a few seconds - you'll see a success message when done
5. The model is now ready! Use the **Quiz** tab to get AI predictions

**Option 2: Train from Command Line**
```bash
cd backend
python train_model.py
```

**Important:** After training via command line, restart the Flask server to load the new model:
- Stop the server (Ctrl+C)
- Run `python app.py` again

### Viewing/Managing Data (Admin)
1. Click the **"Admin"** tab
2. You'll see statistics and a list of all saved cases
3. Click **"Delete"** next to any case to remove it
4. Click **"Refresh Data"** to reload the list
5. Click **"Train Model"** to retrain the model with current data
6. Click **"Export JSON"** to download all data as a JSON file

### Exporting Data
- Go to the **Admin** tab
- Click **"Export JSON"**
- A file will download named `training_data_YYYY-MM-DD.json`
- This JSON contains all cases with their answers and labels

---

## Database & Model Storage

**Database:** The system uses SQLite and creates a file called `training_data.db` in the `backend` directory. This file stores all your training data locally on your computer.

**Trained Model:** After training, a file called `logistic_model.pkl` is created in the `backend` directory. This contains your trained logistic regression model.

---

## API Endpoints

The backend provides these endpoints:
- `POST /predict` - Get AI prediction (requires trained model)
- `POST /api/add-case` - Save a new case
- `GET /api/cases` - Get all cases
- `DELETE /api/cases/<id>` - Delete a case
- `GET /api/export-cases` - Export all cases as JSON
- `POST /api/train-model` - Train the logistic regression model

---

## Customizing the Quiz

If you want to customize the quiz:

1. Open `quiz.html` or `index.html` in a text editor (VS Code, Notepad, TextEdit, etc.).
2. Look for sections labeled **CONFIGURATION** or the question texts.

You can change:
- Question wording  
- Which questions belong to which factor  
- Weights for each question  
- Descriptions for low / medium / high scores  

Save the file and refresh the browser to apply changes.

---

## Troubleshooting

### Common Issues

- **`pip: command not found`**  
  → On Mac, try `pip3`. On Windows, make sure Python is added to PATH or reinstall Python from python.org.

- **`ModuleNotFoundError: No module named 'flask'`**  
  → Run `pip install flask flask-cors` again.

- **Browser says "AI score is not available"**  
  → Make sure `python backend/app.py` is running in a terminal window.

- **"Could not connect to server"**  
  → Make sure `python backend/app.py` is running and you see "Running on http://127.0.0.1:5000"

- **Database errors**  
  → The database file will be created automatically on first use. Make sure you have write permissions in the backend directory.

- **CORS errors**  
  → Make sure `flask-cors` is installed: `pip install flask-cors`

### PyTorch DLL Error on Windows

If you see: `OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed`

**Good News:** The data collection system will still work! The backend will start and you can collect training data. Only the ML prediction feature won't work until PyTorch is fixed.

#### Quick Fixes (try in order):

**Option 1: Reinstall PyTorch (Recommended)**
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Option 2: Install CPU-only version**
```bash
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Option 3: Install Visual C++ Redistributables**  
Download and install from Microsoft:  
https://aka.ms/vs/17/release/vc_redist.x64.exe

**Option 4: Try a different PyTorch version**
```bash
pip uninstall torch
pip install torch==2.0.1
```

**Option 5: Deactivate and recreate virtual environment**
```bash
deactivate
rmdir /s .venv
python -m venv .venv
.venv\Scripts\activate
pip install flask flask-cors torch
```

#### Verify It Works
After trying a fix, run:
```bash
python backend/app.py
```

You should see:
```
✓ ML model loaded successfully
Running on http://127.0.0.1:5000
```

Or if model loading fails but data collection works:
```
Warning: Could not load ML model (...). Data collection features will still work.
Running on http://127.0.0.1:5000
```

Both are fine! The second just means ML predictions won't work, but you can still collect training data.

---

---

## Video Analysis Feature (Research Tool)

### Overview

This project now includes a **Video Fair-Use Risk Analysis** tool that uses state-of-the-art AI to analyze video content and assess fair-use risk.

**Research Question:** How well can vision-language models (VLMs) assess fair-use risk for video content?

This tool evaluates:
- **Video Understanding:** Can GPT-4V analyze frames and audio to understand content?
- **Content Identification:** Can it identify similar/source copyrighted material?
- **Legal Reasoning:** Can it apply the four statutory fair-use factors?

### Setup Requirements

#### 1. OpenAI API Key (Required)

Get an API key from [OpenAI Platform](https://platform.openai.com/api-keys)

**Cost Estimates:**
- GPT-4V (gpt-4o): ~$0.01-0.03 per frame
- Whisper: ~$0.006 per minute of audio
- **Total per video:** ~$0.20-0.50 for a 2-minute video with 30 frames

#### 2. AWS S3 Setup (Required)

**Create an S3 Bucket:**
1. Go to AWS Console → S3 → Create Bucket
2. Choose a name (e.g., `fair-use-videos`)
3. Select region (e.g., `us-east-1`)
4. Keep "Block all public access" **enabled** (we use presigned URLs)

**Configure CORS:**
Go to bucket → Permissions → CORS, add:
```json
[
    {
        "AllowedHeaders": ["*"],
        "AllowedMethods": ["PUT", "POST", "GET"],
        "AllowedOrigins": ["*"],
        "ExposeHeaders": ["ETag"],
        "MaxAgeSeconds": 3000
    }
]
```

**Create IAM User:**
1. Go to IAM → Users → Create User
2. Attach policy: Create custom policy with:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject",
                "s3:DeleteObject"
            ],
            "Resource": "arn:aws:s3:::YOUR-BUCKET-NAME/*"
        }
    ]
}
```
3. Create access keys and save them securely

#### 3. FFmpeg (Required)

**Mac:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html)

#### 4. Environment Variables

Create `backend/.env` file:
```bash
# OpenAI
OPENAI_API_KEY=sk-your-key-here

# AWS S3
AWS_ACCESS_KEY_ID=your-access-key-id
AWS_SECRET_ACCESS_KEY=your-secret-access-key
AWS_S3_BUCKET_NAME=your-bucket-name
AWS_REGION=us-east-1
```

**Security Note:** Never commit `.env` to git! It's already in `.gitignore`.

### Installation

Install all dependencies including video analysis:
```bash
cd backend
pip install -r requirements.txt
```

This installs:
- `openai` - OpenAI API client
- `boto3` - AWS SDK for S3
- `opencv-python-headless` - Video frame extraction
- `ffmpeg-python` - Audio extraction

### Usage

1. Start the Flask backend:
```bash
cd backend
python app.py
```

2. Open `video-analysis.html` in your browser, or navigate via the main site navbar

3. Upload an MP4 video (max 200MB recommended)

4. Wait for processing (typically 30-90 seconds):
   - Video uploads to S3
   - Frames extracted (1-2 FPS, max 30 frames)
   - Audio transcribed with Whisper
   - Content analyzed with GPT-4V
   - Fair-use evaluation generated

5. Review results:
   - Overall risk score (0-100)
   - Similar content identified
   - Four-factor analysis with explanations
   - Confidence/completeness score

### How It Works

1. **Upload:** Video uploads directly to S3 via presigned URL (bypasses Flask/Vercel 4.5MB limit)
2. **Processing:** Backend downloads video, extracts frames (1-2 FPS) and audio
3. **Transcription:** Whisper API transcribes audio to text
4. **Similarity Detection:** GPT-4V analyzes frames to identify source material
5. **Fair-Use Evaluation:** GPT-4V evaluates all four statutory factors:
   - Factor 1: Purpose & Character of Use
   - Factor 2: Nature of Copyrighted Work
   - Factor 3: Amount & Substantiality
   - Factor 4: Effect on Market
6. **Results:** Comprehensive report with risk scores and explanations

### API Endpoints (Video Analysis)

- `GET /api/get-upload-url` - Get presigned S3 URL for video upload
- `POST /api/analyze-video` - Analyze uploaded video
  - Body: `{"video_key": "videos/..."}`
  - Returns: Full analysis with scores and explanations

### Cost Management

**Typical costs per video:**
- Whisper (2 min audio): ~$0.012
- GPT-4V (30 frames): ~$0.30-0.60
- S3 storage: <$0.01/GB/month
- **Total:** ~$0.35-0.65 per video

**Cost optimization tips:**
- Limit to first 2 minutes of long videos
- Sample fewer frames (10-20 instead of 30)
- Use "low detail" mode for some frames
- Set S3 lifecycle policy to auto-delete after 1 day

### Limitations & Important Notes

**⚠️ CRITICAL: This is a RESEARCH TOOL, not legal advice**

- Scores are **heuristic assessments** only
- Lower risk ≠ fair use; higher risk ≠ infringement
- Only a court can make binding fair-use determinations
- Do NOT rely on this for actual copyright decisions

**Technical Limitations:**
- Vercel timeout: 60s (Pro tier) or 10s (Hobby tier)
- Longer videos may require separate backend (not Vercel Functions)
- Maximum video size: ~500MB (constrained by `/tmp` space)
- Frame analysis limited to visual content (no OCR/detailed text reading)

### Troubleshooting Video Analysis

**"Cloud storage not configured"**
→ Check that all AWS environment variables are set in `.env`

**"FFmpeg not found"**
→ Install ffmpeg: `brew install ffmpeg` (Mac) or `apt install ffmpeg` (Linux)

**"Failed to download video"**
→ Check S3 bucket permissions and IAM user access keys

**"OpenAI API error"**
→ Verify `OPENAI_API_KEY` is correct and has credits

**Processing times out**
→ Try shorter video or fewer frames. Consider running Flask on dedicated server instead of Vercel.

**High API costs**
→ Reduce `max_frames` in `video_processor.py` or `fps_sample_rate`

---

## Important Note

This tool is for **education and discussion only**.  
It is **not legal advice**, and it does **not** tell you definitively whether a use is or is not fair use.  
Only a court can do that.

**For the video analysis feature specifically:**  
The AI's assessment is based on pattern recognition and training data, not legal expertise. Treat all scores as experimental research outputs, not legal guidance.





