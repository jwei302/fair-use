# Fair Use Evaluation Tool

**By Eliot Chang, Jeffrey Wei, and Leo Wylonis**  
**Course:** CPSC 1830 - Law, Technology, and Culture  
**Live Demo:** [https://fair-use-f373.vercel.app/](https://fair-use-f373.vercel.app/)

---

## Why We Built This

Fair use is one of the most important and confusing parts of copyright law. If you make or post videos online, you have probably asked yourself some version of "Is this fair use or will it get taken down?" The law does not give a simple checklist but rather looks at flexible factors and balances them in context.

This calculator is meant to turn the abstract legal test into something you can explore interactively. Instead of giving a definitive answer of whether something is fair use, the calculator will help you think through how the fair use factors apply to a specific video and where your choices might create more or less risk.

This calculator is focused on videos because that is where a lot of real-world fair use questions show up—such as reaction videos, commentary, parodies, fan edits, and remixes that reuse other people's clips or audio.

---

## Project Overview

This project consists of **three main components**:

### 1. **Interactive Fair Use Calculator**
An interactive quiz that walks users through the four statutory fair use factors plus a "fifth factor" (good/bad faith). Users answer 25 questions on a 1-9 sliding scale to receive:
- Individual factor scores (1-9 for each of the five factors)
- An overall fair use risk score
- Detailed explanations for each factor

### 2. **Machine Learning Predictor**
A logistic regression model trained on expert-labeled court cases that predicts fair use outcomes based on the 25-question responses. This allows us to:
- Identify which questions/factors are most predictive of fair use outcomes
- Compare human intuition (manual scoring) with data-driven predictions
- Understand the relative importance of different fair use factors in real cases

### 3. **Vision-Language Model (VLM) Analyzer**
A research tool that uses GPT-4V (GPT-4 with Vision) and OpenAI Whisper to analyze uploaded videos and assess fair use risk. This explores:
- **Video Understanding:** Can multimodal AI comprehend video content from frames and audio?
- **Content Identification:** Can AI identify copyrighted source material?
- **Legal Reasoning:** Can AI meaningfully apply the four-factor fair use test?

---

## How the Calculator Works

### The 1-9 Sliding Scale Approach

For each factor, you will answer five short questions about your video. Every question uses a sliding scale from **1-9** where:
- **1** = "this choice is very friendly to fair use"
- **9** = "this choice looks more like infringement risk"

For example, for the "amount used" factor, a question might ask how much of the original audio you use. Sliding to 1 means "none of the original audio," and sliding to 9 means "the entire original audio track."

We chose a 1-9 scale instead of yes/no answers because **fair use is almost never all or nothing**. Courts talk about uses being more or less transformative, more or less commercial, or using more or less of the original. As Professor Brad Rosen repeatedly put it, "Where do you draw the line?" A slider lets you place your video somewhere on the spectrum instead of forcing a simple yes or no.

### The Five Factors

**Factor 1 - Purpose and Character of Use:**  
Is your use transformative? Does it add new meaning or message? Is it commercial or educational?

**Factor 2 - Nature of the Copyrighted Work:**  
Are you using creative fiction or factual/informational content? Published or unpublished?

**Factor 3 - Amount and Substantiality:**  
How much of the original do you use? Did you take the "heart" of the work?

**Factor 4 - Effect on the Market:**  
Could your video substitute for the original? Does it harm current or potential markets?

**Factor 5 - Good/Bad Faith (The "Fifth Factor"):**  
Are you acting fairly and in good faith, or exploitatively?

### Scoring

After answering all 25 questions (5 per factor), the calculator:
1. **Averages** the five answers for each factor to produce 5 factor scores (1-9)
2. **Combines** the factor scores into an overall risk score using a weighted average:
   - Factor 1 (Purpose) and Factor 4 (Market Effect) are weighted more heavily
   - Factors 2 and 3 carry moderate weight
   - Factor 5 carries slightly less weight

This weighting reflects how courts often treat transformation and market harm as especially important.

Your resulting score falls into three bands:
- **1-3:** Leans toward fair use
- **4-6:** Mixed or uncertain
- **7-9:** Leans against fair use

---

## Technical Implementation

### 1. Interactive Calculator (Frontend)

**Technologies:** HTML, CSS, JavaScript  
**Design:** Responsive, modern UI with gradient cards and smooth animations

The quiz interface is built entirely client-side for simplicity and instant feedback. Each of the 25 questions is a range slider (1-9) with explanatory text. After submission, JavaScript calculates:
- Factor-level scores (average of 5 questions per factor)
- Weighted overall score
- Risk classification (Low/Medium/High)
- Color-coded visual feedback

The calculator also displays detailed explanations for each factor based on the user's responses, helping users understand how their choices map to fair use doctrine.

### 2. Machine Learning Model

**Technologies:** Python, Flask, scikit-learn, NumPy  
**Model:** Logistic Regression  
**Data Storage:** SQLite database

#### Why Logistic Regression?

We chose logistic regression for several reasons:
1. **Interpretability:** We can see exactly which questions contribute most to the prediction
2. **Simplicity:** Easy to train with limited data (10-20 labeled cases)
3. **Binary Classification:** Perfect for "fair use" vs. "not fair use" outcomes
4. **Feature Importance:** Allows us to analyze which factors courts weigh most heavily

#### How It Works

1. **Data Collection:** Users (or instructors) can enter court cases into the "Training Data" tab by:
   - Naming the case (e.g., "Campbell v. Acuff-Rose Music")
   - Answering all 25 questions as the court would have
   - Labeling the outcome (Fair Use = 1, Not Fair Use = 0)

2. **Training:** The Flask backend receives training data via `/api/add-case`, stores it in SQLite, and exposes a `/api/train` endpoint. When triggered:
   - Fetches all labeled cases from the database
   - Converts the 25 answers into feature vectors
   - Trains a logistic regression classifier
   - Saves the trained model as `logistic_model.pkl`

3. **Prediction:** When a user completes the quiz, the frontend sends their 25 answers to `/predict`. The backend:
   - Loads the trained model
   - Predicts the probability of fair use (0-1)
   - Returns the score to display alongside the manual score

4. **Feature Analysis:** The model's coefficients reveal which questions are most predictive. In our initial training with 10 expert cases, we found:
   - **Top 5 Questions:** Q15 (Substitute for original), Q13 (Heart of work), Q16 (Replace original), Q20 (Licensing markets), Q18 (Effect on sales/views)
   - **Pattern:** Most predictive questions come from **Factor 4 (Market Effect)**, suggesting this factor has historically been most decisive

This aligns with legal scholarship noting that market harm is often dispositive in fair use cases.

### 3. Vision-Language Model Analyzer

**Technologies:** Python, Flask, OpenAI API (GPT-4V + Whisper), HTML5 Canvas API, Web Audio API  
**Deployment:** Vercel serverless functions

#### Motivation

Can multimodal AI systems assess fair use? As VLMs become more sophisticated, we wanted to explore:
- Whether AI can identify copyrighted source material from video frames
- Whether AI can apply nuanced legal tests like "transformativeness"
- Whether AI explanations are grounded in observable evidence or hallucinated

This is a **research question**, not a production tool. We're evaluating the capabilities and limitations of state-of-the-art VLMs in legal reasoning contexts.

#### How It Works

**Step 1: Video Processing (Client-Side)**
- User uploads an MP4 video (max 200MB recommended)
- JavaScript extracts frames at 1 fps using HTML5 `<video>` and `<canvas>` APIs (max 30 frames)
- Frames are encoded as base64 JPEG images (768×768px)
- The raw video file is also encoded as base64 for audio transcription

**Step 2: Content Identification (Server-Side)**
- Backend sends 8 representative frames to **GPT-4V (gpt-4o)** with a prompt asking:
  - What copyrighted work(s) does this video use?
  - What is the confidence level?
  - What evidence supports this identification?
- Model returns structured JSON with identified works

**Step 3: Audio Transcription (Server-Side)**
- Backend sends the video file to **OpenAI Whisper API** for speech-to-text
- Whisper supports all common video codecs (MP4, MOV, etc.)
- Returns full transcript of spoken content

**Step 4: Fair Use Evaluation (Server-Side)**
- Backend sends 10 frames (high detail), transcript, and identified content to **GPT-4V** with a detailed prompt that:
  - **Defines** each of the four fair use factors (1-2 sentences each)
  - Provides **scoring guidance** (0-100 scale, where 0=strong fair use, 100=high infringement risk)
  - Includes an **in-context learning example** (worked-through scenario with expected output)
  - Requests **structured JSON output** with scores and explanations for each factor

**Step 5: Results Display**
- Frontend displays:
  - Overall fair use risk score (0-100) with risk level (Low/Moderate/High)
  - Confidence score (reflects completeness of analysis)
  - Individual factor scores and detailed AI-generated explanations
  - Identified similar content from Stage 1

#### Design Rationale

- **Two-stage approach:** Separates fact-finding (content ID) from legal analysis, mirroring how human experts reason
- **In-context learning:** Provides a worked example to calibrate the model's scoring and explanation style
- **Low temperature (0.3):** Reduces randomness for consistent, deterministic assessments
- **JSON structure:** Forces organized output and enables reliable parsing
- **Confidence scoring:** Acknowledges when evidence is incomplete (e.g., no audio, unclear source)

### 4. Backend Architecture

**Technologies:** Flask, Flask-CORS, Vercel (serverless deployment)

The backend serves several purposes:
1. **API for Quiz ML Model:**
   - `/predict` - Returns logistic regression prediction
   - `/api/add-case` - Stores training cases
   - `/api/train` - Trains the model
   - `/api/get-cases` - Retrieves all training data

2. **API for VLM Video Analysis:**
   - `/api/analyze-video-client` - Receives frames and video, calls OpenAI APIs, returns analysis

3. **Static File Serving:**
   - Serves `index.html` and associated assets
   - Handles client-side routing for the quiz interface

**Deployment:**  
The project is deployed on [Vercel](https://vercel.com), which automatically deploys from the main branch. Serverless functions in `/api` handle backend logic, while the frontend is served as static HTML/CSS/JS.

---

## Project Structure

```
fair-use/
├── index.html              # Main frontend (all tabs: Quiz, VLM, Admin, Resources, Process)
├── backend/
│   ├── app.py             # Flask server for local development
│   ├── train_model.py     # Training script (local execution)
│   ├── model_def.py       # Model definition and prediction logic
│   ├── training_data.db   # SQLite database (created automatically)
│   ├── logistic_model.pkl # Trained model (created after training)
│   └── requirements.txt   # Backend Python dependencies
├── api/
│   ├── index.py           # Vercel serverless function (production backend)
│   ├── requirements.txt   # Vercel Python dependencies
│   └── logistic_model.pkl # Trained model (copied to Vercel)
├── vercel.json            # Vercel configuration
├── .vercelignore          # Files to exclude from Vercel deployment
└── README.md              # This file
```

**Key Files:**
- **`index.html`**: Single-page application with tabs for Quiz, VLM Analyzer, Training Data, Admin, Resources, and Process
- **`backend/app.py`**: Flask backend for local development (runs on `http://127.0.0.1:5000`)
- **`api/index.py`**: Production backend for Vercel deployment (serverless functions)
- **`backend/model_def.py`**: Contains logistic regression model code (imported by both Flask and Vercel)
- **`backend/training_data.db`**: SQLite database storing labeled court cases for training

---

## Running Locally

### Prerequisites

- Python 3.8 or later
- Web browser (Chrome, Firefox, Safari, Edge)
- (Optional) OpenAI API key for VLM video analysis

### Step 1: Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

Or manually:
```bash
pip install flask flask-cors scikit-learn numpy openai boto3
```

### Step 2: Set Up Environment Variables (Optional, for VLM)

Create a `.env` file in the project root:

```bash
# OpenAI API Key (required for video analysis)
OPENAI_API_KEY=your-openai-api-key-here
```

### Step 3: Start the Flask Backend

```bash
cd backend
python app.py
```

You should see:
```
Running on http://127.0.0.1:5000
```

**Keep this terminal window open.**

### Step 4: Open the Website

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

The quiz will load with tabs: **Quiz**, **VLM Analyzer**, **Training Data**, **Admin**, **Resources**, **Process**.

---

## How to Use the System

### Taking the Quiz

1. Click the **"Quiz"** tab (default view)
2. Answer all 25 questions using the sliders (1-9 scale)
3. Click **"Evaluate Fair Use"**
4. View your results:
   - **Manual score** (calculated from your answers)
   - **AI model score** (logistic regression prediction, if model is trained)
   - Factor-by-factor breakdown with explanations

### Adding Training Data

1. Click the **"Training Data"** tab
2. Enter a court case name (e.g., "Campbell v. Acuff-Rose Music, 510 U.S. 569 (1994)")
3. Answer all 25 questions as the court would have
4. Select **Fair Use** or **Not Fair Use**
5. Click **"Save Case"** - you'll see a success message

### Training the Model

1. Add at least 2-4 training cases using the Training Data tab
2. Go to the **"Admin"** tab
3. Click **"Train Model"** button
4. Wait a few seconds - you'll see a success message
5. The model is now ready! Return to the **Quiz** tab to get AI predictions

### Using the VLM Analyzer

**Note:** This feature requires an OpenAI API key and incurs costs (~$0.20-0.50 per video).

1. Click the **"VLM Analyzer"** tab
2. Read the research question and methodology
3. Click **"Choose Video File"** and select an MP4 video (max 200MB)
4. Click **"Analyze Video"**
5. Wait 30-60 seconds for processing
6. View results:
   - Overall fair use risk score (0-100)
   - Confidence score
   - Factor-by-factor analysis with AI-generated explanations
   - Identified similar content

---

## Deployment

The project is deployed on **Vercel** at [https://fair-use-f373.vercel.app/](https://fair-use-f373.vercel.app/).

To deploy your own instance:

1. Fork this repository
2. Sign up for [Vercel](https://vercel.com)
3. Import your forked repository
4. Add environment variables in Vercel dashboard (if using VLM):
   - `OPENAI_API_KEY` - Your OpenAI API key
5. Deploy!

Vercel will automatically redeploy on every push to the main branch.

---

## Disclaimer

**This tool is for educational and research purposes only. It is NOT legal advice.**

The scores, predictions, and explanations provided by this tool are heuristic assessments based on:
- Typical patterns in fair use case law
- Machine learning trained on limited expert-labeled data
- AI models that use pattern recognition, not legal expertise

**Lower risk scores do not guarantee fair use. Higher risk scores do not prove infringement. Only courts can make definitive fair use determinations.**

If you have a real copyright dispute or legal question, consult a qualified attorney. This tool cannot and does not replace professional legal counsel.

For the VLM video analysis feature specifically: The AI's assessment is based on observable video content and pattern recognition, not legal expertise. Treat all scores as experimental research outputs exploring the capabilities of multimodal AI, not as guidance for actual copyright decisions.