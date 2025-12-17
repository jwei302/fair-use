# Fair Use Evaluation Tool

**By Eliot Chang, Jeffrey Wei, and Leo Wylonis**  
**Course:** CPSC 1830 - Law, Technology, and Culture  
**Live Demo:** [https://fair-use-f373.vercel.app/](https://fair-use-f373.vercel.app/)

---

## Why We Built This

Fair use is one of the most important and confusing parts of copyright law. If you make or post videos online, you have probably asked yourself some version of "Is this fair use or will it get taken down?" The law does not give a simple checklist but rather looks at flexible factors and balances them in context.

This calculator turns the abstract legal test into something you can explore interactively. Instead of giving a definitive answer, it helps you think through how the fair use factors apply to a specific video and where your choices might create more or less risk. We focused on videos because that is where many real-world fair use questions arise—reaction videos, commentary, parodies, fan edits, and remixes that reuse other people's clips or audio.

---

## Project Overview

### 1. Interactive Fair Use Calculator
An interactive quiz that walks users through the four statutory fair use factors plus a "fifth factor" (good/bad faith). Users answer 25 questions on a 1-9 sliding scale to receive individual factor scores, an overall fair use risk score, and detailed explanations for each factor.

### 2. Machine Learning Predictor
A logistic regression model trained on expert-labeled court cases that predicts fair use outcomes. This allows us to identify which questions are most predictive of outcomes, compare human intuition with data-driven predictions, and understand the relative importance of different factors in real cases.

### 3. Vision-Language Model (VLM) Analyzer
A research tool using GPT-4V and OpenAI Whisper to analyze uploaded videos and assess fair use risk. This explores whether multimodal AI can comprehend video content, identify copyrighted source material, and meaningfully apply the four-factor fair use test.

---

## How the Calculator Works

### The 1-9 Sliding Scale

For each factor, you answer five questions using a sliding scale from **1** (very friendly to fair use) to **9** (looks more like infringement risk). For example, for "amount used," sliding to 1 means "none of the original audio" while 9 means "the entire original audio track."

We chose a 1-9 scale instead of yes/no answers because **fair use is almost never all or nothing**. Courts talk about uses being more or less transformative, more or less commercial, or using more or less of the original. As Professor Brad Rosen repeatedly put it, "Where do you draw the line?" A slider lets you place your video somewhere on the spectrum.

### The Five Factors

The calculator evaluates **Factor 1** (Purpose and Character), **Factor 2** (Nature of the Copyrighted Work), **Factor 3** (Amount and Substantiality), **Factor 4** (Effect on the Market), and **Factor 5** (Good/Bad Faith). After answering all 25 questions, the calculator averages the five answers for each factor and combines them using a weighted average where Factor 1 and Factor 4 are weighted more heavily, reflecting how courts often treat transformation and market harm as especially important. Results fall into three bands: 1-3 (leans toward fair use), 4-6 (mixed/uncertain), and 7-9 (leans against fair use).

---

## Technical Implementation

### Interactive Calculator (Frontend)
Built with HTML, CSS, and JavaScript, the quiz interface runs entirely client-side for simplicity and instant feedback. Each question is a range slider with explanatory text. After submission, JavaScript calculates factor-level scores, weighted overall score, risk classification, and color-coded visual feedback.

### Machine Learning Model
We chose **logistic regression** for its interpretability (we can see which questions contribute most), simplicity (easy to train with limited data), and suitability for binary classification. The workflow is: (1) users enter court cases with 25 answers and labels via the Training Data tab, (2) the Flask backend stores cases in SQLite and trains the model when triggered, (3) when users complete the quiz, the backend predicts fair use probability and returns it alongside the manual score, and (4) the model's coefficients reveal feature importance.

In our initial training with 10 expert cases, the top 5 most predictive questions were Q15 (Substitute for original), Q13 (Heart of work), Q16 (Replace original), Q20 (Licensing markets), and Q18 (Effect on sales/views). Most predictive questions come from **Factor 4 (Market Effect)**, suggesting this factor has historically been most decisive—aligning with legal scholarship noting that market harm is often dispositive.

### Vision-Language Model Analyzer
This research tool explores whether multimodal AI can assess fair use. The pipeline has five steps: (1) client-side video processing extracts frames at 1 fps using HTML5 Canvas API (max 30 frames) and encodes the video as base64, (2) backend sends 8 frames to GPT-4V asking it to identify copyrighted works and return structured JSON, (3) backend sends the video to Whisper API for speech-to-text transcription, (4) backend sends 10 frames, transcript, and identified content to GPT-4V with a detailed prompt defining the four factors, providing scoring guidance (0-100 scale), including an in-context learning example, and requesting structured JSON output, and (5) frontend displays overall risk score, confidence score, individual factor scores with AI-generated explanations, and identified similar content.

The design uses a two-stage approach (separating content ID from legal analysis), in-context learning (to calibrate scoring style), low temperature 0.3 (for consistency), JSON structure (for reliable parsing), and confidence scoring (acknowledging incomplete evidence).

### Backend Architecture
The Flask backend serves multiple APIs: `/predict` (logistic regression prediction), `/api/add-case` (stores training cases), `/api/train` (trains model), `/api/get-cases` (retrieves training data), and `/api/analyze-video-client` (VLM video analysis). The project deploys on Vercel, which automatically deploys from the main branch using serverless functions in `/api` for backend logic and serving the frontend as static HTML/CSS/JS.

---

## Project Structure

```
fair-use/
├── index.html              # Main frontend (Quiz, VLM, Admin, Resources, Process tabs)
├── backend/
│   ├── app.py             # Flask server for local development
│   ├── train_model.py     # Training script
│   ├── model_def.py       # Model definition and prediction logic
│   ├── training_data.db   # SQLite database (auto-created)
│   ├── logistic_model.pkl # Trained model
│   └── requirements.txt   # Backend dependencies
├── api/
│   ├── index.py           # Vercel serverless function (production backend)
│   ├── requirements.txt   # Vercel dependencies
│   └── logistic_model.pkl # Trained model (copied to Vercel)
├── vercel.json            # Vercel configuration
└── README.md              # This file
```

---

## Running Locally

**Prerequisites:** Python 3.8+, web browser, and optionally an OpenAI API key for VLM analysis.

**Step 1: Install Dependencies**
```bash
cd backend
pip install -r requirements.txt
```

**Step 2: (Optional) Set Up Environment Variables**  
Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your-openai-api-key-here
```

**Step 3: Start Flask**
```bash
cd backend
python app.py
```
You should see `Running on http://127.0.0.1:5000`. Keep this terminal open.

**Step 4: Open Website**  
Navigate to `http://127.0.0.1:5000` in your browser.

---

## Usage

**Taking the Quiz:** Click the Quiz tab, answer all 25 questions using sliders, click "Evaluate Fair Use," and view your manual score, AI model score (if trained), and factor-by-factor breakdown.

**Training the Model:** Add at least 2-4 court cases via the Training Data tab (enter case name, answer 25 questions, select Fair Use or Not Fair Use, save). Then go to the Admin tab and click "Train Model." The model is now ready for predictions.

**Using VLM Analyzer:** This requires an OpenAI API key and costs ~$0.20-0.50 per video. Click the VLM Analyzer tab, read the methodology, choose an MP4 video (max 200MB), click "Analyze Video," and wait 30-60 seconds for results showing overall risk score, confidence score, factor analysis, and identified content.

---

## Deployment

The project is deployed on **Vercel** at [https://fair-use-f373.vercel.app/](https://fair-use-f373.vercel.app/). To deploy your own instance: fork the repo, sign up for Vercel, import your fork, add environment variables (`OPENAI_API_KEY` if using VLM), and deploy. Vercel auto-redeploys on every push to main.

---

## Disclaimer

**This tool is for educational and research purposes only. It is NOT legal advice.**

The scores and predictions are heuristic assessments based on typical patterns in fair use case law, machine learning trained on limited data, and AI pattern recognition—not legal expertise. **Lower risk scores do not guarantee fair use. Higher risk scores do not prove infringement. Only courts can make definitive fair use determinations.** If you have a real copyright dispute, consult a qualified attorney. This tool cannot replace professional legal counsel.

For the VLM feature specifically, the AI's assessment is based on observable video content and pattern recognition, not legal expertise. Treat all scores as experimental research outputs exploring the capabilities of multimodal AI.

---

## Acknowledgments

Developed for **CPSC 1830: Law, Technology, and Culture** at Yale University. Special thanks to Professor Brad Rosen for guidance on fair use doctrine, the U.S. Copyright Office and Stanford Fair Use Center for educational resources, and OpenAI for providing GPT-4V and Whisper APIs.

---

## Contact

**Eliot Chang, Jeffrey Wei, Leo Wylonis**  
CPSC 1830 - Law, Technology, and Culture, Yale University  
**Live Demo:** [https://fair-use-f373.vercel.app/](https://fair-use-f373.vercel.app/)
