# Data Collection System - Setup Instructions

## Quick Start

### 1. Install Dependencies
Make sure you're in the project directory and run:

```bash
pip install -r backend/requirements.txt
```

Or manually:
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
- Go back to the main project directory
- Open `quiz.html` in your web browser (double-click it)
- The quiz will load with the new tabs: Quiz, Training Data, Admin, Resources, Process

## How to Use

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

**Important:** After training, restart the Flask server to load the new model:
- Stop the server (Ctrl+C)
- Run `python app.py` again

### Viewing/Managing Data (Admin)
1. Click the **"Admin"** tab
2. You'll see statistics and a list of all saved cases
3. Click **"Delete"** next to any case to remove it
4. Click **"Refresh Data"** to reload the list
5. Click **"Train Model"** to retrain the model with current data
6. Click **"Export JSON"** to download all data as a JSON file

### Using AI Predictions
Once you've trained a model:
1. Go to the **"Quiz"** tab
2. Answer all 25 questions
3. Click **"Evaluate Fair Use"**
4. You'll see both:
   - **Manual score** (calculated from your answers)
   - **AI model score** (logistic regression prediction)

### Exporting Data
- Go to the **Admin** tab
- Click **"Export JSON"**
- A file will download named `training_data_YYYY-MM-DD.json`
- This JSON contains all cases with their answers and labels

## Database & Model Storage

**Database:** The system uses SQLite and creates a file called `training_data.db` in the `backend` directory. This file stores all your training data locally on your computer.

**Trained Model:** After training, a file called `logistic_model.pkl` is created in the `backend` directory. This contains your trained logistic regression model.

## File Structure

```
fair_use_calculator/
├── quiz.html              ← Main website (open in browser)
├── backend/
│   ├── app.py            ← Flask server (run this)
│   ├── train_model.py    ← Training script (can run manually)
│   ├── model_def.py      ← Logistic regression model code
│   ├── training_data.db  ← Database (created automatically)
│   ├── logistic_model.pkl ← Trained model (created after training)
│   └── requirements.txt  ← Dependencies
```

## Troubleshooting

- **"Could not connect to server"** → Make sure `python backend/app.py` is running
- **Database errors** → The database file will be created automatically on first use
- **CORS errors** → Make sure `flask-cors` is installed: `pip install flask-cors`

## API Endpoints

The backend provides these endpoints:
- `POST /predict` - Get AI prediction (requires trained model)
- `POST /api/add-case` - Save a new case
- `GET /api/cases` - Get all cases
- `DELETE /api/cases/<id>` - Delete a case
- `GET /api/export-cases` - Export all cases as JSON
- `POST /api/train-model` - Train the logistic regression model

