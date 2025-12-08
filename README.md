# Fair Use Evaluation Quiz

This project is a simple website that asks 25 questions to help you think about whether a use of copyrighted material might qualify as **fair use**. It provides:

- An overall manual score (percentage)
- Scores and explanations for each of the four fair use factors
- A fifth “other considerations” score
- An optional machine-learning score (using PyTorch)

You **do not need programming experience** to run it.

---

## Files in the Project

```text
quiz.html              ← main website (open in browser)
app.py                 ← optional backend to run ML model
model_def.py           ← neural network definition
create_dummy_model.py  ← creates a sample ML model
model.pt               ← saved PyTorch model (created by the script)
```

## Requirements
- Python 3.8 or later  
- A web browser (Chrome, Firefox, Safari, Edge, etc.)  
- Internet is **not** required after install  
- No hosting, no servers, no paid services  

---

## 1. Install Dependencies

### Windows
1. Open **Command Prompt** (search for “cmd” in the Start menu).
2. Run:
   ```bash
   pip install flask torch flask-cors
   ```

### Mac
1. Open **Terminal** (Applications → Utilities → Terminal).
2. Run:
   ```bash
   pip install flask torch flask-cors
   ```

## 2. Create the Model (First Time Only)

In the same folder as `create_dummy_model.py`, run:

### Windows
```bash
python create_dummy_model.py
```

### Mac
```bash
python create_dummy_model.py
# or, if needed:
# python3 create_dummy_model.py
```

## 3. Start the ML Backend (Optional)

If you want the AI score:

### Windows
```bash
python app.py
```

### Mac
```bash
python app.py
# or:
# python3 app.py
```

You should see something like:
```bash
Running on http://127.0.0.1:5000
```

Leave this window open while using the quiz.

If you skip this step, the quiz still works using the manual score only.
The page will simply say the AI score is “not available”.

## 4. Open the Quiz Website

- Locate `quiz.html` in your file explorer (Finder on Mac, File Explorer on Windows).
- Double-click `quiz.html`.
- It should open in your default web browser.
- Move the sliders, then click **“Evaluate Fair Use”** to see your results.

---

## Editing Questions or Explanations

If you want to customize the quiz:

1. Open `quiz.html` in a text editor (VS Code, Notepad, TextEdit, etc.).
2. Look for sections labeled **CONFIGURATION** or the question texts.

You can change:
- Question wording  
- Which questions belong to which factor  
- Weights for each question  
- Descriptions for low / medium / high scores  

Save the file and refresh the browser to apply changes.

---

## Troubleshooting

- **`pip: command not found`**  
  → On Mac, try `pip3`. On Windows, make sure Python is added to PATH or reinstall Python from python.org.

- **`ModuleNotFoundError: No module named 'flask' or 'torch'`**  
  → Run `pip install flask torch` again.

- **Browser says “AI score is not available”**  
  → Make sure `app.py` is running in a terminal window.

---

## Important Note

This tool is for **education and discussion only**.  
It is **not legal advice**, and it does **not** tell you definitively whether a use is or is not fair use.  
Only a court can do that.





