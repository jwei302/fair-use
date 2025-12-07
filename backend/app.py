from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from model_def import QuizNet

app = Flask(__name__)
CORS(app)   # <-- REQUIRED so browser is allowed to POST from quiz.html

INPUT_DIM = 20  # number of quiz inputs (later change to 20)

# Load model
model = QuizNet(INPUT_DIM)
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    answers = data["answers"]  # list of numbers
    
    x = torch.tensor([answers], dtype=torch.float32)
    with torch.no_grad():
        score = model(x).item()

    return jsonify({"score": score})

if __name__ == "__main__":
    app.run(port=5000, debug=False)
