import torch
from model_def import QuizNet

INPUT_DIM = 20  # match number of quiz questions

model = QuizNet(INPUT_DIM)

# initialize with random weights
torch.save(model.state_dict(), "model.pt")

print("Dummy model created: model.pt")
