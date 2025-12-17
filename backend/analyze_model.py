"""Analyze feature importance from trained logistic regression model"""
import pickle
import numpy as np
import os

MODEL_FILE = "logistic_model.pkl"

if not os.path.exists(MODEL_FILE):
    print("❌ No trained model found!")
    print("   Train a model first using train_model.py")
    exit(1)

# Load model
with open(MODEL_FILE, 'rb') as f:
    model = pickle.load(f)

# Get coefficients
coefficients = model.coef_[0]

# Question texts
question_texts = {
    1: "Transformative nature", 2: "Comment/critique/parody", 3: "Commercial vs noncommercial",
    4: "Creative effort added", 5: "Intent", 6: "Creativity of original", 7: "Published status",
    8: "Finished work", 9: "Creative vs factual parts", 10: "Type of work",
    11: "Amount of video used", 12: "Amount of audio used", 13: "Heart of work",
    14: "Continuous vs broken", 15: "Substitute for original", 16: "Replace original",
    17: "Same audience", 18: "Effect on sales/views", 19: "Same market space",
    20: "Licensing markets", 21: "Tone", 22: "Judge/jury perception",
    23: "Honesty about relationship", 24: "Disclaimers", 25: "Creator perception"
}

# Calculate feature importance (absolute value of coefficients)
feature_importance = [(abs(coef), q_num, coef) for q_num, coef in enumerate(coefficients, 1)]
feature_importance.sort(reverse=True)

print("="*70)
print("LOGISTIC REGRESSION MODEL - FEATURE IMPORTANCE")
print("="*70)
print("\nTop Questions by Impact (highest to lowest):")
print("-"*70)
for rank, (abs_coef, q_num, coef) in enumerate(feature_importance, 1):
    direction = "favors fair use (lower values)" if coef < 0 else "favors infringement (higher values)"
    print(f"{rank:2d}. Q{q_num:2d} (coef={abs_coef:6.3f}): {question_texts[q_num]:35s} - {direction}")

print("\n" + "="*70)
print("Interpretation:")
print("  - Negative coefficient: Lower values (closer to 1) favor fair use")
print("  - Positive coefficient: Higher values (closer to 9) favor infringement")
print("  - Larger absolute value = bigger impact on prediction")
print("="*70)

