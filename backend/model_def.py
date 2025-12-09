"""
Logistic Regression Model for Fair Use Prediction
Uses scikit-learn's LogisticRegression
"""
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np
import os

MODEL_FILE = "logistic_model.pkl"

def create_model():
    """Create a new logistic regression model"""
    # Using higher max_iter and solver for better convergence
    model = LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        random_state=42
    )
    return model

def save_model(model):
    """Save the trained model to disk"""
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to {MODEL_FILE}")

def load_model():
    """Load a trained model from disk"""
    if not os.path.exists(MODEL_FILE):
        return None
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    return model

def predict(model, answers):
    """
    Make a prediction using the model
    
    Args:
        model: Trained logistic regression model
        answers: List of 25 answer values (1-9)
    
    Returns:
        float: Probability score (0-1) representing fair use likelihood
    """
    if model is None:
        return None
    
    # Convert to numpy array and reshape for sklearn
    X = np.array(answers).reshape(1, -1)
    
    # Get probability of positive class (fair use = 1)
    prob = model.predict_proba(X)[0][1]
    
    return float(prob)
