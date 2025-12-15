"""
Logistic Regression Model Definition
Using scikit-learn for simpler deployment
"""
import pickle
import os

def load_model():
    """Load the trained model from disk"""
    model_path = os.path.join(os.path.dirname(__file__), 'logistic_model.pkl')
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        print(f"Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict(model, answers):
    """
    Make a prediction using the loaded model
    Args:
        model: The loaded scikit-learn model
        answers: List of 25 numerical answers (1-9)
    Returns:
        float: Probability score (0-1) that this is fair use
    """
    try:
        import numpy as np
        # Reshape to 2D array as sklearn expects
        X = np.array(answers).reshape(1, -1)
        # Get probability of positive class (fair use)
        prob = model.predict_proba(X)[0][1]
        return float(prob)
    except Exception as e:
        print(f"Prediction error: {e}")
        return None
