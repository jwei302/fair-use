"""
Train Logistic Regression Model on Training Data
Loads data from SQLite database and trains the model
"""
import sqlite3
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from model_def import create_model, save_model, load_model

DB_PATH = "training_data.db"

def load_training_data():
    """Load all cases from the database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT answers, label
        FROM cases
        ORDER BY created_at
    ''')
    rows = c.fetchall()
    conn.close()
    
    if len(rows) == 0:
        print("❌ No training data found in database!")
        print("   Add some cases using the Training Data tab first.")
        return None, None
    
    X = []
    y = []
    
    for row in rows:
        answers = json.loads(row[0])  # Parse JSON string
        label = row[1]
        
        # Validate we have 25 answers
        if len(answers) != 25:
            print(f"⚠️  Warning: Skipping case with {len(answers)} answers (expected 25)")
            continue
        
        X.append(answers)
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✓ Loaded {len(X)} training cases")
    print(f"  - Fair Use (1): {np.sum(y == 1)} cases")
    print(f"  - Not Fair Use (0): {np.sum(y == 0)} cases")
    
    return X, y

def train_model():
    """Train the logistic regression model"""
    print("\n" + "="*50)
    print("Training Logistic Regression Model")
    print("="*50 + "\n")
    
    # Load data
    X, y = load_training_data()
    if X is None:
        return False
    
    if len(X) < 2:
        print("❌ Need at least 2 training cases to train a model!")
        print("   Add more cases using the Training Data tab.")
        return False
    
    # Check if we have both classes
    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        print("❌ Need at least one case of each label (Fair Use and Not Fair Use)!")
        print(f"   Currently all cases are labeled as: {'Fair Use' if unique_labels[0] == 1 else 'Not Fair Use'}")
        print("   Add cases with the opposite label using the Training Data tab.")
        return False
    
    # Split into train/test sets (80/20)
    if len(X) >= 4:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"✓ Split data: {len(X_train)} train, {len(X_test)} test\n")
    else:
        # If too few samples, use all for training
        print(f"⚠️  Only {len(X)} cases available. Using all for training.\n")
        X_train, X_test, y_train, y_test = X, X, y, y
    
    # Create and train model
    print("Training model...")
    model = create_model()
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✓ Training complete!")
    print(f"  Accuracy: {accuracy:.2%}")
    
    if len(X_test) > 0 and len(np.unique(y_test)) > 1:
        print("\n" + "="*50)
        print("Classification Report:")
        print("="*50)
        print(classification_report(y_test, y_pred, 
                                   target_names=['Not Fair Use', 'Fair Use']))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
    
    # Save model
    save_model(model)
    
    print("\n" + "="*50)
    print("Model is ready to use!")
    print("="*50)
    print("\nRestart the Flask server (app.py) to load the new model.\n")
    
    return True

if __name__ == "__main__":
    import sys
    success = train_model()
    if not success:
        sys.exit(1)  # Exit with error code if training failed

