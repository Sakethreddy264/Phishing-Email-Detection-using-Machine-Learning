import pickle
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

def train_model(data_file, model_file):
    with open(data_file, "rb") as f:
        X, y, _ = pickle.load(f)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Check class distribution before SMOTE
    print(f"Before SMOTE: {Counter(y_train)}")
    
    # Apply SMOTE only if the minority class has enough samples
    if len(set(y_train)) > 1 and min(Counter(y_train).values()) > 6:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE: {Counter(y_train)}")
    else:
        print("Skipping SMOTE: Not enough samples for minority class")
    
    # Hyperparameter tuning
    param_grid = {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]}
    model = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
    model.fit(X_train, y_train)
    
    best_model = model.best_estimator_
    
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Best Model Accuracy: {accuracy:.2f}")
    
    with open(model_file, "wb") as f:
        pickle.dump(best_model, f)
    
    print(f"Model saved to {model_file}")

if __name__ == "__main__":
    train_model("data/preprocessed_data.pkl", "models/phishing_detector.pkl")
