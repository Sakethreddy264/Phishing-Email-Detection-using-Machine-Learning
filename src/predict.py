import pickle
import numpy as np

def load_pickle(file):
    try:
        with open(file, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {file}: {e}")
        return None

def predict_email(model_file, vectorizer_file, email_text):
    model = load_pickle(model_file)
    data = load_pickle(vectorizer_file)

    if model is None or data is None:
        return "Error loading model or vectorizer."

    _, _, vectorizer = data
    email_vector = vectorizer.transform([email_text]).toarray()

    # Predict with probability
    prob = model.predict_proba(email_vector)[0]
    prediction = "Phishing" if prob[1] > 0.5 else "Not Phishing"
    
    return f"{prediction} (Confidence: {prob[1]:.2f})"

if __name__ == "__main__":
    email_text = """Subject: Meeting Reminder - Project Discussion

Hi Team,

This is a reminder for our project discussion scheduled for tomorrow at 2 PM in the conference room.  
Please bring your updates and any questions you might have.  

Best regards,  
John Doe  
Project Manager"""
    result = predict_email("models/phishing_detector.pkl", "data/preprocessed_data.pkl", email_text)
    print(f"Prediction: {result}")
