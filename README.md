# **Phishing Email Detection Using Machine Learning**

## ** Project Overview**
This project is a **Phishing Email Detection System** that uses **Machine Learning (ML)** to classify emails as **Phishing** or **Not Phishing**. The model is trained using a dataset of phishing and legitimate emails and uses **TF-IDF (Text Vectorization) and a Gradient Boosting Classifier** to make predictions.

---
## ** Project Structure**
```
Phishing-Email-Detection/
│── models/
│   ├── phishing_detector.pkl      # Trained Machine Learning model
│
│── data/
│   ├── phishing_emails.csv          # Raw email dataset (Phishing & Non-phishing)
│   ├── preprocessed_data.pkl        # Processed email data
│
│── src/
│   ├── preprocess.py                 # Cleans and vectorizes email data
│   ├── train.py                      # Trains the machine learning model
│   ├── predict.py                     # Predicts if an email is phishing or not
│
│── requirements.txt                 # Python dependencies
│── README.md                         # Documentation
```

---

## ** Installation & Setup**

### **1️ Install Dependencies**
Make sure you have **Python 3.8+** and **pip** installed. Run:
```bash
pip install -r requirements.txt
```

---

### **2️ Preprocess the Data**
Run the following command to clean and vectorize the dataset:
```bash
python src/preprocess.py
```
_Expected Output:_
```plaintext
Preprocessed data saved to data/preprocessed_data.pkl
```
This step creates `data/preprocessed_data.pkl` for training.

---

### **3️ Train the Model**
Run the training script to train the phishing detection model:
```bash
python src/train.py
```
_Example Output:_
```plaintext
Best Model Accuracy: 0.85
Model saved to models/phishing_detector.pkl
```
> ⚠️ If accuracy is 0.00, your dataset is too small. Add more non-phishing emails to `phishing_emails.csv` and try again.

---

### **4️ Run Predictions on Emails**
Run this command to check if an email is phishing:
```bash
python src/predict.py
```
_Example Output:_
```plaintext
Prediction: Not Phishing (Confidence: 0.87)
