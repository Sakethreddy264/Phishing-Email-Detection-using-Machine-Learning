import pandas as pd
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-z\s]", "", text)  # Remove special characters & numbers
    text = " ".join([word for word in text.split() if word not in STOPWORDS])  # Remove stopwords
    return text

def preprocess_data(input_file, output_file):
    data = pd.read_csv(input_file)

    if "text" not in data.columns or "label" not in data.columns:
        raise ValueError("CSV must have 'text' and 'label' columns.")

    data["text"] = data["text"].fillna("").apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    X = vectorizer.fit_transform(data["text"]).toarray()
    y = data["label"].values

    with open(output_file, "wb") as f:
        pickle.dump((X, y, vectorizer), f)

    print(f"Preprocessed data saved to {output_file}")

if __name__ == "__main__":
    preprocess_data("data/phishing_emails.csv", "data/preprocessed_data.pkl")
