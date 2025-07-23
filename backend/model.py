# backend/model.py

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# training model
def train_model(data_path='data/emotion_data.csv', model_path='models/text_clf_model.pkl'):
    df = pd.read_csv(data_path)
    X, y = df['text'], df['label']

    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB())
    ])
    model.fit(X, y)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"✅ Model saved: {model_path}")

# loading model
def load_model(model_path='models/text_clf_model.pkl'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file doesn't exist: {model_path}")
    return joblib.load(model_path)

# predict
def predict(text, model):
    return model.predict([text])[0]

# test
if __name__ == '__main__':
    train_model()
    model = load_model()
    sample = "I'm so happy today!"
    print(f"[prediction] \"{sample}\" → {predict(sample, model)}")
