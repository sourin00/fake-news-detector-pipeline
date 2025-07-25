import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
import joblib
import numpy as np
from pathlib import Path


class TfidfLogRegModel:
    """Baseline TF-IDF + Logistic Regression model"""

    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.classifier = LogisticRegression(
            random_state=42,
            max_iter=1000
        )
        self.is_trained = False

    def fit(self, texts, labels):
        """Train the model"""
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.is_trained = True
        return self

    def predict(self, texts):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X)

    def predict_proba(self, texts):
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)

    def save(self, path):
        """Save model to disk"""
        joblib.dump({
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'is_trained': self.is_trained
        }, path)

    def load(self, path):
        """Load model from disk"""
        data = joblib.load(path)
        self.vectorizer = data['vectorizer']
        self.classifier = data['classifier']
        self.is_trained = data['is_trained']
        return self


class BertFakeNewsModel:
    """BERT-based fake news detection model"""

    def __init__(self, model_name='distilbert-base-uncased', num_labels=2):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def encode_texts(self, texts, max_length=512):
        """Tokenize and encode texts"""
        return self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )

    def predict(self, texts, batch_size=16):
        """Make predictions on texts"""
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                encoded = self.encode_texts(batch_texts)

                # Move to device
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                batch_predictions = torch.argmax(outputs.logits, dim=-1)
                predictions.extend(batch_predictions.cpu().numpy())

        return np.array(predictions)

    def predict_proba(self, texts, batch_size=16):
        """Get prediction probabilities"""
        self.model.eval()
        all_probas = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                encoded = self.encode_texts(batch_texts)

                # Move to device
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probas = torch.softmax(outputs.logits, dim=-1)
                all_probas.extend(probas.cpu().numpy())

        return np.array(all_probas)

    def save(self, path):
        """Save model and tokenizer"""
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path):
        """Load model and tokenizer"""
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(self.device)
        return self


class EnsembleModel:
    """Ensemble of TF-IDF and BERT models"""

    def __init__(self):
        self.tfidf_model = None
        self.bert_model = None
        self.weights = [0.3, 0.7]  # TF-IDF weight, BERT weight

    def load_models(self, tfidf_path, bert_path):
        """Load both models"""
        self.tfidf_model = TfidfLogRegModel().load(tfidf_path)
        self.bert_model = BertFakeNewsModel()
        self.bert_model.load(bert_path)

    def predict_proba(self, texts):
        """Ensemble prediction probabilities"""
        if self.tfidf_model is None or self.bert_model is None:
            raise ValueError("Models not loaded")

        # Get predictions from both models
        tfidf_proba = self.tfidf_model.predict_proba(texts)
        bert_proba = self.bert_model.predict_proba(texts)

        # Weighted average
        ensemble_proba = (
                self.weights[0] * tfidf_proba +
                self.weights[1] * bert_proba
        )

        return ensemble_proba

    def predict(self, texts):
        """Ensemble predictions"""
        proba = self.predict_proba(texts)
        return np.argmax(proba, axis=1)
