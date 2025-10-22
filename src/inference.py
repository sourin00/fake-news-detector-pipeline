import torch
import joblib
import numpy as np
from pathlib import Path
from src.model import TfidfLogRegModel, BertFakeNewsModel, EnsembleModel
from src.data_utils import preprocess_text, advanced_preprocess_for_tfidf
import logging

logger = logging.getLogger(__name__)


class FakeNewsDetector:
    """Main inference class for fake news detection"""

    def __init__(self):
        self.tfidf_model = None
        self.bert_model = None
        self.ensemble_model = None
        self.available_models = []

    def load_tfidf_model(self, model_path='models/tfidf_logreg.joblib'):
        """Load TF-IDF model"""
        try:
            self.tfidf_model = TfidfLogRegModel()
            self.tfidf_model.load(model_path)
            self.available_models.append('tfidf')
            logger.info("TF-IDF model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading TF-IDF model: {e}")

    def load_bert_model(self, model_path='models/bert'):
        """Load BERT model"""
        try:
            if Path(model_path).exists():
                self.bert_model = BertFakeNewsModel()
                self.bert_model.load(model_path)
                self.available_models.append('bert')
                logger.info("BERT model loaded successfully")
            else:
                logger.warning(f"BERT model path {model_path} does not exist")
        except Exception as e:
            logger.error(f"Error loading BERT model: {e}")

    def load_ensemble_model(self):
        """Load ensemble model (requires both TF-IDF and BERT)"""
        try:
            if self.tfidf_model is not None and self.bert_model is not None:
                self.ensemble_model = EnsembleModel()
                self.ensemble_model.tfidf_model = self.tfidf_model
                self.ensemble_model.bert_model = self.bert_model
                self.available_models.append('ensemble')
                logger.info("Ensemble model loaded successfully")
            else:
                logger.warning("Cannot load ensemble model: both TF-IDF and BERT models required")
        except Exception as e:
            logger.error(f"Error loading ensemble model: {e}")

    def load_all_models(self):
        """Load all available models"""
        self.load_tfidf_model()
        self.load_bert_model()
        self.load_ensemble_model()

        if not self.available_models:
            logger.warning("No models loaded successfully")
        else:
            logger.info(f"Available models: {', '.join(self.available_models)}")

    def predict_single(self, text, model_type='ensemble'):
        """Predict single text sample"""
        if isinstance(text, list):
            text = text[0] if text else ""

        # Preprocess text
        if model_type == 'tfidf' and self.tfidf_model:
            processed_text = advanced_preprocess_for_tfidf(text)
            prediction = self.tfidf_model.predict([processed_text])[0]
            probabilities = self.tfidf_model.predict_proba([processed_text])[0]

        elif model_type == 'bert' and self.bert_model:
            processed_text = preprocess_text(text)
            prediction = self.bert_model.predict([processed_text])[0]
            probabilities = self.bert_model.predict_proba([processed_text])[0]

        elif model_type == 'ensemble' and self.ensemble_model:
            processed_text = preprocess_text(text)
            prediction = self.ensemble_model.predict([processed_text])[0]
            probabilities = self.ensemble_model.predict_proba([processed_text])[0]

        else:
            # Fallback to available model
            if 'tfidf' in self.available_models:
                return self.predict_single(text, 'tfidf')
            elif 'bert' in self.available_models:
                return self.predict_single(text, 'bert')
            else:
                raise ValueError("No models available for prediction")

        return {
            'prediction': int(prediction),
            'label': 'Real' if prediction == 1 else 'Fake',
            'confidence': float(max(probabilities)),
            'probabilities': {
                'Real': float(probabilities[1]),
                'Fake': float(probabilities[0])
            },
            'model_used': model_type
        }

    def predict_batch(self, texts, model_type='ensemble'):
        """Predict batch of texts"""
        results = []
        for text in texts:
            result = self.predict_single(text, model_type)
            results.append(result)
        return results

    def get_model_info(self):
        """Get information about loaded models"""
        info = {
            'available_models': self.available_models,
            'tfidf_loaded': self.tfidf_model is not None,
            'bert_loaded': self.bert_model is not None,
            'ensemble_loaded': self.ensemble_model is not None
        }
        return info


# Global detector instance
detector = FakeNewsDetector()


def initialize_detector():
    """Initialize the global detector"""
    global detector
    detector.load_all_models()
    return detector


def predict_fake_news(text, model_type='ensemble'):
    """Convenience function for prediction"""
    global detector
    if not detector.available_models:
        detector = initialize_detector()

    return detector.predict_single(text, model_type)


def get_available_models():
    """Get list of available models"""
    global detector
    if not detector.available_models:
        detector = initialize_detector()

    return detector.available_models
