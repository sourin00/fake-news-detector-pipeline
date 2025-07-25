import os

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import argparse
from pathlib import Path
import logging
from src.model import TfidfLogRegModel, BertFakeNewsModel
from src.data_utils import load_sample_data, preprocess_text, load_fake_real_news_kaggle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FakeNewsDataset(Dataset):
    """Dataset class for fake news data"""

    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,  # Reduced from 512
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def train_tfidf_model():
    """Train the baseline TF-IDF model"""
    logger.info("Training TF-IDF baseline model...")


    # df = load_sample_data() # Load LOCAL data

    # Load Kaggle dataset
    df = load_fake_real_news_kaggle()
    # texts = df['text'].apply(preprocess_text).tolist() # Load LOCAL data

    # Kaggle dataset may have 'title' and 'text' columns
    texts = (df['title'].fillna('') + " " + df['text'].fillna('')).tolist()
    labels = df['label'].tolist()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Train model
    model = TfidfLogRegModel()
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    logger.info(f"TF-IDF Model Accuracy: {accuracy:.4f}")

    # Save model
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model.save(os.path.join(models_dir, 'tfidf_logreg.joblib'))
    logger.info(f"TF-IDF model saved to {os.path.join(models_dir, 'tfidf_logreg.joblib')}")

    return model, accuracy


def train_bert_model(epochs=3, batch_size=16, learning_rate=2e-5, gradient_accumulation_steps=1):
    """Train the BERT model with optional gradient accumulation"""
    logger.info("Training BERT model...")

    # df = load_sample_data() # Load LOCAL data

    # Load Kaggle dataset
    df = load_fake_real_news_kaggle()
    # texts = df['text'].apply(preprocess_text).tolist() # Load LOCAL data

    # Kaggle dataset may have 'title' and 'text' columns
    texts = (df['title'].fillna('') + " " + df['text'].fillna('')).tolist()
    labels = df['label'].tolist()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Initialize model and tokenizer
    model = BertFakeNewsModel()

    # Create datasets
    train_dataset = FakeNewsDataset(X_train, y_train, model.tokenizer)
    test_dataset = FakeNewsDataset(X_test, y_test, model.tokenizer)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Set up training
    optimizer = AdamW(model.model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    scaler = torch.amp.GradScaler('cuda')

    train_loader = DataLoader(train_dataset, batch_size=132, shuffle=True, num_workers=6, pin_memory=True)

    model.model.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
        optimizer.zero_grad()
        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)

            with torch.amp.autocast('cuda'):
                outputs = model.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            total_loss += loss.item() * gradient_accumulation_steps
            progress_bar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})

        avg_loss = total_loss / len(train_loader)
        logger.info(f'Epoch {epoch + 1} Average Loss: {avg_loss:.4f}')

    # Evaluation
    model.model.eval()
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    logger.info(f"BERT Model Accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, predictions, target_names=['Real', 'Fake']))

    # Save model
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models', 'bert')
    os.makedirs(models_dir, exist_ok=True)
    model.save(models_dir)
    logger.info(f"BERT model saved to {models_dir}/")

    return model, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train fake news detection models')
    parser.add_argument('--model', choices=['tfidf', 'bert', 'both'], default='both',
                        help='Which model to train')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs for BERT')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')

    args = parser.parse_args()

    # Create models directory
    Path('models').mkdir(exist_ok=True)

    if args.model in ['tfidf', 'both']:
        train_tfidf_model()

    if args.model in ['bert', 'both']:
        train_bert_model(args.epochs, args.batch_size, args.learning_rate)


if __name__ == '__main__':
    main()
