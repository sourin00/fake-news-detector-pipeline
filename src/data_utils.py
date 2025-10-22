import os

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


def preprocess_text(text, max_length=512):
    """Advanced text preprocessing for both TF-IDF and BERT models"""
    if pd.isna(text):
        return ""

    # Convert to string and lowercase
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', ' ', text)

    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)

    # Truncate if too long (for BERT compatibility)
    if len(text.split()) > max_length // 2:  # Rough word count limit
        text = ' '.join(text.split()[:max_length // 2])

    return text.strip()


def advanced_preprocess_for_tfidf(text):
    """More aggressive preprocessing for TF-IDF (stemming, stopword removal)"""
    text = preprocess_text(text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove stopwords and stem
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words and len(word) > 2]

    return ' '.join(words)


def load_sample_data():
    """Load sample fake news dataset"""
    # Create sample data (in production, load from actual dataset)
    fake_news_samples = [
        "Breaking: Scientists discover that vaccines contain microchips for government surveillance!",
        "SHOCKING: Local politician caught embezzling millions, mainstream media won't report!",
        "URGENT: New study shows 5G towers cause coronavirus, government coverup exposed!",
        "EXCLUSIVE: Celebrity death was actually assassination by secret society!",
        "WARNING: Popular food brand contains dangerous chemicals, FDA hiding the truth!",
        "LEAKED: Government plans to control population through water fluoridation!",
        "REVEALED: Moon landing was fake, new evidence proves Hollywood production!",
        "ALERT: Climate change is hoax created by corporations to increase profits!",
        "EXPOSED: Election results were manipulated by foreign hackers!",
        "CONFIRMED: Big pharma hiding cancer cure to maintain profits!"
    ]

    real_news_samples = [
        "The Federal Reserve announced a 0.25% interest rate increase following today's meeting.",
        "Local university researchers publish findings on renewable energy efficiency in peer-reviewed journal.",
        "City council approves budget allocation for infrastructure improvements over next fiscal year.",
        "Weather service issues severe thunderstorm warning for metropolitan area through evening.",
        "Stock market closes mixed today with technology sector showing modest gains.",
        "Health officials report seasonal flu vaccination campaign reaches 70% coverage target.",
        "Transportation department announces completion of highway construction project ahead of schedule.",
        "Annual economic report shows moderate growth in employment rates across multiple sectors.",
        "Educational board implements new curriculum standards following extensive community input.",
        "Environmental agency releases quarterly air quality assessment for urban areas."
    ]

    # Create balanced dataset
    data = []
    for text in fake_news_samples:
        data.append({'text': text, 'label': 0})  # 1 for fake

    for text in real_news_samples:
        data.append({'text': text, 'label': 1})  # 0 for real

    # Add more synthetic data for better training
    additional_fake = [
        "Scientists warn that popular smartphone apps are secretly recording conversations!",
        "BREAKING: Ancient aliens built pyramids, new archaeological evidence surfaces!",
        "Government whistleblower reveals secret mind control experiments on citizens!",
        "SHOCK: Popular social media platform selling user data to foreign governments!",
        "Local mayor involved in massive corruption scandal, authorities refuse to investigate!"
    ]

    additional_real = [
        "Regional hospital announces expansion of emergency services to meet growing demand.",
        "Local business association reports steady growth in small business registrations.",
        "University study examines impact of remote work on employee productivity.",
        "Municipal water treatment facility completes scheduled maintenance upgrades.",
        "Community college launches new vocational training programs for in-demand skills."
    ]

    for text in additional_fake:
        data.append({'text': text, 'label': 1})

    for text in additional_real:
        data.append({'text': text, 'label': 0})

    return pd.DataFrame(data)


def load_external_dataset(filepath):
    """Load external dataset (CSV format expected)"""
    try:
        df = pd.read_csv(filepath)
        # Assume columns are 'text' and 'label'
        # Adjust column names if needed
        if 'title' in df.columns and 'text' not in df.columns:
            df['text'] = df['title']

        # Ensure binary labels (0=real, 1=fake)
        if df['label'].dtype == 'object':
            label_map = {'real': 0, 'fake': 1, 'REAL': 0, 'FAKE': 1}
            df['label'] = df['label'].map(label_map)

        return df
    except Exception as e:
        print(f"Error loading external dataset: {e}")
        return load_sample_data()

def load_fake_real_news_kaggle(data_dir=None):
    """
    Load and merge Fake.csv and True.csv from the Kaggle dataset.
    Assign label 0 to fake news and 1 to real news.
    Returns a combined DataFrame with columns: 'title', 'text', 'label'.
    """
    if data_dir is None:
        # Use the directory of this file as base, then go up one and into 'data'
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, 'data')
    fake_path = os.path.join(data_dir, 'Fake.csv')
    true_path = os.path.join(data_dir, 'True.csv')

    # Read CSV files
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    # Assign labels
    fake_df['label'] = 0
    true_df['label'] = 1

    # Select relevant columns (some datasets may have more)
    fake_cols = [col for col in ['title', 'text', 'label'] if col in fake_df.columns]
    true_cols = [col for col in ['title', 'text', 'label'] if col in true_df.columns]

    fake_df = fake_df[fake_cols]
    true_df = true_df[true_cols]

    # Combine
    combined_df = pd.concat([true_df, fake_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    return combined_df


def get_data_stats(df):
    """Get basic statistics about the dataset"""
    stats = {
        'total_samples': len(df),
        'fake_samples': len(df[df['label'] == 1]),
        'real_samples': len(df[df['label'] == 0]),
        'avg_text_length': df['text'].str.len().mean(),
        'max_text_length': df['text'].str.len().max(),
        'min_text_length': df['text'].str.len().min()
    }
    return stats
