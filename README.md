# Fake News Detector Pipeline

A robust, production-ready pipeline for detecting fake news using both traditional machine learning (TF-IDF + Logistic Regression) and advanced deep learning (BERT) models, with an ensemble option for best performance. Includes a Streamlit web app for interactive analysis.

---

## Features
- **Multiple Models:**
  - TF-IDF + Logistic Regression (fast, interpretable)
  - BERT-based neural network (high accuracy)
  - Ensemble (combines both for best results)
- **Interactive Web App:** Streamlit UI for text/news analysis
- **Batch and Single Prediction:** API-ready inference code
- **Dockerized:** Easy deployment with Docker and Docker Compose
- **Kaggle Dataset Integration:** Uses [Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- **Extensible:** Modular code for easy experimentation

---

## Architecture

```
[Data] --(preprocessing)--> [TF-IDF Model] --\
                                 |           >-- [Ensemble] --> [Prediction]
[Data] --(preprocessing)--> [BERT Model] --/
```

- **src/data_utils.py:** Data loading, preprocessing, and statistics
- **src/model.py:** Model classes (TF-IDF, BERT, Ensemble)
- **src/inference.py:** Unified inference API
- **src/train.py:** Training scripts for both models
- **app/streamlit_app.py:** Streamlit web interface

---

## Quickstart

### 1. Clone the Repository
```bash
git clone <repo-url>
cd fake-news-detector-pipeline
```

### 2. Download Kaggle Dataset
- Download [Fake.csv](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset) and [True.csv](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset) from Kaggle.
- Place both files in the `data/` directory:
  - `data/Fake.csv`
  - `data/True.csv`

### 3. Build and Run with Docker (Recommended)
```bash
docker-compose up --build
```
- The app will be available at [http://localhost:8501](http://localhost:8501)

### 4. Or Run Locally (Python 3.9+)
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Download Trained Models
Click on this [Link](https://drive.google.com/drive/folders/1ulOvoqsBunVndOVOmgDTTyqiBgrKG8nJ?usp=drive_link) to download the pre trained models folder and paste in the root directory of the project.
Once done you can run the application locally without re training the models which is a time taking process.

## Model Training

### Train TF-IDF Model
```bash
python src/train.py --model tfidf
```
- Saves model to `models/tfidf_logreg.joblib`

### Train BERT Model
```bash
python src/train.py --model bert --epochs 3 --batch_size 16
```
- Saves model to `models/bert/`

### Train Both
```bash
python src/train.py --model both
```

- **Note:** Training BERT requires a GPU for reasonable speed.

---

## Prediction & Inference

### From Python
```python
from src.inference import predict_fake_news

result = predict_fake_news("Your news text here", model_type='ensemble')
print(result)
# Output: {'prediction': 1, 'label': 'Real', 'confidence': 0.98, ...}
```
- `model_type`: 'tfidf', 'bert', or 'ensemble'

### From Streamlit Web App
- Go to [http://localhost:8501](http://localhost:8501)
- Paste/type news text, upload a file, or use examples
- Select model and analyze

---

## Project Structure

```
├── app/
│   └── streamlit_app.py         # Streamlit web app
├── data/
│   ├── Fake.csv                # Kaggle fake news
│   └── True.csv                # Kaggle real news
├── models/
│   ├── tfidf_logreg.joblib     # Trained TF-IDF model
│   └── bert/                   # Trained BERT model
├── src/
│   ├── data_utils.py           # Data loading/preprocessing
│   ├── inference.py            # Inference API
│   ├── model.py                # Model classes
│   └── train.py                # Training scripts
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Data Preprocessing
- **Text cleaning:** Lowercasing, URL/email removal, punctuation, etc.
- **TF-IDF:** Additional stemming and stopword removal
- **BERT:** Truncation to 512 tokens

---

## Evaluation
- Accuracy, classification report, and confidence scores are shown in the app and logs.
- Example expected accuracies:
  - TF-IDF: ~90%
  - BERT: ~98%
  - Ensemble: ~99%

---

## Customization
- **Add new models:** Extend `src/model.py` and update `src/inference.py`
- **Change dataset:** Place new CSVs in `data/` and adjust `src/data_utils.py` if needed
- **Tune hyperparameters:** Edit `src/train.py` or pass CLI args

---

## Deployment
- **Docker Compose:** Handles all dependencies, caching, and healthchecks
- **Environment Variables:**
  - `TRANSFORMERS_CACHE` for HuggingFace models
  - `PYTHONPATH` for module resolution

---

## FAQ

**Q: Can I use my own dataset?**
- Yes! Place your CSV in `data/` and adjust `src/data_utils.py:load_external_dataset()` as needed.

**Q: How do I add more models?**
- Implement a new class in `src/model.py` and update the inference logic.

**Q: Is GPU required?**
- Strongly recommended for BERT training/inference, but TF-IDF works on CPU.

---

## 📚 References
- [Kaggle Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [Streamlit](https://streamlit.io/)


