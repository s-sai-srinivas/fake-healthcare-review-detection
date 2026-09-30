# AI-Based Detection of Fake Reviews in Online Healthcare Platforms

Machine learning pipeline that detects fraudulent reviews on online healthcare platforms using NLP — built as my Master's dissertation project (MSc Computer Science, University of East London).

## Overview

Patients rely heavily on online reviews when choosing healthcare providers, and fake reviews undermine that trust. This project builds an automated detection system:

- **Text preprocessing** — lowercasing, punctuation/digit removal, stopword removal, WordNet lemmatization (NLTK)
- **Feature extraction** — TF-IDF vectorization (5,000 features)
- **Rule-based labelling** — the Kaggle dataset was unlabelled, so reviews were labelled positive/negative from ratings
- **Models compared** — Logistic Regression, Multinomial Naive Bayes, Linear SVM
- **Evaluation** — accuracy, precision, recall, F1, confusion matrices, ROC/AUC curves

## Results

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Logistic Regression | 0.83 | 0.86 | 0.83 | 0.81 |
| Naive Bayes | 0.79 | 0.80 | 0.79 | 0.76 |
| **SVM (LinearSVC)** | **0.83** | **0.83** | **0.83** | **0.83** |

SVM was the most balanced model, with the strongest minority-class performance — best suited for catching deceptive reviews.

## Dataset

[Hospital Reviews Dataset — Kaggle](https://www.kaggle.com/datasets/junaid6731/hospital-reviews-dataset)

Download `data_health_Online.csv` and place it in this directory (or set `DATA_PATH`).

## Run

```bash
pip install -r requirements.txt
python fake_review_detection.py
```

The script prints per-model classification reports and shows confusion matrices, metric line graphs, and ROC curves.

## Tech

Python · scikit-learn · NLTK · pandas · matplotlib · seaborn

## Future work

Transformer-based models (BERT/DistilBERT), behavioural features (reviewer history, timing patterns), and cross-platform generalization.
