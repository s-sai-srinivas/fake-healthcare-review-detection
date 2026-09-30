# ============================================

import pandas as pd
import numpy as np
import re
import string
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc

nltk.download('stopwords')
nltk.download('wordnet')

# ============================================
# 1. LOAD DATA
# ============================================
# Dataset: https://www.kaggle.com/datasets/junaid6731/hospital-reviews-dataset
# Download data_health_Online.csv and place it next to this script (or set DATA_PATH)
import os
DATA_PATH = os.environ.get('DATA_PATH', 'data_health_Online.csv')
df = pd.read_csv(DATA_PATH)

# ============================================
# 2. FIX DATASET
# ============================================
df = df.drop(columns=['Unnamed: 3'], errors='ignore')

text_column = 'Feedback'

df = df[['Feedback', 'Ratings']].dropna()

df['label'] = df['Ratings'].apply(lambda x: 1 if x >= 3 else 0)

print("Label Distribution:\n", df['label'].value_counts())

# ============================================
# 3. TEXT CLEANING
# ============================================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

df['clean_review'] = df[text_column].apply(clean_text)

# ============================================
# 4. TF-IDF
# ============================================
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['clean_review'])
y = df['label']

# ============================================
# 5. TRAIN TEST SPLIT
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
# FUNCTIONS FOR LINE GRAPHS
# ============================================
def plot_metrics(model_name, acc, prec, rec, f1):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    values = [acc, prec, rec, f1]

    plt.figure()
    plt.plot(metrics, values, marker='o')
    plt.title(f"{model_name} - Performance Metrics (Line Graph)")
    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.grid()
    plt.show()

def plot_roc(y_test, y_scores, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1],[0,1],'--')
    plt.title(f"{model_name} - ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid()
    plt.show()

# ============================================
# 6. LOGISTIC REGRESSION
# ============================================
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:,1]

print("\nLOGISTIC REGRESSION")
print(classification_report(y_test, y_pred_lr))

cm = confusion_matrix(y_test, y_pred_lr)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

plot_metrics("Logistic Regression",
             accuracy_score(y_test, y_pred_lr),
             precision_score(y_test, y_pred_lr, average='weighted'),
             recall_score(y_test, y_pred_lr, average='weighted'),
             f1_score(y_test, y_pred_lr, average='weighted'))

plot_roc(y_test, y_prob_lr, "Logistic Regression")

# ============================================
# 7. NAIVE BAYES
# ============================================
nb = MultinomialNB()
nb.fit(X_train, y_train)

y_pred_nb = nb.predict(X_test)
y_prob_nb = nb.predict_proba(X_test)[:,1]

print("\nNAIVE BAYES")
print(classification_report(y_test, y_pred_nb))

plot_metrics("Naive Bayes",
             accuracy_score(y_test, y_pred_nb),
             precision_score(y_test, y_pred_nb, average='weighted'),
             recall_score(y_test, y_pred_nb, average='weighted'),
             f1_score(y_test, y_pred_nb, average='weighted'))

plot_roc(y_test, y_prob_nb, "Naive Bayes")

# ============================================
# 8. SVM
# ============================================
svm = LinearSVC()
svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)
scores = svm.decision_function(X_test)

print("\nSVM")
print(classification_report(y_test, y_pred_svm))

plot_metrics("SVM",
             accuracy_score(y_test, y_pred_svm),
             precision_score(y_test, y_pred_svm, average='weighted'),
             recall_score(y_test, y_pred_svm, average='weighted'),
             f1_score(y_test, y_pred_svm, average='weighted'))

plot_roc(y_test, scores, "SVM")

print("\n✅ SUCCESS: Models trained with generated labels")
# ============================================
# COMPARATIVE ANALYSIS VISUALIZATION
# ============================================

import matplotlib.pyplot as plt
import pandas as pd

# ============================================
# 1. MANUALLY ENTER YOUR RESULTS
# ============================================

data = {
    'Model': ['Logistic Regression', 'Naive Bayes', 'SVM'],

    'Accuracy': [0.83, 0.79, 0.83],

    'Precision': [0.86, 0.80, 0.83],   # weighted avg
    'Recall': [0.83, 0.79, 0.83],      # weighted avg
    'F1 Score': [0.81, 0.76, 0.83]     # weighted avg
}

df_results = pd.DataFrame(data)

# ============================================
# 2. ACCURACY COMPARISON (LINE GRAPH)
# ============================================
plt.figure()
plt.plot(df_results['Model'], df_results['Accuracy'], marker='o')
plt.title("Model Accuracy Comparison (Line Graph)")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.xticks(rotation=20)
plt.grid()
plt.show()

# ============================================
# 3. GROUPED METRICS COMPARISON (LINE GRAPH)
# ============================================
metrics = ['Precision', 'Recall', 'F1 Score']

plt.figure()
df_results.set_index('Model')[metrics].plot(kind='line', marker='o')

plt.title("Model Performance Comparison (Line Graph)")
plt.ylabel("Score")
plt.xticks(rotation=20)
plt.legend()
plt.grid()
plt.show()

# ============================================
# 4. F1 SCORE COMPARISON (LINE GRAPH)
# ============================================
plt.figure()
plt.plot(df_results['Model'], df_results['F1 Score'], marker='o')
plt.title("F1 Score Comparison Across Models (Line Graph)")
plt.xlabel("Model")
plt.ylabel("F1 Score")
plt.xticks(rotation=20)
plt.grid()
plt.show()
