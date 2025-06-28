# NLP
NLP project for classifying product reviews (positive or negative) using Logistic Regression and Random Forest classifiers.
# 🚗 Product Review Sentiment Analysis

NLP project for classifying product reviews (positive or negative) using Logistic Regression and Random Forest classifiers.

---

## 📋 Description

Supervised learning project that predicts the sentiment of product reviews based on TF-IDF vectorized text. 
Two machine learning models (Logistic Regression and Random Forest) are compared based on performance, speed, and interpretability.

---

## 📁 Project Structure

```
.
├── data/                           # Dataset
├── notebooks/
│   └── NLP_automotive.ipynb        # Main notebook with full pipeline (preprocessing, training, evaluation)
├── utils.py                        # Helper functions and custom visualizations
├── requirements.txt                # Full list of project dependencies
└── README.md                       # Project documentation
```

---

## 📊 Dataset

The dataset contains 10,000+ product reviews, each labeled as **positive** or **negative**.

* **Input**: `processed_text` column with preprocessed review content.
* **Target**: `sentiment_label` (binary: 0 = negative, 1 = positive).

The dataset is **highly imbalanced**: \~89% positive, \~11% negative.

---

## 🧠 Models Used

### 1. Logistic Regression

* Hyperparameter optimization with `GridSearchCV` for regularization strength `C`.
* `class_weight='balanced'` used to mitigate class imbalance.
* Probabilistic predictions and interpretable coefficients (top words).

### 2. Random Forest

* Parameters: `n_estimators=100`, `max_depth=20`, `class_weight='balanced'`.
* Fast training and robust classification.
* No direct interpretability on text features.

---

## 🔬 Methodology

### Preprocessing

* TF-IDF vectorization (max\_features=5000, unigrams + bigrams, sublinear\_tf).
* Stratified train/test split (75/25).
* Class weight computation to address imbalance.

### Training & Evaluation

* Time of training recorded for both models.
* Predictions include class and probability.
* Evaluation metrics:

  * Accuracy
  * F1-Score (weighted)
  * AUC-ROC
  * Precision & Recall (per class)
  * Confusion Matrix

### Utility Functions

All training routines, metric formatting, and visualizations are implemented in `utils.py`, which also contains the necessary library imports for modular execution.

---

## 📈 Results Summary

| Metric                  | Logistic Regression | Random Forest |
| ----------------------- | ------------------- | ------------- |
| F1-Score (weighted)     | 0.864               | **0.877**     |
| Accuracy                | 0.856               | **0.896**     |
| AUC-ROC                 | **0.825**           | 0.824         |
| Precision (Positive)    | **0.935**           | 0.911         |
| Recall (Positive)       | 0.899               | **0.977**     |
| Precision (Negative)    | 0.392               | **0.587**     |
| Recall (Negative)       | **0.511**           | 0.252         |
| Training Time (seconds) | 2.89                | **0.36**      |

---

## 📊 Visualizations Included

* Confusion matrices displayed side-by-side
* ROC curve for both models
* Bar plot comparing F1-score, Accuracy, AUC-ROC
* Top 5 most influential words in Logistic Regression (positive/negative)
* Table of classification metrics per class
* Summary table comparing key metrics of both models

---

## 🔍 Key Insights

* Random Forest achieved higher overall performance and training speed.
* Logistic Regression offered better interpretability and better recall on the minority class.
* Logistic Regression slightly outperformed in AUC-ROC.

---

## ⚙️ Dependencies

All required libraries are listed in the `requirements.txt` file. Install them with:

```bash
pip install -r requirements.txt
```

Main dependencies include:

* `scikit-learn` – Machine learning algorithms and metrics
* `pandas`, `numpy` – Data manipulation and computation
* `matplotlib`, `seaborn`, `plotly` – Data visualization
* `nltk`, `spacy`, `stop-words` – NLP and text preprocessing
* `gensim` – Topic modeling support (for extensions)
* `wordcloud`, `tqdm`, `scipy` – Utility libraries


