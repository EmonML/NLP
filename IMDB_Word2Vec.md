# 🎬 IMDB Sentiment Analysis — Boosting + Stacking Ensemble

A complete NLP sentiment classification project using **Word2vec, TruncatedSVD, five Boosting algorithms, and a Stacking Ensemble** for classifying IMDB movie reviews as **Positive** or **Negative**.

---

## 📌 Project Overview

Sentiment analysis is a Natural Language Processing (NLP) task that determines the emotional polarity of a piece of text.

In this project, IMDB movie reviews are classified into two categories:

* 🟢 Positive
* 🔴 Negative

The project compares multiple boosting algorithms and then combines them using a **Stacking Ensemble**.

The complete workflow is:

```text
Movie Review
     ↓
Text Cleaning
     ↓
Word2Vec Vectorization
     ↓
TruncatedSVD
     ↓
 ┌───────────────┬────────────────────┬───────────────┐
 │   AdaBoost    │ Gradient Boosting   │    XGBoost    │
 ├───────────────┼────────────────────┼───────────────┤
 │   CatBoost    │      LightGBM       │               │
 └───────────────┴────────────────────┴───────────────┘
                     ↓
             Stacking Classifier
                     ↓
          Logistic Regression
            Meta-Learner
                     ↓
            Final Prediction
```

---

# 🎯 Objectives

The main objectives of this project are:

1. Perform sentiment analysis on IMDB movie reviews.
2. Clean and preprocess raw text data.
3. Convert text into numerical features using Word2Vec.
4. Reduce the dimensionality of Word2Vec features using TruncatedSVD.
5. Train multiple boosting algorithms.
6. Compare individual model performance.
7. Build a Stacking Ensemble.
8. Use Logistic Regression as the meta-learner.
9. Evaluate models using multiple classification metrics.
10. Save trained models and preprocessing artifacts for future inference.

---

# 📂 Dataset

Dataset load from HuggingFace.



| Column      | Description                 |
| ----------- | --------------------------- |
| `review`    | IMDB movie review text      |
| `sentiment` | Positive/Negative sentiment |

Supported sentiment labels include:

```text
positive
negative
```

or:

```text
1
0
```

# 🧹 Data Preprocessing

The project performs several preprocessing operations.

### 1. Missing-value removal

Rows containing missing review or sentiment values are removed.

### 2. Duplicate removal

Duplicate records are removed from both training and testing datasets.

### 3. Text normalization

Reviews are converted to lowercase.

### 4. HTML removal

HTML tags such as:

```text
<br />
```

are removed.

### 5. URL removal

URLs and web addresses are removed.

### 6. Special-character cleaning

Unnecessary characters are removed while preserving useful sentiment-related characters such as:

```text
!
?
'
```

---

# 🔢 Word2Vec Feature Extraction

After cleaning the text, **Word2Vec Vectorization** is applied.

The project uses:

```python
w2v_model = Word2Vec(
    sentences=train_tokens,
    vector_size=W2V_VECTOR_SIZE,
    window=W2V_WINDOW,
    min_count=W2V_MIN_COUNT,
    workers=W2V_WORKERS,
    sg=1,          # Skip-gram
    negative=10,
    epochs=10,
    seed=SEED
)

```


# 📉 TruncatedSVD

Word2Vec generates a high-dimensional sparse matrix.

Using the complete Word2Vec matrix directly with tree-based boosting algorithms can become computationally expensive.

Therefore, the project applies:

```python
TruncatedSVD(n_components=300)
```

This converts the high-dimensional TF-IDF representation into a compact dense feature representation.

### Pipeline

```text
Word2Vec
  ↓
High-dimensional Sparse Matrix
  ↓
TruncatedSVD
  ↓
300-dimensional Dense Features
```

---

# 🚀 Boosting Models

Five different boosting algorithms are implemented.

## 1. AdaBoost

AdaBoost combines multiple weak learners sequentially and focuses more on incorrectly classified samples.

---

## 2. Gradient Boosting

Gradient Boosting builds trees sequentially where each new tree attempts to correct the errors of previous trees.

---

## 3. XGBoost

**XGBoost (Extreme Gradient Boosting)** is an optimized gradient boosting framework.

It provides:

* Regularization
* Efficient tree construction
* Parallel computation
* Histogram-based tree learning

The project uses:

```python
tree_method='hist'
```

---

## 4. CatBoost

CatBoost is another gradient boosting framework designed for efficient and robust machine learning.

It is particularly popular for tabular machine-learning problems.

---

## 5. LightGBM

LightGBM is a gradient boosting framework optimized for speed and memory efficiency.

The project uses:

```python
LGBMClassifier
```

---

# 🧩 Stacking Ensemble

After training the individual boosting models, they are combined using:

```python
StackingClassifier
```

The base estimators are:

```text
AdaBoost
GradientBoosting
XGBoost
CatBoost
LightGBM
```

The final meta-learner is:

```text
Logistic Regression
```

The architecture is:

```text
                   ┌── AdaBoost ────────┐
                   │                    │
                   ├── GradientBoosting ┤
                   │                    │
Word2Vec + SVD ──────┼── XGBoost ─────────┼──> Logistic Regression
                   │                    │       Meta-Learner
                   ├── CatBoost ────────┤
                   │                    │
                   └── LightGBM ────────┘
```

The stacking model uses:

```python
stack_method='predict_proba'
```

and:

```python
cv=5
```

This allows the meta-learner to learn from the probability outputs generated by the base models using cross-validation.

---

# 📊 Model Evaluation

The project evaluates the models using:

### Accuracy

Measures the proportion of correctly classified reviews.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Classification Report

The final stacking model produces a classification report containing:

* Precision
* Recall
* F1-score
* Support

### Confusion Matrix

The confusion matrix shows:

```text
                 Predicted
              Negative Positive

Actual Negative    TN       FP
       Positive    FN       TP
```

### ROC Curve

The ROC curve illustrates the relationship between:

* True Positive Rate
* False Positive Rate

### ROC-AUC

ROC-AUC measures the model's ability to distinguish between positive and negative reviews.

---

# 📈 Visualizations

The notebook includes several visualizations:

* Boosting model accuracy comparison
* ROC curves for individual boosting models
* Accuracy vs ROC-AUC comparison
* Confusion matrix
* Final Stacking Ensemble ROC curve

---

# 🔮 Prediction

A custom prediction function is included:

```python
predict_sentiment(review)
```

Example:

```python
predict_sentiment(
    "This movie was fantastic. The acting was excellent and I really enjoyed it."
)
```

Example output structure:

```python
{
    'sentiment': 'Positive',
    'positive_probability': 0.95,
    'negative_probability': 0.05
}
```

The actual probability values depend on the trained model and dataset.

---

# 💾 Saved Artifacts

The notebook saves the trained components inside:

```text
artifacts/
```

The generated files include:

```text
artifacts/
│
├── Word2Vec_vectorizer.joblib
├── svd_transformer.joblib
├── stacking_imdb_model.joblib
├── model_comparison.csv
│
├── adaboost_model.joblib
├── gradientboosting_model.joblib
├── xgboost_model.joblib
├── catboost_model.joblib
└── lightgbm_model.joblib
```

These artifacts allow the preprocessing pipeline and trained models to be reused without retraining from scratch.

---

# 🏗️ Project Structure

Recommended GitHub structure:

```text
IMDB-Boosting-Stacking-Ensemble/
│
├── IMDB_Boosting_Stacking_Ensemble.ipynb
│
├── data/
│   ├── IMDB_Dataset load from HuggingFace.
│   
│
├── artifacts/
│   ├── Word2Vec_vectorizer.joblib
│   ├── svd_transformer.joblib
│   ├── stacking_imdb_model.joblib
│   ├── model_comparison.csv
│   ├── adaboost_model.joblib
│   ├── gradientboosting_model.joblib
│   ├── xgboost_model.joblib
│   ├── catboost_model.joblib
│   └── lightgbm_model.joblib
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Data Processing

* NumPy
* Pandas

### NLP

* Word2Vec
* N-grams
* Text preprocessing

### Dimensionality Reduction

* TruncatedSVD

### Machine Learning

* Scikit-learn

### Boosting

* AdaBoost
* Gradient Boosting
* XGBoost
* CatBoost
* LightGBM

### Visualization

* Matplotlib
* Seaborn

### Model Serialization

* Joblib

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/IMDB-Boosting-Stacking-Ensemble.git
```

Move into the project directory:

```bash
cd IMDB-Boosting-Stacking-Ensemble
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Or directly:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk xgboost catboost lightgbm joblib
```

---

# ▶️ Running the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
IMDB_Boosting_Stacking_Ensemble.ipynb
```

The Dataset is necessary:

```python
from dataset load_dataset
dataset = load_dataset("stanfordnlp/imdb")
```

Then run the notebook cells sequentially.

---

# ⚙️ Complete Machine Learning Pipeline

```text
                IMDB Dataset
                     │
                     ▼
              Data Validation
                     │
                     ▼
             Duplicate Removal
                     │
                     ▼
              Text Cleaning
                     │
                     ▼
             Word2Vec Vectorizer
               1-2 Gram
                     │
                     ▼
               TruncatedSVD
                300 Features
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
      AdaBoost    XGBoost    CatBoost
          │          │          │
          ├──── Gradient ───────┤
          │       Boosting      │
          │                     │
          └───── LightGBM ─────┘
                     │
                     ▼
             StackingClassifier
                     │
                     ▼
             Logistic Regression
                Meta-Learner
                     │
                     ▼
              Final Prediction
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       Accuracy    ROC-AUC   Confusion Matrix
```

---

# 🌟 Key Features

* ✅ IMDB sentiment classification
* ✅ Separate train/test datasets
* ✅ Text preprocessing
* ✅ TF-IDF feature extraction
* ✅ Unigram + Bigram features
* ✅ TruncatedSVD dimensionality reduction
* ✅ Five boosting algorithms
* ✅ Stacking ensemble
* ✅ Logistic Regression meta-learner
* ✅ Accuracy comparison
* ✅ ROC-AUC comparison
* ✅ Classification report
* ✅ Confusion matrix
* ✅ ROC curve
* ✅ Prediction function
* ✅ Model serialization with Joblib
* ✅ GitHub-ready project structure

---

# 📚 What This Project Demonstrates

This project demonstrates how traditional NLP and ensemble machine learning can be combined into a complete text-classification pipeline.

You can learn:

```text
NLP
 │
 ├── Text Cleaning
 ├── TF-IDF
 └── N-grams
       │
       ▼
Feature Engineering
       │
       └── TruncatedSVD
              │
              ▼
      Ensemble Learning
              │
       ├── AdaBoost
       ├── GradientBoosting
       ├── XGBoost
       ├── CatBoost
       └── LightGBM
              │
              ▼
       Stacking Ensemble
              │
              ▼
      Logistic Regression
              │
              ▼
      Sentiment Prediction
```

---

# 🚀 Future Improvements

Possible extensions include:

* Hyperparameter tuning using GridSearchCV
* RandomizedSearchCV
* Word2Vec features
* FastText embeddings
* BERT embeddings
* DistilBERT
* Transformer-based sentiment classification
* Feature selection
* Calibration of predicted probabilities
* Cross-validation comparison
* Streamlit deployment
* FastAPI REST API
* Docker deployment
* Cloud deployment

---

# ⚠️ Important Note

The notebook's final model performance should be reported using the **actual results produced after running the notebook**.

Therefore, accuracy, precision, recall, F1-score and ROC-AUC values should not be hard-coded into this README unless they have been obtained from the executed experiment.

---

# 👨‍💻 Author

**Md Emon Islam**

Machine Learning & Deep Learning Enthusiast

Areas of interest:

* Machine Learning
* Deep Learning
* Natural Language Processing
* Computer Vision
* Generative AI
* AI Engineering

---

# ⭐ Project Summary

**IMDB Boosting + Stacking Ensemble** is an NLP machine-learning project that combines **TF-IDF + TruncatedSVD** with five boosting algorithms and a **StackingClassifier** to perform movie-review sentiment classification.

The project provides a complete workflow from raw text preprocessing to feature engineering, model training, ensemble learning, evaluation, prediction, and model persistence.

