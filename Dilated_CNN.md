# 🎬 IMDb Sentiment Analysis using Improved Dilated CNN

A Deep Learning based Natural Language Processing (NLP) project for
classifying IMDb movie reviews into **Positive** and **Negative**
sentiments using an **Improved Dilated Convolutional Neural Network
(Dilated CNN)**.

---

## 📌 Project Overview

Sentiment Analysis is a Natural Language Processing (NLP) task that
determines whether a given piece of text expresses a positive or
negative opinion.

In this project, the IMDb movie review dataset is used to build a
deep learning based binary sentiment classification model.

The model uses an improved **Dilated CNN architecture** with:

- TextVectorization
- Bigram features
- Word Embeddings
- Residual Dilated CNN blocks
- Multiple dilation rates
- Batch Normalization
- Spatial Dropout
- Global Max Pooling
- Global Average Pooling
- Attention-based Pooling
- Fully Connected layers
- Validation-based threshold tuning

The complete pipeline is designed to be suitable for experimentation,
learning, GitHub portfolio projects, and further deployment.

---

# 🎯 Objective

The main objective of this project is to develop a robust deep learning
model that can automatically classify IMDb movie reviews as:

- **Positive**
- **Negative**

The project also focuses on reducing misclassification by improving
text preprocessing, model architecture, pooling strategy, training
strategy, and classification threshold selection.

---

# 🧠 Model Architecture

The overall architecture can be summarized as:

```text
Raw IMDb Review
      │
      ▼
Text Cleaning
      │
      ▼
TextVectorization
      │
      ├── Vocabulary
      └── Bigram Features
      │
      ▼
Token IDs
      │
      ▼
Embedding Layer
      │
      ▼
Spatial Dropout
      │
      ▼
Residual Dilated CNN
      │
      ├── Dilation Rate = 1
      ├── Dilation Rate = 2
      ├── Dilation Rate = 4
      ├── Dilation Rate = 8
      └── Dilation Rate = 16
      │
      ▼
Multiple Pooling Strategies
      │
      ├── Global Max Pooling
      ├── Global Average Pooling
      └── Attention Pooling
      │
      ▼
Concatenation
      │
      ▼
Dense Layer
      │
      ▼
Batch Normalization
      │
      ▼
Dropout
      │
      ▼
Dense Layer
      │
      ▼
Sigmoid Output
      │
      ▼
Positive / Negative
