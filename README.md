# 📰 Fake News Detection using Deep Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-Framework-orange?logo=tensorflow&logoColor=white) ![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-brightgreen?logo=kaggle&logoColor=white) ![Status](https://img.shields.io/badge/Status-Completed-success?style=flat)
 
---

## 📌 Overview  
This project implements a **Fake News Detection System** using **Deep Learning (LSTM, BiLSTM, GRU)**.  
It classifies news articles as **Real** or **Fake** by leveraging **NLP preprocessing, TF-IDF, embeddings, and RNN architectures**.  

The notebook is fully reproducible on **Kaggle/Colab** and includes both **classical ML approaches** and **deep learning pipelines**.

---

## 🚀 Features
- ✅ Data preprocessing (cleaning, stopword removal, tokenization)  
- ✅ TF-IDF + Label Encoding for baseline models  
- ✅ Deep Learning models:  
  - LSTM  
  - BiLSTM  
  - GRU  
- ✅ Evaluation with **confusion matrix & classification report**  
- ✅ Kaggle dataset integration  

---

## 📂 Dataset  
Dataset used: [Fake News Detection Dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)  

- **True.csv** → Real news articles  
- **Fake.csv** → Fake news articles  
- Final dataset: Balanced & merged with labels  

---

## ⚙️ Tech Stack
- **Python 3.10+**  
- **TensorFlow / Keras** (Deep Learning models)  
- **Scikit-learn** (TF-IDF, Label Encoding, Evaluation)  
- **Pandas, NumPy** (Data processing)  
- **Matplotlib, Seaborn** (Visualization)  

---

## 📊 Model Architectures
1. **LSTM**  
   - Embedding layer  
   - LSTM (128 units)  
   - Dense + Sigmoid  

2. **GRU**  
   - Embedding layer  
   - GRU (128 units)  
   - Dense + Sigmoid  

3. **BiLSTM**  
   - Embedding layer  
   - Bidirectional LSTM (128 units)  
   - Dense + Sigmoid  

---

## 📈 Results  

- **LSTM**: High accuracy with sequential dependencies  
- **GRU**: Faster training, competitive accuracy  
- **BiLSTM**: Best performance (captures past + future context)  
### Training Performance  

| **Epoch** | **Train Loss** | **Train Accuracy** | **Val Loss** | **Val Accuracy** |
|-----------|----------------|---------------------|--------------|------------------|
| 1         | 0.1497         | 93.95%              | 0.0350       | 98.90%           |
| 2 (partial)| 0.0114        | 99.64%              | -            | -                |

✅ The model quickly converges, achieving **~99% validation accuracy by Epoch 1**.  
✅ Further epochs suggest even higher performance, showing strong generalization.  

---

## 📸 Sample Visualizations
- 📊 Categorical distribution of Fake vs Real news  
- 🧾 Confusion matrix for predictions  

---

## ▶️ Usage
### 1️⃣ Clone repo
```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
