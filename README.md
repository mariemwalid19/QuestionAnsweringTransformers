#  Task 6 — Question Answering with Transformers

##  Project Overview

This project implements a **Question Answering (QA) system** using **Transformer-based models** (DistilBERT) fine-tuned on the **SQuAD v1.1 dataset**.
The system can take a **context (passage)** and a **question**, then extract the most relevant **answer span**.

A **Streamlit interface** is built to allow interactive testing and evaluation.

---

##  Tools & Libraries

* **Python 3.12**
* [Hugging Face Transformers](https://huggingface.co/transformers/)
* [Tokenizers](https://github.com/huggingface/tokenizers)
* [Datasets / Evaluate](https://huggingface.co/docs/evaluate)
* **Pandas**
* **Streamlit**

---

##  Dataset

* **SQuAD v1.1 (Stanford Question Answering Dataset)** from Kaggle.
* Train set used for fine-tuning (sample: 2,000 examples).
* Dev set used for evaluation (sample: 500 examples).

---

##  Model Training

* **Base model:** `distilbert-base-uncased`
* **Fine-tuning task:** Question Answering (span prediction)
* **Training config (quick run):**

  * Batch size: 8
  * Epochs: 1
  * Learning rate: 3e-5
  * Gradient accumulation: 1
  * CPU training (lightweight, no GPU)
* **Output directory:** `distilbert-qa-quick-model/`

---

##  Results

### Training Loss

* Step 100 → **2.57**
* Step 200 → **2.25**
* Final Training Loss \~ **2.25**

### Evaluation Metrics (dev set, 500 examples)

* **Exact Match (EM):** 41.0%
* **F1 Score:** 54.9%
* **Eval Loss:** 2.50

 The model is able to extract reasonable answers despite short training.

---

##  Streamlit App

A simple **Streamlit web app** was built for interaction.

### Features

* Paste a **context/passage** and enter a **question** → get the extracted answer + confidence score.
* Upload a **SQuAD v1.1 JSON file** → evaluate EM & F1 (sample size configurable).
* Option to use either the **local fine-tuned model** (`distilbert-qa-quick-model`) or Hugging Face pre-trained models (BERT, RoBERTa, ALBERT).

### Run the App

```bash
streamlit run streamlit_app.py
```

---

##  Bonus (Model Comparison)

* **DistilBERT fine-tuned (local)**: EM = 41.0, F1 = 54.9
* **BERT-base (pretrained, no fine-tuning)**: typically \~20–30 EM, \~40 F1 (worse without training).
* **RoBERTa-base (pretrained)**: slightly stronger, but slower to load.

---

##  Covered Topics

* Question Answering
* Span Extraction
* Transformer-based NLP
* Model Evaluation (EM & F1)
* Interactive Deployment (Streamlit)

---

##  Next Improvements

* Train for **more epochs** on full SQuAD → higher EM & F1.
* Try **larger models** (`bert-base`, `roberta-base`).
* Add **GPU acceleration** for faster training/evaluation.
