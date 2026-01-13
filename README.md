# Question Answering with Transformers (Generative QA)

## Project Overview

This project implements a **Generative Question Answering system** using **Transformer-based Seq2Seq models (T5 / FLAN-T5)**.
Instead of extracting an exact answer span from the text, the system **generates a natural-language answer** based on semantic understanding of the provided context.

The application accepts a **context (short or long document)** and a **question**, then produces a **clear, paraphrased answer** written in the model’s own words.

A **Streamlit web interface** is provided for interactive testing with pasted text or uploaded TXT files.

---

## Tools & Libraries

* **Python 3.x**
* **Hugging Face Transformers**
* **Hugging Face Tokenizers**
* **Streamlit**
* **PyTorch (via Transformers)**

---

## Model

* **Model type:** Seq2Seq (Text-to-Text)
* **Primary model used:** `google/flan-t5-small`
* **Task:** Generative Question Answering (not span extraction)

The model is prompted to:

* Answer questions **only using the given context**
* **Paraphrase** information instead of copying sentences
* Produce **coherent natural-language answers**, even for multi-sentence responses

---

## Key Design Choices

* **Generative QA instead of extractive QA**
  Answers are generated as full sentences rather than copied spans from the context.

* **Chunking for long documents**
  Long contexts are split into overlapping chunks to stay within model limits.

* **Prompt-controlled answer style**
  The user can choose how the answer is written:

  * Direct answer
  * Detailed explanation
  * Step-by-step reasoning
  * Summarize then answer

* **Hallucination control**
  The prompt explicitly restricts answers to the provided context and discourages unsupported claims.

---

## Streamlit Application

A lightweight Streamlit app is used to test the system interactively.

### Features

* Paste a **context/passage** directly into the app
* Upload a **TXT file** as context
* Enter a custom **question**
* Choose an **answer style**
* Control **maximum answer length**
* Get a **clear, paraphrased answer** generated from the context

### Run the App

```bash
streamlit run streamlit_app.py
```

---

## Example Behavior

**Input:**

* Context: A multi-paragraph article about Artificial Intelligence
* Question: *How has recent technological progress contributed to the growth of AI, and what challenges remain?*

**Output:**

* A coherent paragraph explaining:

  * How GPUs, cloud computing, and large datasets accelerated AI development
  * What challenges still exist, such as bias, hallucinations, and lack of true understanding

The answer is **factually grounded in the context** and **not copied verbatim**.

---

## Covered Topics

* Generative Question Answering
* Transformer-based NLP (T5 / FLAN-T5)
* Prompt engineering for controlled generation
* Handling long documents with chunking
* Interactive deployment using Streamlit

---

## Limitations

* The project uses a **small model (flan-t5-small)**, so answers may be concise.
* The system does not perform formal evaluation (EM / F1), since answers are **generative**, not extractive.
* Extremely long or ambiguous questions may require a larger model for best results.

---

## Possible Improvements

* Upgrade to **flan-t5-base** or **flan-t5-large** for richer answers
* Add automatic **answer confidence scoring**
* Implement **answer comparison across chunks** for better aggregation
* Add logging or evaluation on a custom QA dataset

---

### Final Note

This project focuses on **practical, human-readable question answering**, prioritizing clarity and correctness over benchmark scores.
It demonstrates how modern Transformer models can be used to build a usable QA system with minimal infrastructure.
