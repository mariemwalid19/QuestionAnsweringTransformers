# app.py
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import os, json
import evaluate
import pandas as pd
from pathlib import Path
from typing import Dict, List

st.set_page_config(page_title="QA Demo — SQuAD / Transformers", layout="wide")

local_model_path = r"E:\Elevvo\QuestionAnsweringTransformers\distilbert-qa-quick-model"

@st.cache_resource(show_spinner=False)
def load_pipeline(model_source: str, use_local: bool = True):
    if use_local and Path(model_source).exists():
        model_path = str(Path(model_source).resolve())
        st.info(f" Using local model from: {model_path}")
    else:
        st.warning(f" Local folder '{model_source}' not found. Falling back to Hugging Face Hub.")
        model_path = model_source

    qa = pipeline("question-answering", model=model_path, tokenizer=model_path, device=-1)
    return qa


def predict_answer(qa_pipeline, question: str, context: str, max_answer_len: int = 64):
    if not question or not context:
        return {"answer": "", "score": 0.0}
    try:
        out = qa_pipeline({"question": question, "context": context}, max_answer_len=max_answer_len)
        return {"answer": out.get("answer", ""), "score": float(out.get("score", 0.0))}
    except Exception as e:
        return {"answer": f"ERROR: {e}", "score": 0.0}

def read_squad_json(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        js = json.load(f)
    rows = []
    for art in js.get("data", []):
        for para in art.get("paragraphs", []):
            ctx = para["context"]
            for qa in para["qas"]:
                rows.append({
                    "id": qa.get("id"),
                    "question": qa.get("question"),
                    "context": ctx,
                    "answers_text": [a["text"] for a in qa.get("answers", [])],
                    "answers_start": [a["answer_start"] for a in qa.get("answers", [])]
                })
    return rows

def evaluate_on_squad(qa_pipeline, squad_rows: List[Dict], sample: int = None):
    metric = evaluate.load("squad")
    preds, refs = [], []
    rows = squad_rows if sample is None else squad_rows[:sample]
    for i, r in enumerate(rows):
        q = r["question"]
        ctx = r["context"]
        gold = r["answers_text"]
        try:
            out = qa_pipeline({"question": q, "context": ctx}, max_answer_len=60)
            pred_txt = out.get("answer", "")
        except Exception:
            pred_txt = ""
        preds.append({"id": str(i), "prediction_text": pred_txt})
        refs.append({"id": str(i), "answers": {"text": gold, "answer_start": r.get("answers_start", [])}})
    return metric.compute(predictions=preds, references=refs)

st.title("Task 6 — Question Answering (SQuAD) — Streamlit Demo")
st.markdown(
    """
This app:  
- loads a fine-tuned QA model (default: local `distilbert-qa-quick-model/` if present),  
- answers a user question given a passage, and  
- can evaluate EM & F1 on an uploaded SQuAD v1.1 JSON file (or a sample of it).
"""
)

st.sidebar.header("Model & Options")
use_local_checkbox = st.sidebar.checkbox("Prefer local model if present", value=True)
model_choice = st.sidebar.selectbox(
    "Model (local folder or HF model name):",
    options=[local_model_path, "distilbert-base-uncased", "bert-base-uncased", "roberta-base", "albert-base-v2"],
    index=0
)
st.sidebar.markdown("If you select a Hugging Face name the model will be downloaded (internet required).")

with st.spinner("Loading model pipeline (cached)..."):
    qa_pipe = load_pipeline(model_choice, use_local=use_local_checkbox)

st.subheader("Interactive QA")
col1, col2 = st.columns([1, 1])
with col1:
    context_text = st.text_area("Paste the context / passage here (long texts OK)", height=260)
with col2:
    question_text = st.text_input("Question", value="")
    max_len = st.slider("Max answer length (tokens)", 5, 128, 64)
    if st.button("Get Answer"):
        if not context_text.strip() or not question_text.strip():
            st.warning("Please paste a context and type a question.")
        else:
            with st.spinner("Running model..."):
                res = predict_answer(qa_pipe, question_text.strip(), context_text.strip(), max_answer_len=max_len)
            st.markdown("**Answer:**")
            st.success(res["answer"])
            st.write(f"Confidence score: {res['score']:.4f}")

st.markdown("---")
st.subheader("Batch Evaluation (Exact Match & F1)")

st.markdown(
    """
Upload a **SQuAD v1.1 JSON** file (train or dev).  
For speed, the app evaluates a small sample by default (you can increase or evaluate full set if you have CPU/GPU time).
"""
)

uploaded_file = st.file_uploader("Upload SQuAD v1.1 JSON (optional)", type=["json"])
sample_size = st.number_input("Number of examples to evaluate (small → fast)", min_value=20, max_value=5000, value=200, step=20)
evaluate_button = st.button("Run Evaluation on Uploaded File")

if uploaded_file is not None:
    tmp_path = Path("uploaded_squad.json")
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("Uploaded SQuAD file saved.")
    try:
        squad_rows = read_squad_json(tmp_path)
        st.info(f"Parsed {len(squad_rows)} QA pairs from uploaded file.")
    except Exception as e:
        st.error(f"Failed to parse JSON: {e}")
        squad_rows = []
else:
    squad_rows = []

if evaluate_button:
    if not squad_rows:
        st.error("Upload a SQuAD JSON file first.")
    else:
        st.info(f"Running evaluation (sample={sample_size}) — this may take a while.")
        with st.spinner("Running QA pipeline over examples..."):
            metrics = evaluate_on_squad(qa_pipe, squad_rows, sample=min(sample_size, len(squad_rows)))
        st.success("Evaluation complete.")
        st.write(metrics)

st.markdown("---")
st.subheader("Tips & Notes")
st.markdown(
    """
- For a fast demo, use `distilbert-base-uncased` or your local `distilbert-qa-quick-model/`.  
- For best accuracy, fine-tune `bert-base-uncased`/`roberta-base` on the whole SQuAD train set (requires GPU & more time).  
- The evaluation shown here uses the SQuAD v1.1 metric (Exact Match & F1) as requested in Task 6.  
"""
)
