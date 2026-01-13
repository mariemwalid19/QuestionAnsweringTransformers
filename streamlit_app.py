import os
os.environ["STREAMLIT_DISABLE_WATCHDOG"] = "true"

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path
import re

if "file_text" not in st.session_state:
    st.session_state.file_text = ""

st.set_page_config(
    page_title="Generative Question Answering — T5", 
    layout="wide"
)

local_model_path = "google/flan-t5-small"

@st.cache_resource(show_spinner=False)
def load_generator(model_name: str):
    if Path(model_name).exists():
        model_path = str(Path(model_name).resolve())
        st.info(f"Loading local model from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)
    else:
        st.info(f"Loading model from Hugging Face Hub: {model_name}")
        generator = pipeline("text2text-generation", model=model_name, tokenizer=model_name, device=-1)
    return generator


def split_text_into_chunks(text, max_words=300, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunks.append(" ".join(words[start:end]))
        start += max_words - overlap
    return chunks


def answer_with_chunking(generator, question, context, max_len, answer_style):

    if answer_style == "Direct answer":
        instruction = "Answer the question clearly and briefly in your own words."
    elif answer_style == "Detailed explanation":
        instruction = (
            "Answer BOTH parts of the question in one paragraph. "
            "First explain how recent technological progress helped AI grow, "
            "then explain what challenges still remain."
        )
    elif answer_style == "Reasoned answer (step by step)":
        instruction = "Explain the answer step by step in simple language."
    else:
        instruction = "Summarize the relevant ideas, then answer clearly."

    base_prompt = (
        "You are a knowledgeable assistant. Use ONLY the context provided to answer.\n"
        "Write the answer as a coherent paragraph (not bullet points) and paraphrase "
        "— do NOT copy long sentences verbatim from the context.\n\n"
        "Question:\n{question}\n\n"
        "Context:\n{chunk}\n\n"
        "Instruction:\n{instruction}\n"
    )

    chunks = split_text_into_chunks(context, max_words=350, overlap=50)
    answers = []

    for chunk in chunks:
        prompt = base_prompt.format(
            question=question,
            chunk=chunk,
            instruction=instruction
        )

        output = generator(
            prompt,
            max_length=max_len,
            num_beams=4,
            do_sample=False
        )

        ans = output[0]["generated_text"].strip()

        if ans and len(ans.split()) >= 12:
            answers.append(ans)

    if not answers:
        return "The context does not contain enough information to answer the question."

    return max(answers, key=lambda x: len(x.split()))


st.title("Generative Question Answering — T5")
st.markdown(
    """
This app:
- loads a T5-style generative QA model,
- accepts long documents or articles as context,
- generates answers based on semantic understanding (not span extraction).
"""
)


st.sidebar.header("Model & Options")
use_local_checkbox = st.sidebar.checkbox("Prefer local model if present", value=True)
model_choice = st.sidebar.selectbox(
    "Model (local folder or HF model name):",
    options=[
        local_model_path, 
        "t5-small",
        "google/flan-t5-small"
        ],
    index=0
)
st.sidebar.markdown("Only Seq2Seq (T5-style) models are supported.")

with st.spinner("Loading model pipeline (cached)..."):
    model_to_load = model_choice if use_local_checkbox is False else model_choice
    generator = load_generator(model_to_load)

st.subheader("Interactive QA")
uploaded_file = st.file_uploader(
    "Upload a TXT file (optional)",
    type=["txt"],
    key="txt_uploader"
)

if uploaded_file and st.session_state.file_text == "":
    try:
        st.session_state.file_text = uploaded_file.read().decode("utf-8")
        st.success("TXT file loaded successfully.")
    except Exception as e:
        st.error(f"Failed to read file: {e}")


col1, col2 = st.columns([1, 1])
with col1:
    context_text = st.text_area("Paste the context / passage here (long texts OK)", height=300)
with col2:
    question_text = st.text_input("Question", value="")
    max_len = st.slider("Max answer length (tokens)", 32, 512, 128)
    answer_style = st.selectbox("Answer style", options=[
        "Direct answer", "Detailed explanation", "Reasoned answer (step by step)", "Summarize then answer"
    ])
    if st.button("Get Answer"):
        final_context = st.session_state.file_text.strip() if st.session_state.file_text.strip() else context_text.strip()
        if not final_context or not question_text.strip():
            st.warning("Please provide a context (text or file) and a question.")
        else:
            with st.spinner("Running model..."):
                ans = answer_with_chunking(generator, question_text.strip(), final_context, max_len, answer_style)
            st.markdown("**Answer:**")
            st.success(ans)

st.markdown("---")
st.subheader("Tips & Notes")
st.markdown(
    """
- If the model cannot find a clear answer in the context it will return 'Insufficient information in context.'
"""
)
