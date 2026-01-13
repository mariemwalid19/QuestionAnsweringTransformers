import os
os.environ["STREAMLIT_DISABLE_WATCHDOG"] = "true"

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path

if "file_text" not in st.session_state:
    st.session_state.file_text = ""

st.set_page_config(
    page_title="Generative Question Answering — T5", 
    layout="wide"
)

local_model_path = "google/flan-t5-small"

@st.cache_resource(show_spinner=False)
def load_generator(model_name: str):
    generator = pipeline(
        "text2text-generation",
        model=model_name,
        tokenizer=model_name,
        device=-1
    )
    return generator


def predict_answer(generator, question: str, context: str, max_len: int):
    if not question or not context:
        return ""
    prompt = f"question: {question} context: {context}"
    output = generator(
        prompt,
        max_length = max_len,
        num_beams = 4,
        do_sample = False
    )
    return output[0]["generated_text"]


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
    chunks = split_text_into_chunks(context)
    answers = []

    if answer_style == "Direct answer":
        instruction = "Answer the question directly."
    elif answer_style == "Detailed explanation":
        instruction = "Answer in a detailed and explanatory way."
    elif answer_style == "Reasoned answer (step by step)":
        instruction = "Answer step by step and explain your reasoning."
    elif answer_style == "Summarize then answer":
        instruction = "Summarize the relevant information first, then answer the question."
    else:
        instruction = "Answer clearly and concisely."


    for chunk in chunks:
        prompt = f"""
You are an expert reader.
        
Task:
Answer the question based only on the context.
        
Rules:
- Combine multiple ideas.
- Do not copy sentences verbatim.
- Provide a clear, complete answer.
        
Question:
{question}
        
Context:
{chunk}
        
Instruction:
{instruction}
"""
        output = generator(
            prompt,
            max_length=max_len,
            num_beams=4,
            do_sample=False
        )

        answer = output[0]["generated_text"].strip()
        if answer:
            answers.append(answer)

    if not answers:
        return "No answer found."

    def score_answer(ans):
        length_score = min(len(ans.split()), 80)
        structure_bonus = 20 if ans.count(".") >= 1 else 0
        idea_bonus = 10 if ans.count(",") >= 1 else 0
        return length_score + structure_bonus + idea_bonus

    return max(answers, key=score_answer)


st.title("Generative Question Answering — T5")
st.markdown(
    """
This app:
- loads a fine-tuned **generative QA model (T5)**,
- accepts long documents or articles as context,
- generates answers based on semantic understanding.
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
st.sidebar.markdown("Only **Seq2Seq (T5-style)** models are supported.")

with st.spinner("Loading model pipeline (cached)..."):
    generator = load_generator(model_choice)

st.subheader("Interactive QA")
uploaded_file = st.file_uploader(
    "Upload a TXT file (optional)",
    type=["txt"],
    key="txt_uploader"
)

if uploaded_file and st.session_state.file_text == "":
    st.session_state.file_text = uploaded_file.read().decode("utf-8")
    st.success("TXT file loaded successfully.")


col1, col2 = st.columns([1, 1])
with col1:
    context_text = st.text_area("Paste the context / passage here (long texts OK)", height=260)
with col2:
    question_text = st.text_input("Question", value="")
    max_len = st.slider("Max answer length (tokens)", 5, 128, 64)
    answer_style = st.selectbox(
        "Answer style",
        options=[
            "Direct answer",
            "Detailed explanation",
            "Reasoned answer (step by step)",
            "Summarize then answer"
        ]
    )
    if st.button("Get Answer"):
        final_context = ""

        if st.session_state.file_text.strip():
            final_context = st.session_state.file_text
        else:
            final_context = context_text

        if not final_context.strip() or not question_text.strip():
            st.warning("Please provide a context (text or file) and a question.")
        else:
            with st.spinner("Running model..."):
                answer = answer_with_chunking(
                    generator,
                    question_text.strip(),
                    final_context.strip(),
                    max_len,
                    answer_style
                )
            st.markdown("**Answer:**")
            st.success(answer)

st.markdown("---")
st.subheader("Tips & Notes")
st.markdown(
    """
**Notes:**
- This system performs **generative question answering** using a text-to-text Transformer.
- Answers are generated, not extracted verbatim from the passage.
- Large documents can be handled via chunking (future enhancement).
"""
)
