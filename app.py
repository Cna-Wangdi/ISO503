import re
import pickle

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------ CONFIG (must match training) ------------
MAX_SEQ_LEN = 200  # same as in sentiment_pipeline.py

def basic_clean(text: str) -> str:
    """
    Same cleaning logic as training script.
    """
    text = text.lower()
    # keep letters, digits, spaces and apostrophes
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def load_sentiment_model():
    # uses the .keras model you saved
    model = load_model("sentiment_lstm_model.keras")
    return model


@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer


model = load_sentiment_model()
tokenizer = load_tokenizer()

# ------------ UI ------------
st.title("ISY503 Sentiment Analysis Demo")
st.write(
    "Enter a customer review below and click **Analyse Sentiment** "
    "to see if the model predicts it as a positive or negative review."
)

user_text = st.text_area("Review text:", height=150)

if st.button("Analyse Sentiment"):
    if not user_text.strip():
        st.warning("Please type a review first.")
    else:
        cleaned = basic_clean(user_text)
        seq = tokenizer.texts_to_sequences([cleaned])
        pad = pad_sequences(seq, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")
        prob = float(model.predict(pad, verbose=0)[0][0])

        if prob >= 0.5:
            label = "Positive review"
        else:
            label = "Negative review"

        st.subheader(label)
        st.write(f"Model confidence: **{prob:.3f}**")