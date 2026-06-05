import streamlit as st
import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = load_model("lstm_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load max length
with open("max_len.pkl", "rb") as f:
    max_len = pickle.load(f)


# Function to predict next word
def predict_next_word(text):

    seq = tokenizer.texts_to_sequences([text])[0]

    seq = pad_sequences(
        [seq],
        maxlen=max_len - 1,
        padding='pre'
    )

    pred = model.predict(seq, verbose=0)

    pred_index = np.argmax(pred)

    next_word = ""

    for word, index in tokenizer.word_index.items():
        if index == pred_index:
            next_word = word
            break

    return next_word


# Streamlit UI
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Next Word Prediction using LSTM")

st.write("Enter a sentence and predict the next word.")

user_input = st.text_input(
    "Enter Text",
    placeholder="life is"
)

if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:

        word = predict_next_word(user_input)

        st.success(f"Predicted Next Word: **{word}**")

