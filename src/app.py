import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ======================
# Page config
# ======================

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detection System")

st.write(
    "Detect whether a news text is **Real** or **Fake** using Transformer models."
)


# ======================
# Device
# ======================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================
# Model selection
# ======================

model_option = st.selectbox(
    "Choose model",
    ["BERT", "RoBERTa"]
)

if model_option == "BERT":
    model_path = "results/bert_results"
else:
    model_path = "results/roberta_results"


# ======================
# Load model
# ======================

@st.cache_resource
def load_model(model_path):

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    model.to(device)

    model.eval()

    return tokenizer, model


tokenizer, model = load_model(model_path)


# ======================
# Prediction
# ======================

def predict(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():

        outputs = model(**inputs)

    logits = outputs.logits

    probs = torch.softmax(logits, dim=1)

    pred = torch.argmax(probs, dim=1).item()

    confidence = probs[0][pred].item()

    label = "Real News" if pred == 0 else "Fake News"

    return label, confidence, probs


# ======================
# Text input
# ======================

text = st.text_area(
    "Enter news text",
    height=150
)


# ======================
# Detect button
# ======================

if st.button("Detect"):

    if text.strip() == "":
        st.warning("Please enter some text")

    else:

        with st.spinner("Analyzing news..."):

            label, confidence, probs = predict(text)

        if label == "Fake News":

            st.error(f"⚠️ {label}")

        else:

            st.success(f"✅ {label}")

        st.write(f"Confidence: **{confidence*100:.2f}%**")

        st.write("### Class probabilities")

        st.write(f"Real: {probs[0][0].item()*100:.2f}%")
        st.write(f"Fake: {probs[0][1].item()*100:.2f}%")


# ======================
# Examples
# ======================

st.markdown("---")

st.write("### Example texts")

example_fake = "Scientists confirm that drinking bleach cures COVID-19."

example_real = "The World Health Organization confirmed vaccines are safe and effective."


col1, col2 = st.columns(2)

with col1:

    if st.button("Test Fake Example"):

        label, confidence, _ = predict(example_fake)

        st.write(example_fake)

        st.write(label, f"{confidence*100:.2f}%")

with col2:

    if st.button("Test Real Example"):

        label, confidence, _ = predict(example_real)

        st.write(example_real)

        st.write(label, f"{confidence*100:.2f}%")