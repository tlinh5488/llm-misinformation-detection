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
st.markdown(
"""
Detect whether a news article is **Real** or **Fake** using fine-tuned Transformer models.

Models available:
- **BERT (fine-tuned)**
- **RoBERTa / DistilRoBERTa (fine-tuned)**
"""
)

st.markdown("---")


# ======================
# Device
# ======================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================
# Model selection
# ======================

model_option = st.selectbox(
    "Select Model",
    ["BERT", "RoBERTa"]
)

if model_option == "BERT":
    model_path = "results/bert_results"
    model_info = "BERT-base fine-tuned on FakeNews dataset"
else:
    model_path = "results/roberta_results"
    model_info = "RoBERTa / DistilRoBERTa fine-tuned on FakeNews dataset"

st.info(model_info)


# ======================
# Load model
# ======================

@st.cache_resource
def load_model(path):

    tokenizer = AutoTokenizer.from_pretrained(path)

    model = AutoModelForSequenceClassification.from_pretrained(path)

    model.to(device)
    model.eval()

    return tokenizer, model


tokenizer, model = load_model(model_path)


# ======================
# Prediction function
# ======================

def predict(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=96
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
# Input text
# ======================

text = st.text_area(
    "Enter News Article",
    placeholder="Paste a news article or statement here...",
    height=180
)


# ======================
# Detect button
# ======================

if st.button("🔍 Detect Fake News"):

    if text.strip() == "":
        st.warning("Please enter a news text first.")
    else:

        with st.spinner("Analyzing text..."):

            label, confidence, probs = predict(text)

        st.markdown("## Prediction Result")

        if label == "Fake News":
            st.error(f"⚠️ {label}")
        else:
            st.success(f"✅ {label}")

        st.write(f"Confidence: **{confidence*100:.2f}%**")

        st.markdown("### Probability Distribution")

        real_prob = probs[0][0].item()
        fake_prob = probs[0][1].item()

        st.write("Real News")
        st.progress(real_prob)

        st.write("Fake News")
        st.progress(fake_prob)


# ======================
# Example texts
# ======================

st.markdown("---")
st.markdown("### Example News")

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