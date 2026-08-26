import streamlit as st
import pickle
import re
import string
import os

# Set page title and layout
st.set_page_config(page_title="Spam Email Detector", page_icon="🛡️", layout="centered")


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@st.cache_resource
def load_model():
    model_path = os.path.join("outputs", "spam_model.pkl")
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data['model'], data['vectorizer']


st.title("🛡️ AI Spam Email Detector")
st.write("Enter an email or message below to verify whether it is **Spam** or **Ham (Legitimate)**.")

# Example email buttons for quick testing
st.markdown("##### Quick Test Examples:")
col1, col2, col3 = st.columns(3)
example_text = ""
if col1.button("🎁 Claim Lottery"):
    example_text = "Congratulations! You won a $1,000,000 cash prize. Click here to claim your reward immediately!"
if col2.button("📅 Meeting Invite"):
    example_text = "Hi Ade, can we reschedule our sync meeting to 3 PM this Thursday? Let me know."
if col3.button("⚠️ Urgent Account"):
    example_text = "URGENT: Your bank security credentials have been suspended. Verify your identity now."

# Input text box
email_input = st.text_area("Email Content:", value=example_text, height=140, placeholder="Paste email text here...")

if st.button("🔍 Analyze Email", type="primary"):
    if not email_input.strip():
        st.warning("Please enter some text before analyzing.")
    else:
        try:
            model, vectorizer = load_model()
            cleaned = clean_text(email_input)
            features = vectorizer.transform([cleaned]).toarray()

            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            confidence = probabilities[prediction] * 100

            st.divider()
            if prediction == 1:
                st.error(f"🚨 **Prediction: SPAM** ({confidence:.1f}% confidence)")
                st.progress(int(confidence))
                st.info(
                    "⚠️ This message contains language patterns commonly associated with phishing or scam attempts.")
            else:
                st.success(f"✅ **Prediction: HAM / Legitimate** ({confidence:.1f}% confidence)")
                st.progress(int(confidence))
                st.info("🛡️ This message appears safe and resembles normal personal/business correspondence.")

        except Exception as e:
            st.error(f"Error analyzing email: {e}. Make sure 'outputs/spam_model.pkl' exists.")