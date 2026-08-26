import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Multi-Category Email & Threat Classifier",
    page_icon="🛡️",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background: #1e2530;
        padding: 16px;
        border-radius: 10px;
        border-left: 5px solid #4e8cff;
        margin-bottom: 12px;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    model_path = os.path.join("outputs", "multiclass_spam_model.pkl")
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


pipeline = load_pipeline()

st.title("🛡️ Multi-Category Threat & Email Classifier")
st.caption("Categorizes email text into Primary, Spam, Phishing, or Promotional threats using an MLP Classifier.")

if pipeline is None:
    st.error("Model artifact not found. Please run `python3 model.py` to generate the model file.")
    st.stop()

model = pipeline["model"]
vectorizer = pipeline["vectorizer"]
encoder = pipeline["encoder"]

# Preset test templates
presets = {
    "Select a preset sample...": "",
    "Primary: Code Review Request": "Hi team, please review the PR submitted for the caching layer optimization. Let me know if changes are needed.",
    "Spam: Lottery Winner Scam": "Congratulations! You won $1,000,000 in the international cash draw. Claim your funds immediately by replying now!",
    "Phishing: Urgent Account Suspension": "SECURITY ALERT: Unauthorized login attempt detected. Your bank access is suspended. Click here to verify your identity and password.",
    "Promotions: Discount Coupon": "Flash Deal: Get 50% off all developer certifications this week only with promo code SAVE50. Limited time offer!"
}

selected_preset = st.selectbox("⚡ Quick Test Presets:", list(presets.keys()))
default_text = presets[selected_preset] if selected_preset != "Select a preset sample..." else ""

email_input = st.text_area(
    "Enter Email Text to Analyze:",
    value=default_text,
    height=160,
    placeholder="Paste email header and body here..."
)

col1, col2 = st.columns([1, 4])
with col1:
    classify_btn = st.button("🚀 Analyze Email", use_container_width=True, type="primary")

if classify_btn and email_input.strip():
    # Preprocess & Predict
    vec_input = vectorizer.transform([email_input])
    pred_idx = model.predict(vec_input)[0]
    pred_label = encoder.inverse_transform([pred_idx])[0]

    # Probabilities
    probs = model.predict_proba(vec_input)[0]
    prob_df = pd.DataFrame({
        "Category": encoder.classes_,
        "Confidence Score": probs
    }).sort_values(by="Confidence Score", ascending=False)

    st.markdown("---")

    # Category Badges & Color Cues
    status_config = {
        "Primary": ("✅ Primary Email", "#28a745", "Legitimate personal or business communication."),
        "Spam": ("⚠️ Spam Message", "#ffc107", "Unsolicited promotional scam or commercial advertisement."),
        "Phishing": ("🚨 High-Risk Phishing Threat", "#dc3545",
                     "Malicious attempt to harvest credentials or financial data."),
        "Promotions": ("🏷️ Marketing / Promotion", "#17a2b8",
                       "Legitimate marketing newsletter, coupon, or sale announcement.")
    }

    badge_title, badge_color, badge_desc = status_config.get(pred_label, (pred_label, "#6c757d", ""))

    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        st.subheader("Classification Result")
        st.markdown(
            f"""
            <div style="background-color: {badge_color}22; border: 2px solid {badge_color}; border-radius: 8px; padding: 18px; margin-bottom: 12px;">
                <h3 style="color: {badge_color}; margin: 0;">{badge_title}</h3>
                <p style="margin-top: 8px; font-size: 15px; color: #ddd;">{badge_desc}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.metric("Top Classification Confidence", f"{max(probs) * 100:.2f}%")

    with res_col2:
        st.subheader("Category Probabilities")
        st.bar_chart(prob_df.set_index("Category"), color="#4e8cff")

elif classify_btn and not email_input.strip():
    st.warning("Please enter text or select a preset sample before analyzing.")