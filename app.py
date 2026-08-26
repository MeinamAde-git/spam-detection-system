import os
import re
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Multi-Category Threat & Explainable Classifier",
    page_icon="🛡️",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .metric-card {
        background: #1e2530;
        padding: 16px;
        border-radius: 10px;
        margin-bottom: 12px;
    }
    .highlight-spam {
        background-color: rgba(220, 53, 69, 0.35);
        color: #ff8b94;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
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

st.title("🛡️ Threat Detection & Explainable AI (XAI)")
st.caption("Classifies emails across 4 categories and surfaces suspicious keywords driving the threat score.")

if pipeline is None:
    st.error("Model artifact not found. Please ensure `outputs/multiclass_spam_model.pkl` exists.")
    st.stop()

model = pipeline["model"]
vectorizer = pipeline["vectorizer"]
encoder = pipeline["encoder"]

# Presets
presets = {
    "Select a preset sample...": "",
    "Phishing: Urgent Account Suspension": "SECURITY ALERT: Unauthorized login attempt detected. Your bank access is suspended. Click here to verify your identity and password immediately.",
    "Spam: Lottery Winner Scam": "Congratulations! You won $1,000,000 in the international cash draw. Claim your funds immediately by replying now!",
    "Promotions: Limited Discount": "Flash Deal: Get 50% off all developer certifications this week only with promo code SAVE50. Limited time offer!",
    "Primary: Code Review Sync": "Hi team, please review the PR submitted for the caching layer optimization. Let me know if changes are needed."
}

selected_preset = st.selectbox("⚡ Quick Test Presets:", list(presets.keys()))
default_text = presets[selected_preset] if selected_preset != "Select a preset sample..." else ""

email_input = st.text_area(
    "Enter Email Text to Analyze:",
    value=default_text,
    height=150,
    placeholder="Paste email text here..."
)

col_btn, _ = st.columns([1, 4])
with col_btn:
    analyze_btn = st.button("🚀 Analyze Email", use_container_width=True, type="primary")


def highlight_top_keywords(text, vectorizer, top_n=6):
    words = re.findall(r'\b\w+\b', text)
    feature_names = set(vectorizer.get_feature_names_out())

    # Identify matching vocabulary words in input
    detected = [w for w in words if w.lower() in feature_names]
    highlight_set = set(detected[:top_n])

    highlighted_text = text
    for word in highlight_set:
        pattern = re.compile(rf'\b({re.escape(word)})\b', re.IGNORECASE)
        highlighted_text = pattern.sub(r'<span class="highlight-spam">\1</span>', highlighted_text)

    return highlighted_text, list(highlight_set)


if analyze_btn and email_input.strip():
    vec_input = vectorizer.transform([email_input])
    pred_idx = model.predict(vec_input)[0]
    pred_label = encoder.inverse_transform([pred_idx])[0]
    probs = model.predict_proba(vec_input)[0]

    prob_dict = dict(zip(encoder.classes_, probs))
    threat_score = int((prob_dict.get("Phishing", 0) + prob_dict.get("Spam", 0)) * 100)

    status_config = {
        "Primary": ("✅ Primary Email", "#28a745", "Legitimate personal or business communication."),
        "Spam": ("⚠️ Spam Message", "#ffc107", "Unsolicited promotional scam or commercial advertisement."),
        "Phishing": ("🚨 High-Risk Phishing Threat", "#dc3545",
                     "Malicious attempt to harvest credentials or sensitive details."),
        "Promotions": ("🏷️ Marketing / Promotion", "#17a2b8", "Legitimate newsletter, deal, or commercial update.")
    }

    badge_title, badge_color, badge_desc = status_config.get(pred_label, (pred_label, "#6c757d", ""))

    st.markdown("---")

    # Threat Metrics Bar
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Predicted Category", pred_label)
    with m2:
        st.metric("Model Confidence", f"{max(probs) * 100:.1f}%")
    with m3:
        st.metric("Overall Threat Index", f"{threat_score}/100")

    st.progress(threat_score / 100)

    res_col1, res_col2 = st.columns([1.2, 1])

    with res_col1:
        st.subheader("🔍 Explainability & Token Analysis")
        annotated_text, flagged_tokens = highlight_top_keywords(email_input, vectorizer)

        st.markdown(
            f"""
            <div style="background-color: #1a1e24; border-radius: 8px; padding: 16px; border: 1px solid #333; line-height: 1.6;">
                {annotated_text}
            </div>
            """,
            unsafe_allow_html=True
        )
        if flagged_tokens:
            st.caption(f"**Key Driver Tokens Identified:** `{', '.join(flagged_tokens)}`")

    with res_col2:
        st.subheader("📊 Probability Distribution")
        prob_df = pd.DataFrame({
            "Category": encoder.classes_,
            "Score": probs
        }).sort_values(by="Score", ascending=False)
        st.bar_chart(prob_df.set_index("Category"), color="#4e8cff")

elif analyze_btn and not email_input.strip():
    st.warning("Please enter text or select a preset before analyzing.")