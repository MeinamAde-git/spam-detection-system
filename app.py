import os
import re
from urllib.parse import urlparse
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="ThreatIntel: Multi-Class Email & Security Heuristics Scanner",
    page_icon="🛡️",
    layout="wide"
)

# Custom Theme and UI Elements
st.markdown("""
    <style>
    .metric-card {
        background: #1e2530;
        padding: 16px;
        border-radius: 8px;
        border-left: 4px solid #4e8cff;
        margin-bottom: 10px;
    }
    .highlight-spam {
        background-color: rgba(220, 53, 69, 0.35);
        color: #ff8b94;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-tag {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
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

st.title("🛡️ ThreatIntel: Multi-Class & Heuristic Email Scanner")
st.caption(
    "Combines Scikit-Learn MLP classification with real-time heuristic security audits, URL inspection, and psychological trigger detection.")

if pipeline is None:
    st.error("Model pipeline not found. Please ensure `outputs/multiclass_spam_model.pkl` exists.")
    st.stop()

model = pipeline["model"]
vectorizer = pipeline["vectorizer"]
encoder = pipeline["encoder"]


# ----------------- HEURISTIC ENGINES ----------------- #

def analyze_urls(text):
    """Extracts and evaluates URLs for suspicious phishing characteristics."""
    url_pattern = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')
    found_urls = url_pattern.findall(text)

    shorteners = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "buff.ly", "ow.ly"}
    results = []

    for u in found_urls:
        parsed = urlparse(u if u.startswith("http") else "http://" + u)
        domain = parsed.netloc.lower()

        is_ip = bool(re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', domain))
        is_shortener = any(s in domain for s in shorteners)
        has_at_symbol = "@" in u
        is_suspicious_tld = domain.endswith((".xyz", ".top", ".club", ".work", ".click", ".buzz"))

        risk_flags = []
        if is_ip:
            risk_flags.append("Raw IP Address Domain")
        if is_shortener:
            risk_flags.append("Obfuscated URL Shortener")
        if has_at_symbol:
            risk_flags.append("Credentials in URL (@ sign)")
        if is_suspicious_tld:
            risk_flags.append("High-Risk TLD")

        results.append({
            "url": u,
            "domain": domain,
            "flags": risk_flags,
            "is_malicious": len(risk_flags) > 0
        })
    return results


def analyze_heuristics(text):
    """Calculates structural metrics and psychological triggers."""
    caps_count = sum(1 for c in text if c.isupper())
    letters_count = sum(1 for c in text if c.isalpha())
    caps_ratio = (caps_count / letters_count * 100) if letters_count > 0 else 0

    exclamation_count = text.count("!")
    dollar_count = text.count("$")

    urgency_keywords = ["urgent", "immediately", "immediate", "suspended", "action required", "24 hours",
                        "unauthorized", "verify now", "expires"]
    lure_keywords = ["congratulations", "winner", "cash", "free gift", "claim now", "guaranteed", "inheritance",
                     "$1,000,000"]

    detected_urgency = [w for w in urgency_keywords if re.search(rf'\b{w}\b', text, re.I)]
    detected_lures = [w for w in lure_keywords if re.search(rf'\b{w}\b', text, re.I)]

    return {
        "caps_ratio": caps_ratio,
        "exclamation_count": exclamation_count,
        "dollar_count": dollar_count,
        "detected_urgency": detected_urgency,
        "detected_lures": detected_lures
    }


def highlight_top_keywords(text, vectorizer, top_n=6):
    words = re.findall(r'\b\w+\b', text)
    feature_names = set(vectorizer.get_feature_names_out())
    detected = [w for w in words if w.lower() in feature_names]
    highlight_set = set(detected[:top_n])

    highlighted_text = text
    for word in highlight_set:
        pattern = re.compile(rf'\b({re.escape(word)})\b', re.IGNORECASE)
        highlighted_text = pattern.sub(r'<span class="highlight-spam">\1</span>', highlighted_text)

    return highlighted_text, list(highlight_set)


# ----------------- UI CONTROLS ----------------- #

presets = {
    "Select a preset sample...": "",
    "🚨 Phishing with Malicious URL & Urgency": "SECURITY ALERT: Unauthorized login attempt detected from 192.168.1.1. Your bank access is suspended immediately. Verify password here: http://bit.ly/secure-bank-login-auth",
    "⚠️ Spam: Cash Giveaway": "CONGRATULATIONS! You won $1,000,000 in our international cash draw! Claim your guaranteed payout now at http://185.220.101.5/payout!",
    "🏷️ Promotions: Flash Discount": "Flash Deal: Get 50% off all developer cloud certifications this week only with code SAVE50. Visit https://store.example.com/deals to redeem.",
    "✅ Primary: Code Architecture Review": "Hi team, please review the PR submitted for the caching layer optimization. Let me know if further adjustments are required before merge."
}

selected_preset = st.selectbox("⚡ Load Security Preset Test:", list(presets.keys()))
default_text = presets[selected_preset] if selected_preset != "Select a preset sample..." else ""

email_input = st.text_area(
    "Email Content Inspector:",
    value=default_text,
    height=150,
    placeholder="Paste email text, raw headers, or messages with links..."
)

col_run, _ = st.columns([1, 4])
with col_run:
    run_btn = st.button("🔍 Run Full Security Audit", use_container_width=True, type="primary")

if run_btn and email_input.strip():
    # Model Inference
    vec_input = vectorizer.transform([email_input])
    pred_idx = model.predict(vec_input)[0]
    pred_label = encoder.inverse_transform([pred_idx])[0]
    probs = model.predict_proba(vec_input)[0]
    prob_dict = dict(zip(encoder.classes_, probs))

    # Heuristic Audits
    url_audit = analyze_urls(email_input)
    heuristics = analyze_heuristics(email_input)

    # Dynamic Composite Threat Index
    base_threat = int((prob_dict.get("Phishing", 0) + prob_dict.get("Spam", 0)) * 75)
    heuristic_penalty = (len(url_audit) * 10) + (len(heuristics["detected_urgency"]) * 5) + (
        5 if heuristics["caps_ratio"] > 30 else 0)
    total_threat_score = min(100, base_threat + heuristic_penalty)

    st.markdown("---")

    # Top Scorecards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Classification", pred_label)
    with c2:
        st.metric("ML Model Confidence", f"{max(probs) * 100:.1f}%")
    with c3:
        st.metric("Composite Threat Index", f"{total_threat_score}/100")
    with c4:
        st.metric("Detected URLs", f"{len(url_audit)}")

    st.progress(total_threat_score / 100)

    # Main Dashboard Columns
    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        st.subheader("📝 Explainable Content & Token Highlighter")
        annotated_text, flagged_tokens = highlight_top_keywords(email_input, vectorizer)
        st.markdown(
            f"""
            <div style="background-color: #1a1e24; border-radius: 8px; padding: 16px; border: 1px solid #333; line-height: 1.6; font-size: 15px;">
                {annotated_text}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🔗 URL & Domain Intelligence")
        if url_audit:
            for item in url_audit:
                domain_status = "🚨 Flagged" if item["is_malicious"] else "✅ Standard"
                st.write(f"**URL:** `{item['url']}` — **Status:** {domain_status}")
                if item["flags"]:
                    for flag in item["flags"]:
                        st.markdown(
                            f'<span class="badge-tag" style="background-color:#721c24; color:#f8d7da;">⚠️ {flag}</span>',
                            unsafe_allow_html=True)
                else:
                    st.caption("No direct blacklist patterns or URL obfuscation detected.")
        else:
            st.info("No embedded links found in the text.")

    with col_right:
        st.subheader("📊 Category Probabilities")
        prob_df = pd.DataFrame({
            "Category": encoder.classes_,
            "Confidence": probs
        }).sort_values(by="Confidence", ascending=False)
        st.bar_chart(prob_df.set_index("Category"), color="#4e8cff")

        st.subheader("⚡ Heuristic Triggers")
        st.write(
            f"**Caps Intensity:** `{heuristics['caps_ratio']:.1f}%` | **Exclamations:** `{heuristics['exclamation_count']}` | **Currency Symbols:** `{heuristics['dollar_count']}`")

        if heuristics["detected_urgency"]:
            st.markdown(f"**Urgency Cues:** `{'`, `'.join(heuristics['detected_urgency'])}`")
        if heuristics["detected_lures"]:
            st.markdown(f"**Financial/Lure Cues:** `{'`, `'.join(heuristics['detected_lures'])}`")
        if not heuristics["detected_urgency"] and not heuristics["detected_lures"]:
            st.caption("No psychological urgency or lure keywords triggered.")

elif run_btn and not email_input.strip():
    st.warning("Please provide email text before running the security scan.")