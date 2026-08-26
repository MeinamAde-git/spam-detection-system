import os
import re
from urllib.parse import urlparse
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="ThreatIntel: AI & Heuristic Email Security Platform",
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

st.title("🛡️ ThreatIntel: Multi-Class Email Security Platform")
st.caption(
    "Enterprise email defense platform integrating Scikit-Learn MLP classification, URL audits, and batch telemetry scanning.")

if pipeline is None:
    st.error("Model pipeline not found. Please ensure `outputs/multiclass_spam_model.pkl` exists.")
    st.stop()

model = pipeline["model"]
vectorizer = pipeline["vectorizer"]
encoder = pipeline["encoder"]


# ----------------- SECURITY ENGINES ----------------- #

def analyze_urls(text):
    url_pattern = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')
    found_urls = url_pattern.findall(str(text))
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
            risk_flags.append("Raw IP Address")
        if is_shortener:
            risk_flags.append("URL Shortener")
        if has_at_symbol:
            risk_flags.append("Embedded Credentials (@)")
        if is_suspicious_tld:
            risk_flags.append("Suspicious TLD")

        results.append({
            "url": u,
            "domain": domain,
            "flags": risk_flags,
            "is_malicious": len(risk_flags) > 0
        })
    return results


def analyze_heuristics(text):
    text_str = str(text)
    caps_count = sum(1 for c in text_str if c.isupper())
    letters_count = sum(1 for c in text_str if c.isalpha())
    caps_ratio = (caps_count / letters_count * 100) if letters_count > 0 else 0

    exclamation_count = text_str.count("!")
    dollar_count = text_str.count("$")

    urgency_keywords = ["urgent", "immediately", "immediate", "suspended", "action required", "24 hours",
                        "unauthorized", "verify now", "expires"]
    lure_keywords = ["congratulations", "winner", "cash", "free gift", "claim now", "guaranteed", "inheritance",
                     "$1,000,000"]

    detected_urgency = [w for w in urgency_keywords if re.search(rf'\b{w}\b', text_str, re.I)]
    detected_lures = [w for w in lure_keywords if re.search(rf'\b{w}\b', text_str, re.I)]

    return {
        "caps_ratio": caps_ratio,
        "exclamation_count": exclamation_count,
        "dollar_count": dollar_count,
        "detected_urgency": detected_urgency,
        "detected_lures": detected_lures
    }


def highlight_top_keywords(text, vectorizer, top_n=6):
    words = re.findall(r'\b\w+\b', str(text))
    feature_names = set(vectorizer.get_feature_names_out())
    detected = [w for w in words if w.lower() in feature_names]
    highlight_set = set(detected[:top_n])

    highlighted_text = str(text)
    for word in highlight_set:
        pattern = re.compile(rf'\b({re.escape(word)})\b', re.IGNORECASE)
        highlighted_text = pattern.sub(r'<span class="highlight-spam">\1</span>', highlighted_text)

    return highlighted_text, list(highlight_set)


# ----------------- APP MODES ----------------- #

tab1, tab2 = st.tabs(["🔍 Single Email Inspector", "📁 Bulk CSV Threat Scanner"])

# ================= TAB 1: SINGLE EMAIL INSPECTOR =================
with tab1:
    presets = {
        "Select a preset sample...": "",
        "🚨 Phishing with Suspicious URL & Urgency": "SECURITY ALERT: Unauthorized login attempt detected from 192.168.1.1. Your bank access is suspended immediately. Verify password here: http://bit.ly/secure-bank-login-auth",
        "⚠️ Spam: Cash Lottery Scam": "CONGRATULATIONS! You won $1,000,000 in our international cash draw! Claim your payout now at http://185.220.101.5/payout!",
        "🏷️ Promotions: Flash Discount": "Flash Deal: Get 50% off all developer cloud certifications this week only with code SAVE50. Visit https://store.example.com/deals to redeem.",
        "✅ Primary: Code Architecture Review": "Hi team, please review the PR submitted for the caching layer optimization. Let me know if further adjustments are required before merge."
    }

    selected_preset = st.selectbox("⚡ Load Security Preset Test:", list(presets.keys()))
    default_text = presets[selected_preset] if selected_preset != "Select a preset sample..." else ""

    email_input = st.text_area(
        "Email Content Inspector:",
        value=default_text,
        height=140,
        placeholder="Paste raw email content or messages with links..."
    )

    col_run, _ = st.columns([1, 4])
    with col_run:
        run_btn = st.button("🔍 Run Full Security Audit", use_container_width=True, type="primary")

    if run_btn and email_input.strip():
        vec_input = vectorizer.transform([email_input])
        pred_idx = model.predict(vec_input)[0]
        pred_label = encoder.inverse_transform([pred_idx])[0]
        probs = model.predict_proba(vec_input)[0]
        prob_dict = dict(zip(encoder.classes_, probs))

        url_audit = analyze_urls(email_input)
        heuristics = analyze_heuristics(email_input)

        base_threat = int((prob_dict.get("Phishing", 0) + prob_dict.get("Spam", 0)) * 75)
        heuristic_penalty = (len(url_audit) * 10) + (len(heuristics["detected_urgency"]) * 5) + (
            5 if heuristics["caps_ratio"] > 30 else 0)
        total_threat_score = min(100, base_threat + heuristic_penalty)

        st.markdown("---")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Classification", pred_label)
        with c2:
            st.metric("ML Model Confidence", f"{max(probs) * 100:.1f}%")
        with c3:
            st.metric("Threat Index", f"{total_threat_score}/100")
        with c4:
            st.metric("Detected URLs", f"{len(url_audit)}")

        st.progress(total_threat_score / 100)

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
                f"**Caps Intensity:** `{heuristics['caps_ratio']:.1f}%` | **Exclamations:** `{heuristics['exclamation_count']}` | **Symbols:** `{heuristics['dollar_count']}`")
            if heuristics["detected_urgency"]:
                st.markdown(f"**Urgency Cues:** `{'`, `'.join(heuristics['detected_urgency'])}`")
            if heuristics["detected_lures"]:
                st.markdown(f"**Lure Cues:** `{'`, `'.join(heuristics['detected_lures'])}`")
            if not heuristics["detected_urgency"] and not heuristics["detected_lures"]:
                st.caption("No psychological urgency or lure triggers found.")

# ================= TAB 2: BULK CSV PROCESSOR =================
with tab2:
    st.subheader("📁 Bulk Dataset Auditor")
    st.caption("Upload a CSV file containing an email text column to scan thousands of records in batch.")

    # Downloadable Sample Template
    sample_df = pd.DataFrame({
        "email_text": [
            "Hi team, please review the pull request for the caching layer by EOD.",
            "URGENT: Unauthorized login detected from unfamiliar device. Click here to verify your account password: http://bit.ly/bank-auth",
            "Congratulations! You won $5,000,000 in our international lottery! Claim your cash prize now at http://192.168.1.1/claim",
            "Summer Flash Sale: 40% discount on all cloud developer packages with coupon code SUMMER40.",
            "Can we reschedule our sync meeting to 2 PM this afternoon?"
        ]
    })

    st.download_button(
        "📥 Download Sample CSV Template",
        sample_df.to_csv(index=False).encode("utf-8"),
        "sample_email_batch.csv",
        "text/csv"
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"**Loaded {len(batch_df)} records.**")

        # Select target text column
        text_column = st.selectbox("Select the column containing email text:", batch_df.columns)

        if st.button("⚡ Run Batch Analysis", type="primary"):
            with st.spinner("Processing batch classifications and threat scores..."):
                raw_texts = batch_df[text_column].fillna("").astype(str).tolist()

                # Model inference
                batch_vecs = vectorizer.transform(raw_texts)
                pred_indices = model.predict(batch_vecs)
                predictions = encoder.inverse_transform(pred_indices)
                probabilities = model.predict_proba(batch_vecs)
                max_confidences = [round(float(max(p)) * 100, 2) for p in probabilities]

                # Heuristic evaluations
                url_counts = [len(analyze_urls(t)) for t in raw_texts]
                urgency_counts = [len(analyze_heuristics(t)["detected_urgency"]) for t in raw_texts]

                # Threat indices
                threat_scores = []
                for i, p_arr in enumerate(probabilities):
                    p_dict = dict(zip(encoder.classes_, p_arr))
                    base = int((p_dict.get("Phishing", 0) + p_dict.get("Spam", 0)) * 75)
                    penalty = (url_counts[i] * 10) + (urgency_counts[i] * 5)
                    threat_scores.append(min(100, base + penalty))

                # Populate results
                batch_df["Predicted_Category"] = predictions
                batch_df["Confidence_%"] = max_confidences
                batch_df["Threat_Index"] = threat_scores
                batch_df["Detected_URLs"] = url_counts
                batch_df["Urgency_Triggers"] = urgency_counts

                st.success("Batch audit complete!")

                # Batch Analytics
                bc1, bc2, bc3 = st.columns(3)
                with bc1:
                    st.metric("Total Emails Processed", len(batch_df))
                with bc2:
                    flagged_count = int((batch_df["Predicted_Category"].isin(["Spam", "Phishing"])).sum())
                    st.metric("High-Risk Threats Detected", flagged_count)
                with bc3:
                    avg_threat = round(float(batch_df["Threat_Index"].mean()), 1)
                    st.metric("Average Threat Score", f"{avg_threat}/100")

                st.subheader("Category Breakdown")
                st.bar_chart(batch_df["Predicted_Category"].value_counts())

                st.subheader("Processed Data Preview")
                st.dataframe(batch_df, use_container_width=True)

                # Export Button
                csv_export = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Export Enriched Audit Results (CSV)",
                    csv_export,
                    "threatintel_audit_results.csv",
                    "text/csv",
                    type="primary"
                )