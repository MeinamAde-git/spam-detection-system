import pandas as pd
import numpy as np


def create_multiclass_dataset(n_samples=2500, random_state=42):
    """
    Generates a synthetic multi-class email dataset across 4 categories:
    - Primary (0)
    - Spam (1)
    - Phishing (2)
    - Promotions (3)
    """
    np.random.seed(random_state)

    primary_samples = [
        "Hi team, attached is the revised architectural blueprint for the backend API.",
        "Are we still on for our 1:1 sync tomorrow morning at 10 AM?",
        "Please review the PR submitted for the caching layer optimization.",
        "Hey, are you free this weekend to grab coffee and catch up?",
        "Can you send over the meeting notes from yesterday's client review?",
        "The project deliverables have been submitted to the evaluation portal.",
        "Let me know when you finish reviewing the code changes."
    ]

    spam_samples = [
        "Congratulations! You won $1,000,000 in the international cash draw. Claim now!",
        "Double your investment in 24 hours guaranteed with no risk! Reply now!",
        "Urgent: You have unclaimed inheritance funds waiting for immediate wire transfer.",
        "Earn $5,000 weekly working 2 hours from home. Click here to sign up instantly!",
        "Claim your free gift card now before time runs out! No purchase necessary.",
        "You are our lucky visitor today! Click to claim your luxury vacation voucher."
    ]

    phishing_samples = [
        "SECURITY ALERT: Unauthorized login attempt detected. Verify your account password here immediately.",
        "Your bank account access has been suspended due to suspicious activity. Click here to verify your identity.",
        "Urgent: Your cloud storage subscription expired. Enter your credit card to avoid permanent data deletion.",
        "IT Support: Immediate password reset required for your corporate enterprise account. Click this link.",
        "Action Required: We detected an unrecognized device accessing your wallet. Confirm your PIN now.",
        "Your package delivery failed due to incorrect address. Update your payment details to reschedule."
    ]

    promo_samples = [
        "Big Summer Sale! Get up to 60% off all developer courses and certifications this week.",
        "Flash Deal: Free shipping on all tech accessories with promo code SAVE50.",
        "Exclusive subscriber discount: Upgrade to Pro Plan and get 3 months free!",
        "Our weekly newsletter is here: Top 10 Python tips, community highlights, and discounts.",
        "Limited time offer: 40% off annual subscription. Don't miss out!",
        "New arrival in store! Check out the latest hardware tools and enjoy early-bird pricing."
    ]

    records = []
    classes = [
        ("Primary", primary_samples),
        ("Spam", spam_samples),
        ("Phishing", phishing_samples),
        ("Promotions", promo_samples)
    ]

    samples_per_class = n_samples // 4

    for label_name, templates in classes:
        chosen = np.random.choice(templates, size=samples_per_class, replace=True)
        for text in chosen:
            # Minor text variation
            noise = np.random.choice(["", " Please check.", " Reply asap.", " Thanks.", " Details inside."])
            records.append({"text": text + noise, "category": label_name})

    df = pd.DataFrame(records)
    return df.sample(frac=1, random_state=random_state).reset_index(drop=True)


if __name__ == "__main__":
    df = create_multiclass_dataset()
    print("Dataset generated successfully!")
    print(df["category"].value_counts())