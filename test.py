import pandas as pd
import pickle
import scipy.sparse as sp
import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import MarkupResemblesLocatorWarning
import warnings
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Load model và vectorizer
with open("outputs/models/linear_svm.pkl", "rb") as f:
    svm = pickle.load(f)

with open("outputs/models/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

stop_words = set(stopwords.words("english"))

# =====================
# HÀM CLEAN TEXT
# =====================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = BeautifulSoup(str(text), "html.parser").get_text()
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

# =====================
# HÀM EXTRACT METADATA
# =====================
def extract_metadata(subject, body, sender):
    body = str(body)
    sender = str(sender)
    subject = str(subject)

    urls = re.findall(r'http[s]?://\S+', body)
    url_count = len(urls)
    has_url = 1 if url_count > 0 else 0

    urgency_words = ["urgent", "immediately", "asap", "critical",
                     "action required", "wire transfer", "confidential"]
    has_urgency = 1 if any(w in body.lower() for w in urgency_words) else 0

    subject_length = len(subject)

    free_email = ["gmail", "yahoo", "hotmail", "outlook"]
    free_email_sender = 1 if any(f in sender.lower() for f in free_email) else 0

    return [url_count, has_url, has_urgency, subject_length, free_email_sender]

# =====================
# HÀM EXPLAIN PREDICTION
# =====================
def explain_prediction(subject, body, sender):
    body_lower = str(body).lower()
    sender_lower = str(sender).lower()

    signals = []

    # Check URL
    urls = re.findall(r'http[s]?://\S+', str(body))
    if urls:
        signals.append(f"⚠️  Contains URL: {urls[0][:50]}")

    # Check urgency keywords
    urgency_words = ["urgent", "immediately", "asap",
                     "critical", "wire transfer", "confidential"]
    found = [w for w in urgency_words if w in body_lower]
    if found:
        signals.append(f"⚠️  Urgency keywords: {', '.join(found)}")

    # Check sender free email
    free_email = ["gmail", "yahoo", "hotmail", "outlook"]
    if any(f in sender_lower for f in free_email):
        signals.append(f"⚠️  Sender uses free email: {sender}")

    # Subject length
    signals.append(f"📏 Subject length: {len(str(subject))} chars")

    if len(signals) > 1:
        print("  Signals detected:")
        for s in signals:
            print(f"    {s}")
    else:
        print("  Signals detected: None — looks clean ✅")

# =====================
# LOAD FILE CSV
# =====================
df = pd.read_csv("data/demo_emails.csv")
print(f"Loaded {len(df)} emails\n")
print("=" * 60)

label_map = {0: "BENIGN", 1: "PHISHING", 2: "BEC"}

# =====================
# PREDICT TỪNG EMAIL
# =====================
for i, row in df.iterrows():
    subject = str(row["subject"])
    body    = str(row["body"])
    sender  = str(row["sender"])

    # Clean text
    text = clean_text(subject) + " " + clean_text(body)

    # TF-IDF
    X_tfidf = tfidf.transform([text])

    # Metadata
    metadata = extract_metadata(subject, body, sender)
    metadata_sparse = sp.csr_matrix([metadata])

    # Ghép features
    X_combined = sp.hstack([X_tfidf, metadata_sparse])

    # Predict
    pred = svm.predict(X_combined)[0]
    prob = svm.predict_proba(X_combined)[0]

    print(f"Email {i+1}:")
    print(f"  Sender:     {sender}")
    print(f"  Subject:    {subject}")
    print(f"  Prediction: {label_map[pred]}")
    print(f"  Confidence: Benign {prob[0]:.1%} | Phishing {prob[1]:.1%} | BEC {prob[2]:.1%}")

    # So sánh với label thật
    if "label" in df.columns:
        true_label = str(row["label"]).upper()
        pred_label = label_map[pred]
        status = "✅ CORRECT" if true_label == pred_label else "❌ WRONG"
        print(f"  True Label: {true_label} → {status}")

    # Phân tích dấu hiệu
    explain_prediction(subject, body, sender)
    print("=" * 60)