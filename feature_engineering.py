import pandas as pd
import numpy as np
import re
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

os.makedirs("outputs/models", exist_ok=True)

# Load data
print("Đang load data...")
df = pd.read_csv("data/emails_processed.csv")
print(f"Loaded {len(df)} emails")

#fill missing
df["text"] = df["text"].fillna("")
df["sender"] = df["sender"].fillna("")
df["subject"] = df["subject"].fillna("")
df["body"] = df["body"].fillna("")

# =====================
# 1. TF-IDF
# =====================
print("\nĐang tạo TF-IDF features...")

tfidf = TfidfVectorizer(
    max_features=10000, # chỉ lấy 10,000 từ phổ biến nhất
    ngram_range=(1, 2),  # unigram và bigram
    min_df=2,           # từ phải xuất hiện ít nhất 2 lần
    sublinear_tf=True   # dùng log để giảm ảnh hưởng của từ xuất hiện quá nhiều
)

X_tfidf = tfidf.fit_transform(df["text"])
print(f"TF-IDF shape: {X_tfidf.shape}")

# =====================
# 2. METADATA FEATURES
# =====================
print("\nĐang tạo metadata features...")

def extract_metadata(row):
    body = str(row["body"])
    subject = str(row["subject"])
    sender = str(row["sender"])

    #đếm số lượng url trong body
    urls = re.findall(r'http[s]?://\S+', body)
    url_count = len(urls)
    has_url = 1 if url_count > 0 else 0

    #kiểm tra từ urgency
    urgency_words = [
        "urgent", "immediately", "asap", "critical",
        "action required", "wire transfer", "confidential"
    ]
    has_urgency = 1 if any(w in body.lower() for w in urgency_words) else 0

    #Độ dài subject
    subject_length = len(subject)

    # Sender dùng dịch vụ email miễn phí (gmail/yahoo/hotmail/outlook)
    # Dấu hiệu thường gặp ở BEC: kẻ tấn công giả danh sếp nhưng gửi từ mail cá nhân
    free_email = ["gmail", "yahoo", "hotmail", "outlook"]
    free_email_sender = 1 if any(f in sender.lower() for f in free_email) else 0

    return pd.Series([
        url_count,
        has_url,
        has_urgency,
        subject_length,
        free_email_sender
    ])

# Áp dụng lên toàn bộ dataframe
metadata = df.apply(extract_metadata, axis=1)
metadata.columns = [
    "url_count",
    "has_url",
    "has_urgency",
    "subject_length",
    "free_email_sender"
]
print("Metadata features:")
print(metadata.head(3))

# =====================
# 3. COMBINE FEATURES
# =====================
print("\nGhép TF-IDF + Metadata...")

# Chuyển metadata thành sparse matrix để ghép với TF-IDF
metadata_sparse = sp.csr_matrix(metadata.values)

#ghép lại
X_combined = sp.hstack([X_tfidf, metadata_sparse])
print(f"Combined shape: {X_combined.shape}")

# =====================
# 4. LABEL ENCODING
# =====================
print("\nEncoding labels...")

label_map = {
    "benign": 0,
    "phishing": 1,
    "bec": 2
}
y = df["label"].map(label_map)
print(y.value_counts())

# =====================
# 5. LƯU LẠI
# =====================
print("\nLưu features...")

# Lưu TF-IDF vectorizer để dùng lại sau
with open("outputs/models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

# Lưu features và labels
sp.save_npz("outputs/models/X_combined.npz", X_combined)
np.save("outputs/models/y_labels.npy", y.values)

print("\n=== KẾT QUẢ ===")
print(f"X_combined shape: {X_combined.shape}")
print(f"y shape: {y.shape}")
print(f"Labels: {label_map}")
print("\n✅ Done! Saved features to outputs/models/")