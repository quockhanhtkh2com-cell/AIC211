import pandas as pd
import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#Download NLTK data lần đầu 
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

#Load data
print("Đang load data...")
df = pd.read_csv("data/dataset.csv")
print(f"Loaded {len(df)} emails")

# =====================
# 1. CHUẨN HÓA LABEL
# =====================
print("\nChuẩn hóa label....")
df["label"] = df["label"].str.lower()
print(df["label"].value_counts())

# =====================
# 2. XỬ LÝ MISSING VALUES
# =====================
print("\nXử lý missing values...")

df["sender"] = df["sender"].fillna("")
df["subject"] = df["subject"].fillna("")
df = df.dropna(subset=["body"])
df = df[df["body"].str.strip() != ""]
print(f"Sau khi xử lý missing: {len(df)} emails")

# =====================
# 3. HÀM LÀM SẠCH TEXT
# =====================
stop_words = set(stopwords.words("english"))

def clean_text(text):
    # Kiểm tra NaN trước khi xử lý
    if pd.isna(text):
        return ""
    # Bước 1: Xóa HTML tags
    # Ví dụ: <div>Hello</div> → Hello
    text = BeautifulSoup(text, "html.parser").get_text()
    # Bước 2: Chuyển về chữ thường
    # Ví dụ: "URGENT Transfer" → "urgent transfer"
    text = text.lower()
    # Bước 3: Xóa ký tự đặc biệt, chỉ giữ chữ và số
    # Ví dụ: "hello@world!" → "hello world"
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    # Bước 4: Xóa khoảng trắng thừa
    # Ví dụ: "hello   world" → "hello world"
    text = re.sub(r"\s+", " ", text).strip()
    # Bước 5: Tokenize — tách thành từng từ
    # Ví dụ: "hello world" → ["hello", "world"]
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    # Ghép lại thành chuỗi
    return " ".join(tokens)

# =====================
# 4. ÁP DỤNG CLEAN TEXT
# =====================
print("\nĐang làm sạch text... (có thể mất vài phút)")

df["subject_clean"] = df["subject"].apply(clean_text)
df["body_clean"] = df["body"].apply(clean_text)

# Ghép subject + body thành 1 cột text
df["text"] = df["subject_clean"] + " " + df["body_clean"]

# =====================
# 5. KIỂM TRA KẾT QUẢ
# =====================
print("\n=== KẾT QUẢ PREPROCESSING ===")
print("Shape:", df.shape)
print("\nLabel counts:")
print(df["label"].value_counts())
print("\nSample text sau khi clean:")
print(df["text"].iloc[0][:200])

# =====================
# 6. LƯU FILE
# =====================
df.to_csv("data/emails_processed.csv", index=False)
print("\n✅ Done! Saved to data/emails_processed.csv")