import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
from bs4 import MarkupResemblesLocatorWarning   #tắt cảnh báo th k gì to tát
import warnings
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

os.makedirs("outputs/plots", exist_ok=True)

print("Đang load data...")
df = pd.read_csv("data/emails_processed.csv")
print(f"Load {len(df)} emails")

# =====================
# 1. BIỂU ĐỒ PHÂN BỐ CLASS
# =====================
print("\nVẽ biểu đồ phân bố class...")

plt.figure(figsize=(8, 5))
ax = sns.countplot(
    x="label",
    data=df,
    palette={"benign": "green", "bec": "red", "phishing": "orange"}
)

for p in ax.patches:
    ax.annotate(
        f'{int(p.get_height()):,}', #lấy từng cột đổi qua số nguyên với dấu phẩy dễ đọc
        (p.get_x() + p.get_width() / 2., p.get_height()), #đặt số điểm giữa cột
        ha='center', va='bottom', fontsize=12
    )

plt.title("Class Distribution", fontsize=14)
plt.xlabel("Label")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/plots/class_distribution.png")
print("✅ Saved class_distribution.png")

# =====================
# 2. WORD CLOUD
# =====================
print("\nVẽ word clouds...")

for label in ["benign", "phishing", "bec"]:
    text = " ".join(df[df["label"] == label]["text"].dropna().tolist())

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=100,
        colormap="viridis"
    ).generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud — {label.upper()}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"outputs/plots/wordcloud_{label}.png")
    plt.close()
    print(f"✅ Saved wordcloud_{label}.png")

# =====================
# 3. PHÂN BỐ ĐỘ DÀI EMAIL
# =====================
print("\nVẽ biểu đồ độ dài email...")

df["text_length"] = df["text"].fillna("").apply(len)

plt.figure(figsize=(10, 5))
for label, color in zip(
    ["benign", "phishing", "bec"],
    ["green", "orange", "red"]
):
    subset = df[df["label"] == label]["text_length"]
    subset=subset[subset <= 5000]
    plt.hist(subset, bins=50, alpha=0.5, label=label, color=color)

plt.title("Email Length Distribution", fontsize=14)
plt.xlabel("Number of Characters")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/plots/email_length_distribution.png")
print("✅ Saved email_length_distribution.png")

print("\n✅ EDA hoàn thành! Kiểm tra thư mục outputs/plots/")