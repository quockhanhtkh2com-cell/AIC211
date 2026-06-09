import numpy as np
import scipy.sparse as sp
import pandas as pd
import pickle

# =====================
# LOAD DATA & MODEL
# =====================
print("Đang load data...")

X_test = sp.load_npz("outputs/models/X_test.npz")
test_indices = np.load("outputs/models/test_indices.npy")
y_test = np.load("outputs/models/y_test.npy")
# Load dataset gốc để lấy text email
df = pd.read_csv("data/emails_processed.csv")

# Load best model — Linear SVM
with open("outputs/models/linear_svm.pkl", "rb") as f:
    svm = pickle.load(f)

label_names = ["benign", "phishing", "bec"]
label_map = {0: "benign", 1: "phishing", 2: "bec"}

print("✅ Loaded!")

# =====================
# DỰ ĐOÁN
# =====================
y_pred = svm.predict(X_test)

# =====================
# LẤY INDEX CỦA TEST SET  Ghép email gốc với kết quả model đoán để biết email nào bị đoán sai
# =====================
# Tập test được chia NGẪU NHIÊN (stratify) trong train_models.py,
# nên phải lấy đúng các dòng theo test_indices, không phải 15% dòng cuối
df_test = df.iloc[test_indices].copy()
df_test["y_true"] = y_test             #Thêm cột đáp án thật vào df_test
df_test["y_pred"] = y_pred             #Thêm cột đáp án model đoán vào df_test

# =====================
# 1. FALSE NEGATIVE — BEC bị bỏ sót
# =====================
print("\n=== FALSE NEGATIVE (BEC bị bỏ sót) ===")

fn_bec = df_test[
    (df_test["y_true"] == 2) &   # thật là BEC
    (df_test["y_pred"] != 2)     # model đoán không phải BEC
]

print(f"Số BEC bị bỏ sót: {len(fn_bec)}")
print(f"Tổng BEC thật: {(df_test['y_true'] == 2).sum()}")
print(f"BEC Recall: {1 - len(fn_bec)/(df_test['y_true'] == 2).sum():.4f}")

if len(fn_bec) > 0:
    print("\nVí dụ email BEC bị bỏ sót:")
    for i, row in fn_bec.head(3).iterrows(): #lấy 3 dòng email đầu và duyệt qua từng dòng
        print(f"\n--- Email {i} ---")
        print(f"Subject: {row['subject']}")
        print(f"Body:    {str(row['body'])[:200]}")
        print(f"Đoán là: {label_map[row['y_pred']]}")

# =====================
# 2. FALSE POSITIVE — Benign bị flag nhầm
# =====================
print("\n=== FALSE POSITIVE (Benign bị flag nhầm thành BEC) ===")

fp_bec = df_test[
    (df_test["y_true"] == 0) &
    (df_test["y_pred"] == 2)
]

print(f"Số benign bị flag nhầm thành BEC: {len(fp_bec)}")

if len(fp_bec) > 0:
    print("\nVí dụ email benign bị flag nhầm:")
    for i, row in fp_bec.head(3).iterrows():
        print(f"\n--- Email {i} ---")
        print(f"Subject: {row['subject']}")
        print(f"Body:    {str(row['body'])[:200]}")

# =====================
# 3. PHÂN TÍCH TẠI SAO SAI
# =====================
print("\n=== PHÂN TÍCH LỖI ===")

# Tất cả email bị sai
all_wrong = df_test[df_test["y_true"] != df_test["y_pred"]]
print(f"Tổng số email đoán sai: {len(all_wrong)}")
print(f"Tổng số email test:     {len(df_test)}")
print(f"Error rate:             {len(all_wrong)/len(df_test):.4f}")

print("\nPhân loại lỗi:")
for true_label in range(3):
    for pred_label in range(3):
        if true_label != pred_label:
            count = len(df_test[
                (df_test["y_true"] == true_label) &
                (df_test["y_pred"] == pred_label)
            ])
            if count > 0:
                print(f"  {label_map[true_label]} → {label_map[pred_label]}: {count} emails")

# =====================
# 4. LƯU REPORT
# =====================
print("\nLưu error analysis report...")

with open("outputs/error_analysis.txt", "w", encoding="utf-8") as f:
    f.write("=== ERROR ANALYSIS REPORT ===\n\n")

    f.write("1. FALSE NEGATIVE — BEC bị bỏ sót\n")
    f.write(f"   Số BEC bị bỏ sót: {len(fn_bec)}\n")
    f.write(f"   Tổng BEC thật:    {(df_test['y_true'] == 2).sum()}\n")
    f.write(f"   BEC Recall:       {1 - len(fn_bec)/(df_test['y_true'] == 2).sum():.4f}\n\n")

    f.write("2. FALSE POSITIVE — Benign bị flag nhầm\n")
    f.write(f"   Số benign bị flag nhầm thành BEC: {len(fp_bec)}\n\n")

    f.write("3. PHÂN LOẠI LỖI\n")
    for true_label in range(3):
        for pred_label in range(3):
            if true_label != pred_label:
                count = len(df_test[
                    (df_test["y_true"] == true_label) &
                    (df_test["y_pred"] == pred_label)
                ])
                if count > 0:
                    f.write(f"   {label_map[true_label]} → {label_map[pred_label]}: {count} emails\n")

print("✅ Saved to outputs/error_analysis.txt")
print("\n✅ Error Analysis hoàn thành!")