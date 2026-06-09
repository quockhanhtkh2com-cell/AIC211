import numpy as np  #xử lý toán học
import scipy.sparse as sp #làm ma trận
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score   #chia dữ liệu để train và test thành nhiều fold để cross validation, và chạy tự đọng để trả về đánh giá
from sklearn.naive_bayes import MultinomialNB    #Thuật toán ML
from sklearn.svm import LinearSVC #model cho bài toán phân loại văn bản
from sklearn.calibration import CalibratedClassifierCV  #chuyển output thành xác suất dự đoán
from xgboost import XGBClassifier #dùng thuật toán boosting cho độ chính xác cao
from sklearn.metrics import classification_report #báo cáo đánh giá model (precision, recall, f1-score, support)
import pickle  #lưu và tải model xuống file
import os

os.makedirs("outputs/models", exist_ok=True)

# =====================
# LOAD FEATURES
# =====================
print("Đang load features...")
X = sp.load_npz("outputs/models/X_combined.npz") #dùng ma trận nên sp
y = np.load("outputs/models/y_labels.npy") #chỉ là list nên dùng np
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# =====================
# TRAIN/VAL/TEST SPLIT
# =====================
print("\nChia data...")

# Chia theo INDEX để biết chính xác email nào nằm trong tập test
# (error_analysis.py cần index này để ghép đúng email gốc với dự đoán)
idx = np.arange(X.shape[0])

# Chia 70% train, 30% tạm
idx_train, idx_temp = train_test_split(
    idx,
    test_size=0.30,
    random_state=42,
    stratify=y  #giữ tỉ lệ class như nhau
)

# Chia 30% tạm → 15% val(kiểm tra trong lúc học), 15% test(kiểm tra cuối)
idx_val, idx_test = train_test_split(
    idx_temp,
    test_size=0.50,
    random_state=42,
    stratify=y[idx_temp]
)

X = X.tocsr()  # csr mới cắt dòng theo index được
X_train, y_train = X[idx_train], y[idx_train]
X_val, y_val = X[idx_val], y[idx_val]
X_test, y_test = X[idx_test], y[idx_test]

print(f"Train: {X_train.shape[0]} emails") #in ra bao nhiêu mail, k in hàng [1]
print(f"Val: {X_val.shape[0]} emails")
print(f"Test: {X_test.shape[0]} emails")

# =====================
# MODEL 1: NAIVE BAYES
# =====================
print("\n--- Training Naive Bayes ---")

nb = MultinomialNB()
nb.fit(X_train, y_train) #học model

nb_val_score = nb.score(X_val, y_val)  #đoán trong quá trình học
print(f"Validation Accuracy: {nb_val_score:.4f}")   #đoán được trong val bao nhiêu %
print(classification_report(y_val, nb.predict(X_val), target_names=["benign", "phishing", "bec"]))
#biết đúng cái gì, sai cái gì

# =====================
# MODEL 2: LINEAR SVM
# =====================
print("\n--- Training Linear SVM ---")

svm = CalibratedClassifierCV(
    LinearSVC(
        class_weight="balanced",
        max_iter=2000,
        random_state=42
    )
)
svm.fit(X_train, y_train)

svm_val_score = svm.score(X_val, y_val)
print(f"Validation Accuracy: {svm_val_score:.4f}")
print(classification_report(y_val, svm.predict(X_val), target_names=["benign", "phishing", "bec"]))

# =====================
# MODEL 3: XGBOOST
# =====================
print("\n--- Training XGBoost ---")

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="mlogloss",
    random_state=42
)
xgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

xgb_val_score = xgb.score(X_val, y_val)
print(f"Validation Accuracy: {xgb_val_score:.4f}")
print(classification_report(y_val, xgb.predict(X_val), target_names=["benign", "phishing", "bec"]))

# =====================
# SO SÁNH 3 MODELS
# =====================
print("\n=== SO SÁNH 3 MODELS ===")
print(f"Naive Bayes Accuracy: {nb_val_score:.4f}")
print(f"Linear SVM Accuracy: {svm_val_score:.4f}")
print(f"XGBoost: Accuracy: {xgb_val_score:.4f}")

# =====================
# LƯU 3 MODELS
# =====================
print("\nLưu models...")

with open("outputs/models/naive_bayes.pkl", "wb") as f:
    pickle.dump(nb, f)
with open("outputs/models/linear_svm.pkl", "wb") as f:
    pickle.dump(svm, f)
with open("outputs/models/xgboost.pkl", "wb") as f:
    pickle.dump(xgb, f)

# Lưu dạng sparse (.npz): bản dense .npy của ~9.500 x 10.005 sẽ nặng hàng trăm MB
sp.save_npz("outputs/models/X_test.npz", X_test)
np.save("outputs/models/y_test.npy", y_test)
np.save("outputs/models/test_indices.npy", idx_test)

print("✅ Done! Saved models to outputs/models/")