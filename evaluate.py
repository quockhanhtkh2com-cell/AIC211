import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score,
    classification_report
)

os.makedirs("outputs/plots", exist_ok=True)

# =====================
# LOAD DATA & MODELS
# =====================
print("Đang load models và data...")

X_test = sp.load_npz("outputs/models/X_test.npz")
y_test = np.load("outputs/models/y_test.npy")

with open("outputs/models/naive_bayes.pkl", "rb") as f:
    nb = pickle.load(f)
with open("outputs/models/linear_svm.pkl", "rb") as f:
    svm = pickle.load(f)
with open("outputs/models/xgboost.pkl", "rb") as f:
    xgb = pickle.load(f)

label_names = ["benign", "phishing", "bec"]
print("✅ Loaded!")

# =====================
# CONFUSION MATRIX      cho thấy model sai cụ thể ở đâu
# =====================
print("\nVẽ Confusion Matrix...")

models = {
    "Naive Bayes": nb,
    "Linear SVM": svm,
    "XGBoost": xgb
}

for name, model in models.items():
    y_pred = model.predict(X_test) #Cho model dự đoán trên tập test
    cm = confusion_matrix(y_test, y_pred) #So sánh đáp án đúng vs đáp án model đoán

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(  #Chuyển ma trận số thành biểu đồ màu đẹp
        confusion_matrix=cm,   #ma trận số
        display_labels=label_names  #tên class
    )
    disp.plot(ax=ax, colorbar=False, cmap=("Blues"))
    ax.set_title(f"Confusion Matrix - {name}", fontsize=13)
    plt.tight_layout()

    filename = name.lower().replace(" ", "_")
    plt.savefig(f"outputs/plots/confusion_matrix_{filename}.png")
    plt.close()
    print(f"✅ Saved confusion_matrix_{filename}.png")

# =====================
# PR CURVE (BEST MODEL)   #cho thấy model tốt đến mức nào
# =====================
print("\nVẽ PR Curve cho Linear SVM...")

# Lấy xác suất từ SVM
y_prob = svm.predict_proba(X_test)

plt.figure(figsize=(9, 6))
colors = ["green", "orange", "red"]

for i, (label, color) in enumerate(zip(label_names, colors)): #duyệt 3 class và ghép cặp list lại, emurate số 0, 1, 2 vào
    precision, recall, _ = precision_recall_curve( #nhận 3 giá trị mà chỉ cần 2 nên thêm _ vào
        (y_test == i).astype(int),
        y_prob[:, i]
    )
    ap = average_precision_score(
        (y_test == i).astype(int),  #so sánh phần tử trong test với i và chuyển T/F thành 1/0
        y_prob[:, i]      #lấy tất cả dòng của cột i
    )
    plt.plot(recall, precision,
             label=f"{label} (AP={ap:.2f})",
             color=color, linewidth=2)
    
plt.title("PR Curve - Linear SVM", fontsize=14)
plt.xlabel("Recall")   #trrong tất cả email spam tìm đc bao nhiêu
plt.ylabel("Precision") #trong tất cả cái email đã sét thì đúng bao nhiêu
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/plots/pr_curve_svm.png")
plt.close()
print("✅ Saved pr_curve_svm.png")

# =====================     Accuracy   → đánh giá TỔNG THỂ cả 3 class
# BẢNG SO SÁNH MODELS       BEC Recall → đánh giá riêng KHẢ NĂNG PHÁT HIỆN BEC
# =====================     BEC F1     → đánh giá riêng SỰ CÂN BẰNG của BEC
print("\n=== BẢNG SO SÁNH CUỐI CÙNG ===")
print(f"{'Model':<15} {'Accuracy':<12} {'BEC Recall':<12} {'BEC F1':<10}") #căn lề
print("-" * 50)

for name, model in models.items():
    y_pred = model.predict(X_test)
    report = classification_report(  #Hàm của sklearn tự động tính Precision, Recall, F1 cho từng class và trả về kết quả
        y_test, y_pred,
        target_names=label_names,
        output_dict=True
    )
    accuracy = report["accuracy"]
    bec_recall = report["bec"]["recall"]
    bec_f1 = report["bec"]["f1-score"]
    print(f"{name:<15} {accuracy:<12.4f} {bec_recall:<12.4f} {bec_f1:<10.4f}")

print("\n✅ Evaluate hoàn thành! Kiểm tra outputs/plots/")
