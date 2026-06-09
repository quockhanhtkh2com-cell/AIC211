import pandas as pd

df = pd.read_csv("data/dataset.csv")

print("=" * 50)
print("FILE HAM:")
print("=== SHAPE ===")
print(df.shape)
print("\n=== COLUMNS ===")
print(df.columns.tolist()) #Chuyển sang dạng list cho dễ đọc
print("\n=== 3 DÒNG ĐẦU ===")
print(df.head(3))
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())
print("\n=== LABELS ===")
df = pd.read_csv("data/dataset.csv")
print(df["label"].value_counts())
