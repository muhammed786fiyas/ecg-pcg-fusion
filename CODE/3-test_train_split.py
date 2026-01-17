import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# ----------------------------
# Paths
# ----------------------------
mat_dir = r"E:\PROJECTS\CARDIAC-PROJECT-UPDATED\DATASET\2-MATLAB DATA"
labels_csv = r"E:\PROJECTS\CARDIAC-PROJECT-UPDATED\DATASET\2-MATLAB DATA\LABELS.csv"

output_base = r"E:\PROJECTS\CARDIAC-PROJECT-UPDATED\DATASET\SPLIT_DATA"
train_dir = os.path.join(output_base, "train")
test_dir = os.path.join(output_base, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# ----------------------------
# Load labels
# ----------------------------
labels_df = pd.read_csv(labels_csv)

# ----------------------------
# Stratified split (70–30)
# ----------------------------
train_df, test_df = train_test_split(
    labels_df,
    test_size=0.30,
    random_state=42,
    stratify=labels_df["label"]
)

# ----------------------------
# Copy MAT files
# ----------------------------
def copy_files(df, destination):
    for _, row in df.iterrows():
        mat_file = row["record"] + ".mat"
        src = os.path.join(mat_dir, mat_file)
        dst = os.path.join(destination, mat_file)

        # if os.path.exists(src):
        #     shutil.copy(src, dst)
        # else:
        #     print(f"Missing file: {mat_file}")

copy_files(train_df, train_dir)
copy_files(test_df, test_dir)

# ----------------------------
# Save split labels
# ----------------------------
# train_df.to_csv(os.path.join(output_base, "train_labels.csv"), index=False)
# test_df.to_csv(os.path.join(output_base, "test_labels.csv"), index=False)

print("✅ 70–30 stratified train–test split completed")
print(f"Train samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(train_df["label"].value_counts(normalize=True))
print(test_df["label"].value_counts(normalize=True))