import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 🔹 Path to extracted Haute-Garonne dataset
# FEATURES_CSV = "./haute_garonne_features.csv"
DATA_DIR = "../data/Haute-Garonne"

# 🔹 Load extracted dataset
df = pd.read_csv(FEATURES_CSV)

# 🔹 Encode species into unique numeric labels
label_encoder = LabelEncoder()
df["Species_ID"] = label_encoder.fit_transform(df["Species"])

# 🔹 Split dataset into train (80%) and validation (20%)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["Species_ID"], random_state=42)

# 🔹 Save train.txt
with open(os.path.join(DATA_DIR, "train.txt"), "w") as train_file:
    for _, row in train_df.iterrows():
        train_file.write(f"{row['Image Path']}: {row['Species_ID']}\n")

# 🔹 Save val.txt
with open(os.path.join(DATA_DIR, "val.txt"), "w") as val_file:
    for _, row in val_df.iterrows():
        val_file.write(f"{row['Image Path']}: {row['Species_ID']}\n")

print("✅ train.txt and val.txt successfully created!")