import os
import random
from sklearn.model_selection import train_test_split

# 🔹 Path to dataset
DATA_DIR = "."
OUTPUT_DIR = "."

# 🔹 Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🔹 Dictionary to store species with unique numeric labels
species_to_id = {}
species_id_counter = 0

# 🔹 Collect image paths and assign labels
image_list = []
for species_class in os.listdir(DATA_DIR):  # Aves, Insecta
    class_path = os.path.join(DATA_DIR, species_class)

    if os.path.isdir(class_path):  # Ensure it's a directory
        for species in os.listdir(class_path):  # Species directories
            species_path = os.path.join(class_path, species)

            if os.path.isdir(species_path):
                if species not in species_to_id:
                    species_to_id[species] = species_id_counter
                    species_id_counter += 1

                # Get all images in this species folder
                for img_file in os.listdir(species_path):
                    img_path = os.path.join(species_path, img_file)
                    
                    # Store image path and species ID
                    image_list.append((img_path, species_to_id[species]))

# 🔹 Shuffle dataset to avoid bias
random.shuffle(image_list)

# 🔹 Split into train (80%) and validation (20%)
train_data, val_data = train_test_split(image_list, test_size=0.1, random_state=42, stratify=[x[1] for x in image_list])

# 🔹 Save train.txt
train_txt_path = os.path.join(OUTPUT_DIR, "train.txt")
with open(train_txt_path, "w") as train_file:
    for img_path, species_id in train_data:
        train_file.write(f"{img_path}: {species_id}\n")

# 🔹 Save val.txt
val_txt_path = os.path.join(OUTPUT_DIR, "val.txt")
with open(val_txt_path, "w") as val_file:
    for img_path, species_id in val_data:
        val_file.write(f"{img_path}: {species_id}\n")

print(f"✅ train.txt and val.txt created in {OUTPUT_DIR}!")
print(f"🔹 Total Images: {len(image_list)} | Training: {len(train_data)} | Validation: {len(val_data)}")
print(f"🔹 Species Count: {len(species_to_id)}")
