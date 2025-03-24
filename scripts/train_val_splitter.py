import os
import random
import json
from sklearn.model_selection import train_test_split
from typing import Dict

def save_data(file_path, data):
    with open(file_path, "w") as file:
        for img_path, species_id in data:
            file.write(f"{img_path}: {species_id}\n")

DATA_DIR = "/home/tom-maverick/Documents/02_GitHub/cvpr18-inaturalist-transfer/data/inat2017/"
OUTPUT_DIR = DATA_DIR 

os.makedirs(OUTPUT_DIR, exist_ok=True)

species_to_id = {}
species_id_counter = 0

image_list = []
species_dict: Dict[int, str] = {}

for species_class in os.listdir(DATA_DIR):  
    class_path = os.path.join(DATA_DIR, species_class)

    if os.path.isdir(class_path):  
        for species in os.listdir(class_path):  
            species_path = os.path.join(class_path, species)

            if os.path.isdir(species_path):
                if species not in species_to_id and species == "Acanthis flammea":
                    if species_class in ["Aves"]:
                    # if species_class in ["Aves", "Insecta"]:
                        species_to_id[species] = species_id_counter
                        if species not in species_dict.keys() :
                            species_dict[species_id_counter] = species
                        species_id_counter += 1

                        for img_file in os.listdir(species_path):
                            img_path = os.path.join(species_path, img_file)
                            image_list.append((img_path, species_to_id[species])) 

class_path = os.path.join(DATA_DIR, "Other")
# The id has already been plus 1 from the loop above
other_id = species_id_counter 

if os.path.isdir(class_path):  
    for species in os.listdir(class_path):  
        species_path = os.path.join(class_path, species)

        if os.path.isdir(species_path):
            if species not in species_to_id:
                # if species_class in ["Aves", "Insecta"]:
                if species not in species_dict.keys():
                    species_dict[species_id_counter] = species

                species_to_id[species] = other_id 

                for img_file in os.listdir(species_path):
                    img_path = os.path.join(species_path, img_file)
                    image_list.append((img_path, species_to_id[species]))

save_data(os.path.join(OUTPUT_DIR, "dataset_manifest.txt"), image_list)
with open(os.path.join(OUTPUT_DIR, "dataset_species_labels.json"), "w", encoding='utf-8') as file:
    json.dump(species_dict, file, indent=2)

# random.shuffle(image_list)

train_data, val_data = train_test_split(
    image_list, test_size=0.99, random_state=42, stratify=[x[1] for x in image_list]
)

save_data(os.path.join(OUTPUT_DIR, "train.txt"), train_data)
save_data(os.path.join(OUTPUT_DIR, "val.txt"), val_data)

print(f"train.txt and val.txt created in {OUTPUT_DIR}")
print(f"Total species: {species_id_counter + 1}")
print(f"Total Images: {len(image_list)} | Training: {len(train_data)} | Validation: {len(val_data)}")
