import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

def identifying_dominant_species(properties_json_path: str, class_to_analyze: str) -> List[str]:
    # 🔹 Load JSON Data
    with open(properties_json_path, "r", encoding='utf-8') as file:
        species_data = json.load(file)

    if class_to_analyze not in species_data:
        print(f"Class '{class_to_analyze}' not found.")
        return []

    # 🔹 Extract species names and image counts, then sort by image count
    species_dict: Dict[str, int] = species_data[class_to_analyze]
    sorted_species = sorted(species_dict.items(), key=lambda x: x[1])
    species_names, image_counts = zip(*sorted_species)

    # 🔹 Compute CDF
    sorted_images = np.array(image_counts)  # X-axis values
    cdf_values = np.arange(1, len(sorted_images) + 1) / len(sorted_images)  # Y-axis values

    # 🔹 Find the image count threshold where CDF = 0.5
    index_50_percent = np.argmax(cdf_values >= 0.5)
    threshold_image_count = sorted_images[index_50_percent]

    # 🔹 Identify species beyond 50% CDF
    dominant_species = [species for species, count in sorted_species if count >= threshold_image_count]

    print(f"📌 {len(dominant_species)} species contribute to more than 50% of the dataset.")
    print("🔹 These species are:")
    print(dominant_species)

    return dominant_species
