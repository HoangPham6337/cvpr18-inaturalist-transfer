import json
import numpy as np
from typing import Dict, Optional
from scripts.utility import prepare_data_cdf_ppf
from collections import defaultdict


def identifying_dominant_species(properties_json_path: str, threshold: float, classes_to_analyze: list[str]) -> Optional[Dict[str, list[str]]]:
    species_data: Dict[str, list[str]] = defaultdict(list)
    for species_class in classes_to_analyze:
        result = prepare_data_cdf_ppf(properties_json_path, species_class)
        if result is None:
            print(f"ERROR: Data preparation failed for {species_class}")
            return None
        species_names, sorted_image_counts = result
        total_images = sum(sorted_image_counts)
        cumulative_images = np.cumsum(sorted_image_counts) 
        cdf_values = cumulative_images / total_images
        sorted_images = np.array(sorted_image_counts)
        filtered_index = np.argmax(cdf_values >= threshold)
        thresholded_image_count = sorted_images[filtered_index]

        dominant_species = [species for species, count in zip(species_names, sorted_image_counts) if count >= thresholded_image_count]
        species_data[species_class] = dominant_species
    return species_data