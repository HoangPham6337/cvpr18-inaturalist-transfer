import os
import sys
import json
from typing import Dict, List
from utility import SpeciesDict, write_species_to_json, read_species_from_json
from collections import defaultdict

CLASS_LIST = [
    "Reptilia",
    "Aves",
    "Mollusca",
    "Insecta",
    "Fungi",
    "Plantae",
    "Chromista",
    "Arachnida",
    "Protozoa",
    "Animalia",
    "Actinopterygii",
    "Amphibia",
    "Mammalia",
]


def extract_species_from_directory(data_path: str) -> SpeciesDict:
    """
    Extracts species from a dataset directory, organizing them by taxonomic class.

    Args:
        repo_path: Path to the dataset repository.

    Returns:
        SpeciesDict (Dict[str, list[str]]): Dictionary containing species as keys and their species as values.

    Raises:
        FileNotFoundError: If the repository path does not exists.
    """
    species_dict: SpeciesDict = {}
    total_species = 0

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset repository not found: {data_path}")

    try:
        class_folders = [
            d
            for d in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, d))
        ]

        for class_name in class_folders:
            class_path = os.path.join(data_path, class_name)

            species_list = [
                d
                for d in os.listdir(class_path)
                if os.path.isdir(os.path.join(class_path, d))
            ]
            species_dict[class_name] = species_list

            species_extract = len(species_list)
            total_species += species_extract
            print(f"Extracted {species_extract} species from {class_path}")
    except Exception as e:
        print(f"Error while extracting species: {e}")

    print(f"\nData extracted from {data_path}")
    print(f"Extracted {len(species_dict.keys())} classes: {list(species_dict.keys())}")
    print(f"Extracted {total_species} species")
    return species_dict


def extract_species_from_json(
    json_file_path: str, target_classes: list[str] = ["Aves", "Insecta"]
) -> SpeciesDict:
    """
    Extracts species data from a JSON file, filtering only the specified classes.

    Args:
        json_file_path: Path to the JSON file containing species data
        target_classes: Species class to include

    Returns:
        SpeciesDict: A dictionary with filtered species data.
    """
    if not os.path.isfile(json_file_path):
        raise FileNotFoundError(f"Error: File '{json_file_path}' not found.")

    data = read_species_from_json(json_file_path)

    filtered_data: SpeciesDict = {
        key: data[key] for key in target_classes if key in data
    }

    total_species = sum(len(species) for species in filtered_data.values())

    print(f"\nExtracting data from {json_file_path}")
    for species_class, species in filtered_data.items():
        print(f"Extracted {len(species)} species from {species_class}")

    print(
        f"Extracted {len(filtered_data.keys())} classes: {list(filtered_data.keys())}"
    )
    print(f"Extracted {total_species} species")
    return filtered_data


def get_dataset_properties(data_path: str, output_file_path: str) -> None:
    """
    Extracts dataset properties by counting the number of images per species
    and saves the data to a JSON file.

    Args:
        data_path: The root directory of the dataset
        output_file_name: The path of the exported JSON file
    """
    dataset_properties: Dict[str, Dict[str, int]] = defaultdict(dict)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset repository not found: {data_path}")

    try:
        species_class_folders = [
            d
            for d in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, d))
        ]

        for species_class in species_class_folders:
            species_class_path = os.path.join(data_path, species_class)
            for species in os.listdir(species_class_path):
                species_path = os.path.join(species_class_path, species)
                species_images = [
                    img
                    for img in os.listdir(species_path)
                    if img.lower().endswith(".jpg")
                ]
                dataset_properties[species_class][species] = len(species_images)
    except Exception as e:
        print(f"Error while extracting species: {e}")

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    with open(output_file_path, "w", encoding="utf-8") as file:
        json.dump(dataset_properties, file, indent=2)

    print(f"Dataset properties saved to {output_file_path}")


if __name__ == "__main__":
    SPECIES_OUTPUT_PATH = "output/haute_garonne_other.json"
    OUTPUT_PATH = "output/haute-garonne_other_properties.json"
    DEFAULT_DATA_PATH: str = "."
    CHECK = False

    data_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_PATH

    if len(sys.argv) <= 1:
        print("Using the current working directory as default path.")

    try:
        if os.path.isfile(data_path) and ".json" in data_path:
            species_data = extract_species_from_json(data_path)
        else:
            species_data = extract_species_from_directory(data_path)
        if not CHECK:
            write_species_to_json(SPECIES_OUTPUT_PATH, species_data)
            get_dataset_properties(data_path, OUTPUT_PATH)
    except Exception as e:
        print(f"Error processing dataset: {e}")
