import os
import sys
import json
from typing import Dict, List
from collections import defaultdict
from scripts.utility import SpeciesDict, write_species_to_json, read_species_from_json

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
            if class_name == "species_lists":
                continue
            class_path = os.path.join(data_path, class_name)
            species_list = [
                d
                for d in os.listdir(class_path)
                if os.path.isdir(os.path.join(class_path, d))
            ]
            species_dict[class_name] = species_list

            species_extract = len(species_list)
            total_species += species_extract
            print(f"{class_path}: {species_extract}")
    except Exception as e:
        print(f"Error while extracting species: {e}")

    print(f"Extracted {len(species_dict.keys())} classes: {list(species_dict.keys())}")
    print(f"Total {total_species} species")
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

    print(f"Extracting data from {json_file_path}")
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
            if species_class == "species_lists":
                continue
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


def run_dataset_analysis(config, mod_dataset_created = False):
    original_data_path = config["paths"]["src_dataset"]
    if mod_dataset_created:
        modified_data_path = config["paths"]["dst_dataset"]
        inter_data_path = config["paths"]["inter_dataset"]
    else:
        modified_data_path = config["paths"]["web_crawl_output_json"]
    output_dir = config["paths"]["output_dir"]

    ori_species_output_path = os.path.join(output_dir, f"{os.path.basename(original_data_path)}_species.json")
    ori_prop_output_path = os.path.join(output_dir, f"{os.path.basename(original_data_path)}_properties.json")


    mod_species_output_path = os.path.join(output_dir, f"{os.path.basename(modified_data_path).split('.')[0]}_species.json")
    mod_prop_output_path = os.path.join(output_dir, f"{os.path.basename(modified_data_path)}_properties.json")

    species_data = extract_species_from_directory(original_data_path)
    write_species_to_json(ori_species_output_path, species_data)
    get_dataset_properties(original_data_path, ori_prop_output_path)
    print()

    if mod_dataset_created:
        inter_species_output_path = os.path.join(output_dir, f"{os.path.basename(inter_data_path)}_species.json")
        inter_prop_output_path = os.path.join(output_dir, f"{os.path.basename(inter_data_path)}_properties.json")


        species_data = extract_species_from_directory(modified_data_path)
        write_species_to_json(mod_species_output_path, species_data)
        get_dataset_properties(modified_data_path, mod_prop_output_path)

        print()

        species_data = extract_species_from_directory(inter_data_path)
        write_species_to_json(inter_species_output_path, species_data)
        get_dataset_properties(inter_data_path, inter_prop_output_path)
    else:
        species_data = extract_species_from_json(modified_data_path)
        write_species_to_json(mod_species_output_path, species_data)


def run_dataset_small_analysis(config):
    small_data_path = config["paths"]["dst_dataset_small"]
    output_dir = config["paths"]["output_dir"]

    small_species_output_path = os.path.join(output_dir, f"{os.path.basename(small_data_path)}_species.json")
    small_prop_output_path = os.path.join(output_dir, f"{os.path.basename(small_data_path)}_properties.json")

    species_data = extract_species_from_directory(small_data_path)
    write_species_to_json(small_species_output_path, species_data)
    get_dataset_properties(small_data_path, small_prop_output_path)