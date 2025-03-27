import json
import os
import shutil
import yaml
import glob
from tqdm import tqdm
from typing import Dict, Tuple, Optional

SpeciesDict = Dict[str, list[str]]

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

class FailedOperation(Exception):
    """Raise when an operation failed due to external reasons."""
    def __init__(self, message, *args):
        self.message = message
        super(FailedOperation, self).__init__(*args)


def load_config(config_path: str):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def write_species_to_json(file_output_path: str, species_data: SpeciesDict) -> None:
    """
    Writes species and subspecies data to a JSON file.

    Args:
        file_output_path: The path where the JSON file will be saved.
        species_data (Dict[str, list[str]]): Dictionary containing species as keys and their species as values.

    Raises:
        IOError: If an error occur while writing to the file.
    """
    try:
        os.makedirs(os.path.dirname(file_output_path), exist_ok=True)

        with open(file_output_path, "w", encoding="utf-8") as f:
            json.dump(species_data, f, indent=4)
        
        print(f"JSON file is dumped to {file_output_path}")
    except IOError as e:
        print(f"Error writing to file {file_output_path}: {e}")


def read_species_from_json(file_input_path: str) -> SpeciesDict:
    """
    Reads species and subspecies data from a JSON file.

    Args:
        file_output_path: The path to the JSON file.
    
    Returns:
        SpeciesDict (Dict[str, list[str]]): Dictionary containing species as keys and their species as values.

    Raises:
        IOError: If the file cannot be read or does not exist.
        JSONDecodeError: If the file is not a valid JSON.
    """
    try:
        with open(file_input_path, "r", encoding="utf-8") as f:
            species_data = json.load(f)

            print(f"Successfully loaded species data from {file_input_path}")
            return species_data

    except IOError as e:
        print(f"Error reading {file_input_path}: {e}")
        return {}

    except json.JSONDecodeError as e:
        print(f"Invalid JSON format in {file_input_path}: {e}")
        return {}

def copy_matched_species(
    src_dataset: str,
    dst_dataset: str,
    matched_species: SpeciesDict,
    total_matches: int,
    included_class: list[str]
) -> None:
    species_counter = 0
    for class_name, species_set in matched_species.items():
        if class_name not in included_class:
            continue
        for species_name in tqdm(species_set, f"Copy data in {class_name}"):
            src_dir = os.path.join(src_dataset, class_name, species_name)
            dst_dir = os.path.join(dst_dataset, class_name, species_name)

            if os.path.exists(src_dir):
                species_counter += 1
                os.makedirs(dst_dir, exist_ok=True)

                # print(f"{species_counter}/{total_matches} Copied: {class_name}/{species_name}")
                for item in os.listdir(src_dir):
                    src_file = os.path.join(src_dir, item)
                    dst_file = os.path.join(dst_dir, item)
                    if os.path.isfile(src_file):
                        if not os.path.isfile(dst_file):
                            shutil.copy2(src_file, dst_dir)
                            # print(f"{species_counter}/{total_matches} Copied: {class_name}/{species_name}/{item}")
                        else:
                            # print(f"{species_counter}/{total_matches} File exists - skipping: {class_name}/{species_name}/{item}")
                            pass
            else:
                print(f"{species_counter}/{total_matches} Missing source directory: {src_dir}")

    if (total_matches != species_counter):
        raise FailedOperation(f"Failed to copy all matches species {total_matches} != {species_counter}")


def prepare_data_cdf_ppf(properties_json_path: str, class_to_analyze: str) -> Optional[Tuple[list[str], list[int]]]:
    """Loads species image data from JSON dataset properties file, sorts by image count, and prepares data for CDF/PPF calculations.

    Args:
        properties_json_path: Path to the dataset properties JSON file.
        class_to_analyze: The target class (e.g., 'Aves', 'Insecta').

    Returns:
        Optional[Tuple[List[str], List[int]]]: Sorted species names and corresponding image counts.
    """
    try:
        with open(properties_json_path, "r", encoding="utf-8") as file:
            species_data = json.load(file)
    except FileNotFoundError:
        print(f"File not found: {properties_json_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Not a valid JSON file: {e}")
        return None

    species_images_data: Dict[str, int] = species_data.get(class_to_analyze, {})
    if not species_images_data:
        print(f"ERROR: Class '{class_to_analyze}' not found or contains no data")
        return None
    sorted_species = sorted(species_images_data.items(), key=lambda x: x[1])
    species_names, image_counts = zip(*sorted_species)

    return list(species_names), list(image_counts)


def cleanup(config):
    output_dir = config["paths"]["output_dir"]
    inter_dataset = config["paths"]["inter_dataset"]
    dst_dataset = config["paths"]["dst_dataset"]
    dst_dataset_small = config["paths"]["dst_dataset_small"]
    datasets = [inter_dataset, dst_dataset, dst_dataset_small]
    files = glob.glob(f"{output_dir}/*")
    for path in files:
        print(f"Removing: {path}")
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
            
    for dataset in datasets:
        shutil.rmtree(dataset)