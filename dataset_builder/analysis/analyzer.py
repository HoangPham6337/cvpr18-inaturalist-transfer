import os
from typing import List
from dataset_builder.analysis.scanner import (
    _scan_image_counts,
    _scan_species_list,
    _filter_species_from_json,
)
from dataset_builder.core.utility import write_data_to_json, _is_json_file, SpeciesDict


def _summarize_species_data(
    species_dict: SpeciesDict, source: str, verbose: bool = False
):
    total_species = sum(len(species) for species in species_dict.values())
    print(f"Extracted from: {source}")
    if verbose:
        for species_class, species in species_dict.items():
            print(f"\t{species_class}: {len(species)} species")
    print(f"Total extracted species: {total_species}")


def run_analyze_dataset(
    data_path: str,
    output_dir: str,
    prefix: str,
    target_classes: List[str],
    verbose: bool = False,
    overwrite: bool = False,
):
    """
    Analyzes a dataset (folder or JSON) and outputs:
    - species list
    - image count per species (if folder)

    Args:
        data_path: Path to dataset root or JSON species file
        output_dir: Folder to save results
        prefix: Prefix for output filenames
        target_classes: Class filters (used for both folder and JSON)
        verbose: Whether to print detailed per-class info
    """

    os.makedirs(output_dir, exist_ok=True)
    species_output_path = os.path.join(output_dir, f"{prefix}_species.json")

    if (
        os.path.isfile(species_output_path)
        and _is_json_file(species_output_path)
        and not overwrite
    ):
        print(f"{species_output_path} already exists, skipping analyzing dataset.")
        return

    species_name, species_count = "Species list", "Properties"
    counts_path = os.path.join(output_dir, f"{prefix}_composition.json")

    if data_path.lower().endswith(".json"):
        species_dict = _filter_species_from_json(data_path, target_classes, verbose)
        _summarize_species_data(species_dict, data_path, verbose)
        write_data_to_json(species_output_path, species_name, species_dict)
    else:
        species_dict, total_species = _scan_species_list(data_path)
        print(f"Total extracted species: {total_species}")
        image_counts = _scan_image_counts(data_path)

        write_data_to_json(species_output_path, species_name, species_dict)
        write_data_to_json(counts_path, species_count, image_counts)
