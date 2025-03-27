import json
import os
from typing import Set, Tuple

from scripts.utility import SpeciesDict, read_species_from_json, write_species_to_json, FailedOperation


def aggregate_all_species(species_data: SpeciesDict) -> Set[str]:
    species_set: Set[str] = set()
    # species_class can be used for filtering, however I have already processed
    # the input data beforehand so it's still here for when ever we need to do
    # the filtering in here
    for species_class, species_list in species_data.items():
        for species in species_list:
            species_set.add(species)
    return species_set


def cross_reference(
    species_dict_1: SpeciesDict,
    species_dict_2: SpeciesDict,
    species_set_1: Set[str],
    species_set_2: Set[str],
    output_path: str,
) -> Tuple[SpeciesDict, int]:
    """
    Cross-reference two species dataset, identifying matched and unmatched species, and exports
    the results to a JSON file.

    Args:
        species_dict_1: SpeciesDict
        species_dict_2: SpeciesDict
        species_set_1 (Set[str]): Aggregated set of all species from dataset 1
        species_set_2 (Set[str]): Aggregated set of all species from dataset 2
        output_path (str): Path to save the output JSON file

    Returns:
        Tuple[SpeciesDict, int]: A dictionary containing species class as keys and their species
        as values and the total number of matches.
    """

    matches, unmatched = find_set_matches_differences(species_set_1, species_set_2)
    # Union class to cover all unique classes from both dicts
    all_classes = set(species_dict_1) | set(species_dict_2)
    all_matched_dict: SpeciesDict = {}

    try:
        os.makedirs(os.path.basename(output_path), exist_ok=True)
        output_data = {
            "total_matched": len(matches),
            "total_unmatched": len(unmatched),
            "class_comparison": {},
        }

        for class_name in all_classes:
            species_set_1 = set(species_dict_1.get(class_name, []))
            species_set_2 = set(species_dict_2.get(class_name, []))

            class_matches = species_set_1 & species_set_2
            all_matched_dict[class_name] = list(class_matches)
            not_matches = (species_set_1 | species_set_2) - class_matches

            output_data["class_comparison"][class_name] = {  # type: ignore
                "matched": sorted(class_matches),
                "unmatched": sorted(not_matches),
            }

        json_file_path = os.path.join(output_path, "cross_reference.json")
        with open(json_file_path, "w", encoding="utf-8") as file:
            json.dump(output_data, file, indent=2)

        print(f"Cross-reference result saved to {output_path}/{json_file_path}")

    except IOError as e:
        print(f"Failed to write to file: {e}")

    return all_matched_dict, len(matches)


def find_set_matches_differences(
    set_1: Set[str], set_2: Set[str]
) -> Tuple[Set[str], Set[str]]:
    """
    Computes the intersection and symmetric difference of two datasets.

    Args:
        set_1: The first set of data
        set_2: The second set of data

    Returns:
        Tuple[Set[str], Set[str]]: A tuple containing
            - `matches`: Elements that exist in both sets
            - `not_matches`: Elements that are unique to either sets
    """
    matches = set_1 & set_2
    not_matches = set_1 ^ set_2
    return matches, not_matches


def run_cross_reference(config):
    output_dir = config["paths"]["output_dir"]
    json_1 = os.path.basename(config["paths"]["src_dataset"])
    json_1_path = f"{os.path.join(output_dir, json_1)}_species.json"
    json_2_path = config["paths"]["web_crawl_output_json"]

    dataset_1 = read_species_from_json(json_1_path)
    dataset_2 = read_species_from_json(json_2_path)
    if dataset_1 == {} or dataset_2 == {}:
        raise FailedOperation("")
    species_set_1 = aggregate_all_species(dataset_1)
    species_set_2 = aggregate_all_species(dataset_2)

    match_species, total_matches = cross_reference(
        dataset_1, dataset_2, species_set_1, species_set_2, output_dir 
    )

    write_species_to_json(os.path.join(output_dir, "matched_species.json"), match_species)