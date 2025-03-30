import os

from typing import List
from dataset_builder.core.exceptions import FailedOperation
from dataset_builder.core.log import log
from dataset_builder.core.utility import (
    read_species_from_json,
    write_data_to_json,
    _is_json_file,
)
from dataset_builder.analysis.matching import (
    cross_reference_set,
)


def run_cross_reference(
    output_file_path: str,
    # output_dir: str,
    json_1_path: str,
    json_2_path: str,
    data_1_name: str,
    data_2_name: str,
    target_classes: List[str],
    verbose: bool = False,
    overwrite: bool = False
):
    # file_name = f"matched_species_{data_1_name}_{data_2_name}.json"
    # output_path = os.path.join(output_dir, file_name)
    display_name = f"Matched species between {data_1_name} and {data_2_name}"

    if os.path.isfile(output_file_path) and not overwrite:
        print(f"{output_file_path} already exists, skipping web crawl.")
        return

    if not _is_json_file(json_1_path) or not _is_json_file(json_2_path):
        raise FailedOperation("JSON file not found.")

    dataset_1 = read_species_from_json(json_1_path)
    dataset_2 = read_species_from_json(json_2_path)

    if not dataset_1 or not dataset_2:
        raise FailedOperation(
            "One or both species dataset are empty. Cross-reference aborted."
        )

    match_species, total_matches = cross_reference_set(dataset_1, dataset_2, target_classes)

    log(f"Total matches: {total_matches}", verbose)


    write_data_to_json(output_file_path, display_name, match_species)
