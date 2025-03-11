import os
import sys
import shutil
from typing import Set, Tuple, Dict

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

def parse_species_file(filepath: str) -> Dict[str, Set[str]]:
    species_by_class: Dict[str, Set[str]] = {}
    current_class = ""

    with open(filepath, 'r') as file:
        for line in file:
            line = line.rstrip()
            if line and not line.startswith('\t') and "subspecies" not in line:
                current_class = line.strip()
                species_by_class[current_class] = set()
            elif line.startswith("\t"):
                species_name = line.strip()
                species_by_class[current_class].add(species_name)
    
    return species_by_class


def cross_reference(output_path: str, file_1: str, file_2: str) -> Tuple[Dict[str, Set[str]], int]:
    species_dict_1 = parse_species_file(FILE_1)
    species_dict_2 = parse_species_file(FILE_2)

    all_classes = set(species_dict_1) | set(species_dict_2)  # Union class to cover all unique classes from both dicts
    all_matched_dict: Dict[str, Set[str]] = {}

    try:
        os.makedirs("output", exist_ok=True)
        with open(output_path, "w") as file:
            matches_a, un_matches_a = cross_reference_aggregate(file_1, file_2)
            file.write(f"Total matched: {len(matches_a)} species\n")
            file.write(f"Total unmatched: {len(un_matches_a)} species\n\n")

            for class_name in all_classes:
                species_set_1: Set[str] = species_dict_1.get(class_name, set())
                species_set_2: Set[str] = species_dict_2.get(class_name, set())

                matches: Set[str] = species_set_1 & species_set_2
                all_matched_dict[class_name]= matches
                not_matches: Set[str] = (species_set_1 | species_set_2) - matches

                file.write(f"Class: {class_name}\n")
                file.write(f"\tMatched: {len(matches)}\n")
                for s in sorted(matches):
                    file.write(f"\t\t{s}\n")

                file.write(f"\tUnmatched: {len(not_matches)}\n")
                file.write(f"\t'-' means in {FILE_1} but not in {FILE_2} and vice-versa.\n")
                for s in sorted(not_matches):
                    mark = '-' if s in species_set_1 else '+'
                    file.write(f"\t\t{mark} {s}\n")
    except IOError:
        print("Something went wrong, cannot write to file!")

    return all_matched_dict, len(matches_a)


def copy_matched_species(src_dataset: str, dst_dataset: str, matched_species: Dict[str, Set[str]], total_matches: int) -> None:
    counter = 1
    for class_name, species_set in matched_species.items():
        for species_name in species_set:
            src_dir = os.path.join(src_dataset, class_name, species_name)
            dst_dir = os.path.join(dst_dataset, class_name, species_name)

            if os.path.exists(src_dir):
                os.makedirs(dst_dir, exist_ok=True)

                for item in os.listdir(src_dir):
                    src_file = os.path.join(src_dir, item)
                    if os.path.isfile(src_file):
                        shutil.copy2(src_file, dst_dir)
                print(f"{counter}/{total_matches} Copied: {class_name}/{species_name}")
                counter += 1
            else:
                print(f"Missing source directory: {src_dir}")


def parse_species_file_aggregate(filepath: str) -> Set[str]:
    species_set: Set[str] = set()
    with open(filepath, "r") as file:
        for line in file:
            line = line.rstrip()
            if (
                line
                and not line[0].isdigit()
                and not line.endswith("subspecies")
                and not line.startswith("\t")
                and line not in CLASS_LIST
            ):
                species_set.add(line.strip())
            elif line.startswith("\t"):
                species_set.add(line.strip())
        return species_set


def cross_reference_aggregate(file_1: str, file_2: str) -> Tuple[Set[str], Set[str]]:
    set_1: Set[str] = parse_species_file_aggregate(file_1)
    set_2: Set[str] = parse_species_file_aggregate(file_2)

    matches = set_1 & set_2
    not_matches = set_1 ^ set_2

    return matches, not_matches


if __name__ == "__main__":
    OUTPUT_PATH = "output/cross_reference.txt"
    FILE_1 = "output/species.txt"
    FILE_2 = "output/haute_garonne-species.txt"
    SRC_DATASET = "/media/tom-maverick/Dataset/train_val_images"
    DST_DATASET = "/media/tom-maverick/Dataset/INaturalMatched"

    try:
        if "-c" == sys.argv[1]:
            print("Copying data, please be patient.")
            match_species, total_matches = cross_reference(OUTPUT_PATH, FILE_1, FILE_2)
            copy_matched_species(SRC_DATASET, DST_DATASET, match_species, total_matches)
            print("Finish copying data")
    except IndexError:
        cross_reference(OUTPUT_PATH, FILE_1, FILE_2)
        print(f"Cross-reference result is save to {OUTPUT_PATH}")
