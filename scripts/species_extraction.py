import os
import sys
from typing import Dict, List

SpeciesDict = Dict[str, list[str]]

def extract_species(repo_path: str) -> SpeciesDict:
    species_dict: SpeciesDict = {}

    for class_name in next(os.walk(repo_path))[1]:
        class_path = os.path.join(repo_path, class_name)
        species = [item for item in next(os.walk(class_path))[1]]
        species_dict[class_name] = species

    return species_dict

def pretty_print_all_species(species_dict: SpeciesDict) -> None:
    for species, subspecies in species_dict.items():
        print(f"\"{species}\"", end=", ")
        # for s in subspecies:
        #     print(f"\t{s}")
    print()

def write_species_to_file(file_path: str, species_dict: SpeciesDict) -> None:
    try:
        os.makedirs("output", exist_ok=True)
        with open(file_path, "w") as f:
            for species, subspecies in species_dict.items():
                f.writelines(f"{species}\n")
                f.writelines(f"{len(subspecies)} subspecies\n")
                for s in subspecies:
                    f.writelines(f"\t{s}\n")
    except IOError:
        print("Something gone wrong, check the file.")

if __name__ == "__main__":
    FILE_PATH = "output/species.txt"
    repo_path: str = ""
    try:
        repo_path = sys.argv[1]
    except IndexError:
        repo_path = "."

    species_data: Dict[str, List[str]] = extract_species(repo_path)

    if "-v" in sys.argv:
        pretty_print_all_species(species_data)
    else:
        write_species_to_file(FILE_PATH, species_data)