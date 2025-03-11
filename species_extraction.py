import os
import sys
from typing import Dict, List

def extract_species(repo_path: str) -> Dict[str, List[str]]:
    species_dict: Dict[str, list[str]] = {}

    for root, dirs, _ in os.walk(repo_path):
        species_name: str = os.path.basename(root)

        subspecies = [dir for dir in dirs]
        
        if subspecies:
            species_dict[species_name] = subspecies

    return species_dict

def pretty_print_all_species(species_dict: Dict[str, List[str]]) -> None:
    for species, subspecies in species_dict.items():
        print(f"Species: {species}")
        for s in subspecies:
            print(f" ├── {s}")

def write_species_to_file(species_dict: Dict[str, List[str]]) -> None:
    try:
        with open("species.txt", "w") as f:
            for species, subspecies in species_dict.items():
                f.writelines(f"{species} - {len(subspecies)} subspecies\n")
                for s in subspecies:
                    f.writelines(f" ├── {s}\n")
    except IOError:
        print("Something gone wrong, check the file.")

if __name__ == "__main__":
    repo_path: str = ""
    if sys.argv[1]:
        repo_path = sys.argv[1]
    else:
        repo_path = "."

    species_data: Dict[str, List[str]] = extract_species(repo_path)

    # pretty_print_all_species(species_data)
    write_species_to_file(species_data)