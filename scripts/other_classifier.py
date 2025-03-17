import os
import shutil
from typing import Optional


def species_path_extract(data_path: str) -> list[str]:
    """
    Extracts paths to species directories within a given dataset directory.
    
    Args:
        data_path: The base directory containing class folders.

    Returns:
        List[str]: A list of full paths to species directory.

    Raises:
        FileNotFoundError: If the given directory does not exists or is not a directory.
    """
    species_folder_path: list[str] = []
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Directory not found: {data_path}")

    for class_entry in os.scandir(data_path):
        if class_entry.is_dir():
            class_path = class_entry.path
            for species_entry in os.scandir(class_path):
                if species_entry.is_dir():
                    species_folder_path.extend(species_entry.path)
    return species_folder_path


def copy_file(
    species_paths: list[str],
    out_dir: str,
    isOther: bool,
    message: Optional[str] = None,
):
    """
    Copies files from species directories into a structured output directory.

    Args:
        species_paths: List of species directory paths.
        out_dir: The output directory where files should be copied.
        is_other: Determines if species belong to the 'Other' category.
        other_dir: Name of the 'Other' category folder. Defaults to "Other".
        message: Message to display before processing. Defaults to None.
    """
    total = len(species_paths)
    if message:
        print(message)
    
    for idx, species in enumerate(species_paths, 1):
        species_parts = species.split(os.sep)

        if isOther:
            species_base = species_parts[-1]
            species_out_dir = os.path.join(out_dir, OTHER_DIR, species_base)
        else:
            species_base = os.path.join(species_parts[-2], species_parts[-1])
            species_out_dir = os.path.join(out_dir, species_base)

        os.makedirs(species_out_dir, exist_ok=True)

        print(f"{idx}/{total} Copying {species} to {species_out_dir}")

        for file in os.listdir(species):
            src = os.path.join(species, file)
            dst = os.path.join(species_out_dir, file)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)


if __name__ == "__main__":
    REPO_DIR = os.getcwd()

    INAT_DIR = os.path.join(REPO_DIR, "data/inat2017")
    HG_DIR = os.path.join(REPO_DIR, "data/haute_garonne")
    OUTPUT_DIR = os.path.join(REPO_DIR, "data/inat2017_other")
    OTHER_DIR = os.path.join(OUTPUT_DIR, "Other")

    if not os.path.exists(OTHER_DIR):
        os.makedirs(OTHER_DIR)

    hg_folder_path: list[str] = species_path_extract(HG_DIR)
    hg_species: list[str] = [os.path.basename(f) for f in hg_folder_path]
    inat_folder_path: list[str] = species_path_extract(INAT_DIR)
    other_folder_path: list[str] = [
        f for f in inat_folder_path if os.path.basename(f) not in hg_species
    ]

    message = "Stage 1: Copying species not in Haute-Garonne to 'Other'"
    copy_file(other_folder_path, OUTPUT_DIR, True, message)
    message = "Stage 2: Copying species in Haute-Garonne to their new folder"
    copy_file(hg_folder_path, OUTPUT_DIR, False, message)
