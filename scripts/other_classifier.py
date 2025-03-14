import os
import shutil
from typing import Optional


def species_path_extract(dir: str) -> list[str]:
    species_folder_path: list[str] = []
    for species_class in os.listdir(dir):
        class_path = os.path.join(dir, species_class)
        if os.path.isdir(class_path):
            for species in os.listdir(class_path):
                species_path = os.path.join(class_path, species)
                if os.path.isdir(species_path):
                    species_folder_path.append(species_path)
    return species_folder_path


def copy_file(
    species_paths: list[str],
    out_dir: str,
    isOther: bool,
    message: Optional[str] = None,
):
    total = len(species_paths)
    if message:
        print(message)
    for idx, species in enumerate(species_paths, 1):
        if isOther:
            species_base = species.split("/")[-1]
        else:
            species_base = "/".join(species.split("/")[-2:])
        species_out_dir = os.path.join(out_dir, species_base)

        if not os.path.exists(species_out_dir):
            os.makedirs(species_out_dir)

        print(f"{idx}/{total} Copying {species} to {out_dir}/{species_base}")

        for file in os.listdir(species):
            src = os.path.join(species, file)
            dst = os.path.join(species_out_dir)
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
    copy_file(other_folder_path, OUTPUT_DIR, True)
    message = "Stage 2: Copying species in Haute-Garonne to their new folder"
    copy_file(hg_folder_path, OUTPUT_DIR, False, message)
