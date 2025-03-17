import json
import pandas as pd
from typing import Set, Optional, Dict
from cross_reference import aggregate_all_species
from utility import read_species_from_json

import matplotlib.pyplot as plt
from matplotlib_venn import venn2


def venn_diagram(
    set_1: Set[str],
    set_1_name: str,
    set_2: Set[str],
    set_2_name: str,
    diagram_name: str,
    save_path: Optional[str] = None,
) -> None:

    only_dataset_1 = len(set_1 - set_2)
    only_dataset_2 = len(set_2 - set_1)
    shared_species = len(set_1 & set_2)
    no_in_common_species = len(set_1 ^ set_2)

    venn = venn2([set_1, set_2], set_labels=(set_1_name, set_2_name))

    venn.get_label_by_id("10").set_text(only_dataset_1)
    venn.get_label_by_id("01").set_text(only_dataset_2)
    venn.get_label_by_id("11").set_text(shared_species)

    summary_text = (
        f"Total species in {set_1_name}: {len(set_1)}\n"
        f"Total species in {set_2_name}: {len(set_2)}\n"
        f"Total shared species: {shared_species}\n"
        f"Total species that is not in common: {no_in_common_species}"
    )

    plt.text(
        0,
        -0.6,
        summary_text,
        ha="center",
        va="top",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.5),
    )

    plt.title(diagram_name)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Venn diagram saved to {save_path}")
    plt.show()

def class_composition_bar_chart(properties_json_path: str, class_to_analyze: str) -> None:
    with open(properties_json_path, "r", encoding='utf-8') as file:
        species_data = json.load(file)

    species_dict: Dict[str, int] = species_data[class_to_analyze]

    if class_to_analyze not in species_data:
        print(f"Class '{class_to_analyze}' not found.")
        return

    species_df = pd.DataFrame(species_dict.items(), columns=['Species', 'Image Count'])
    total_images = species_df["Image Count"].sum()
    species_df = species_df.sort_values(by="Image Count", ascending=False)

    species_df["Percentage"] = (species_df["Image Count"] / total_images) * 100

    labels = species_df["Species"]
    image_counts = species_df["Image Count"]
    percentages = species_df["Percentage"]

    fig, ax = plt.subplots(figsize=(15, len(labels) * 0.3))
    ax.barh(labels, image_counts)
    ax.set_xlabel("Number of images")
    ax.set_title(f"Species distribution within class: {class_to_analyze}")

    for i, (count, percentage) in enumerate(zip(image_counts, percentages)):
        ax.text(count + 1, i, f"{percentage:.2f}%", va="center")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    OUTPUT_PATH = "output"
    FILE_1 = "output/iNat2017_Aves_Insecta_Full.json"
    FILE_2 = "output/iNaturalist_HG_Aves_Insecta.json"
    PROPERTIES_FILE = "output/haute-garonne_properties.json"

    dataset_1 = read_species_from_json(FILE_1)
    dataset_2 = read_species_from_json(FILE_2)
    species_set_1 = aggregate_all_species(dataset_1)
    species_set_2 = aggregate_all_species(dataset_2)

    venn_diagram(
        species_set_1,
        "INaturalist 2017 Aves and Insecta",
        species_set_2,
        "Haute-Garonne Aves and Insecta",
        "Species Overlap Between Datasets",
    )

    class_composition_bar_chart(PROPERTIES_FILE, "Aves")