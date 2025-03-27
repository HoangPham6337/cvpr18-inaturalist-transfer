import json
import os
import pandas as pd
import numpy as np
from typing import Set, Optional, Dict
from scripts.analysis.cross_reference import aggregate_all_species
from scripts.utility import read_species_from_json, prepare_data_cdf_ppf

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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Venn diagram saved to {save_path}")
    plt.close()
    # plt.show()

def class_composition_bar_chart(properties_json_path: str, class_to_analyze: str, save_path: Optional[str] = None) -> None:
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

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Bar chart saved to {save_path}")
    plt.close()


def visualizing_cdf(properties_json_path: str, class_to_analyze: str, save_path: Optional[str] = None) -> None:
    result = prepare_data_cdf_ppf(properties_json_path, class_to_analyze)


    if result is None:
        print(f"ERROR: Data preparation failed for {class_to_analyze}")
        return
    species_names, sorted_image_counts = result

    ecdf = False
    if ecdf:
        cdf_values = np.arange(1, len(sorted_image_counts) + 1) / len(sorted_image_counts)
    else:
        total_images = sum(sorted_image_counts)
        cumulative_images = np.cumsum(sorted_image_counts) 
        cdf_values = cumulative_images / total_images

    plt.figure(figsize=(12, 6))
    plt.plot(sorted_image_counts, cdf_values, marker=".", linestyle="-")

    plt.xlabel("Number of Images")
    plt.ylabel("Cumulative Probability")
    plt.title(f"Cumulative Distribution Function (CDF) of {class_to_analyze} Image Counts")
    plt.grid(axis="x")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"CDF plot saved to {save_path}")

    plt.close()


def visualizing_ppf(properties_json_path: str, class_to_analyze: str, save_path: Optional[str] = None) -> None:
    result = prepare_data_cdf_ppf(properties_json_path, class_to_analyze)
    if result is None:
        print(f"ERROR: Data preparation failed for {class_to_analyze}")
        return

    species_names, sorted_image_counts = result

    total_images = sum(sorted_image_counts)
    cumulative_images = np.cumsum(sorted_image_counts) 
    cdf_values = cumulative_images / total_images

    plt.figure(figsize=(6, 12))
    plt.plot(cdf_values, sorted_image_counts, marker='.', linestyle="-")
    for x, y, species in zip(cdf_values, sorted_image_counts, species_names):
        plt.hlines(y, xmin=0, xmax=x, colors="gray", linestyles="dashed", alpha=0.5)
        # plt.text(x, y, species, fontsize=8, rotation=45, ha="right")
    plt.xlabel("Cumulative Probability")
    plt.yticks(ticks=sorted_image_counts, labels=species_names, fontsize=10)
    plt.ylabel("Species")
    plt.title(f"Percent Point Function (PPF) of {class_to_analyze} Image Counts")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"PPF plot saved to {save_path}")
    plt.close()


def run_visualization(config):
    src_dataset_name = config["paths"]["src_dataset"].split(os.sep)[-1]
    dst_dataset_name = config["paths"]["dst_dataset"].split(os.sep)[-1]
    inter_dataset_name = config["paths"]["inter_dataset"].split(os.sep)[-1]
    small_dataset_name = config["paths"]["dst_dataset_small"].split(os.sep)[-1]

    output_dir = config["paths"]["output_dir"]
    export_dir = os.path.join(output_dir, "plots")

    os.makedirs(export_dir, exist_ok=True)
    included_classes = config["train_val_split"]["included_classes"]

    file_1 = os.path.join(output_dir, f"{src_dataset_name}_species.json")  # Original dataset
    file_2 = os.path.join(output_dir, f"{dst_dataset_name}_species.json")  # Filtered original dataset
    file_3 = os.path.join(output_dir, f"{inter_dataset_name}.json")  # Original dst dataset
    file_4 = os.path.join(output_dir, f"{small_dataset_name}_species.json")  # Filtered dst dataset

    properties_file_1 = os.path.join(output_dir, f"{src_dataset_name}_properties.json")
    properties_file_2 = os.path.join(output_dir, f"{dst_dataset_name}_properties.json")

    dataset_1 = read_species_from_json(file_1)
    dataset_2 = read_species_from_json(file_2)
    dataset_3 = read_species_from_json(file_3)
    dataset_4 = read_species_from_json(file_4)

    species_set_1 = aggregate_all_species(dataset_1)
    species_set_2 = aggregate_all_species(dataset_2)
    species_set_3 = aggregate_all_species(dataset_3)
    species_set_4 = aggregate_all_species(dataset_4)


    for species_class in included_classes:
        for dataset_name, properties_file in [
            (src_dataset_name, properties_file_1),
            (dst_dataset_name, properties_file_2),
        ]:
            base_filename = f"{dataset_name}_{species_class}"

            # class_composition_bar_chart(
            #     properties_file,
            #     species_class,
            #     save_path=os.path.join(export_dir, "composition", f"{base_filename}_bar.png")
            # )

            # visualizing_cdf(
            #     properties_file,
            #     species_class,
            #     save_path=os.path.join(export_dir, "cdf", f"{base_filename}_cdf.png")
            # )

            # visualizing_ppf(
            #     properties_file,
            #     species_class,
            #     save_path=os.path.join(export_dir, "pdf", f"{base_filename}_ppf.png")
            # )

    venn_diagram(
        species_set_1,
        src_dataset_name,
        species_set_3,
        inter_dataset_name,
        "Species Overlap Between Datasets",
        save_path=os.path.join(export_dir, f"{src_dataset_name}_vs_{inter_dataset_name}_venn.png")
    )

    venn_diagram(
        species_set_2,
        dst_dataset_name,
        species_set_4,
        small_dataset_name,
        "Species Overlap Between Datasets",
        save_path=os.path.join(export_dir, f"{dst_dataset_name}_vs_{small_dataset_name}_venn.png")
    )