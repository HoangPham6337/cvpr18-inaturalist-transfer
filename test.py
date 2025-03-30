import json
import os
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from dataset_builder.analysis.matching import _aggregate_all_species
from dataset_builder.core.utility import read_species_from_json, _prepare_data_cdf_ppf

import matplotlib.pyplot as plt
from matplotlib_venn import venn2


def print_on_plot(species_needed: int, cdf_reached: int, species_num: int):
    plt.scatter(species_needed, cdf_reached, color="black", zorder=5)
    plt.text(species_needed + 1, cdf_reached, f"{species_needed} species ({cdf_reached*100:.1f}% images)\n{species_needed / species_num * 100:.1f}% species", fontsize=20, va="top")


def plot_axh_line(y: float, text: str):
    plt.axhline(y, color="red", linestyle="--", alpha=0.5)
    plt.text(0, y, text, va="bottom", ha="left", color="red", fontsize=12)



def visualizing_ppf(properties_json_path: str, class_to_analyze: str, save_path: Optional[str] = None) -> None:
    result = _prepare_data_cdf_ppf(properties_json_path, class_to_analyze)
    if result is None:
        print(f"ERROR: Data preparation failed for {class_to_analyze}")
        return

    species_names, sorted_image_counts = result

    total_images = sum(sorted_image_counts)
    cumulative_images = np.cumsum(sorted_image_counts) 
    cdf_values = cumulative_images / total_images

    species_num = len(species_names)
    species_indices = np.arange(1, species_num + 1)

    thresholds = [0.5, 0.8, 0.9]
    plt.figure(figsize=(30, 18))

    for threshold in thresholds:
        idx = np.argmax(cdf_values >= threshold)
        species_needed = int(idx + 1)
        cdf_reached = cdf_values[idx]

        plt.plot(species_indices, cdf_values, marker='.', linestyle="-")

        print_on_plot(species_needed, cdf_reached, species_num)

    plt.scatter(species_num, cdf_values[species_num - 1], color="black", zorder=5)
    plt.text(species_num + 1, cdf_values[species_num - 1], f"{species_num} species", fontsize=20, va="top")

    plt.xlabel("Number of species (ranked by image count)", fontsize=20)
    plt.ylabel("Cumulative percentage of images", fontsize=20)
    plt.title(f"Cumulative Composition Curve for {class_to_analyze}")

    plot_axh_line(0.5, "50%")
    plot_axh_line(0.8, "80%")
    plot_axh_line(0.9, "90%")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"PPF plot saved to {save_path}")
        plt.close()
    else:
        plt.show()


def class_composition_bar_chart(properties_json_path: str, class_to_analyze: str, save_path: Optional[str] = None) -> None:
    with open(properties_json_path, "r", encoding='utf-8') as file:
        species_data = json.load(file)

    species_dict: Dict[str, int] = species_data[class_to_analyze]

    if class_to_analyze not in species_data:
        print(f"Class '{class_to_analyze}' not found.")
        return

    species_df = pd.DataFrame(species_dict.items(), columns=['Species', 'Image Count'])
    species_df = species_df.sort_values(by="Image Count", ascending=True)

    total_images = species_df["Image Count"].sum()
    species_df["Percentage"] = (species_df["Image Count"] / total_images) * 100
    percentages = species_df["Percentage"]

    labels = species_df["Species"]
    image_counts = species_df["Image Count"]

    fig, ax = plt.subplots(figsize=(18, len(labels) * 0.3))
    ax.barh(labels, image_counts)
    ax.set_xlabel("Number of images")
    ax.set_title(f"Species distribution within class: {class_to_analyze}")
    ax.text(100, len(image_counts) + 1, f"Number of images: {total_images}", fontsize=20)

    for i, (count, percentage) in enumerate(zip(image_counts, percentages)):
        ax.text(count + 1, i, f"{count} / {percentage:.2f}%", va="center")
    
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Bar chart saved to {save_path}")
        plt.close()
    else:
        plt.show()


# visualizing_ppf("output/haute_garonne_composition.json", "Aves")
class_composition_bar_chart("output/train_val_images_composition.json", ['Actinopterygii', 'Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Chromista', 'Fungi', 'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Protozoa', 'Reptilia'], "./test.png")