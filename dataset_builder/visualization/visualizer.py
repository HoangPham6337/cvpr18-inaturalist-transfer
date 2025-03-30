import json
import os
import pandas as pd
import numpy as np
import multiprocessing
from typing import List, Optional, Dict
from dataset_builder.analysis.matching import _aggregate_all_species
from dataset_builder.core.utility import read_species_from_json, _prepare_data_cdf_ppf, log

import matplotlib.pyplot as plt
from matplotlib_venn import venn2


def _print_on_plot(species_needed: int, cdf_reached: int, species_num: int):
    plt.scatter(species_needed, cdf_reached, color="black", zorder=5)
    plt.text(species_needed + 1, cdf_reached, f"{species_needed} species ({cdf_reached*100:.1f}% images)\n{species_needed / species_num * 100:.1f}% species", fontsize=20, va="top")


def _plot_axh_line(y: float, text: str):
    plt.axhline(y, color="red", linestyle="--", alpha=0.5)
    plt.text(0, y, text, va="bottom", ha="left", color="red", fontsize=12)


def venn_diagram(
    dataset_1_path: str,
    dataset_2_path: str,
    set_1_name: str,
    set_2_name: str,
    diagram_name: str,
    save_path: Optional[str] = None,
) -> None:
    dataset_1 = read_species_from_json(dataset_1_path)
    dataset_2 = read_species_from_json(dataset_2_path)

    set_1 = _aggregate_all_species(dataset_1)
    set_2 = _aggregate_all_species(dataset_2)

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

def _class_composition_bar_chart(properties_json_path: str, class_to_analyze: str, save_path: Optional[str] = None, verbose: bool = False) -> None:
    with open(properties_json_path, "r", encoding='utf-8') as file:
        species_data = json.load(file)

    # Allow for failing since we still can generate other plot
    if class_to_analyze not in species_data:
        log(f"Class '{class_to_analyze}' not found.", True, "WARNING")
        return

    species_dict: Dict[str, int] = species_data[class_to_analyze]

    species_df = pd.DataFrame(species_dict.items(), columns=['Species', 'Image Count'])
    species_df = species_df.sort_values(by="Image Count", ascending=True)

    total_images = species_df["Image Count"].sum()
    species_df["Percentage"] = (species_df["Image Count"] / total_images) * 100
    percentages = species_df["Percentage"]

    labels = species_df["Species"]
    image_counts = species_df["Image Count"]

    fig, ax = plt.subplots(figsize=(18, min(int(len(labels) * 0.3), 200)))
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
        log(f"Bar chart saved to {save_path}", verbose)
        plt.close()
    else:
        plt.show()



def _visualizing_ppf(properties_json_path: str, class_to_analyze: str, save_path: Optional[str] = None, verbose: bool = False) -> None:
    result = _prepare_data_cdf_ppf(properties_json_path, class_to_analyze)
    # Allow for failing
    if result is None:
        log(f"No data found for {class_to_analyze}", True, "WARNING")
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

        _print_on_plot(species_needed, cdf_reached, species_num)

    plt.scatter(species_num, cdf_values[species_num - 1], color="black", zorder=5)
    plt.text(species_num + 1, cdf_values[species_num - 1], f"{species_num} species", fontsize=20, va="top")

    plt.xlabel("Number of species (ranked by image count)", fontsize=20)
    plt.ylabel("Cumulative percentage of images", fontsize=20)
    plt.title(f"Cumulative Composition Curve for {class_to_analyze}")

    _plot_axh_line(0.5, "50%")
    _plot_axh_line(0.8, "80%")
    _plot_axh_line(0.9, "90%")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        log(f"PPF plot saved to {save_path}", verbose)
        plt.close()
    else:
        plt.show()


def _visualize_class(properties_file: str, species_class: str, export_dir: str, dataset_name: str, verbose: bool = False) -> None:
    export_dir = os.path.join(export_dir, dataset_name)
    base_filename = f"{dataset_name}_{species_class}"
    print(f"Processing {base_filename}")

    # Generate the composition bar chart
    _class_composition_bar_chart(
        properties_file,
        species_class,
        save_path=os.path.join(export_dir, "composition", f"{base_filename}_bar.png"),
        verbose=verbose
    )

    # Generate the PPF visualization
    _visualizing_ppf(
        properties_file,
        species_class,
        save_path=os.path.join(export_dir, "ppf", f"{base_filename}_ppf.png"),
        verbose=verbose
    )


def run_visualization(src_dataset_path: str, dst_dataset_path: str, output_dir: str, target_classes_src: List[str], target_classes_dst: List[str], verbose: bool = False):
    src_dataset_name = src_dataset_path.split(os.sep)[-1]
    dst_dataset_name = dst_dataset_path.split(os.sep)[-1]

    output_dir = output_dir
    export_dir = os.path.join(output_dir, "plots")
    os.makedirs(os.path.join(output_dir, "plots", "composition"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots", "ppf"), exist_ok=True)

    os.makedirs(export_dir, exist_ok=True)

    src_file = os.path.join(output_dir, f"{src_dataset_name}_species.json")  # Original dataset
    dst_file = os.path.join(output_dir, f"{dst_dataset_name}_species.json")  # Filtered original dataset

    properties_file_1 = os.path.join(output_dir, f"{src_dataset_name}_composition.json")
    properties_file_2 = os.path.join(output_dir, f"{dst_dataset_name}_composition.json")
    
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(
            _visualize_class,
            [(properties_file_1, species_class, export_dir, src_dataset_name, verbose) for species_class in target_classes_src]
        )
        
        pool.starmap(
            _visualize_class,
            [(properties_file_2, species_class, export_dir, dst_dataset_name, verbose) for species_class in target_classes_dst]
        )

    # venn_diagram(
    #     src_file,
    #     dst_file,
    #     src_dataset_name,
    #     dst_dataset_name,
    #     "Species Overlap Between Datasets",
    #     save_path=os.path.join(export_dir, f"{src_dataset_name}_vs_{dst_dataset_name}_venn.png")
    # )