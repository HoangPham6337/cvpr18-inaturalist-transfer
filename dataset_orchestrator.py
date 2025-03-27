import traceback
import os
from scripts.utility import load_config, cleanup, FailedOperation
from scripts.data_prepare.haute_garonne_web_crawl import run_crawl
from scripts.data_prepare.copy_matched_species import run_copy
from scripts.data_prepare.train_val_splitter import run_split
from scripts.data_prepare.other_dataset_builder import run_other_dataset_builder_big
from scripts.data_prepare.other_dataset_builder import run_other_dataset_builder_small
from scripts.analysis.dataset_analyzer import run_dataset_analysis
from scripts.analysis.dataset_analyzer import run_dataset_small_analysis
from scripts.analysis.cross_reference import run_cross_reference
from scripts.analysis.visualizer import run_visualization


config = load_config("./config.yaml")

def run_stage(stage_name: str, func):
    separator = "#" * 50
    print(f"\n{separator}")
    print(f"{stage_name}...")
    print(f"{separator}")
    func()
    print(f"{stage_name} completed.")
    print(f"{separator}\n")


os.makedirs(config["paths"]["src_dataset"], exist_ok=True)
os.makedirs(config["paths"]["inter_dataset"], exist_ok=True)
os.makedirs(config["paths"]["dst_dataset"], exist_ok=True)
os.makedirs(config["paths"]["dst_dataset_small"], exist_ok=True)
os.makedirs(config["paths"]["output_dir"], exist_ok=True)


try:
    run_stage("Crawling the web to obtain the dataset", lambda: run_crawl(config))
    run_stage("Getting datasets properties", lambda: run_dataset_analysis(config))
    run_stage("Matching species between datasets", lambda: run_cross_reference(config))
    run_stage("Copying matching species", lambda: run_copy(config))
    run_stage("Creating dataset with 'Other'", lambda: run_other_dataset_builder_big(config))
    run_stage("Getting datasets properties", lambda: run_dataset_analysis(config, mod_dataset_created=True))
    run_stage("Creating small dataset with 'Other'", lambda: run_other_dataset_builder_small(config))
    run_stage("Getting small datasets properties", lambda: run_dataset_small_analysis(config))
    run_stage("Generating dataset manifests", lambda: run_split(config))
    run_stage("Generating visualization", lambda: run_visualization(config))

except FailedOperation as failedOp:
    print(failedOp.message, "\n")
    print(traceback.format_exc())
    print("Cleaning up")
    cleanup(config)
    exit()