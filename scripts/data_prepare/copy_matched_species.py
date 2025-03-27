from scripts.utility import copy_matched_species, read_species_from_json 


def run_copy(config):
    src = config["paths"]["src_dataset"]
    dst = config["paths"]["inter_dataset"]
    matched_json_path = config["paths"]["matched_species_json"]
    included_classes = config["train_val_split"]["included_classes"]

    matched_species = read_species_from_json(matched_json_path)
    counter = 0
    for species_class in matched_species.keys():
        if species_class in included_classes:
            for _ in matched_species.get(species_class, []):
                counter += 1
    matched_number = counter
    copy_matched_species(src, dst, matched_species, matched_number, included_classes)