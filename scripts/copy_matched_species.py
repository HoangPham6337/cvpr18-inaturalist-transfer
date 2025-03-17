from utility import copy_matched_species, read_species_from_json

SRC_DATASET = (
    "/home/tom-maverick/Documents/02_GitHub/cvpr18-inaturalist-transfer/data/inat2017/"
)
DST_DATASET = "/home/tom-maverick/Documents/02_GitHub/cvpr18-inaturalist-transfer/data/haute_garonne/"


matched_species = read_species_from_json("output/matched_species_Aves_Insecta.json")

copy_matched_species(SRC_DATASET, DST_DATASET, matched_species, 284)
