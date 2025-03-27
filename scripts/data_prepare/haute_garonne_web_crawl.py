import time
import os
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, Optional

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from scripts.utility import write_species_to_json, SpeciesDict, FailedOperation


def extract_species_by_class_web(url: str) -> SpeciesDict:
    """
    Extracts species names grouped by their taxonomic class from a webpage.

    Args:
        url: The URL of the page containing species classification.

    Returns:
        SpeciesDict (Dict[str, list[str]]): Dictionary containing species as keys and their species as values.
    """
    data: SpeciesDict = defaultdict(list)

    try:
        response: requests.Response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to fetch {url} cause of: {e}")
        return data

    soup = BeautifulSoup(response.text, "html.parser")

    for section in soup.select("h2.title"):
        class_tag: Optional[Tag] = section.select_one(".othernames .sciname")
        if not class_tag:
            continue

        class_name: str = class_tag.text.strip()
        species_list = section.find_next_sibling('ul', class_='listed_taxa')
        if not isinstance(species_list, Tag): continue

        for species in species_list.select("li.clear"):
            scientific_tag: Optional[Tag] = species.select_one(".sciname")

            if scientific_tag:
                scientific_name = scientific_tag.text.strip()
                data[class_name].append(scientific_name)
    
    # print(f"Extracted {sum(len(v) for v in data.values())} species across {len(data)} classes: {list(data.keys())}")
    return data


def scrape_species_data(total_pages: int, base_url: str, delay: int) -> SpeciesDict:
    """
    Scrapes species data from multiple pages and aggregates them by taxonomic class.

    Returns:
        SpeciesDict (Dict[str, list[str]]): Dictionary containing species as keys and their species as values.
    
    Raises:
        requests.RequestsException: The script cannot fetch the page
    """
    all_species: SpeciesDict = defaultdict(list)

    for page_number in tqdm(range(1, total_pages + 1)):
        url = f"{base_url}{page_number}&view=plain"

        try:
            # print(f"Processing Page {page_number}/{total_pages}: ", end="")
            page_species = extract_species_by_class_web(url)
            for class_name, species_list in page_species.items():
                all_species[class_name].extend(species_list)

        except requests.RequestException as e:
            print(f"Failed to fetch page {page_number}: {e}")

        time.sleep(delay)

    return all_species


def run_crawl(config):
    base_url = config["web_crawl"]["base_url"]
    total_pages = config["web_crawl"]["total_pages"]
    delay = config["web_crawl"]["delay_between_requests"]
    output_path = config["paths"]["web_crawl_output_json"]

    if os.path.isfile(output_path):
        raise FailedOperation(f"{output_path} already exists, please rename or delete it.\nFailed to crawl data from web.")

    else:
        try:
            species_data = scrape_species_data(total_pages, base_url, delay)
        except KeyboardInterrupt:
            print("Operation canceled.")

        write_species_to_json(output_path, species_data)