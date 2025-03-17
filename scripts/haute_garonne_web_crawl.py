import time
from collections import defaultdict
from typing import Dict, Optional

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from scripts.dataset_analyzer import write_species_to_json
from utility import SpeciesDict


def extract_species_by_class(url: str) -> SpeciesDict:
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
    
    print(f"Extracted {sum(len(v) for v in data.values())} species across {len(data)} classes: {list(data.keys())}")
    return data


def scrape_species_data() -> SpeciesDict:
    """
    Scrapes species data from multiple pages and aggregates them by taxonomic class.

    Returns:
        SpeciesDict (Dict[str, list[str]]): Dictionary containing species as keys and their species as values.
    
    Raises:
        requests.RequestsException: The script cannot fetch the page
    """
    all_species: SpeciesDict = defaultdict(list)

    for page_number in range(1, TOTAL_PAGES + 1):
        url = f"{BASE_URL}{page_number}&view=plain"

        try:
            print(f"Processing Page {page_number}/{TOTAL_PAGES}: ", end="")
            page_species = extract_species_by_class(url)
            for class_name, species_list in page_species.items():
                all_species[class_name].extend(species_list)

        except requests.RequestException as e:
            print(f"Failed to fetch page {page_number}: {e}")

        time.sleep(DELAY_BETWEEN_REQUEST)

    return all_species


if __name__ == "__main__":
    TOTAL_PAGES = 104
    FILE_PATH = "output/iNaturalist_All_Species_Full.json"
    BASE_URL: str = "https://www.inaturalist.org/check_lists/32961-Haute-Garonne-Check-List?page="
    DELAY_BETWEEN_REQUEST = 1
    
    species_data = scrape_species_data()

    write_species_to_json(FILE_PATH, species_data)

    print(f"\nTotal classes: {len(species_data)}")
    print(f"Total species extracted: {sum(len(v) for v in species_data.values())}")