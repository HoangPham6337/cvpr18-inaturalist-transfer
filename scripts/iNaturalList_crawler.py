import requests
import time
from bs4 import BeautifulSoup, Tag, NavigableString
from typing import Optional, Dict
from collections import defaultdict
from species_extraction import SpeciesDict, write_species_to_file


def extract_species_by_class(url: str) -> SpeciesDict:
    data: SpeciesDict = defaultdict(list)

    try:
        response: requests.Response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to fetch {url} cause of: {e}")
        return data
        
    soup = BeautifulSoup(response.text, 'html.parser')

    for section in soup.select('h2.title'):
        class_span: Optional[Tag] = section.select_one('.othernames .sciname')
        if not class_span: continue

        class_name: str = class_span.text.strip()
        species_list_sibling: Optional[Tag | NavigableString] = section.find_next_sibling('ul', class_='listed_taxa')
        if not isinstance(species_list_sibling, Tag): continue

        species_list: Tag = species_list_sibling

        for species in species_list.select('li.clear'):
            scientific_span: Optional[Tag] = species.select_one('.sciname')

            if scientific_span is None:
                continue

            scientific_name = scientific_span.text.strip()
            data[class_name].append(scientific_name)

    return data

if __name__ == "__main__":
    TOTAL_PAGES = 104
    FILE_PATH = "output/haute_garonne-species.txt"
    base_url: str = "https://www.inaturalist.org/check_lists/32961-Haute-Garonne-Check-List?page="
    all_species: SpeciesDict = defaultdict(list)

    for page_number in range(1, TOTAL_PAGES + 1):
        url = f"{base_url}{page_number}&view=plain"
        page_species = extract_species_by_class(url)
        for class_name, species_list in page_species.items():
            all_species[class_name].extend(species_list)

        print(f"Processed Page {page_number}/{TOTAL_PAGES}")
        time.sleep(1)

    write_species_to_file(FILE_PATH, all_species)
