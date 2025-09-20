import requests
from bs4 import BeautifulSoup
import time
import json
import urllib.parse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "ner_config_wiki.json"
PROGRESS_FILE = BASE_DIR / "progress_wiki.json"

START_CATEGORIES = [
    "Category:Chief_executives",
    "Category:Technology_company_founders",
    "Category:Women_chief_executives"
]

WIKIPEDIA_BASE = "https://en.wikipedia.org/wiki/"
HEADERS = {"User-Agent": "Mozilla/5.0"}

MAX_DEPTH = 2
SAVE_INTERVAL = 20

def load_config():
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("Config file is corrupted. Reinitializing.")
    return {"role_keywords": {}, "known_people": {}, "non_person_terms": []}

def save_config(config):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

def load_progress():
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("Progress file is corrupted. Reinitializing.")
    return {}

def save_progress(progress):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

def get_category_members(category_url):
    try:
        r = requests.get(category_url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        links = soup.select("#mw-pages a, #mw-subcategories a")
        return [link.get("href") for link in links if link.get("href")]
    except Exception as e:
        print(f"Failed to fetch {category_url}: {e}")
        return []

def get_subcategories(category_url):
    try:
        r = requests.get(category_url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        links = soup.select("#mw-subcategories a")
        return [link.get("href") for link in links if link.get("href") and link.get("href").startswith("/wiki/Category:")]
    except Exception as e:
        print(f"Failed to fetch subcategories from {category_url}: {e}")
        return []

def should_skip_category(category_name):
    lower = category_name.lower()
    return any(term in lower for term in [
        "century", "births", "deaths", "ancient", "medieval", "renaissance",
        "lists", "timeline", "history of", "by year", "by decade",
        "publishers_(people)_by_century", "businesswomen_by_century"
    ])

def is_person_page(page_url):
    try:
        r = requests.get(page_url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        infobox = soup.find("table", {"class": "infobox"})
        if infobox and ("person" in infobox.get("class", []) or "biography" in infobox.get("class", [])):
            return True
        return "Born" in soup.text and "Occupation" in soup.text
    except Exception:
        return False

def extract_name_from_url(url):
    raw = url.split("/wiki/")[-1]
    decoded = urllib.parse.unquote(raw)
    no_brackets = re.sub(r"\s*\(.*?\)", "", decoded)
    return no_brackets.replace("_", " ").strip().lower()

def scrape_category(category, depth=0, known_people=None, progress=None):
    if depth > MAX_DEPTH or category in progress.get("visited", []):
        return
    if should_skip_category(category):
        print(f"Skipping irrelevant category: {category}")
        return

    print(f"Scraping: {category} (depth {depth})")
    progress.setdefault("visited", []).append(category)
    save_progress(progress)

    category_url = WIKIPEDIA_BASE + category
    members = get_category_members(category_url)

    for link in members:
        if link.startswith("/wiki/Category:"):
            subcat = link.replace("/wiki/", "")
            scrape_category(subcat, depth + 1, known_people, progress)
        elif link.startswith("/wiki/"):
            page_url = WIKIPEDIA_BASE + link.replace("/wiki/", "")
            name = extract_name_from_url(link)
            if name in known_people:
                if "BUSINESS-EXECUTIVE-MANAGER" not in known_people[name]:
                    known_people[name].append("BUSINESS-EXECUTIVE-MANAGER")
            else:
                if is_person_page(page_url):
                    known_people[name] = ["BUSINESS-EXECUTIVE-MANAGER"]
                    print(f"Added: {name}")
                    if len(known_people) % SAVE_INTERVAL == 0:
                        config = {"role_keywords": {}, "known_people": dict(sorted(known_people.items())), "non_person_terms": []}
                        save_config(config)
                        print(f"Saved after {len(known_people)} entries.")
            time.sleep(4)

def scrape_national_business_categories(known_people, progress):
    root_category = "Category:Businesspeople_by_nationality"
    print(f"\nStarting crawl from {root_category}")
    root_url = WIKIPEDIA_BASE + root_category
    country_categories = get_subcategories(root_url)

    for cat_link in country_categories:
        country_cat = cat_link.replace("/wiki/", "")
        if country_cat in progress.get("visited", []):
            continue
        if should_skip_category(country_cat):
            print(f"Skipping irrelevant country category: {country_cat}")
            continue

        print(f"\nCountry category: {country_cat}")
        scrape_category(country_cat, depth=0, known_people=known_people, progress=progress)

        subcats = get_subcategories(WIKIPEDIA_BASE + country_cat)
        for subcat_link in subcats:
            subcat_name = subcat_link.replace("/wiki/", "")
            if should_skip_category(subcat_name):
                print(f"Skipping irrelevant subcategory: {subcat_name}")
                continue
            if any(keyword in subcat_name.lower() for keyword in ["executive", "founder", "entrepreneur", "ceo", "manager"]):
                print(f"Subcategory: {subcat_name}")
                scrape_category(subcat_name, depth=1, known_people=known_people, progress=progress)

        time.sleep(4)

def run_scraper():
    config = load_config()
    progress = load_progress()
    known_people = {k.lower(): v for k, v in config.get("known_people", {}).items()}

    for category in START_CATEGORIES:
        scrape_category(category, depth=0, known_people=known_people, progress=progress)

    scrape_national_business_categories(known_people, progress)

    config["known_people"] = dict(sorted(known_people.items()))
    save_config(config)
    print(f"\nScraping complete. Total entries: {len(known_people)}")

if __name__ == "__main__":
    run_scraper()
