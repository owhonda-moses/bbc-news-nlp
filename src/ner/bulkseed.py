import requests
import json
import time
from pathlib import Path
from requests.exceptions import RequestException

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "ner_config.json"
PROGRESS_FILE = BASE_DIR / "progress_wiki.json"
OFFSET = 100000

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

# per-category chunk sizes
BASE_LIMITS = {
    "POLITICIAN": 1000,
    "ATHLETE": 1000,
    "BUSINESS-EXECUTIVE-MANAGER": 1000,
    "MUSICIAN": 1000,
    "ACTOR-DIRECTOR": 1000,
    "AUTHOR-WRITER": 1000
}

CATEGORIES = {
    "BUSINESS-EXECUTIVE-MANAGER": ["Q15978631", "Q1123240", "Q194622", "Q27686"],
    "ATHLETE": ["Q2066131", "Q41583"],
    "MUSICIAN": ["Q639669", "Q177220", "Q2259451", "Q1281618", "Q970948", "Q36834", "Q753163", "Q488205"],
    "AUTHOR-WRITER": ["Q6625963", "Q49757", "Q28389" "Q214917", "Q1086863", "Q1930187", "Q36180", "Q482980"],
    "POLITICIAN": ["Q82955"],
    "ACTOR-DIRECTOR": ["Q33999", "Q2526255"]
}



# QUERY_TEMPLATE = """
# SELECT DISTINCT ?person ?personLabel WHERE {{
#   ?person wdt:P31 wd:Q5;
#           wdt:P106 {occupation} .
#   OPTIONAL {{ ?person wdt:P2031 ?careerStart. }}
#   OPTIONAL {{ ?person wdt:P2032 ?careerEnd. }}
#   FILTER(!BOUND(?careerStart) || ?careerStart <= "2005-12-31"^^xsd:dateTime)
#   FILTER(!BOUND(?careerEnd) || ?careerEnd >= "2000-01-01"^^xsd:dateTime)
#   SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
# }}
# LIMIT {limit}
# OFFSET {offset}
# """


QUERY_TEMPLATE = """
SELECT DISTINCT ?person ?personLabel WHERE {{
  ?person wdt:P31 wd:Q5;
          wdt:P106/wdt:P279* {occupation} .
  OPTIONAL {{ ?person wdt:P2031 ?careerStart. }}
  OPTIONAL {{ ?person wdt:P2032 ?careerEnd. }}
  FILTER(!BOUND(?careerStart) || ?careerStart <= "2005-12-31"^^xsd:dateTime)
  FILTER(!BOUND(?careerEnd) || ?careerEnd >= "2000-01-01"^^xsd:dateTime)
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT {limit}
OFFSET {offset}
"""

def run_query(query, retries=20, delay=5, context=""):
    headers = {"Accept": "application/sparql-results+json"}
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(
                SPARQL_ENDPOINT,
                params={"query": query},
                headers=headers,
                timeout=120
            )
            r.raise_for_status()
            try:
                return r.json()
            except json.JSONDecodeError as je:
                print(f"  JSON decode error on attempt {attempt}/{retries} for {context}: {je}")
                if attempt < retries:
                    print(f"    Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    print(f" Skipping chunk due to persistent JSON errors at {context}.")
                    return None  # skip chunk
        except RequestException as e:
            if hasattr(e, "response") and e.response is not None:
                status = e.response.status_code
                reason = e.response.reason
                msg = f"{status} {reason}"
            else:
                msg = str(e).split("\n")[0]  # non-HTTP errors

            print(f"  Error on attempt {attempt}/{retries} for {context}: {msg}")
            backoff = 30 if "429" in msg else delay
            if attempt < retries:
                print(f"    Retrying in {backoff}s...")
                time.sleep(backoff)
            else:
                raise

def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"role_keywords": {}, "known_people": {}, "non_person_terms": []}

def save_config(config):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_progress(progress):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

def harvest(start_fresh=False):
    if start_fresh:
        print("Starting with cleared progress and known_people.")
        progress = {}
        config = {"role_keywords": {}, "known_people": {}, "non_person_terms": []}
    else:
        progress = load_progress()
        config = load_config()

    known_people = {k.lower(): v for k, v in config.get("known_people", {}).items()}
    # normalize to list format
    for k, v in known_people.items():
        if not isinstance(v, list):
            known_people[k] = [v]

    for role, occ_ids in CATEGORIES.items():
        limit = BASE_LIMITS[role]
        print(f"\n== Starting category: {role} (chunk size {limit}) ==")
        for occ_id in occ_ids:
            key = f"{role}:{occ_id}"

            if progress.get(key) == "done":
                print(f"Skipping {occ_id} (already complete).")
                continue

            offset = progress.get(key, 0)
            print(f"Fetching occupation {occ_id} starting at offset {offset}...")

            while True:
                if offset >= OFFSET:
                    print(f"Offset limit reached for {role}:{occ_id}. Stopping.")
                    progress[key] = "done"
                    save_progress(progress)
                    break

                print(f"  Querying offset {offset}")
                query = QUERY_TEMPLATE.format(occupation=f"wd:{occ_id}", limit=limit, offset=offset)
                context = f"{role} ({occ_id}) offset {offset}"
                data = run_query(query, context=context)
                results = data["results"]["bindings"]

                if not results:
                    print("  No more results for this occupation.")
                    progress[key] = "done"
                    save_progress(progress)
                    break

                added_this_chunk = 0
                for row in results:
                    name = row["personLabel"]["value"].strip().lower()
                    if not name:
                        continue

                    if name in known_people:
                        # append
                        if isinstance(known_people[name], list):
                            if role not in known_people[name]:
                                known_people[name].append(role)
                                added_this_chunk += 1
                        else:
                            # convert to list and append
                            if known_people[name] != role:
                                known_people[name] = [known_people[name], role]
                                added_this_chunk += 1
                    else:
                        known_people[name] = [role]
                        added_this_chunk += 1

                offset += limit
                print(f"  Added {added_this_chunk} new names in this chunk. Total so far: {len(known_people)}")

                # save known_people only if changed
                if added_this_chunk > 0:
                    config["known_people"] = dict(sorted(known_people.items()))
                    save_config(config)

                # save updated offset
                if progress.get(key) != offset:
                    progress[key] = offset
                    save_progress(progress)

                # pause to avoid 429s
                time.sleep(5)

            print(f"Saved progress after {occ_id}: {len(known_people)} total entries.")

    print("\nHarvesting complete.")
    print(f"Final total entries: {len(known_people)}")

if __name__ == "__main__":
    harvest(start_fresh=False) # set True to start from scratch
