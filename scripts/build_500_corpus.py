#!/usr/bin/env python3
"""Build a 500-novel corpus from Project Gutenberg for the RLM experiment.

Queries the Gutendex API for the most popular English fiction,
estimates publication decades from author birth/death years,
and selects ~50 novels per decade (1800s-1890s).

Outputs a shell-includable NOVELS array and a JSON manifest.

Usage:
    python scripts/build_500_corpus.py
    python scripts/build_500_corpus.py --target-per-decade 50
"""

import json
import math
import sys
import time
import urllib.request
import urllib.parse
from collections import defaultdict

API_BASE = "https://gutendex.com/books/"
TARGET_PER_DECADE = 50
DECADES = ["1800s", "1810s", "1820s", "1830s", "1840s", "1850s", "1860s", "1870s", "1880s", "1890s"]

# Well-known novels with hardcoded publication years (avoids heuristic errors)
KNOWN_PUB_YEARS = {
    84: 1818,     # Frankenstein
    2701: 1851,   # Moby-Dick
    1342: 1813,   # Pride and Prejudice
    768: 1847,    # Wuthering Heights
    11: 1865,     # Alice's Adventures in Wonderland
    1260: 1847,   # Jane Eyre
    43: 1886,     # Jekyll and Hyde
    145: 1871,    # Middlemarch
    98: 1859,     # A Tale of Two Cities
    1400: 1861,   # Great Expectations
    174: 1890,    # Picture of Dorian Gray
    345: 1897,    # Dracula
    161: 1811,    # Sense and Sensibility
    121: 1817,    # Northanger Abbey
    158: 1815,    # Emma
    141: 1814,    # Mansfield Park
    105: 1817,    # Persuasion
    1661: 1892,   # Adventures of Sherlock Holmes
    244: 1887,    # A Study in Scarlet
    120: 1883,    # Treasure Island
    135: 1862,    # Les Miserables
    2554: 1866,   # Crime and Punishment
    2600: 2600,   # War and Peace -- will be caught by range filter
    1399: 1877,   # Anna Karenina
    766: 1850,    # David Copperfield
    730: 1843,    # Oliver Twist
    46: 1843,     # A Christmas Carol
    1023: 1837,   # Pickwick Papers
    599: 1848,    # Vanity Fair
    514: 1868,    # Little Women
    16: 1726,     # Gulliver's Travels -- too early
    110: 1891,    # Tess of the d'Urbervilles
    36: 1898,     # War of the Worlds
    35: 1895,     # The Time Machine
    76: 1884,     # Huckleberry Finn
    74: 1876,     # Tom Sawyer
    219: 1899,    # Heart of Darkness
    1952: 1892,   # The Yellow Wallpaper
    160: 1899,    # The Awakening
    541: 1920,    # Age of Innocence -- too late
    209: 1898,    # Turn of the Screw
    25344: 1850,  # The Scarlet Letter
    1695: 1850,   # Scarlet Letter (alt ID)
    82: 1819,     # Ivanhoe
    969: 1848,    # Tenant of Wildfell Hall
    767: 1847,    # Agnes Grey
    940: 1826,    # Last of the Mohicans
    107: 1874,    # Far from the Madding Crowd
    143: 1886,    # Mayor of Casterbridge
    122: 1878,    # Return of the Native
    2610: 1831,   # Hunchback of Notre-Dame
    1184: 1844,   # Count of Monte Cristo
    996: 1605,    # Don Quixote -- too early
    1727: -800,   # The Odyssey -- way too early
    2600: 1869,   # War and Peace (fix)
    4300: 1818,   # Northanger Abbey (alt)
    203: 1841,    # Uncle Tom's Cabin? No -- Deerslayer
    58585: 1899,  # McTeague
}


def fetch_page(page: int) -> dict:
    """Fetch one page of results from Gutendex."""
    params = urllib.parse.urlencode({
        "languages": "en",
        "topic": "fiction",
        "sort": "popular",
        "mime_type": "text/plain",
        "page": page,
    })
    url = f"{API_BASE}?{params}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return json.loads(resp.read())
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                print(f"  Failed to fetch page {page}: {e}", file=sys.stderr)
                return {"results": []}


def estimate_pub_year(book: dict) -> int | None:
    """Estimate publication year from author dates or hardcoded mapping."""
    book_id = book["id"]

    # Check hardcoded first
    if book_id in KNOWN_PUB_YEARS:
        return KNOWN_PUB_YEARS[book_id]

    # Use author dates as heuristic
    authors = book.get("authors", [])
    if not authors:
        return None

    author = authors[0]
    birth = author.get("birth_year")
    death = author.get("death_year")

    if birth and death:
        # Estimate: midpoint of writing career ≈ birth + 35,
        # but cap at death - 5 (can't publish posthumously much)
        est = min(birth + 35, death - 5)
        return est
    elif birth:
        return birth + 35
    elif death:
        return death - 15
    return None


def get_decade(year: int) -> str | None:
    """Map a year to its decade label."""
    if year < 1800 or year >= 1900:
        return None
    decade_start = (year // 10) * 10
    return f"{decade_start}s"


def get_text_url(book: dict) -> str | None:
    """Get plain text download URL."""
    formats = book.get("formats", {})
    # Prefer UTF-8 plain text
    for key in ["text/plain; charset=utf-8", "text/plain"]:
        if key in formats:
            return formats[key]
    return None


def sanitize_filename(title: str) -> str:
    """Convert title to a safe filename."""
    import re
    # Take first ~60 chars of title
    name = title[:60].lower()
    name = re.sub(r'[^a-z0-9]+', '_', name)
    name = name.strip('_')
    return name + ".txt"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build 500-novel Gutenberg corpus")
    parser.add_argument("--target-per-decade", type=int, default=TARGET_PER_DECADE)
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--output-json", default="scripts/500_novel_corpus.json")
    parser.add_argument("--output-shell", default="scripts/500_novel_corpus.sh")
    args = parser.parse_args()

    target = args.target_per_decade
    decade_buckets: dict[str, list] = {d: [] for d in DECADES}
    seen_ids: set[int] = set()
    seen_titles: set[str] = set()  # Deduplicate by normalized title

    print(f"Fetching English fiction from Gutendex (target: {target} per decade)...")

    for page in range(1, args.max_pages + 1):
        # Check if all decades are full
        if all(len(bucket) >= target for bucket in decade_buckets.values()):
            print(f"All decades filled at page {page}.")
            break

        print(f"  Page {page}...", end=" ", flush=True)
        data = fetch_page(page)
        results = data.get("results", [])
        if not results:
            print("no results, stopping.")
            break

        added = 0
        for book in results:
            book_id = book["id"]
            if book_id in seen_ids:
                continue

            # Must have plain text
            text_url = get_text_url(book)
            if not text_url:
                continue

            # Estimate publication decade
            pub_year = estimate_pub_year(book)
            if pub_year is None:
                continue
            decade = get_decade(pub_year)
            if decade is None:
                continue

            # Skip if decade full
            if len(decade_buckets[decade]) >= target:
                continue

            # Deduplicate by title (different editions of same book)
            title_norm = book["title"].lower().split(";")[0].split(":")[0].strip()
            if title_norm in seen_titles:
                continue

            # Record it
            author = book["authors"][0] if book["authors"] else {}
            entry = {
                "id": book_id,
                "title": book["title"],
                "author": author.get("name", "Unknown"),
                "birth_year": author.get("birth_year"),
                "death_year": author.get("death_year"),
                "pub_year_est": pub_year,
                "decade": decade,
                "download_count": book.get("download_count", 0),
                "text_url": text_url,
                "filename": sanitize_filename(book["title"]),
            }
            decade_buckets[decade].append(entry)
            seen_ids.add(book_id)
            seen_titles.add(title_norm)
            added += 1

        print(f"{added} added. Totals: {', '.join(f'{d}={len(b)}' for d, b in decade_buckets.items())}")

        if not data.get("next"):
            print("No more pages.")
            break

        time.sleep(0.5)  # Rate limit

    # Summary
    total = sum(len(b) for b in decade_buckets.values())
    print(f"\n=== Corpus: {total} novels ===")
    for decade in DECADES:
        bucket = decade_buckets[decade]
        print(f"  {decade}: {len(bucket)} novels")

    # Write JSON manifest
    manifest = {
        "total": total,
        "target_per_decade": target,
        "decades": {d: decade_buckets[d] for d in DECADES},
    }
    with open(args.output_json, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nJSON manifest: {args.output_json}")

    # Write shell array
    with open(args.output_shell, "w") as f:
        f.write("# Auto-generated by build_500_corpus.py\n")
        f.write("# Do not edit manually.\n\n")
        f.write("NOVELS=(\n")
        for decade in DECADES:
            f.write(f"    # {decade}\n")
            for entry in decade_buckets[decade]:
                f.write(f'    "{entry["decade"]} {entry["id"]} {entry["filename"]}"\n')
            f.write("\n")
        f.write(")\n")
    print(f"Shell array:   {args.output_shell}")


if __name__ == "__main__":
    main()
