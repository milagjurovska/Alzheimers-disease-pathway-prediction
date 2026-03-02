"""
fetch_uniprot.py
================
Fetch Alzheimer's disease-associated human proteins from UniProt REST API.

Queries reviewed (Swiss-Prot) human proteins with the "Alzheimer disease" keyword
and retrieves comprehensive annotations including GO terms, pathway cross-references,
binding sites, interactions, and cofactor/catalytic activity (with ChEBI links).

Output: data/raw/uniprot_ad_proteins.json
"""

import json
import os
import sys
import time
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://rest.uniprot.org/uniprotkb/search"

QUERY = '(taxonomy_id:9606) AND (reviewed:true) AND (alzheimer)'

FIELDS = ",".join([
    "accession",
    "gene_names",
    "protein_name",
    "length",
    # Function
    "cc_function",
    "cc_disease",
    "cc_pathway",
    "cc_catalytic_activity",
    "cc_cofactor",
    "ft_binding",
    "cc_activity_regulation",
    # Interaction
    "cc_interaction",
    "cc_subunit",
    # Location
    "cc_subcellular_location",
    # Gene Ontology
    "go_p",
    "go_f",
    "go_c",
    "go_id",
    # Keywords
    "keyword",
    # Pathway cross-references
    "xref_reactome",
    "xref_kegg",
    # PPI cross-references
    "xref_string",
    "xref_intact",
    # Chemistry cross-references
    "xref_drugbank",
    "xref_chembl",
])

PAGE_SIZE = 500
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "uniprot_ad_proteins.json")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_all_proteins() -> list[dict]:
    """Page through UniProt search results and collect all entries."""
    all_results: list[dict] = []

    params = {
        "query": QUERY,
        "fields": FIELDS,
        "format": "json",
        "size": PAGE_SIZE,
    }

    url = BASE_URL
    page = 1

    while url:
        print(f"  Fetching page {page} (URL: {url[:60]}...) …", end=" ", flush=True)

        # For the first page, we use params. For subsequent pages, the URL from the Link header
        # already contains the encoded query/cursor/params.
        response = requests.get(url, params=params if page == 1 else None)
        response.raise_for_status()

        data = response.json()
        results = data.get("results", [])
        all_results.extend(results)
        print(f"got {len(results)} entries (total so far: {len(all_results)})")

        # Use requests' built-in link parsing
        next_link = response.links.get("next", {}).get("url")
        url = next_link
        page += 1

        if url:
            time.sleep(0.5)  # polite rate-limiting

    return all_results

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("UniProt Fetcher — Alzheimer's Disease Human Proteins")
    print("=" * 60)
    print(f"Query : {QUERY}")
    print(f"Fields: {len(FIELDS.split(','))} fields")
    print()

    proteins = fetch_all_proteins()

    if not proteins:
        print("ERROR: No proteins returned. Check your query or network connection.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(proteins, f, indent=2, ensure_ascii=False)

    print()
    print(f"✓ Saved {len(proteins)} protein entries → {OUTPUT_FILE}")
    print()

    # Quick summary
    accessions = [p.get("primaryAccession", "?") for p in proteins]
    print(f"Sample accessions: {', '.join(accessions[:10])}")
    if len(accessions) > 10:
        print(f"  … and {len(accessions) - 10} more")


if __name__ == "__main__":
    main()
