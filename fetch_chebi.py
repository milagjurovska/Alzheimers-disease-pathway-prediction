"""
fetch_chebi.py
==============
Fetch detailed compound information from ChEBI for all ChEBI IDs found in
the UniProt protein data (from catalytic activity, cofactors, binding sites).

Uses the EBI OLS4 REST API to retrieve compound metadata.

Input : data/raw/uniprot_ad_proteins.json
Output: data/raw/chebi_compounds.json
"""

import json
import os
import re
import sys
import time

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
UNIPROT_FILE = os.path.join(RAW_DIR, "uniprot_ad_proteins.json")
OUTPUT_FILE = os.path.join(RAW_DIR, "chebi_compounds.json")

OLS4_BASE = "https://www.ebi.ac.uk/ols4/api/ontologies/chebi/terms"
CHEBI_IRI_PREFIX = "http://purl.obolibrary.org/obo/CHEBI_"

# ---------------------------------------------------------------------------
# Step 1: Extract ChEBI IDs from UniProt data
# ---------------------------------------------------------------------------

CHEBI_PATTERN = re.compile(r"CHEBI:\d+", re.IGNORECASE)


def extract_chebi_ids_from_uniprot(proteins: list[dict]) -> set[str]:
    """
    Walk through each protein entry and find all ChEBI identifiers in
    catalytic activity, cofactor, and binding-site annotations.
    """
    chebi_ids: set[str] = set()

    for protein in proteins:
        # --- catalytic activity comments ---
        for comment in protein.get("comments", []):
            if comment.get("commentType") == "CATALYTIC ACTIVITY":
                reaction = comment.get("reaction", {})
                for xref in reaction.get("reactionCrossReferences", []):
                    if xref.get("database") == "ChEBI":
                        chebi_ids.add(xref["id"])
                text = reaction.get("name", "")
                chebi_ids.update(CHEBI_PATTERN.findall(text))

            elif comment.get("commentType") == "COFACTOR":
                for cofactor in comment.get("cofactors", []):
                    cid = cofactor.get("cofactorCrossReference", {}).get("id")
                    if cid:
                        chebi_ids.add(cid)
                    name = cofactor.get("name", "")
                    chebi_ids.update(CHEBI_PATTERN.findall(name))

        # --- binding site features ---
        for feature in protein.get("features", []):
            if feature.get("type") == "Binding site":
                ligand = feature.get("ligand", {})
                if isinstance(ligand, dict):
                    lid = ligand.get("ligandCrossReference", {}).get("id")
                    if lid and "CHEBI" in lid.upper():
                        chebi_ids.add(lid)

    # Normalise: ensure uppercase "CHEBI:NNNNN"
    normalised = set()
    for cid in chebi_ids:
        cid_clean = cid.strip()
        if cid_clean:
            normalised.add(cid_clean.upper() if cid_clean.upper().startswith("CHEBI:") else cid_clean)
    return normalised


# ---------------------------------------------------------------------------
# Step 2: Fetch compound details from ChEBI via OLS4 API
# ---------------------------------------------------------------------------

def _get_first(lst, default=None):
    """Safely get first item from a list, or default."""
    if isinstance(lst, list) and lst:
        return lst[0]
    return default


def fetch_compound_details(chebi_ids: set[str]) -> list[dict]:
    """
    For each ChEBI ID, query the EBI OLS4 API to get compound metadata.
    """
    compounds: list[dict] = []
    sorted_ids = sorted(chebi_ids)
    total = len(sorted_ids)

    for i, cid in enumerate(sorted_ids, 1):
        numeric_id = cid.replace("CHEBI:", "")
        iri = f"{CHEBI_IRI_PREFIX}{numeric_id}"
        print(f"  [{i}/{total}] Fetching {cid} …", end=" ", flush=True)

        try:
            resp = requests.get(
                OLS4_BASE,
                params={"iri": iri},
                headers={"Accept": "application/json"},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            terms = data.get("_embedded", {}).get("terms", [])
            if not terms:
                print("SKIP (not found in OLS4)")
                continue

            term = terms[0]
            annotation = term.get("annotation", {})

            name = term.get("label", "")
            description = _get_first(term.get("description", []), "")
            formula = _get_first(annotation.get("formula", []))
            mass = _get_first(annotation.get("mass", []))
            charge = _get_first(annotation.get("charge", []))
            monoisotopic_mass = _get_first(annotation.get("monoisotopicmass", []))
            smiles = _get_first(annotation.get("smiles", []))
            inchikey = _get_first(annotation.get("inchikey", []))

            # Database cross-references
            db_xrefs = []
            for xref in term.get("obo_xref", []):
                if isinstance(xref, dict):
                    db_xrefs.append({
                        "database": xref.get("database", ""),
                        "id": xref.get("id", ""),
                    })

            # Parse ontology parents from ancestors link
            parents = []
            parents_link = (
                term.get("_links", {})
                .get("parents", {})
                .get("href")
            )
            if parents_link:
                try:
                    pr = requests.get(parents_link, headers={"Accept": "application/json"}, timeout=15)
                    if pr.status_code == 200:
                        parent_terms = pr.json().get("_embedded", {}).get("terms", [])
                        for pt in parent_terms:
                            parents.append({
                                "chebi_id": pt.get("obo_id", ""),
                                "name": pt.get("label", ""),
                            })
                except Exception:
                    pass

            compound = {
                "chebi_id": cid,
                "name": name,
                "description": description,
                "formula": formula,
                "mass": float(mass) if mass else None,
                "charge": float(charge) if charge is not None else None,
                "monoisotopic_mass": float(monoisotopic_mass) if monoisotopic_mass else None,
                "smiles": smiles,
                "inchikey": inchikey,
                "ontology_parents": parents,
                "database_xrefs": db_xrefs,
            }
            compounds.append(compound)
            print(f"OK ({name})")

        except Exception as e:
            print(f"ERROR ({e})")

        time.sleep(0.3)  # polite rate-limiting

    return compounds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("ChEBI Fetcher — Small Molecules from AD Protein Data")
    print("=" * 60)

    # Load UniProt data
    if not os.path.exists(UNIPROT_FILE):
        print(f"ERROR: UniProt data not found at {UNIPROT_FILE}")
        print("       Run fetch_uniprot.py first.")
        sys.exit(1)

    with open(UNIPROT_FILE, "r", encoding="utf-8") as f:
        proteins = json.load(f)
    print(f"Loaded {len(proteins)} proteins from UniProt data.")
    print()

    # Extract ChEBI IDs
    chebi_ids = extract_chebi_ids_from_uniprot(proteins)
    print(f"Found {len(chebi_ids)} unique ChEBI IDs in the protein data.")
    if not chebi_ids:
        print("No ChEBI IDs found — check the UniProt data for cofactor/catalytic annotations.")
        sys.exit(0)

    print()
    print("Fetching compound details from ChEBI via OLS4 API …")
    print()

    # Fetch from ChEBI
    compounds = fetch_compound_details(chebi_ids)

    # Save
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(compounds, f, indent=2, ensure_ascii=False)

    print()
    print(f"✓ Saved {len(compounds)} compound entries → {OUTPUT_FILE}")
    print(f"  (from {len(chebi_ids)} unique ChEBI IDs)")


if __name__ == "__main__":
    main()
