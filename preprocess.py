"""
preprocess.py
=============
Parse raw UniProt + ChEBI JSON data and produce clean, structured CSVs
ready for downstream ML modelling.

Outputs (in data/processed/):
  - proteins.csv            : one row per AD-associated protein
  - chemicals.csv           : one row per ChEBI compound
  - protein_chemical_links.csv : protein ↔ compound relationships
  - protein_pathway_map.csv : protein ↔ pathway relationships

Usage:
    python preprocess.py              # run preprocessing
    python preprocess.py --verify     # verify output integrity
"""

import argparse
import json
import os
import re
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

UNIPROT_FILE = os.path.join(RAW_DIR, "uniprot_ad_proteins.json")
CHEBI_FILE = os.path.join(RAW_DIR, "chebi_compounds.json")

CHEBI_PATTERN = re.compile(r"CHEBI:\d+", re.IGNORECASE)

# ---------------------------------------------------------------------------
# 1. Parse UniProt JSON → proteins DataFrame
# ---------------------------------------------------------------------------


def _safe_text(value) -> str:
    """Extract text from a UniProt annotation value (may be dict or list)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "; ".join(_safe_text(v) for v in value if v)
    if isinstance(value, dict):
        # free-text comment blocks
        texts = value.get("texts", [])
        if texts:
            return "; ".join(t.get("value", "") for t in texts if isinstance(t, dict))
        return value.get("value", str(value))
    return str(value)


def _extract_go_terms(entry: dict, aspect: str) -> str:
    """
    Extract GO terms for a given aspect (go_p -> P:, go_f -> F:, go_c -> C:).
    UniProt JSON stores GO terms in the uniProtKBCrossReferences array.
    """
    prefix = {"go_p": "P:", "go_f": "F:", "go_c": "C:"}.get(aspect, "")
    terms = []
    
    for xref in entry.get("uniProtKBCrossReferences", []):
        if xref.get("database") == "GO":
            props = xref.get("properties", [])
            go_id = xref.get("id", "")
            term_name = ""
            for prop in props:
                if prop.get("key") == "GoTerm":
                    term_name = prop.get("value", "")
                    break
            
            # GO terms in UniProt are prefixed with F:, P:, or C:
            if term_name.startswith(prefix):
                # Clean prefix from name for better visualization
                display_name = term_name[len(prefix):].strip()
                terms.append(f"{go_id}:{display_name}" if display_name else go_id)
                
    return "; ".join(terms)


def _extract_comments(entry: dict, comment_type: str) -> str:
    """Extract text from comments of a given type."""
    texts = []
    for comment in entry.get("comments", []):
        if comment.get("commentType", "").upper() == comment_type.upper():
            # Free-text comments
            for t in comment.get("texts", []):
                if isinstance(t, dict):
                    texts.append(t.get("value", ""))
                elif isinstance(t, str):
                    texts.append(t)
            # Molecule-level (e.g. subcellular location)
            for loc in comment.get("subcellularLocations", []):
                location = loc.get("location", {})
                if isinstance(location, dict):
                    texts.append(location.get("value", ""))
            # Disease
            disease = comment.get("disease", {})
            if disease:
                desc = disease.get("description", "")
                name = disease.get("diseaseId", "")
                if name:
                    texts.append(f"{name}: {desc}" if desc else name)
            # Catalytic activity
            reaction = comment.get("reaction", {})
            if reaction:
                texts.append(reaction.get("name", ""))
            # Cofactor
            for cof in comment.get("cofactors", []):
                texts.append(cof.get("name", ""))
            # Pathway
            if comment_type.upper() == "PATHWAY":
                # pathway comments can have free text
                pass  # already captured by texts
    return "; ".join(t for t in texts if t)


def _extract_xrefs(entry: dict, db_name: str) -> str:
    """Extract cross-reference IDs for a given database."""
    ids = []
    for xref in entry.get("uniProtKBCrossReferences", []):
        if xref.get("database", "").upper() == db_name.upper():
            ids.append(xref.get("id", ""))
    return "; ".join(ids) if ids else ""


def _extract_keywords(entry: dict) -> str:
    """Extract keyword values."""
    kws = entry.get("keywords", [])
    if isinstance(kws, list):
        return "; ".join(
            kw.get("name", kw) if isinstance(kw, dict) else str(kw)
            for kw in kws
        )
    return str(kws)


def _count_features(entry: dict, feature_type: str) -> int:
    """Count features of a given type (e.g. 'Binding site')."""
    return sum(
        1 for f in entry.get("features", [])
        if f.get("type", "").upper() == feature_type.upper()
    )


def _count_interactions(entry: dict) -> int:
    """Count binary protein-protein interactions."""
    for comment in entry.get("comments", []):
        if comment.get("commentType") == "INTERACTION":
            interactions = comment.get("interactions", [])
            return len(interactions)
    return 0


def _extract_chebi_from_entry(entry: dict) -> str:
    """Collect all ChEBI IDs referenced in a protein entry."""
    chebi_ids: set[str] = set()
    for comment in entry.get("comments", []):
        ctype = comment.get("commentType", "")
        if ctype == "CATALYTIC ACTIVITY":
            reaction = comment.get("reaction", {})
            for xref in reaction.get("reactionCrossReferences", []):
                if xref.get("database") == "ChEBI":
                    chebi_ids.add(xref["id"].upper())
            chebi_ids.update(m.upper() for m in CHEBI_PATTERN.findall(reaction.get("name", "")))
        elif ctype == "COFACTOR":
            for cof in comment.get("cofactors", []):
                cid = cof.get("cofactorCrossReference", {}).get("id")
                if cid:
                    chebi_ids.add(cid.upper())
    for feat in entry.get("features", []):
        if feat.get("type") == "Binding site":
            ligand = feat.get("ligand", {})
            if isinstance(ligand, dict):
                lid = ligand.get("ligandCrossReference", {}).get("id")
                if lid and "CHEBI" in lid.upper():
                    chebi_ids.add(lid.upper())
    return "; ".join(sorted(chebi_ids))


def parse_uniprot(proteins: list[dict]) -> pd.DataFrame:
    """Convert raw UniProt JSON entries into a flat DataFrame."""
    rows = []
    for entry in proteins:
        row = {
            "uniprot_id": entry.get("primaryAccession", ""),
            "gene_name": (
                entry.get("genes", [{}])[0].get("geneName", {}).get("value", "")
                if entry.get("genes") else ""
            ),
            "protein_name": (
                entry.get("proteinDescription", {})
                .get("recommendedName", {})
                .get("fullName", {})
                .get("value", "")
            ),
            "length": entry.get("sequence", {}).get("length", 0),

            # Functional annotations
            "function_text": _extract_comments(entry, "FUNCTION"),
            "disease_text": _extract_comments(entry, "DISEASE"),
            "pathway_text": _extract_comments(entry, "PATHWAY"),
            "catalytic_activity": _extract_comments(entry, "CATALYTIC ACTIVITY"),
            "cofactors": _extract_comments(entry, "COFACTOR"),
            "activity_regulation": _extract_comments(entry, "ACTIVITY REGULATION"),
            "subcellular_location": _extract_comments(entry, "SUBCELLULAR LOCATION"),

            # Gene Ontology
            "go_bp": _extract_go_terms(entry, "go_p"),
            "go_mf": _extract_go_terms(entry, "go_f"),
            "go_cc": _extract_go_terms(entry, "go_c"),

            # Keywords
            "keywords": _extract_keywords(entry),

            # Cross-references
            "reactome_ids": _extract_xrefs(entry, "Reactome"),
            "kegg_id": _extract_xrefs(entry, "KEGG"),
            "string_ids": _extract_xrefs(entry, "STRING"),
            "intact_ids": _extract_xrefs(entry, "IntAct"),
            "drugbank_ids": _extract_xrefs(entry, "DrugBank"),
            "chembl_ids": _extract_xrefs(entry, "ChEMBL"),

            # Counts
            "binding_sites_count": _count_features(entry, "Binding site"),
            "interaction_count": _count_interactions(entry),

            # Linked ChEBI IDs
            "chebi_ids": _extract_chebi_from_entry(entry),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Fill missing protein names from alternative/submitted names
    for i, entry in enumerate(proteins):
        if not df.at[i, "protein_name"]:
            alt = (
                entry.get("proteinDescription", {})
                .get("submittedName", [{}])
            )
            if isinstance(alt, list) and alt:
                df.at[i, "protein_name"] = alt[0].get("fullName", {}).get("value", "")

    return df


# ---------------------------------------------------------------------------
# 2. Parse ChEBI JSON → chemicals DataFrame
# ---------------------------------------------------------------------------

def parse_chebi(compounds: list[dict]) -> pd.DataFrame:
    """Convert raw ChEBI data (from OLS4 API) into a flat DataFrame."""
    rows = []
    for c in compounds:
        parents = c.get("ontology_parents", [])
        parent_names = "; ".join(
            p["name"] for p in parents
            if isinstance(p, dict) and p.get("name")
        )
        parent_ids = "; ".join(
            p.get("chebi_id", "") for p in parents
            if isinstance(p, dict)
        )

        # Extract KEGG Compound link if available
        kegg_compound = ""
        for link in c.get("database_xrefs", []):
            db = link.get("database", "")
            if "kegg" in db.lower():
                kegg_compound = link.get("id", "")
                break

        rows.append({
            "chebi_id": c.get("chebi_id", ""),
            "name": c.get("name", ""),
            "description": c.get("description", ""),
            "formula": c.get("formula"),
            "mass": c.get("mass"),
            "charge": c.get("charge"),
            "monoisotopic_mass": c.get("monoisotopic_mass"),
            "smiles": c.get("smiles"),
            "inchikey": c.get("inchikey"),
            "ontology_parents": parent_names,
            "ontology_parent_ids": parent_ids,
            "kegg_compound_id": kegg_compound,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Build protein–chemical link table
# ---------------------------------------------------------------------------

def build_protein_chemical_links(proteins: list[dict]) -> pd.DataFrame:
    """
    Create a long-form table of (uniprot_id, chebi_id, link_type) from
    catalytic activity, cofactor, and binding-site annotations.
    """
    links = []

    for entry in proteins:
        uid = entry.get("primaryAccession", "")

        for comment in entry.get("comments", []):
            ctype = comment.get("commentType", "")

            if ctype == "CATALYTIC ACTIVITY":
                reaction = comment.get("reaction", {})
                for xref in reaction.get("reactionCrossReferences", []):
                    if xref.get("database") == "ChEBI":
                        links.append({
                            "uniprot_id": uid,
                            "chebi_id": xref["id"].upper(),
                            "link_type": "substrate/product",
                        })

            elif ctype == "COFACTOR":
                for cof in comment.get("cofactors", []):
                    cid = cof.get("cofactorCrossReference", {}).get("id")
                    if cid:
                        links.append({
                            "uniprot_id": uid,
                            "chebi_id": cid.upper(),
                            "link_type": "cofactor",
                        })

        for feat in entry.get("features", []):
            if feat.get("type") == "Binding site":
                ligand = feat.get("ligand", {})
                if isinstance(ligand, dict):
                    lid = ligand.get("ligandCrossReference", {}).get("id")
                    if lid and "CHEBI" in lid.upper():
                        links.append({
                            "uniprot_id": uid,
                            "chebi_id": lid.upper(),
                            "link_type": "binding",
                        })

    df = pd.DataFrame(links)
    if not df.empty:
        df = df.drop_duplicates()
    return df


# ---------------------------------------------------------------------------
# 4. Build protein–pathway mapping
# ---------------------------------------------------------------------------

def build_protein_pathway_map(proteins: list[dict]) -> pd.DataFrame:
    """
    Create a long-form table of (uniprot_id, pathway_id, pathway_name, pathway_source).
    Sources: Reactome xrefs, KEGG xrefs, and UniProt pathway comments.
    """
    mappings = []

    for entry in proteins:
        uid = entry.get("primaryAccession", "")

        # Reactome cross-references
        for xref in entry.get("uniProtKBCrossReferences", []):
            if xref.get("database") == "Reactome":
                pathway_id = xref.get("id", "")
                pathway_name = ""
                for prop in xref.get("properties", []):
                    if prop.get("key") == "PathwayName":
                        pathway_name = prop.get("value", "")
                mappings.append({
                    "uniprot_id": uid,
                    "pathway_id": pathway_id,
                    "pathway_name": pathway_name,
                    "pathway_source": "Reactome",
                })

            elif xref.get("database") == "KEGG":
                mappings.append({
                    "uniprot_id": uid,
                    "pathway_id": xref.get("id", ""),
                    "pathway_name": "",
                    "pathway_source": "KEGG",
                })

        # UniProt pathway comments
        for comment in entry.get("comments", []):
            if comment.get("commentType") == "PATHWAY":
                for text in comment.get("texts", []):
                    val = text.get("value", "") if isinstance(text, dict) else str(text)
                    if val:
                        mappings.append({
                            "uniprot_id": uid,
                            "pathway_id": "",
                            "pathway_name": val,
                            "pathway_source": "UniProt",
                        })

    df = pd.DataFrame(mappings)
    if not df.empty:
        df = df.drop_duplicates()
    return df


# ---------------------------------------------------------------------------
# 5. Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline():
    """Execute the full preprocessing pipeline."""
    print("=" * 60)
    print("Preprocessing Pipeline")
    print("=" * 60)

    # ---- Load raw data ----
    if not os.path.exists(UNIPROT_FILE):
        print(f"ERROR: {UNIPROT_FILE} not found. Run fetch_uniprot.py first.")
        sys.exit(1)

    with open(UNIPROT_FILE, "r", encoding="utf-8") as f:
        proteins_raw = json.load(f)
    print(f"Loaded {len(proteins_raw)} proteins from UniProt.")

    chebi_raw = []
    if os.path.exists(CHEBI_FILE):
        with open(CHEBI_FILE, "r", encoding="utf-8") as f:
            chebi_raw = json.load(f)
        print(f"Loaded {len(chebi_raw)} compounds from ChEBI.")
    else:
        print(f"WARNING: {CHEBI_FILE} not found. Chemical data will be empty.")
        print("         Run fetch_chebi.py to populate it.")

    print()

    # ---- Parse ----
    print("Parsing proteins …")
    proteins_df = parse_uniprot(proteins_raw)
    print(f"  → {len(proteins_df)} proteins, {len(proteins_df.columns)} columns")

    print("Parsing chemicals …")
    chemicals_df = parse_chebi(chebi_raw)
    print(f"  → {len(chemicals_df)} compounds, {len(chemicals_df.columns)} columns")

    print("Building protein–chemical links …")
    links_df = build_protein_chemical_links(proteins_raw)
    print(f"  → {len(links_df)} links")

    print("Building protein–pathway map …")
    pathway_df = build_protein_pathway_map(proteins_raw)
    print(f"  → {len(pathway_df)} protein–pathway mappings")

    print()

    # ---- Save ----
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    proteins_path = os.path.join(PROCESSED_DIR, "proteins.csv")
    chemicals_path = os.path.join(PROCESSED_DIR, "chemicals.csv")
    links_path = os.path.join(PROCESSED_DIR, "protein_chemical_links.csv")
    pathway_path = os.path.join(PROCESSED_DIR, "protein_pathway_map.csv")

    proteins_df.fillna("").to_csv(proteins_path, index=False)
    chemicals_df.fillna("").to_csv(chemicals_path, index=False)
    links_df.fillna("").to_csv(links_path, index=False)
    pathway_df.fillna("").to_csv(pathway_path, index=False)

    print("Saved processed datasets:")
    print(f"  ✓ {proteins_path}")
    print(f"  ✓ {chemicals_path}")
    print(f"  ✓ {links_path}")
    print(f"  ✓ {pathway_path}")

    return proteins_df, chemicals_df, links_df, pathway_df


# ---------------------------------------------------------------------------
# 6. Verification
# ---------------------------------------------------------------------------

def verify_outputs():
    """Check that processed files exist and have reasonable content."""
    print("=" * 60)
    print("Output Verification")
    print("=" * 60)
    print()

    files = {
        "proteins.csv": os.path.join(PROCESSED_DIR, "proteins.csv"),
        "chemicals.csv": os.path.join(PROCESSED_DIR, "chemicals.csv"),
        "protein_chemical_links.csv": os.path.join(PROCESSED_DIR, "protein_chemical_links.csv"),
        "protein_pathway_map.csv": os.path.join(PROCESSED_DIR, "protein_pathway_map.csv"),
    }

    all_ok = True
    for name, path in files.items():
        print(f"--- {name} ---")
        if not os.path.exists(path):
            print(f"  ✗ FILE NOT FOUND: {path}")
            all_ok = False
            continue

        df = pd.read_csv(path)
        print(f"  Rows    : {len(df)}")
        print(f"  Columns : {len(df.columns)}")
        print(f"  Columns : {list(df.columns)}")

        if df.empty:
            print(f"  ✗ WARNING: DataFrame is EMPTY")
            all_ok = False
        else:
            # Missing value report
            missing = df.isnull().sum()
            missing_pct = (missing / len(df) * 100).round(1)
            non_empty = df.astype(str).apply(lambda col: (col != "").sum())
            print(f"  Non-empty values per column:")
            for col in df.columns:
                print(f"    {col:35s}: {non_empty[col]:5d} / {len(df)}  ({missing_pct[col]:.1f}% missing)")
            print(f"  ✓ OK")

        print()

    # Spot-check known AD proteins
    proteins_path = files["proteins.csv"]
    if os.path.exists(proteins_path):
        df = pd.read_csv(proteins_path)
        known_genes = ["APP", "PSEN1", "PSEN2", "MAPT", "APOE", "BACE1"]
        found = df[df["gene_name"].isin(known_genes)]["gene_name"].tolist()
        print(f"Known AD gene check: found {len(found)} of {len(known_genes)}")
        print(f"  Found: {found}")
        missing_genes = set(known_genes) - set(found)
        if missing_genes:
            print(f"  Missing: {missing_genes}")
        print()

    if all_ok:
        print("✓ All verification checks passed.")
    else:
        print("✗ Some checks failed — review above.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess AD protein/chemical/pathway data")
    parser.add_argument("--verify", action="store_true", help="Verify processed outputs only")
    args = parser.parse_args()

    if args.verify:
        verify_outputs()
    else:
        run_pipeline()


if __name__ == "__main__":
    main()
