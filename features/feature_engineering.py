"""
feature_engineering.py
======================
Build an ML-ready feature matrix from the processed AD protein data.

This module converts the biologically annotated protein tables into a
numerical feature matrix suitable for supervised pathway-membership
classification.  It is the single source of truth for feature definitions
used by all three downstream models (Random Forest, XGBoost, Neural Network).

Pipeline overview
-----------------
1. Load `proteins.csv`, `protein_pathway_map.csv`, and
   `protein_chemical_links.csv` from ``data/processed/``.
2. Construct **label** column: for each protein, identify its most frequent
   Reactome pathway.  The top-K pathways (``TOP_K_PATHWAYS``) become class
   labels; all remaining proteins are assigned the sentinel label ``"Other"``.
   Proteins with no Reactome annotation are assigned ``"No Pathway"``.
3. Build feature groups:
   - Numeric biophysical features (sequence length, binding-site count,
     protein–protein interaction count).
   - Chemical-link aggregates (substrate/product count, cofactor count,
     binding-molecule count).
   - Boolean functional flags (has disease annotation, catalytic activity,
     pathway comment, DrugBank entry, ChEMBL entry, KEGG entry).
   - UniProt keyword bag-of-words (top-``TOP_KEYWORDS`` keywords encoded as
     binary indicator columns).
   - GO-term bag-of-words for Biological Process, Molecular Function, and
     Cellular Component (top-``TOP_GO`` terms each).
   - Subcellular-localisation one-hot encoding.
4. Impute missing numerics with the column median.
5. Return a clean ``(X, y, feature_names, label_encoder)`` bundle.

Usage
-----
    from features.feature_engineering import build_features

    X, y, feature_names, le = build_features()
    # X : np.ndarray, shape (n_proteins, n_features)
    # y : np.ndarray of int class indices
    # feature_names : list[str]
    # le : sklearn LabelEncoder that maps integers → pathway names
"""

from __future__ import annotations

import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Number of the most-frequent Reactome pathways to use as explicit classes.
# Proteins in all other pathways are merged into the "Other" label.
TOP_K_PATHWAYS: int = 10

# How many top UniProt keywords to encode as binary features.
TOP_KEYWORDS: int = 60

# How many top GO terms (per namespace) to encode as binary features.
TOP_GO: int = 40

# Minimum string length to consider a text field non-empty.
_MIN_TEXT_LEN: int = 2


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _has_text(series: pd.Series) -> pd.Series:
    """Return a 0/1 Series: 1 if the cell contains meaningful text."""
    return series.fillna("").str.strip().str.len().gt(_MIN_TEXT_LEN).astype(int)


def _parse_semicolon_list(series: pd.Series) -> pd.Series:
    """Split a '; '-delimited string column into lists, normalised to lower."""
    return series.fillna("").apply(
        lambda s: [tok.strip().lower() for tok in s.split(";") if tok.strip()]
    )


def _top_tokens(token_lists: pd.Series, top_n: int) -> list[str]:
    """
    Find the ``top_n`` most frequent tokens across all rows.

    Parameters
    ----------
    token_lists:
        Series of lists of string tokens.
    top_n:
        How many distinct tokens to return.

    Returns
    -------
    List of token strings, ordered by frequency descending.
    """
    counter: Counter = Counter()
    for lst in token_lists:
        counter.update(lst)
    return [tok for tok, _ in counter.most_common(top_n)]


def _binarise_token_lists(
    token_lists: pd.Series, vocab: list[str]
) -> pd.DataFrame:
    """
    Convert a Series of token lists into a sparse binary DataFrame.

    Each column in the output corresponds to one vocabulary token; a cell is
    1 if that token appears in the protein's list, 0 otherwise.

    Parameters
    ----------
    token_lists:
        Series of lists of strings (one list per protein).
    vocab:
        Ordered list of unique tokens to keep.

    Returns
    -------
    pd.DataFrame of shape ``(len(token_lists), len(vocab))`` with dtype int8.
    """
    vocab_index = {tok: i for i, tok in enumerate(vocab)}
    n = len(token_lists)
    m = len(vocab)
    matrix = np.zeros((n, m), dtype=np.int8)
    for row_idx, tokens in enumerate(token_lists):
        for tok in tokens:
            col_idx = vocab_index.get(tok)
            if col_idx is not None:
                matrix[row_idx, col_idx] = 1
    return pd.DataFrame(matrix, columns=vocab)


# ---------------------------------------------------------------------------
# Label construction
# ---------------------------------------------------------------------------

def _build_labels(
    proteins: pd.DataFrame,
    pathway_map: pd.DataFrame,
    top_k: int = TOP_K_PATHWAYS,
) -> pd.Series:
    """
    Assign a pathway class label to each protein.

    Strategy
    --------
    For each protein, we look at all its Reactome pathway assignments and
    pick the **most common pathway across the whole dataset** that this
    protein belongs to.  This avoids arbitrary tie-breaking between equally
    plausible memberships.

    The ``top_k`` most-populated pathways become explicit labels.  Any protein
    whose "best" pathway falls outside this set receives the label ``"Other"``.
    Proteins with no Reactome mapping at all receive ``"No Pathway"``.

    Parameters
    ----------
    proteins:
        The proteins DataFrame (one row per protein; must contain
        ``uniprot_id``).
    pathway_map:
        Long-form protein–pathway map (columns: ``uniprot_id``,
        ``pathway_name``, ``pathway_source``).
    top_k:
        Number of top Reactome pathways to keep as distinct labels.

    Returns
    -------
    pd.Series indexed like ``proteins``, with string pathway labels.
    """
    reactome = pathway_map[pathway_map["pathway_source"] == "Reactome"].copy()
    reactome = reactome[reactome["pathway_name"].str.strip().ne("")]

    # Rank pathways by global protein count (most covered first)
    pathway_freq = reactome["pathway_name"].value_counts()
    top_pathways = set(pathway_freq.head(top_k).index.tolist())

    # For each protein, select its highest-ranked Reactome pathway
    reactome["rank"] = reactome["pathway_name"].map(
        lambda name: pathway_freq.get(name, 0)
    )
    best_per_protein = (
        reactome.sort_values("rank", ascending=False)
        .groupby("uniprot_id")["pathway_name"]
        .first()
    )

    labels: list[str] = []
    for uid in proteins["uniprot_id"]:
        pathway = best_per_protein.get(uid, None)
        if pathway is None:
            labels.append("No Pathway")
        elif pathway in top_pathways:
            labels.append(pathway)
        else:
            labels.append("Other")

    label_series = pd.Series(labels, index=proteins.index, name="label")

    # Post-process: merge any class with fewer than MIN_CLASS_SIZE members
    # into "Other" to prevent stratified split failures.
    MIN_CLASS_SIZE = 5
    class_counts = label_series.value_counts()
    tiny_classes = class_counts[class_counts < MIN_CLASS_SIZE].index.tolist()
    if tiny_classes:
        label_series = label_series.replace(
            {cls: "Other" for cls in tiny_classes}
        )

    return label_series



# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def _numeric_features(
    proteins: pd.DataFrame,
    links: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build numerical and boolean feature columns.

    Numeric biophysical
    ~~~~~~~~~~~~~~~~~~~
    ``length``
        Amino-acid sequence length — proxy for protein complexity.
    ``binding_sites_count``
        Number of annotated small-molecule binding sites.
    ``interaction_count``
        Number of documented binary protein–protein interactions in
        UniProt (IntAct-sourced).

    Chemical-link aggregates
    ~~~~~~~~~~~~~~~~~~~~~~~~
    Counts per protein of substrate/product, cofactor, and binding
    relationships derived from the ``protein_chemical_links.csv`` table.

    Boolean functional flags (0/1)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Presence of free-text disease annotation, pathway comment, catalytic
    activity description, DrugBank ID, ChEMBL ID, KEGG ID, and ChEBI IDs.
    These are binary signals that capture whether a protein has been
    experimentally connected to each annotation type.

    Parameters
    ----------
    proteins:
        pandas DataFrame from ``proteins.csv``.
    links:
        pandas DataFrame from ``protein_chemical_links.csv``.

    Returns
    -------
    pd.DataFrame with one row per protein and purely numeric columns.
    """
    df = pd.DataFrame(index=proteins.index)

    # --- Biophysical numerics ---
    df["length"] = pd.to_numeric(proteins["length"], errors="coerce")
    df["binding_sites_count"] = pd.to_numeric(
        proteins["binding_sites_count"], errors="coerce"
    ).fillna(0)
    df["interaction_count"] = pd.to_numeric(
        proteins["interaction_count"], errors="coerce"
    ).fillna(0)

    # --- Chemical link aggregates ---
    # Count each link_type per protein from the long-form links table
    link_counts = (
        links.groupby(["uniprot_id", "link_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    # Normalise expected column names
    expected_link_cols = {
        "substrate/product": "link_substrate_product",
        "cofactor": "link_cofactor",
        "binding": "link_binding",
    }
    uid_to_links = {
        row["uniprot_id"]: row
        for _, row in link_counts.iterrows()
    }
    for src_col, dst_col in expected_link_cols.items():
        df[dst_col] = proteins["uniprot_id"].map(
            lambda uid, col=src_col: uid_to_links.get(uid, {}).get(col, 0)
        ).fillna(0).astype(int)

    # Total chemical links (substrate+cofactor+binding)
    df["link_total"] = (
        df["link_substrate_product"] + df["link_cofactor"] + df["link_binding"]
    )

    # --- Boolean functional flags ---
    df["has_disease_annotation"] = _has_text(proteins["disease_text"])
    df["has_pathway_comment"] = _has_text(proteins["pathway_text"])
    df["has_catalytic_activity"] = _has_text(proteins["catalytic_activity"])
    df["has_cofactors"] = _has_text(proteins["cofactors"])
    df["has_activity_regulation"] = _has_text(proteins["activity_regulation"])
    df["has_drugbank"] = _has_text(proteins["drugbank_ids"])
    df["has_chembl"] = _has_text(proteins["chembl_ids"])
    df["has_kegg"] = _has_text(proteins["kegg_id"])
    df["has_chebi_links"] = _has_text(proteins["chebi_ids"])

    return df


def _keyword_features(
    proteins: pd.DataFrame, top_n: int = TOP_KEYWORDS
) -> pd.DataFrame:
    """
    Encode UniProt keywords as a bag-of-words binary matrix.

    UniProt assigns controlled-vocabulary keywords (e.g. "Kinase",
    "Membrane", "Alternative splicing") that summarise protein function.
    We select the ``top_n`` most frequent keywords across the dataset and
    encode each protein's keyword set as a binary row vector.

    Parameters
    ----------
    proteins:
        DataFrame containing a ``keywords`` column (semicolon-delimited).
    top_n:
        Vocabulary size for keyword features.

    Returns
    -------
    pd.DataFrame with columns prefixed by ``kw_``.
    """
    kw_lists = _parse_semicolon_list(proteins["keywords"])
    vocab = _top_tokens(kw_lists, top_n)
    df = _binarise_token_lists(kw_lists, vocab)
    df.columns = [f"kw_{col.replace(' ', '_')}" for col in df.columns]
    df.index = proteins.index
    return df


def _go_features(
    proteins: pd.DataFrame, top_n: int = TOP_GO
) -> pd.DataFrame:
    """
    Encode GO terms as bag-of-words binary matrices (BP, MF, CC).

    Gene Ontology (GO) terms are the most widely used controlled vocabulary
    for protein function.  We separately encode the three GO namespaces:
    - **go_bp** (Biological Process): what pathway/process the protein
      participates in — very directly linked to the prediction target.
    - **go_mf** (Molecular Function): the biochemical activity (e.g.,
      "kinase activity", "ion binding").
    - **go_cc** (Cellular Component): where in the cell the protein acts.

    Parameters
    ----------
    proteins:
        DataFrame with ``go_bp``, ``go_mf``, ``go_cc`` columns.
    top_n:
        Vocabulary size per GO namespace.

    Returns
    -------
    pd.DataFrame with columns prefixed by ``gobp_``, ``gomf_``, ``gocc_``.
    """
    frames = []
    for col, prefix in [("go_bp", "gobp"), ("go_mf", "gomf"), ("go_cc", "gocc")]:
        tok_lists = _parse_semicolon_list(proteins[col])
        # GO terms look like "GO:0006915:apoptotic process"; take the term name
        name_lists = tok_lists.apply(
            lambda lst: [
                tok.split(":")[-1].strip() if ":" in tok else tok for tok in lst
            ]
        )
        vocab = _top_tokens(name_lists, top_n)
        sub_df = _binarise_token_lists(name_lists, vocab)
        sub_df.columns = [f"{prefix}_{c.replace(' ', '_')}" for c in sub_df.columns]
        sub_df.index = proteins.index
        frames.append(sub_df)
    return pd.concat(frames, axis=1)


def _location_features(proteins: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode subcellular localisation annotations.

    Subcellular location is a powerful predictor of pathway membership:
    e.g., mitochondrial proteins are over-represented in oxidative
    phosphorylation pathways.  We parse the free-text ``subcellular_location``
    field and one-hot encode the most commonly occurring location tokens.

    Parameters
    ----------
    proteins:
        DataFrame containing a ``subcellular_location`` column.

    Returns
    -------
    pd.DataFrame with columns prefixed by ``loc_`` (top-25 locations).
    """
    loc_lists = _parse_semicolon_list(proteins["subcellular_location"])
    vocab = _top_tokens(loc_lists, 25)
    df = _binarise_token_lists(loc_lists, vocab)
    df.columns = [f"loc_{c.replace(' ', '_')}" for c in df.columns]
    df.index = proteins.index
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_features(
    processed_dir: str = PROCESSED_DIR,
    top_k_pathways: int = TOP_K_PATHWAYS,
    top_keywords: int = TOP_KEYWORDS,
    top_go: int = TOP_GO,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str], LabelEncoder]:
    """
    Build the complete ML feature matrix and label vector.

    This is the single entry point used by all three model scripts.  It
    loads the processed CSVs, engineers features, encodes labels, and
    returns everything the models need.

    Parameters
    ----------
    processed_dir:
        Path to the directory containing ``proteins.csv``,
        ``protein_pathway_map.csv``, and ``protein_chemical_links.csv``.
    top_k_pathways:
        Number of Reactome pathways to model as explicit classes; proteins
        in rarer pathways are assigned the ``"Other"`` label.
    top_keywords:
        Size of the UniProt keyword vocabulary (binary features).
    top_go:
        Size of the GO term vocabulary per namespace (binary features).
    verbose:
        If True, print feature construction progress and summary statistics.

    Returns
    -------
    X : np.ndarray, shape (n_proteins, n_features)
        Float64 feature matrix, imputed and ready for any sklearn-compatible
        estimator.
    y : np.ndarray, shape (n_proteins,)
        Integer class labels in [0, n_classes).
    feature_names : list[str]
        Ordered list of feature names corresponding to columns of ``X``.
    le : sklearn.preprocessing.LabelEncoder
        Fitted label encoder; call ``le.inverse_transform(y_pred)`` to convert
        integer predictions back to pathway name strings.
    """
    # ── Load data ──────────────────────────────────────────────────────────
    proteins = pd.read_csv(os.path.join(processed_dir, "proteins.csv"))
    pathway_map = pd.read_csv(os.path.join(processed_dir, "protein_pathway_map.csv"))
    links = pd.read_csv(os.path.join(processed_dir, "protein_chemical_links.csv"))

    if verbose:
        print(f"Loaded {len(proteins)} proteins, {len(pathway_map)} pathway "
              f"mappings, {len(links)} chemical links.")

    # ── Labels ─────────────────────────────────────────────────────────────
    labels = _build_labels(proteins, pathway_map, top_k=top_k_pathways)

    if verbose:
        dist = labels.value_counts()
        print("\nClass distribution:")
        for cls, cnt in dist.items():
            print(f"  {cls:55s}: {cnt:4d} proteins")

    # ── Features ───────────────────────────────────────────────────────────
    num_df = _numeric_features(proteins, links)
    kw_df = _keyword_features(proteins, top_n=top_keywords)
    go_df = _go_features(proteins, top_n=top_go)
    loc_df = _location_features(proteins)

    feature_df = pd.concat([num_df, kw_df, go_df, loc_df], axis=1)

    # ── Impute remaining NaN with column median ─────────────────────────────
    for col in feature_df.columns:
        if feature_df[col].isna().any():
            feature_df[col] = feature_df[col].fillna(feature_df[col].median())

    # ── Convert to arrays ──────────────────────────────────────────────────
    X = feature_df.values.astype(np.float64)
    feature_names = list(feature_df.columns)

    le = LabelEncoder()
    y = le.fit_transform(labels)

    if verbose:
        print(f"\nFeature matrix: {X.shape[0]} proteins x {X.shape[1]} features")
        print(f"Number of classes: {len(le.classes_)}")
        print(f"Classes: {list(le.classes_)}")

    return X, y, feature_names, le


# ---------------------------------------------------------------------------
# CLI for quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    X, y, names, le = build_features(verbose=True)
    print("\nFirst 5 feature names:", names[:5])
    print("Last 5 feature names:", names[-5:])
    print("y[:10]:", y[:10])
    print("Mapped labels:", le.inverse_transform(y[:10]))
