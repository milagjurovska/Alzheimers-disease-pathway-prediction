from __future__ import annotations
import os
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
TOP_K_PATHWAYS: int = 10
TOP_KEYWORDS: int = 60
TOP_GO: int = 40
_MIN_TEXT_LEN: int = 2


def _has_text(series: pd.Series) -> pd.Series:
    return series.fillna("").str.strip().str.len().gt(_MIN_TEXT_LEN).astype(int)


def _parse_semicolon_list(series: pd.Series) -> pd.Series:
    return series.fillna("").apply(
        lambda s: [tok.strip().lower() for tok in s.split(";") if tok.strip()]
    )


def _top_tokens(token_lists: pd.Series, top_n: int) -> list[str]:
    counter: Counter = Counter()
    for lst in token_lists:
        counter.update(lst)
    return [tok for tok, _ in counter.most_common(top_n)]


def _binarise_token_lists(
    token_lists: pd.Series, vocab: list[str]
) -> pd.DataFrame:
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


def _build_labels(
    proteins: pd.DataFrame,
    pathway_map: pd.DataFrame,
    top_k: int = TOP_K_PATHWAYS,
) -> pd.Series:
    reactome = pathway_map[pathway_map["pathway_source"] == "Reactome"].copy()
    reactome = reactome[reactome["pathway_name"].str.strip().ne("")]
    pathway_freq = reactome["pathway_name"].value_counts()
    top_pathways = set(pathway_freq.head(top_k).index.tolist())
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
    MIN_CLASS_SIZE = 5
    class_counts = label_series.value_counts()
    tiny_classes = class_counts[class_counts < MIN_CLASS_SIZE].index.tolist()
    if tiny_classes:
        label_series = label_series.replace(
            {cls: "Other" for cls in tiny_classes}
        )
    return label_series


def _numeric_features(
    proteins: pd.DataFrame,
    links: pd.DataFrame,
) -> pd.DataFrame:
    df = pd.DataFrame(index=proteins.index)
    df["length"] = pd.to_numeric(proteins["length"], errors="coerce")
    df["binding_sites_count"] = pd.to_numeric(
        proteins["binding_sites_count"], errors="coerce"
    ).fillna(0)
    df["interaction_count"] = pd.to_numeric(
        proteins["interaction_count"], errors="coerce"
    ).fillna(0)
    link_counts = (
        links.groupby(["uniprot_id", "link_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
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
    df["link_total"] = (
        df["link_substrate_product"] + df["link_cofactor"] + df["link_binding"]
    )
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
    kw_lists = _parse_semicolon_list(proteins["keywords"])
    vocab = _top_tokens(kw_lists, top_n)
    df = _binarise_token_lists(kw_lists, vocab)
    df.columns = [f"kw_{col.replace(' ', '_')}" for col in df.columns]
    df.index = proteins.index
    return df


def _go_features(
    proteins: pd.DataFrame, top_n: int = TOP_GO
) -> pd.DataFrame:
    frames = []
    for col, prefix in [("go_bp", "gobp"), ("go_mf", "gomf"), ("go_cc", "gocc")]:
        tok_lists = _parse_semicolon_list(proteins[col])
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
    loc_lists = _parse_semicolon_list(proteins["subcellular_location"])
    vocab = _top_tokens(loc_lists, 25)
    df = _binarise_token_lists(loc_lists, vocab)
    df.columns = [f"loc_{c.replace(' ', '_')}" for c in df.columns]
    df.index = proteins.index
    return df


def build_features(
    processed_dir: str = PROCESSED_DIR,
    top_k_pathways: int = TOP_K_PATHWAYS,
    top_keywords: int = TOP_KEYWORDS,
    top_go: int = TOP_GO,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str], LabelEncoder]:
    proteins = pd.read_csv(os.path.join(processed_dir, "proteins.csv"))
    pathway_map = pd.read_csv(os.path.join(processed_dir, "protein_pathway_map.csv"))
    links = pd.read_csv(os.path.join(processed_dir, "protein_chemical_links.csv"))
    if verbose:
        print(f"Loaded {len(proteins)} proteins, {len(pathway_map)} pathway "
              f"mappings, {len(links)} chemical links.")
    labels = _build_labels(proteins, pathway_map, top_k=top_k_pathways)
    if verbose:
        dist = labels.value_counts()
        print("\nClass distribution:")
        for cls, cnt in dist.items():
            print(f"  {cls:55s}: {cnt:4d} proteins")
    num_df = _numeric_features(proteins, links)
    kw_df = _keyword_features(proteins, top_n=top_keywords)
    go_df = _go_features(proteins, top_n=top_go)
    loc_df = _location_features(proteins)
    feature_df = pd.concat([num_df, kw_df, go_df, loc_df], axis=1)
    for col in feature_df.columns:
        if feature_df[col].isna().any():
            feature_df[col] = feature_df[col].fillna(feature_df[col].median())
    X = feature_df.values.astype(np.float64)
    feature_names = list(feature_df.columns)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    if verbose:
        print(f"\nFeature matrix: {X.shape[0]} proteins x {X.shape[1]} features")
        print(f"Number of classes: {len(le.classes_)}")
        print(f"Classes: {list(le.classes_)}")
    return X, y, feature_names, le
if __name__ == "__main__":
    X, y, names, le = build_features(verbose=True)
    print("\nFirst 5 feature names:", names[:5])
    print("Last 5 feature names:", names[-5:])
    print("y[:10]:", y[:10])
    print("Mapped labels:", le.inverse_transform(y[:10]))
