# Alzheimer's Disease Pathway Prediction

Predicting biological pathways associated with Alzheimer's disease through integration of publicly available bioinformatics databases (UniProt, ChEBI, Reactome, KEGG).

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Data Pipeline

```bash
# 1. Fetch AD-associated human proteins from UniProt
python fetch_uniprot.py

# 2. Fetch small-molecule data from ChEBI for linked compounds
python fetch_chebi.py

# 3. Parse, clean, and merge into structured CSVs
python preprocess.py

# 4. (Optional) Verify output integrity
python preprocess.py --verify
```

## Output

All processed data is saved to `data/processed/`:

| File | Description |
|------|-------------|
| `proteins.csv` | One row per AD-associated protein with functional annotations |
| `chemicals.csv` | One row per ChEBI compound linked to the proteins |
| `protein_chemical_links.csv` | Protein ↔ compound relationships |
| `protein_pathway_map.csv` | Protein ↔ pathway relationships (Reactome, KEGG) |

## Project Structure

```
├── data/
│   ├── raw/            # Raw JSON from APIs
│   └── processed/      # Cleaned CSV files
├── fetch_uniprot.py    # UniProt data fetcher
├── fetch_chebi.py      # ChEBI compound fetcher
├── preprocess.py       # Parsing + cleaning + merging
├── requirements.txt
└── README.md
```
