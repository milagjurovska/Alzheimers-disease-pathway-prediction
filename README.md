# Alzheimer's Disease Pathway Inference

## Project Overview
This project performs a systematic analysis and inference of biological pathways associated with Alzheimer's Disease (AD). It integrates data from **UniProt** (human proteins) and **ChEBI** (small molecules/chemicals) to identify key molecular mechanisms, specifically focusing on neurotransmitters, ions, and metabolic processes.

The goal is to move beyond simple protein lists and uncover significantly enriched pathways linked to synaptic signaling, neuroinflammation, and metabolic dysfunction.

## Methodology

### Phase 1: Protein Data Acquisition (UniProt)
- Fetched human proteins associated with "Alzheimer's disease" using the UniProtKB API.
- Query: `(alzheimer) AND (organism_id:9606)`.
- **Data**: Recovered ~1400 proteins with annotations for Function, Subcellular Location, and Disease Involvement.

### Phase 2: Chemical Data Integration (ChEBI)
- Extracted chemical interaction data (Cofactors, Catalytic Activity) from protein records.
- Mapped chemical names to **ChEBI IDs**.
- Fetched detailed chemical classifications (Neurotransmitter, Ion, Metabolite, Cofactor) using the OLS API.
- **Key Outcome**: Identified specific small molecules interacting with AD-related proteins.

### Phase 3: Pathway Analysis & Inference
- Mapped proteins to **KEGG** and **Reactome** pathways.
- Performed enrichment analysis to leverage the frequency of proteins in specific pathways.
- Integrated chemical data to highlight pathways enriched with specific molecular interactions (e.g., Ion channel transport).

## Project Structure

```
├── data/
│   ├── raw/                  # Raw data from APIs (mostly TSV)
│   ├── processed/            # Cleaned CSVs, ChEBI links, and classifications
│   └── results/              # Final analysis output (Pathway frequencies, enrichments)
├── src/
│   ├── fetch_uniprot.py      # Fetches protein data from UniProt
│   ├── preprocess_uniprot.py # Cleans and structures protein data
│   ├── extract_chebi_ids.py  # Extracts ChEBI IDs from protein metadata
│   ├── fetch_chebi.py        # Fetches chemical details from OLS/ChEBI
│   ├── classify_chemicals.py # Classifies chemicals (Ion, Neurotransmitter, etc.)
│   ├── pathway_analysis.py   # Performs pathway mapping and integration
│   └── enrich_pathway_names.py # Adds readable names to pathway IDs
└── README.md
```

## How to Run

1.  **Install Dependencies**:
    ```bash
    pip install pandas requests
    ```

2.  **Execute Pipeline**:
    Run the scripts in the following order to reproduce the analysis:
    
    ```bash
    # 1. Fetch Protein Data
    python src/fetch_uniprot.py
    
    # 2. Preprocess Data
    python src/preprocess_uniprot.py
    
    # 3. Extract Chemical IDs
    python src/extract_chebi_ids.py
    
    # 4. Fetch Chemical Details (Takes time due to API limits)
    python src/fetch_chebi.py
    
    # 5. Classify Chemicals
    python src/classify_chemicals.py
    
    # 6. Analyze Pathways (Statistical Baseline)
    python src/pathway_analysis.py
    
    # 7. Get Pathway Names (Optional)
    python src/enrich_pathway_names.py
    
    # 8. Machine Learning: Feature Engineering
    python src/feature_engineering.py
    
    # 9. Train & Evaluate Model
    python src/train_model.py
    ```

## Results
Key results are stored in `data/results/`:
- `pathway_frequency_named.csv`: Ranked list of statistically enriched pathways.
- `ml_performance_report.txt`: Performance metrics of the Random Forest model (Accuracy, Precision, Recall, Feature Importance).
- `roc_curve.png`: ROC Curve visualization of the model's performance.
- `feature_importance.png`: Bar chart showing which molecular features drive the prediction.
