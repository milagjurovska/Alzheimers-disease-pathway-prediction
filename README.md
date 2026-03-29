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
│   ├── processed/      # Cleaned CSV files
│   ├── results/        # Model evaluation metrics (results_summary.csv)
│   └── visualizations/ # All generated plots
├── features/
│   └── feature_engineering.py  # Feature matrix construction
├── models/
│   ├── evaluate.py         # Shared evaluation utilities
│   ├── random_forest.py    # Random Forest classifier
│   ├── xgboost_model.py    # XGBoost classifier
│   └── neural_network.py   # MLP Neural Network (PyTorch)
├── fetch_uniprot.py
├── fetch_chebi.py
├── preprocess.py
├── visualize_data.py
├── run_models.py           # <- Run all three models from here
└── requirements.txt
```

## Modelling

### Task

**Multi-class pathway membership classification**: given a protein's
biological features, predict which Reactome pathway group it belongs to.
The top-10 most-populated Reactome pathways become explicit classes;
proteins in rarer pathways are merged into an "Other" class; proteins
with no Reactome annotation are assigned "No Pathway".

### Features

The feature matrix (~200 columns) is built by `features/feature_engineering.py`
from four groups:

| Feature group | Columns | Description |
|---|---|---|
| Numeric / biophysical | 9 | Sequence length, binding-site count, interaction count, chemical-link type counts, boolean functional flags |
| UniProt keyword BoW | 60 | Binary presence of the 60 most-frequent controlled-vocabulary keywords |
| GO-term BoW | 120 | Binary presence of top-40 GO terms per namespace (BP, MF, CC) |
| Subcellular location | 25 | One-hot encoding of subcellular location annotations |

### Models

| Model | File | Key features |
|---|---|---|
| **Random Forest** | `models/random_forest.py` | 200-600 trees, RandomizedSearchCV, MDI feature importances |
| **XGBoost** | `models/xgboost_model.py` | Gradient boosting, early stopping, SHAP feature attribution |
| **Neural Network** | `models/neural_network.py` | 3-layer MLP (256->128->64), BatchNorm, Dropout, AdamW + Cosine LR |

### Running the Models

```bash
# Train and evaluate all three models (recommended):
python run_models.py

# Train a single model:
python run_models.py --model rf    # Random Forest only
python run_models.py --model xgb   # XGBoost only
python run_models.py --model nn    # Neural Network only

# Skip a model:
python run_models.py --skip nn
```

### Outputs

After running, the following files are created:

**Metrics**
- `data/results/results_summary.csv` - side-by-side accuracy, macro-F1, weighted-F1

**Plots** (`data/visualizations/`)
- `rf_feature_importances.png` - top-30 features by Gini MDI
- `rf_cv_results.png` - hyperparameter search CV scores
- `rf_confusion_matrix.png`
- `xgb_training_curve.png` - train/validation loss across boosting rounds
- `xgb_shap_summary.png` - SHAP feature attribution
- `xgb_confusion_matrix.png`
- `nn_loss_curves.png` - train/validation cross-entropy per epoch
- `nn_accuracy_curve.png` - train/validation accuracy per epoch
- `nn_confusion_matrix.png`

### Results

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| **Random Forest** | 70.93% | 0.3111 | 0.6618 |
| **XGBoost** | 62.56% | 0.4617 | 0.6171 |
| **Neural Network (MLP)** | 51.54% | 0.4355 | 0.5263 |

- **Random Forest** achieves the highest overall accuracy (70.93%) but the lowest Macro F1 (0.3111), indicating it favors majority classes ("Other", "No Pathway") at the expense of rare pathways.
- **XGBoost** achieves the best Macro F1 (0.4617), demonstrating the most balanced classification across all pathway classes, including minority ones.
- **Neural Network (MLP)** shows competitive Macro F1 (0.4355) but lower accuracy, likely due to the limited dataset size (~906 training samples) relative to its 99,011 parameters.
