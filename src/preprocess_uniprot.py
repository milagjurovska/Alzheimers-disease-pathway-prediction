import pandas as pd
import os
import re

def clean_text(text):
    if pd.isna(text):
        return ""
    # Remove "FUNCTION: ", "SUBCELLULAR LOCATION: ", etc. prefixes if present
    text = re.sub(r'^[A-Z\s]+:\s*', '', str(text))
    # Remove evidence tags like {ECO:0000269|PubMed:12345}
    text = re.sub(r'\{ECO:[^}]+\}', '', text)
    return text.strip()

def preprocess_uniprot_data():
    input_file = os.path.join("data", "raw", "uniprot_alzheimer_human.tsv")
    output_dir = os.path.join("data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "uniprot_proteins_cleaned.csv")
    
    print(f"Loading data from {input_file}...")
    try:
        df = pd.read_csv(input_file, sep='\t')
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print(f"Initial entries: {len(df)}")
    
    # Rename columns for clarity if needed (column names from API are usually clear)
    # Expected columns: Entry, Entry Name, Protein names, Gene Names, Function [CC], Subcellular location [CC], Involvement in disease, Gene Ontology (biological process), ...
    
    # Clean text columns
    text_cols = ['Function [CC]', 'Subcellular location [CC]', 'Involvement in disease', 'Cofactor', 'Catalytic activity']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)
    
    # Filter for proteins clearly linked to Alzheimer's if needed (though query did that)
    # Maybe add a column for "Mentioned in Disease" just as a flag
    
    # save
    print(f"Saving processed data to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done.")

if __name__ == "__main__":
    preprocess_uniprot_data()
