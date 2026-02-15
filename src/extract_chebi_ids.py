import pandas as pd
import re
import os

def extract_chebi_ids():
    input_file = os.path.join("data", "processed", "uniprot_proteins_cleaned.csv")
    output_dir = os.path.join("data", "processed")
    links_file = os.path.join(output_dir, "protein_chebi_links.csv")
    ids_file = os.path.join(output_dir, "unique_chebi_ids.txt")
    
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    
    chebi_pattern = re.compile(r'CHEBI:(\d+)')
    
    protein_links = []
    unique_ids = set()
    
    # Columns to search
    target_cols = ['Cofactor', 'Catalytic activity']
    
    for _, row in df.iterrows():
        entry = row['Entry']
        found_ids = set()
        
        for col in target_cols:
            if col in df.columns and pd.notna(row[col]):
                text = str(row[col])
                matches = chebi_pattern.findall(text)
                for m in matches:
                    chebi_id = f"CHEBI:{m}"
                    found_ids.add(chebi_id)
                    unique_ids.add(chebi_id)
        
        for cid in found_ids:
            protein_links.append({'Entry': entry, 'ChEBI_ID': cid})
            
    # Save links
    links_df = pd.DataFrame(protein_links)
    print(f"Found {len(links_df)} protein-chemical associations.")
    links_df.to_csv(links_file, index=False)
    
    # Save unique IDs
    sorted_ids = sorted(list(unique_ids))
    print(f"Found {len(sorted_ids)} unique ChEBI IDs.")
    with open(ids_file, 'w') as f:
        for cid in sorted_ids:
            f.write(f"{cid}\n")
            
    print(f"Saved links to {links_file}")
    print(f"Saved sorted IDs to {ids_file}")

if __name__ == "__main__":
    extract_chebi_ids()
