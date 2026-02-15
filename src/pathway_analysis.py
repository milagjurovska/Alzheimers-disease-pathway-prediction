import pandas as pd
import os
import re
from collections import Counter

def analyze_pathways():
    proteins_file = os.path.join("data", "processed", "uniprot_proteins_cleaned.csv")
    chemicals_file = os.path.join("data", "processed", "chebi_compounds_classified.csv")
    links_file = os.path.join("data", "processed", "protein_chebi_links.csv")
    output_dir = os.path.join("data", "results")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data...")
    df_prot = pd.read_csv(proteins_file)
    df_chem = pd.read_csv(chemicals_file)
    df_links = pd.read_csv(links_file)
    
    # 1. Pathway Frequency Analysis
    print("Analyzing pathways...")
    pathway_counts = Counter()
    
    # Parse Reactome (format: R-HSA-123456;Name w/ semi-colons... tricky, usually "ID [Optional Name];")
    # UniProt 'Reactome' column often contains: "R-HSA-164843;" or "R-HSA-164843 [Reactome];"
    # Actually, UniProt format is: "DatabaseName:ID;" or just "ID;" depending on 'xref_reactome' field.
    # The 'xref_reactome' field usually returns list of IDs.
    
    # Let's inspect the actual format from a sample row if possible, but assuming semicolon sep
    for _, row in df_prot.iterrows():
        if pd.notna(row.get('Reactome')):
            # Split by ';' and clean
            paths = str(row['Reactome']).split(';')
            for p in paths:
                p = p.strip()
                if p:
                    pathway_counts[p] += 1
        
        if pd.notna(row.get('KEGG')):
             # Split by ';'
            paths = str(row['KEGG']).split(';')
            for p in paths:
                p = p.strip()
                if p:
                    pathway_counts[p] += 1

    # Convert to DataFrame
    pathway_df = pd.DataFrame.from_dict(pathway_counts, orient='index', columns=['Count']).reset_index()
    pathway_df.columns = ['Pathway_ID', 'Count']
    pathway_df = pathway_df.sort_values(by='Count', ascending=False)
    
    print(f"Found {len(pathway_df)} unique pathways.")
    pathway_df.to_csv(os.path.join(output_dir, "pathway_frequency.csv"), index=False)
    
    # 2. Integration: Pathways + Chemicals
    # Focus on specific categories: Neurotransmitter, Ion
    print("Integrating Chemical Data...")
    
    # Create a map of Protein -> [Chemical Categories]
    chem_map = df_chem.set_index('ChEBI_ID')['Category'].to_dict()
    
    # Link proteins to chemical categories
    # df_links has Entry, ChEBI_ID
    df_links['Category'] = df_links['ChEBI_ID'].map(chem_map)
    
    # Count proteins interacting with Neurotransmitters vs Ions by Pathway
    # This is complex: Pathway -> Protein -> Chemical
    
    # Flatten Pathway -> Protein map
    pathway_protein_map = []
    for _, row in df_prot.iterrows():
        entry = row['Entry']
        # Reactome/KEGG
        all_paths = []
        if pd.notna(row.get('Reactome')): all_paths.extend([p.strip() for p in str(row['Reactome']).split(';') if p.strip()])
        if pd.notna(row.get('KEGG')): all_paths.extend([p.strip() for p in str(row['KEGG']).split(';') if p.strip()])
        
        for p in all_paths:
            pathway_protein_map.append({'Pathway_ID': p, 'Entry': entry})
            
    df_pp = pd.DataFrame(pathway_protein_map)
    
    # Join with chemical links
    # df_pp (Pathway, Entry) <-> df_links (Entry, Category)
    df_merged = pd.merge(df_pp, df_links, on='Entry', how='inner')
    
    # Count chemical categories per pathway
    print("Calculating chemical enrichment in pathways...")
    pivot = df_merged.pivot_table(index='Pathway_ID', columns='Category', values='Entry', aggfunc='count', fill_value=0)
    pivot['Total_Proteins'] = df_pp.groupby('Pathway_ID')['Entry'].nunique()
    
    # Filter for top pathways with chemical interactions
    top_chem_pathways = pivot.sort_values(by='Total_Proteins', ascending=False).head(50)
    
    top_chem_pathways.to_csv(os.path.join(output_dir, "pathway_chemical_enrichment.csv"))
    print(f"Saved integration analysis to {output_dir}")

if __name__ == "__main__":
    analyze_pathways()
