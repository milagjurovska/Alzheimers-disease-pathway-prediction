import pandas as pd
import os
import numpy as np

def create_features():
    # File paths
    proteins_file = os.path.join("data", "processed", "uniprot_proteins_cleaned.csv")
    links_file = os.path.join("data", "processed", "protein_chebi_links.csv")
    chem_file = os.path.join("data", "processed", "chebi_compounds_classified.csv")
    pathways_file = os.path.join("data", "results", "pathway_frequency_named.csv")
    output_file = os.path.join("data", "processed", "pathway_features.csv")
    
    print("Loading data...")
    df_prot = pd.read_csv(proteins_file)
    df_links = pd.read_csv(links_file)
    df_chem = pd.read_csv(chem_file)
    df_paths = pd.read_csv(pathways_file)
    
    # 1. Map Proteins to Chemical Categories
    # Create a dict: Entry -> {'Neurotransmitter': 1, 'Ion': 0, ...}
    chem_map = df_chem.set_index('ChEBI_ID')['Category'].to_dict()
    df_links['Category'] = df_links['ChEBI_ID'].map(chem_map)
    
    # One-hot encode chemical categories per protein
    # A protein might have multiple links. We want to know if it *ever* interacts with a category.
    prot_chem_features = df_links.pivot_table(index='Entry', columns='Category', values='ChEBI_ID', aggfunc='count', fill_value=0)
    prot_chem_features = (prot_chem_features > 0).astype(int) # Binary: interacts or not
    
    # Merge chemical features back to protein dataframe
    df_prot = df_prot.merge(prot_chem_features, on='Entry', how='left').fillna(0)
    
    # Ensure all columns exist (in case some categories are missing in data)
    expected_cats = ['Neurotransmitter', 'Ion', 'Cofactor', 'Metabolite']
    for cat in expected_cats:
        if cat not in df_prot.columns:
            df_prot[cat] = 0
            
    # 2. Group Proteins by Pathway
    # We need to rank/process all pathways, not just the top named ones.
    # But for ML, we need names to label them as AD-related. 
    # Let's use the 'pathway_frequency_named.csv' as our master list of pathways to predict.
    
    pathway_features = []
    
    # Create a mapping of Pathway -> [List of Protein Entries]
    # Re-parsing columns 'Reactome' and 'KEGG' from df_prot
    pathway_to_proteins = {}
    
    print("Mapping proteins to pathways...")
    for _, row in df_prot.iterrows():
        entry = row['Entry']
        paths = []
        if pd.notna(row.get('Reactome')): paths.extend([p.strip() for p in str(row['Reactome']).split(';') if p.strip()])
        if pd.notna(row.get('KEGG')): paths.extend([p.strip() for p in str(row['KEGG']).split(';') if p.strip()])
        
        for p in paths:
            if p not in pathway_to_proteins:
                pathway_to_proteins[p] = []
            pathway_to_proteins[p].append(entry)
            
    # 3. Aggregate Features per Pathway
    print("Aggregating features...")
    for pid, entries in pathway_to_proteins.items():
        # Get subset of proteins for this pathway
        subset = df_prot[df_prot['Entry'].isin(entries)]
        
        n_proteins = len(subset)
        if n_proteins < 3: continue # Skip very small pathways
        
        # Chemical Ratios
        ratio_neuro = subset['Neurotransmitter'].mean()
        ratio_ion = subset['Ion'].mean()
        ratio_cofactor = subset['Cofactor'].mean()
        
        # Keyword extraction from Function (simple proxy for mechanistic role)
        # Fraction of proteins with keywords
        def has_keyword(text, kw): return 1 if kw in str(text).lower() else 0
        
        ratio_kinase = subset['Function [CC]'].apply(lambda x: has_keyword(x, 'kinase')).mean()
        ratio_receptor = subset['Function [CC]'].apply(lambda x: has_keyword(x, 'receptor')).mean()
        ratio_transport = subset['Function [CC]'].apply(lambda x: has_keyword(x, 'transport')).mean()
        
        feat = {
            'Pathway_ID': pid,
            'n_proteins': n_proteins,
            'ratio_neurotransmitter': ratio_neuro,
            'ratio_ion': ratio_ion,
            'ratio_cofactor': ratio_cofactor,
            'ratio_kinase': ratio_kinase,
            'ratio_receptor': ratio_receptor,
            'ratio_transport': ratio_transport
        }
        pathway_features.append(feat)
        
    df_features = pd.DataFrame(pathway_features)
    
    # 4. Define Target Variable (Labeling)
    # Join with names to help labeling
    df_features = df_features.merge(df_paths[['Pathway_ID', 'Pathway_Name']], on='Pathway_ID', how='left')
    df_features['Pathway_Name'] = df_features['Pathway_Name'].fillna('')
    
    def get_label(row):
        name = row['Pathway_Name'].lower()
        # Positive Class: Clear AD relevance
        pos_keywords = ['alzheimer', 'amyloid', 'tau', 'synap', 'neuro', 'axon', 'dendri', 'cognitive', 'memory', 'glutamat', 'cholinergic']
        if any(k in name for k in pos_keywords):
            return 1
            
        # Also treat top 10 most frequent pathways as "Core Mechanisms" if not explicitly named
        # We need to access the dataframe rank, but row apply doesn't give context easily.
        # Alternatively, let's relax the keyword search or use the statistical enrichment rank which we can merge in.
        return 0

    # Let's add a rank-based label
    df_features = df_features.sort_values(by='n_proteins', ascending=False)
    top_n_count = int(len(df_features) * 0.05) # Top 5%
    
    # We'll use a combined label: Either Keyword Match OR Top 5% Frequency
    # This ensures we have enough positive samples (statistical inference is a valid ground truth proxy here)
    
    def refined_label(row):
        name = str(row['Pathway_Name']).lower()
        pos_keywords = ['alzheimer', 'amyloid', 'tau', 'synap', 'neuro', 'axon', 'dendri', 'cognitive', 'memory', 'glutamat', 'cholinergic']
        if any(k in name for k in pos_keywords):
            return 1
        return 0
        
    df_features['AD_Related_Keyword'] = df_features.apply(refined_label, axis=1)
    
    # Add top 5% as 1
    df_features['AD_Related'] = 0
    df_features.loc[df_features.index[:top_n_count], 'AD_Related'] = 1
    
    # Combine (Union)
    df_features['AD_Related'] = df_features[['AD_Related', 'AD_Related_Keyword']].max(axis=1)
    
    df_features.drop(columns=['AD_Related_Keyword'], inplace=True)
    
    print("Label distribution:")
    print(df_features['AD_Related'].value_counts())
    
    df_features.to_csv(output_file, index=False)
    print(f"Saved {len(df_features)} pathway feature sets to {output_file}")

if __name__ == "__main__":
    create_features()
