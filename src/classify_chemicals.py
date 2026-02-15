import pandas as pd
import os

def classify_chemical(row):
    name = str(row['Name']).lower()
    defi = str(row['Definition']).lower()
    
    # Neurotransmitters (major ones)
    neurotransmitters = ['glutamate', 'gaba', 'acetylcholine', 'dopamine', 'serotonin', 'norepinephrine', 'epinephrine', 'histamine', 'glycine']
    for nt in neurotransmitters:
        if nt in name:
            return 'Neurotransmitter'
            
    # Ions
    if 'ion' in name or '(+)' in name or '(-)' in name or 'cation' in name or 'anion' in name or 'element' in defi:
        # Refine: biological ions often end in (2+) or (1+) or just chemical symbol
        return 'Ion'
        
    # Cofactors (common ones)
    cofactors = ['nad', 'fad', 'atp', 'gtp', 'coenzyme a', 'heme', 'magnesium', 'zinc', 'iron', 'copper']
    for cf in cofactors:
        if cf in name:
            return 'Cofactor'
            
    # Metabolites (catch-all for small organic molecules)
    return 'Metabolite'

def process_classification():
    input_file = os.path.join("data", "processed", "chebi_compounds.csv")
    output_file = os.path.join("data", "processed", "chebi_compounds_classified.csv")
    
    df = pd.read_csv(input_file)
    print(f"Classifying {len(df)} compounds...")
    
    df['Category'] = df.apply(classify_chemical, axis=1)
    
    print("Classification summary:")
    print(df['Category'].value_counts())
    
    df.to_csv(output_file, index=False)
    print(f"Saved classified data to {output_file}")

if __name__ == "__main__":
    process_classification()
