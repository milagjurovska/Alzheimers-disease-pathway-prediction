import pandas as pd
import requests
import os
import time

def enrich_pathways():
    input_file = os.path.join("data", "results", "pathway_frequency.csv")
    output_file = os.path.join("data", "results", "pathway_frequency_named.csv")
    
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    
    # 1. Fetch KEGG Map (all human pathways)
    print("Fetching KEGG pathway list...")
    kegg_map = {}
    try:
        url = "http://rest.kegg.jp/list/pathway/hsa"
        response = requests.get(url)
        if response.status_code == 200:
            for line in response.text.split('\n'):
                if line.strip():
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        # KEGG ID format: path:hsa00010 -> hsa00010
                        kid = parts[0].replace('path:', '')
                        name = parts[1]
                        kegg_map[kid] = name
            print(f"Fetched {len(kegg_map)} KEGG pathways.")
        else:
            print(f"Failed to fetch KEGG list: {response.status_code}")
    except Exception as e:
        print(f"Error fetching KEGG: {e}")

    # 2. Enrich DataFrame (Focus on top 100 entries to save API calls for Reactome)
    # We will process the top 100 regardless of ID type
    top_n = 100
    df_top = df.head(top_n).copy()
    
    pathway_names = []
    
    print(f"Enriching top {top_n} pathways...")
    
    for pid in df_top['Pathway_ID']:
        name = "Unknown"
        
        # Check KEGG
        # KEGG IDs in UniProt usually "hsa05010"
        if pid in kegg_map:
            name = kegg_map[pid]
        
        # Check Reactome (R-HSA-123456)
        elif pid.startswith("R-HSA-"):
            try:
                # Use Reactome Content Service
                url = f"https://reactome.org/ContentService/data/query/{pid}/displayName"
                # Reactome API can be picky, let's try
                # Note: valid Reactome ID required
                headers = {'Accept': 'text/plain'}
                resp = requests.get(url, headers=headers)
                if resp.status_code == 200:
                    name = resp.text.strip()
            except:
                pass
            time.sleep(0.1) # Be nice
            
        pathway_names.append(name)
        
    df_top['Pathway_Name'] = pathway_names
    
    # Save
    df_top.to_csv(output_file, index=False)
    print(f"Saved enriched top pathways to {output_file}")
    print(df_top[['Pathway_ID', 'Pathway_Name', 'Count']].head(10))

if __name__ == "__main__":
    enrich_pathways()
