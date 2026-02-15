import requests
import pandas as pd
import os
import time

def fetch_chebi_details():
    input_file = os.path.join("data", "processed", "unique_chebi_ids.txt")
    output_dir = os.path.join("data", "processed")
    output_file = os.path.join(output_dir, "chebi_compounds.csv")
    
    with open(input_file, 'r') as f:
        chebi_ids = [line.strip() for line in f if line.strip()]
        
    print(f"Fetching details for {len(chebi_ids)} ChEBI IDs...")
    
    results = []
    
    # Base URL for OLS API (Ontology Lookup Service)
    # Using OLS4 as it's the current standard
    base_url = "https://www.ebi.ac.uk/ols4/api/ontologies/chebi/terms"
    
    for cid in chebi_ids:
        try:
            # Construct IRI for the term. ChEBI IDs in OLS are like http://purl.obolibrary.org/obo/CHEBI_15377
            # Input format is CHEBI:15377
            short_form = cid.replace(":", "_")
            iri = f"http://purl.obolibrary.org/obo/{short_form}"
            
            # Use 'iri' parameter to be precise
            response = requests.get(base_url, params={'iri': iri})
            
            if response.status_code == 200:
                data = response.json()
                if data['_embedded']['terms']:
                    term = data['_embedded']['terms'][0]
                    name = term.get('label', '')
                    description = term.get('description', [])
                    definition = description[0] if description else ""
                    
                    # Get parent terms (to identify if it's an ion, neurotransmitter, etc.)
                    # This might require following links, but let's grab the direct parent URL or label if available in basic term info
                    # OLS often provides 'annotation' or 'description'. 
                    # For deeper hierarchy we might need more calls, but let's start with basic info.
                    
                    results.append({
                        'ChEBI_ID': cid,
                        'Name': name,
                        'Definition': definition,
                        'Stars': term.get('annotation', {}).get('star', [''])[0] if 'annotation' in term else ''
                    })
                    print(f"Fetched: {cid} -> {name}")
                else:
                    print(f"No term found for {cid}")
            else:
                print(f"Failed to fetch {cid}: {response.status_code}")
                
            # Be nice to the API
            time.sleep(0.2)
            
        except Exception as e:
            print(f"Error fetching {cid}: {e}")
            
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Saved compound details to {output_file}")

if __name__ == "__main__":
    fetch_chebi_details()
