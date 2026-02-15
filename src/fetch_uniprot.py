import requests
import os
import sys

def fetch_uniprot_data():
    # Broader query: searches for "alzheimer" in any field for human proteins
    query = "(alzheimer) AND (organism_id:9606)"
    
    params = {
        "query": query,
        "format": "tsv",
        "fields": "accession,id,protein_name,gene_names,cc_function,cc_subcellular_location,cc_disease,go_p,go_f,go_c,cc_cofactor,cc_catalytic_activity,xref_kegg,xref_reactome"
    }
    
    url = "https://rest.uniprot.org/uniprotkb/stream"
    
    output_dir = os.path.join("data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "uniprot_alzheimer_human.tsv")
    
    print(f"Fetching data from UniProt: {url} with params {params}")
    
    try:
        response = requests.get(url, params=params, stream=True)
        response.raise_for_status()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk.decode('utf-8'))
        
        print(f"Successfully saved data to {output_file}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        if 'response' in locals() and response is not None:
             print(f"Response text: {response.text}")
        sys.exit(1)

if __name__ == "__main__":
    fetch_uniprot_data()
