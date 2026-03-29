import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'figure.autolayout': True, 'font.family': 'sans-serif'})
base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, "data", "processed")
output_dir = os.path.join(base_dir, "data", "visualizations")
os.makedirs(output_dir, exist_ok=True)


def generate_charts():
    print("Loading datasets...")
    try:
        proteins = pd.read_csv(os.path.join(data_dir, "proteins.csv"))
        pathways = pd.read_csv(os.path.join(data_dir, "protein_pathway_map.csv"))
        links = pd.read_csv(os.path.join(data_dir, "protein_chemical_links.csv"))
        chemicals = pd.read_csv(os.path.join(data_dir, "chemicals.csv"))
    except FileNotFoundError as e:
        print(f"Error: Could not find processed data files. Ensure preprocess.py has been run. {e}")
        return
    print("Generating Dataset Composition chart...")
    plt.figure(figsize=(8, 6))
    counts = pd.Series({"Proteins": len(proteins), "Chemicals": len(chemicals)})
    sns.barplot(x=counts.index, y=counts.values, palette="muted", hue=counts.index, legend=False)
    plt.title("Alzheimer's Dataset Composition", pad=20)
    plt.ylabel("Unique Count")
    for i, v in enumerate(counts.values):
        plt.text(i, v + 20, str(v), ha='center', fontweight='bold')
    plt.savefig(os.path.join(output_dir, "dataset_composition.png"), dpi=300)
    plt.close()
    print("Generating Etiological Markers chart...")
    markers = ["APP", "MAPT", "PSEN1", "PSEN2", "APOE", "BACE1"]
    marker_data = proteins[proteins['gene_name'].isin(markers)].copy()
    if not marker_data.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=marker_data, x="gene_name", y="interaction_count", palette="viridis", hue="gene_name", legend=False)
        plt.title("Protein-Protein Connectivity of Known AD Etiological Markers", pad=20)
        plt.ylabel("Number of Interacting Proteins (IntAct)")
        plt.xlabel("Key Etiological Genes")
        plt.savefig(os.path.join(output_dir, "etiological_markers.png"), dpi=300)
    plt.close()
    print("Generating Pathway Distribution chart...")
    plt.figure(figsize=(12, 7))
    pathway_counts = pathways['pathway_name'].replace('', pd.NA).dropna().astype(str)
    top_pathways = pathway_counts.value_counts().head(15)
    if not top_pathways.empty:
        sns.barplot(x=top_pathways.values, y=top_pathways.index, hue=top_pathways.index, legend=False)
        plt.title("Top 15 Biological Pathways in Alzheimer's Dataset", pad=20)
        plt.xlabel("Number of Associated Proteins")
        plt.savefig(os.path.join(output_dir, "top_pathways.png"), dpi=300)
    plt.close()
    print("Generating Pathway Sources chart...")
    plt.figure(figsize=(8, 8))
    source_counts = pathways['pathway_source'].value_counts()
    if not source_counts.empty:
        plt.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%',
                startangle=140, colors=sns.color_palette("pastel"), wedgeprops={'edgecolor': 'white'})
        plt.title("Distribution of Integrated Pathway Databases", pad=20)
        plt.savefig(os.path.join(output_dir, "pathway_sources.png"), dpi=300)
    plt.close()
    print("Generating Interaction Types chart...")
    plt.figure(figsize=(8, 8))
    type_counts = links['link_type'].value_counts()
    if not type_counts.empty:
        plt.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
                startangle=140, colors=sns.color_palette("Set3"), wedgeprops={'edgecolor': 'white'})
        plt.title("Protein-Chemical Interaction Categories", pad=20)
        plt.savefig(os.path.join(output_dir, "interaction_types.png"), dpi=300)
    plt.close()
    print("Generating GO Molecular Functions chart...")
    plt.figure(figsize=(12, 7))

    def clean_go(x):
        if not x or pd.isna(x): return []
        return [term.split(':', 1)[1] if ':' in term else term for term in str(x).split('; ')]
    all_go_mf = proteins['go_mf'].apply(clean_go).explode().dropna()
    all_go_mf = all_go_mf[all_go_mf != '']
    top_go_mf = all_go_mf.value_counts().head(12)
    if not top_go_mf.empty:
        sns.barplot(x=top_go_mf.values, y=top_go_mf.index, hue=top_go_mf.index, legend=False)
        plt.title("Primary Molecular Functions (GO Analysis)", pad=20)
        plt.xlabel("Frequency across Dataset")
        plt.savefig(os.path.join(output_dir, "top_go_mf.png"), dpi=300)
    plt.close()
    print("Generating GO Biological Processes chart...")
    plt.figure(figsize=(12, 7))
    all_go_bp = proteins['go_bp'].apply(clean_go).explode().dropna()
    all_go_bp = all_go_bp[all_go_bp != '']
    top_go_bp = all_go_bp.value_counts().head(12)
    if not top_go_bp.empty:
        sns.barplot(x=top_go_bp.values, y=top_go_bp.index, hue=top_go_bp.index, legend=False)
        plt.title("Primary Biological Processes (GO Analysis)", pad=20)
        plt.xlabel("Frequency across Dataset")
        plt.savefig(os.path.join(output_dir, "top_go_bp.png"), dpi=300)
    plt.close()
    print("Generating Protein Length Distribution...")
    plt.figure(figsize=(10, 6))
    sns.histplot(proteins['length'].dropna(), kde=True, color='skyblue')
    plt.title("Distribution of Protein Sequence Lengths")
    plt.xlabel("Amino Acid Count")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, "protein_lengths.png"), dpi=300)
    plt.close()
    print("Generating Chemical Mass Distribution...")
    plt.figure(figsize=(10, 6))
    if 'mass' in chemicals.columns:
        mass_data = pd.to_numeric(chemicals['mass'], errors='coerce').dropna()
        if not mass_data.empty:
            sns.histplot(mass_data, kde=True, color='salmon')
            plt.title("Distribution of Compound Molecular Masses")
            plt.xlabel("Mass (Da)")
            plt.ylabel("Frequency")
            plt.savefig(os.path.join(output_dir, "chemical_masses.png"), dpi=300)
    plt.close()
    print("Generating Hub Proteins chart...")
    plt.figure(figsize=(12, 7))
    hub_counts = links['uniprot_id'].value_counts().head(12)
    id_to_gene = proteins.set_index('uniprot_id')['gene_name'].to_dict()
    hub_labels = [f"{id_to_gene.get(uid, uid)} ({uid})" for uid in hub_counts.index]
    if not hub_counts.empty:
        sns.barplot(x=hub_counts.values, y=hub_labels, hue=hub_labels, legend=False)
        plt.title("Network Hubs: Proteins with Highest Connectivity", pad=20)
        plt.xlabel("Number of Small-Molecule Interactions")
        plt.savefig(os.path.join(output_dir, "hub_proteins.png"), dpi=300)
    plt.close()
    print(f"\nSuccess! Total 10 visualizations saved to: {output_dir}")
if __name__ == "__main__":
    generate_charts()
