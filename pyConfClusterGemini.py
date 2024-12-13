import MDAnalysis as mda
from MDAnalysis.analysis import rms, align
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os,math

def cluster_golh_conformations(pdb_file, epsilon=0.5, min_samples=2, output_dir="clusters"):
    """Clusters conformations of GOLH based on RMSD of C1-C18 carbons and writes all cluster members 
    aligned to the first molecule in the cluster."""
    try:
        universe = mda.Universe(pdb_file)
    except Exception as e:
        raise RuntimeError(f"Error loading PDB: {e}")

    golh = universe.select_atoms("resname GOLH")

    if not golh:
        raise ValueError("No GOLH residues found in the PDB.")

    carbon_atoms = golh.select_atoms("name C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 C14 C15 C16 C17 C18")

    if not carbon_atoms:
        raise ValueError("No C1-C18 carbons found in GOLH residues.")

    n_molecules = len(golh.residues)
    n_carbons = len(carbon_atoms) // n_molecules

    all_carbon_positions = carbon_atoms.positions.reshape(n_molecules, n_carbons, 3)
    carbon_atomgroups = [mol.atoms.select_atoms("name C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 C14 C15 C16 C17 C18") for mol in golh.residues]

    rmsd_matrix = np.zeros((n_molecules, n_molecules))

    for i in range(n_molecules):
        for j in range(i + 1, n_molecules):
            rmsd_matrix[i, j] = math.sqrt(rms.rmsd(carbon_atomgroups[i].positions, carbon_atomgroups[j].positions))
            rmsd_matrix[j, i] = rmsd_matrix[i, j]

    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, metric="precomputed")
    clusters = dbscan.fit_predict(rmsd_matrix)

    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    print(f"Found {n_clusters} clusters")

    os.makedirs(output_dir, exist_ok=True)
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        print(f"Cluster {cluster_id}: {len(cluster_indices)} molecules")

        # Use a LIST to store the molecule's AtomGroups
        cluster_molecules = []

        for mol_index in cluster_indices:
            mol_atoms = golh.residues[mol_index].atoms
            cluster_molecules.append(mol_atoms)  # Append the AtomGroup to the list

        if cluster_molecules: #Check if the list is not empty
            # Select the first molecule in the cluster as the reference
            reference_mol = cluster_molecules[0]
            # Align all molecules in the cluster to the reference
            all_cluster_atoms = mda.AtomGroup([], universe)
            all_cluster_atoms += reference_mol

            for mol in cluster_molecules[1:]:
                # rmsd=align.alignto(mol, reference_mol, select='name C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 C14 C15 C16 C17 C18')
            # Write ALL aligned molecules to a single PDB file
                all_cluster_atoms += mol

            with mda.Writer(os.path.join(output_dir, f'cluster_{cluster_id}.pdb'), len(all_cluster_atoms)) as w:
                w.write(all_cluster_atoms)
            exit(1)
    # for cluster_id in range(n_clusters):
    #     cluster_indices = np.where(clusters == cluster_id)[0]
    #     print(f"Cluster {cluster_id}: {len(cluster_indices)} molecules")

    #     cluster_atoms = mda.AtomGroup([], universe)

    #     for mol_index in cluster_indices:
    #         mol_atoms = golh.residues[mol_index].atoms
    #         cluster_atoms = cluster_atoms + mol_atoms
            
        
        
    #     reference_mol=golh.residues[cluster_indices[0]].atoms

    #     # Align all molecules in the cluster to the reference
    #     for mol_index in cluster_indices[1:]:
    #         mol_atoms = golh.residues[mol_index].atoms
    #         align.alignto(mol_atoms, reference_mol, select='all', weights=None)

    #     # Write all aligned molecules in the cluster to a single PDB file
    #     with mda.Writer(os.path.join(output_dir, f'cluster_{cluster_id}.pdb'), cluster_atoms.n_atoms) as w:
    #         w.write(cluster_atoms)

    plt.scatter(np.arange(n_molecules), clusters)
    plt.xlabel("Molecule Index")
    plt.ylabel("Cluster ID")
    plt.savefig(os.path.join(output_dir, "cluster_plot.png"))
    #plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cluster GOLH conformations based on C1-C18 RMSD.")
    parser.add_argument("-f", "--file", required=True, help="Input PDB file")
    parser.add_argument("-e", "--epsilon", type=float, default=0.5, help="DBSCAN epsilon")
    parser.add_argument("-m", "--min_samples", type=int, default=2, help="DBSCAN min_samples")
    parser.add_argument("-o", "--output", default="clusters", help="Output directory")
    args = parser.parse_args()

    try:
        cluster_golh_conformations(args.file, args.epsilon, args.min_samples, args.output)
    except (RuntimeError, ValueError) as e:
        print(f"Error: {e}")