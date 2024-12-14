import MDAnalysis as mda
from MDAnalysis.analysis import rms
import numpy as np
from sklearn.cluster import DBSCAN  # Import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math, os
from sklearn.neighbors import NearestNeighbors  # Import NearestNeighbors

def cluster_golh_conformations(pdb_file, eps=0.5, min_samples=5, output_dir="clusters"):

    """Clusters conformations of GOLH based on RMSD of C1-C18 carbons using DBSCAN and 
    writes all cluster members to separate PDB files."""
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
            rmsd_matrix[i, j] = rms.rmsd(carbon_atomgroups[i].positions, carbon_atomgroups[j].positions,  superposition=True)
            rmsd_matrix[j, i] = rmsd_matrix[i, j]


    # --- k-distance Graph ---
    neigh = NearestNeighbors(n_neighbors=2)  # 2 neighbors for 2D data (adjust if needed)
    nbrs = neigh.fit(rmsd_matrix)
    distances, indices = nbrs.kneighbors(rmsd_matrix)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.plot(distances)
    plt.title('k-distance Graph')
    plt.xlabel('Data Points sorted by distance')
    plt.ylabel('Epsilon')
    plt.show()
    # --- DBSCAN Clustering ---
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    cluster_labels = dbscan.fit_predict(rmsd_matrix)
    print(cluster_labels)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)  # Exclude noise points
    print(f"Found {n_clusters} clusters")

    # --- 3D Visualization (Optional) ---
    # You can add 3D visualization here if needed, similar to the previous example.

    # --- Write PDB files for each cluster (without MODEL) ---
    os.makedirs(output_dir, exist_ok=True)

    for cluster_id in range(n_clusters):  # DBSCAN labels start from 0
        cluster_members = [mol for i, mol in enumerate(golh.residues) if cluster_labels[i] == cluster_id]
        
        output_filename = os.path.join(output_dir, f"cluster_{cluster_id + 1}.pdb")  # Start file names from 1
        
        with open(output_filename, 'w') as f:  # Open file in write mode
            for molecule in cluster_members:
                for atom in molecule.atoms:
                    # Corrected format for atom line
                    atom_line = f"ATOM  {atom.id:>5d} {atom.name:<4s}{atom.resname:>3s}  {atom.resid:>4d}    " \
                                f"{atom.position[0]:8.3f}{atom.position[1]:8.3f}{atom.position[2]:8.3f}" \
                                f"{1.00:6.2f}{0.00:6.2f}          {atom.element:>2s}  \n"
                    f.write(atom_line)  # Write to file

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cluster GOLH conformations using DBSCAN.")
    parser.add_argument("-f", "--file", required=True, help="Input PDB file")
    parser.add_argument("-e", "--eps", type=float, default=0.5, help="DBSCAN epsilon")
    parser.add_argument("-m", "--min_samples", type=int, default=5, help="DBSCAN min_samples")
    parser.add_argument("-o", "--output", default="clusters", help="Output directory")
    args = parser.parse_args()

    try:
        cluster_golh_conformations(args.file, args.eps, args.min_samples, args.output)
    except (RuntimeError, ValueError) as e:
        print(f"Error: {e}")