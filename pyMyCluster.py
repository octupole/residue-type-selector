import MDAnalysis as mda
from MDAnalysis.analysis import rms
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math,os


def cluster_golh_conformations(pdb_file, method='ward', criterion='maxclust', 
                             n_clusters=None, distance_threshold=None, output_dir="clusters"):

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
            rmsd_matrix[i, j] = rms.rmsd(carbon_atomgroups[i].positions, carbon_atomgroups[j].positions,  superposition=True)
            rmsd_matrix[j, i] = rmsd_matrix[i, j]
    linked = linkage(rmsd_matrix, method=method)  # 'ward', 'single', 'complete', 'average'

    # 2. Plot the dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(linked, 
                orientation='top',
                distance_sort='descending',
                show_leaf_counts=True)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Molecule Index')
    plt.ylabel('Distance')
    #plt.show()

    # 3. Determine the number of clusters and obtain cluster labels
    if n_clusters is not None:
        # Use a fixed number of clusters
        cluster_labels = fcluster(linked, n_clusters, criterion=criterion) 
    elif distance_threshold is not None:
        # Use a distance threshold
        cluster_labels = fcluster(linked, distance_threshold, criterion='distance')
    else:
        raise ValueError("Either n_clusters or distance_threshold must be provided.")

    n_clusters = len(set(cluster_labels))
    print(f"Found {n_clusters} clusters")

    # 4. Prepare data for 3D plotting (no PCA)
    x = np.arange(rmsd_matrix.shape[0])  # Molecule indices for x-axis
    y = np.arange(rmsd_matrix.shape[1])  # Molecule indices for y-axis
    X, Y = np.meshgrid(x, y)
    Z = rmsd_matrix  # RMSD values for z-axis

    # 5. Visualize clusters in 3D
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Different markers for different clusters
    markers = ['o', '^', 's', 'p', '*', 'D', 'v', '<', '>']  # Add more if needed

    for cluster_id in range(1, n_clusters + 1):  # Start from 1 
        cluster_indices = np.where(cluster_labels == cluster_id)
        ax.scatter(X[cluster_indices], 
                   Y[cluster_indices], 
                   Z[cluster_indices], 
                   label=f"Cluster {cluster_id}", 
                   marker=markers[cluster_id % len(markers)])  # Cycle through markers

    # Set labels and title
    ax.set_xlabel('Molecule i')
    ax.set_ylabel('Molecule j')
    ax.set_zlabel('RMSD')
    ax.set_title('Hierarchical Clustering (3D RMSD Matrix)')
    ax.legend()
    #plt.show()
    n_clusters = len(set(cluster_labels))
    print(f"Found {n_clusters} clusters")

    # 4. Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # 5. Write PDB files for each cluster
    # 5. Write PDB files for each cluster (without MODEL)
    for cluster_id in range(1, n_clusters + 1):  # Cluster labels start from 1
        cluster_members = [mol for i, mol in enumerate(golh.residues) if cluster_labels[i] == cluster_id]
        
        output_filename = os.path.join(output_dir, f"cluster_{cluster_id}.pdb")
        
        with open(output_filename, 'w') as f:  # Open file in write mode
            for molecule in cluster_members:
                for atom in molecule.atoms:
                    # Corrected format for atom line
                    atom_line = f"ATOM {atom.id:>5d}  {atom.name:<4s}{atom.resname:>3s}  {atom.resid:>4d}    " \
                                f"{atom.position[0]:8.3f}{atom.position[1]:8.3f}{atom.position[2]:8.3f}" \
                                f"{1.00:6.2f}{0.00:6.2f}          {atom.element:>2s}  \n"
                    f.write(atom_line)  # Write to file    


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cluster GOLH conformations using hierarchical clustering.")
    parser.add_argument("-f", "--file", required=True, help="Input PDB file")
    parser.add_argument("-m", "--method", default="ward", help="Linkage method (ward, single, complete, average)")
    parser.add_argument("-k", "--n_clusters", type=int, help="Number of clusters (if using maxclust)")
    parser.add_argument("-d", "--distance_threshold", type=float, help="Distance threshold (if using distance)")
    parser.add_argument("-o", "--output", default="clusters", help="Output directory")
    args = parser.parse_args()

    try:
        cluster_golh_conformations(args.file, args.method, 'maxclust', args.n_clusters, args.distance_threshold, args.output)
    except (RuntimeError, ValueError) as e:
        print(f"Error: {e}")

