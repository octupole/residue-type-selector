import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
from sklearn.cluster import DBSCAN
import numpy as np
import argparse
from collections import defaultdict

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Compute the number of SDS clusters in a trajectory using MDAnalysis.")
    parser.add_argument("-t", "--topology", required=True, help="Path to the topology file (e.g., .tpr).")
    parser.add_argument("-x", "--trajectory", required=True, help="Path to the trajectory file (e.g., .xtc).")
    parser.add_argument("--eps", type=float, default=4.5, help="DBSCAN eps parameter (distance cutoff for clustering).")
    parser.add_argument("--min_samples", type=int, default=2, help="DBSCAN min_samples parameter (minimum points to form a cluster).")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in femtoseconds for analysis (default: 0.0 fs).")
    parser.add_argument("--end", type=float, default=None, help="End time in femtoseconds for analysis (default: last frame).")
    parser.add_argument("--pymol", action="store_true", help="Output PyMOL selection definitions for clusters.")
    parser.add_argument("--nclust", action="store_true", help="Output the number of SDS atoms in each cluster.")
    args = parser.parse_args()

    # Load the trajectory
    u = mda.Universe(args.topology, args.trajectory)

    # Convert time (in femtoseconds) to frame indices
    dt = u.trajectory.dt  # Time step in femtoseconds
    start_frame = int(args.start / dt)
    end_frame = int(args.end / dt) if args.end is not None else None

    # Adjust the end frame to be inclusive
    if end_frame is not None:
        end_frame += 1  # Make the range inclusive for the end time

    # Select all non-hydrogen atoms in SDS residues
    non_hydrogen_atoms = u.select_atoms("resname SDS and not name H*")

    # Create a list to store the number of clusters over the trajectory
    num_clusters_per_frame = []

    # Loop through the specified frame range
    for ts in u.trajectory[start_frame:end_frame]:
        positions = non_hydrogen_atoms.positions

        # Compute the pairwise distance matrix with PBC
        dist_matrix = distance_array(positions, positions, box=ts.dimensions)

        # Use DBSCAN clustering
        db = DBSCAN(eps=args.eps, min_samples=args.min_samples, metric="precomputed").fit(dist_matrix)

        # Cluster labels
        labels = db.labels_

        # Count clusters (ignoring noise points with label -1)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        num_clusters_per_frame.append(num_clusters)

        # Map atoms in non_hydrogen_atoms to their residues
        atom_to_residue = {i: atom.resid for i, atom in enumerate(non_hydrogen_atoms)}

        # Group residues by cluster and count atoms
        cluster_to_residues = defaultdict(set)
        cluster_to_atom_count = defaultdict(int)
        for atom_index, cluster_label in enumerate(labels):
            if cluster_label != -1:  # Ignore noise
                cluster_to_residues[cluster_label].add(atom_to_residue[atom_index])
                cluster_to_atom_count[cluster_label] += 1

        # Sort clusters by size (number of residues in each cluster)
        sorted_clusters = sorted(cluster_to_residues.items(), key=lambda x: len(x[1]))

        # Print the number of SDS molecules in each cluster
        print(f"Time {ts.time:.2f} fs: Number of SDS clusters = {num_clusters}")
        if args.nclust:
            for cluster_label, residues in sorted_clusters:
                residues_sorted = sorted(residues)
                print(f"  Cluster {cluster_label}: {len(residues)} SDS molecules")
                if args.pymol:
                    selection_definition = " or ".join(f"resi {resid}" for resid in residues_sorted)
                    print(f"    Time {ts.time:.2f} fs: PyMOL Selection: {selection_definition}")

    # Post-processing
    print(f"Average number of clusters over trajectory: {np.mean(num_clusters_per_frame)}")

if __name__ == "__main__":
    main()
