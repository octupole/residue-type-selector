import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN
import MDAnalysis as mda
import argparse

class MoleculeClustering:
    def __init__(self, topology, trajectory, resname, eps=2.0, min_samples=2, grid_size=10):
        self.topology = topology
        self.trajectory = trajectory
        self.resname = resname
        self.eps = eps
        self.min_samples = min_samples
        self.grid_size = grid_size
        self.universe = mda.Universe(topology, trajectory)
        self.residues = self.universe.select_atoms(f"resname {resname}").residues

    @staticmethod
    def compute_rmsd(positions1, positions2):
        """Compute the RMSD between two sets of positions."""
        diff = positions1 - positions2
        return np.sqrt((diff ** 2).sum() / len(positions1))

    def minimize_rmsd(self, positions1, positions2):
        """Minimize RMSD by varying rigid translations and rotations."""
        com1 = positions1.mean(axis=0)
        com2 = positions2.mean(axis=0)
        positions2_centered = positions2 - com2

        min_rmsd = float("inf")
        best_rotation = None
        best_translation = None

        angles = np.linspace(0, 2 * np.pi, self.grid_size)
        for alpha in angles:
            for beta in angles:
                for gamma in angles:
                    rot = R.from_euler("xyz", [alpha, beta, gamma]).as_matrix()
                    rotated_positions2 = positions2_centered @ rot.T
                    rmsd = self.compute_rmsd(positions1, rotated_positions2 + com1)
                    if rmsd < min_rmsd:
                        min_rmsd = rmsd
                        best_rotation = rot
                        best_translation = com1 - com2 @ rot.T

        return min_rmsd, best_rotation, best_translation

    def compute_distance_matrix(self):
        """Compute the distance matrix for all residues."""
        n_residues = len(self.residues)
        distance_matrix = np.zeros((n_residues, n_residues))

        for i in range(n_residues):
            positions1 = self.residues[i].atoms.positions
            for j in range(i + 1, n_residues):
                positions2 = self.residues[j].atoms.positions
                min_rmsd, _, _ = self.minimize_rmsd(positions1, positions2)
                distance_matrix[i, j] = distance_matrix[j, i] = min_rmsd

        return distance_matrix

    def cluster_molecules(self, distance_matrix):
        """Cluster molecules using DBSCAN based on the distance matrix."""
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="precomputed").fit(distance_matrix)
        return clustering.labels_

    def compute_average_conformations(self, labels):
        """Compute the average conformation for each cluster."""
        clusters = {}
        for i, label in enumerate(labels):
            if label == -1:
                continue  # Skip noise
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(self.residues[i].atoms.positions)

        average_conformations = {}
        for label, positions_list in clusters.items():
            average_positions = np.mean(np.array(positions_list), axis=0)
            average_conformations[label] = average_positions

        return clusters, average_conformations

    def run(self):
        print("Computing distance matrix...")
        distance_matrix = self.compute_distance_matrix()

        print("Performing clustering...")
        labels = self.cluster_molecules(distance_matrix)

        print("Computing average conformations...")
        clusters, average_conformations = self.compute_average_conformations(labels)

        for label, positions_list in clusters.items():
            print(f"Cluster {label}: {len(positions_list)} molecules")
            print(f"Average conformation (first few atoms):\n{average_conformations[label][:5]}")

        print("Clustering completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster molecular conformations based on structural similarity.")
    parser.add_argument("-t", "--topology", required=True, help="Path to the topology file (e.g., .tpr).")
    parser.add_argument("-x", "--trajectory", required=True, help="Path to the trajectory file (e.g., .xtc).")
    parser.add_argument("-r", "--resname", required=True, help="Residue name of the molecules to cluster (e.g., SDS).")
    parser.add_argument("--eps", type=float, default=2.0, help="DBSCAN eps parameter (default: 2.0).")
    parser.add_argument("--min_samples", type=int, default=2, help="DBSCAN min_samples parameter (default: 2).")
    parser.add_argument("--grid_size", type=int, default=10, help="Grid size for rotation minimization (default: 10).")
    args = parser.parse_args()

    clustering = MoleculeClustering(
        topology=args.topology,
        trajectory=args.trajectory,
        resname=args.resname,
        eps=args.eps,
        min_samples=args.min_samples,
        grid_size=args.grid_size
    )
    clustering.run()
    