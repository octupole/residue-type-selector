import MDAnalysis as mda
""""
This module provides classes and methods to perform clustering on molecular dynamics trajectories based on RMSD (Root Mean Square Deviation) of selected residues.
Classes:
    ResidueSelector:
        A class to select residues within a specified z-coordinate range.
            __init__(universe):
                Initializes the ResidueSelector with a given MDAnalysis Universe.
            select_residues_in_z_range(selection, z_down, z_up):
                Selects residues within a specified z-coordinate range.
    GOLHClusterer:
        A class to perform clustering on molecular dynamics trajectories based on RMSD of selected residues.
            __init__(topology_file, trajectory_file, z_up, z_down, n_clusters=3, output_dir="clusters", begin=None, end=None):
                Initializes the GOLHClusterer with the given parameters.
                Computes the RMSD matrices for each frame in the trajectory.
                Performs K-means clustering on the RMSD matrix and prints cluster information.
Usage:
    The script can be executed as a standalone program with command-line arguments to specify the input topology and trajectory files, output directory, z-coordinate range, and frame range for analysis.
Example:
    python pyKmeanTrajectory_aimod.py -t topology.tpr -x trajectory.xtc -o clusters -u 60.0 -d 19.0 -b 0 -e 1000
"""
from MDAnalysis.analysis import align, rms
from MDAnalysis.transformations import unwrap
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
import os
from collections import Counter
import warnings
import torch
import cupy as cp

SELECTION_C = "name C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 C14 C15 C16 C17 C18"
#SELECTION_C = "name C1 C3 C5 C7 C9 C11 C13 C15 C17 C18"

dtype = torch.float
device = torch.device("cuda:0")

def rmsd_torch(u, v):
    u = u.view(-1, 3)
    v = v.view(-1, 3)
    u_centered = u - u.mean(dim=0)
    v_centered = v - v.mean(dim=0)
    cov_matrix = torch.mm(u_centered.T, v_centered)
    u_rotated = torch.mm(u_centered, torch.svd(cov_matrix).U)
    return torch.sqrt(((u_rotated - v_centered) ** 2).sum() / len(u))


def rmsd_cupy(u, v):
    u = u.reshape(-1, 3)
    v = v.reshape(-1, 3)
    u_centered = u - u.mean(axis=0)
    v_centered = v - v.mean(axis=0)
    rotation_matrix = cp.linalg.svd(cp.dot(u_centered.T, v_centered))[0]
    u_rotated = cp.dot(u_centered, rotation_matrix.T)
    return cp.sqrt(cp.mean(cp.sum((u_rotated - v_centered) ** 2, axis=1)))




class ResidueSelector:
    def __init__(self, universe):
        self.universe = universe

    def select_residues_in_z_range(self, selection, z_down, z_up):
        z_down = float(z_down)
        z_up = float(z_up)

        selected_residues = [
            res.resindex for res in selection.residues
            if all(z_down < atom.position[2] < z_up for atom in res.atoms)
        ]

        return self.universe.residues[selected_residues].atoms

class GOLHClusterer:
    """"
    A class to perform clustering on molecular dynamics trajectories based on RMSD (Root Mean Square Deviation) of selected residues.
    Attributes:
        topology_file (str): Path to the topology file.
        trajectory_file (str): Path to the trajectory file.
        z_up (float): Upper bound of the z-coordinate range for residue selection.
        z_down (float): Lower bound of the z-coordinate range for residue selection.
        n_clusters (int): Number of clusters to form. Default is 3.
        output_dir (str): Directory to save cluster output files. Default is "clusters".
        begin (int): Starting frame for analysis. Default is None.
        end (int): Ending frame for analysis. Default is None.
    Methods:
        validate_files():
            Validates the existence of topology and trajectory files.
        validate_frame_range():
            Validates and sets the frame range for analysis.
        compute_framewise_rmsd():
            Computes the framewise RMSD for selected residues and accumulates positions.
        write_cluster_outputs(output_dir='clusters'):
            Writes the cluster outputs to the specified directory.
        optimal_clusters(max_clusters=10):
            Determines the optimal number of clusters using silhouette scores.
        perform_clustering():
            Performs k-means clustering on the RMSD matrix and prints cluster information.
        run():
            Executes the full clustering workflow: RMSD computation, optimal cluster determination, clustering, and output writing.
    """
    def __init__(self, topology_file, trajectory_file, z_up, z_down, n_clusters=3, output_dir="clusters", begin=None, end=None, step=1):
        """
        Initialize the KMeans clustering on a molecular dynamics trajectory.

        Parameters:
        topology_file (str): Path to the topology file.
        trajectory_file (str): Path to the trajectory file.
        z_up (float): Upper z-coordinate boundary for residue selection.
        z_down (float): Lower z-coordinate boundary for residue selection.
        n_clusters (int, optional): Number of clusters to form. Default is 3.
        output_dir (str, optional): Directory to save the clustering results. Default is "clusters".
        begin (int, optional): Starting frame for the trajectory analysis. Default is None.
        end (int, optional): Ending frame for the trajectory analysis. Default is None.

        Raises:
        RuntimeError: If there is an error loading the topology or trajectory files.
        """
        self.topology_file = topology_file
        self.trajectory_file = trajectory_file
        self.z_up = z_up
        self.z_down = z_down
        self.n_clusters = n_clusters
        self.output_dir = output_dir
        self.begin = begin
        self.end = end
        self.step = step

        self.validate_files()

        try:
            self.universe = mda.Universe(topology_file, trajectory_file)
        except Exception as e:
            raise RuntimeError(f"Error loading topology/trajectory: {e}")

        self.selector = ResidueSelector(self.universe)
        self.validate_frame_range()
        self.device=torch.device("cuda:0")
        self.dtype = torch.float

    def validate_files(self):
        """
        Validates the existence of the topology and trajectory files.

        Raises:
            FileNotFoundError: If the topology file does not exist.
            FileNotFoundError: If the trajectory file does not exist.
        """
        if not os.path.exists(self.topology_file):
            raise FileNotFoundError(f"Topology file '{self.topology_file}' does not exist.")
        if not os.path.exists(self.trajectory_file):
            raise FileNotFoundError(f"Trajectory file '{self.trajectory_file}' does not exist.")

    def validate_frame_range(self):
        """
        Validates and sets the frame range for the trajectory.

        This method ensures that the `begin` and `end` attributes are within the
        valid range of the trajectory frames. If `begin` or `end` are not set,
        they are initialized to the start and end of the trajectory, respectively.
        If the frame range is out of bounds, a ValueError is raised.

        Raises:
            ValueError: If the frame range is out of bounds.
        """
        if self.begin is None:
            self.begin = 0
        if self.end is None:
            self.end = len(self.universe.trajectory) - 1

        if self.begin < 0 or self.end >= len(self.universe.trajectory):
            raise ValueError("Frame range is out of bounds.")

    def compute_framewise_rmsd(self):
        """
        Compute the Root Mean Square Deviation (RMSD) for each frame in the trajectory.

        This method calculates the RMSD matrices for each frame in the trajectory between
        the specified `begin` and `end` frames. It selects atoms of residues with resname
        "GOLH" or "GOLO" within a specified z-range, aligns them to a reference frame, and
        computes the RMSD for each frame.

        The computed RMSD matrices are stored in `self.rmsd_matrix`, and the average positions
        of the atoms over all frames are stored in `self.average_positions`.

        Attributes:
            rmsd_matrix (np.ndarray): The average RMSD matrix over all frames.
            average_positions (list of np.ndarray): The average positions of the atoms over all frames.

        Returns:
            None
        """
        rmsd_matrices = []
        accumulated_positions = []

        self.universe.trajectory[self.begin]  # Set to the first frame
        golh = self.universe.select_atoms("resname GOLH or resname GOLO")
        golh = self.selector.select_residues_in_z_range(golh, self.z_down, self.z_up)
        self.golh = golh

        carbon_atomgroups = [mol.atoms.select_atoms(SELECTION_C) for mol in golh.residues]
        n_atoms_per_group = [len(ag) for ag in carbon_atomgroups]

        accumulated_positions = [np.zeros((n_atoms, 3)) for n_atoms in n_atoms_per_group]

        ref_positions = [ag.positions.copy() for ag in carbon_atomgroups]
        ref_com = [ag.center_of_mass() for ag in carbon_atomgroups]

        for n, positions in enumerate(ref_positions):
            ref_positions[n] -= ref_com[n]
            ref_com[n] = np.zeros(3)
        self.universe.trajectory.add_transformations(unwrap(self.universe.atoms))
        for ts in self.universe.trajectory[self.begin:self.end + 1:self.step]:
            print(f"Processing step: {ts.frame}, timestep: {ts.time} fs")
            mobile_positions = [ag.positions.copy() for ag in carbon_atomgroups]
            mobile_com = [ag.center_of_mass() for ag in carbon_atomgroups]

            n_molecules = len(golh.residues)
            if n_molecules == 0:
                continue
            for m, ag in enumerate(ref_com):
                translation_vector = ref_com[m] - mobile_com[m]
                mobile_positions[m] += translation_vector

                rot_matrix, _ = align.rotation_matrix(mobile_positions[m], ref_positions[m])
                mobile_positions[m] = mobile_positions[m] @ rot_matrix.T

            for n, pos in enumerate(mobile_positions):
                accumulated_positions[n] += pos
            # Compute RMSD matrix for this frame

                
            rmsd_matrix = squareform(pdist(
                np.array([pos.flatten() for pos in mobile_positions]),
                metric=lambda u, v: rms.rmsd(u.reshape(-1, 3), v.reshape(-1, 3), center=True, superposition=True)
            ))
            rmsd_matrices.append(rmsd_matrix)
        self.rmsd_matrix = np.mean(rmsd_matrices, axis=0)
        self.average_positions = [pos / float(len(rmsd_matrices)) for pos in accumulated_positions]
        
    def write_cluster_outputs(self, output_dir='clusters'):
        """
        Write the cluster outputs to the specified directory.
        This method creates a directory for cluster outputs, selects carbon atoms from the residues,
        and processes each cluster to align and translate the molecules. It then calculates the average
        structure for each cluster and writes it to a PDB file.
        Parameters:
        output_dir (str): The directory where the cluster output files will be saved. Default is 'clusters'.
        Returns:
        None
        """
        os.makedirs(self.output_dir, exist_ok=True)
        carbon_atomgroups = self.golh.residues[0].atoms.select_atoms(SELECTION_C)
        
        for cluster_id in range(self.n_clusters):
            cluster_indices = [i for i, mol in enumerate(self.golh.residues) if self.cluster_labels[i] == cluster_id]
            cluster_members = [mol for i, mol in enumerate(self.golh.residues) if self.cluster_labels[i] == cluster_id]
            
            ref_carbon_atoms = cluster_members[0].atoms.select_atoms(SELECTION_C)  # Select atoms of the reference molecule
            ref_carbon_atoms.positions=self.average_positions[cluster_indices[0]]

            for i, molecule in enumerate(cluster_members):
                if i == 0:
                    continue
                mobile_carbon_atoms = molecule.atoms.select_atoms(SELECTION_C)
                mobile_carbon_atoms.positions=self.average_positions[cluster_indices[i]]
                
                
                # 1. Translation to overlap centers of mass
                translation_vector = ref_carbon_atoms.center_of_mass() - mobile_carbon_atoms.center_of_mass()
                molecule.atoms.translate(translation_vector)

                # 2. Rotation using align.rotation_matrix
                rotation_matrix, rmsd = align.rotation_matrix(mobile_carbon_atoms.positions, ref_carbon_atoms.positions)
                molecule.atoms.rotate(rotation_matrix)

        # --- Calculate and Write Average Structure ---
            avg_positions = np.mean([mol.atoms.select_atoms(SELECTION_C).positions for mol in cluster_members], axis=0)
            avg_universe = mda.Merge(cluster_members[0].atoms.select_atoms(SELECTION_C))  # Use the first molecule as a template
            avg_universe.atoms.positions = avg_positions

            avg_output_filename = os.path.join(output_dir, f"cluster_{cluster_id + 1}_avg.pdb")

            with open(avg_output_filename, 'w') as f:
                for atom in avg_universe.atoms:
                    atom_line = f"ATOM  {atom.id:>5d}  {atom.name:<4s}{atom.resname:>3s} {atom.resid:>4d}    " \
                                f"{atom.position[0]:8.3f}{atom.position[1]:8.3f}{atom.position[2]:8.3f}" \
                                f"{1.00:6.2f}{0.00:6.2f}          {atom.element:>2s}  \n"
                    f.write(atom_line)
                # Write CONECT records for linear chain
                for i in range(1, len(avg_universe.atoms)):
                    f.write(f"CONECT{i:>5d}{i+1:>5d}\n")  # Connect i to i+1
                    if i > 1:  # For atoms after the first, connect to the previous atom as well
                        f.write(f"CONECT{i+1:>5d}{i:>5d}\n")  # Connect i+1 to i


    def optimal_clusters(self, max_clusters=10):
        """
        Determine the optimal number of clusters for KMeans clustering using the silhouette score.

        Parameters:
        max_clusters (int): The maximum number of clusters to test. Default is 10.

        Returns:
        None: The method sets the optimal number of clusters (n_clusters) as an attribute of the class.

        The method iterates over a range of cluster numbers from 2 to max_clusters, performs KMeans clustering
        for each number of clusters, and calculates the silhouette score for each clustering. The number of clusters
        that yields the highest silhouette score is set as the optimal number of clusters (n_clusters).
        """
        scores = []
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=0).fit(self.rmsd_matrix)
            scores.append(
                silhouette_score(self.rmsd_matrix, kmeans.labels_)
            )
        self.n_clusters = np.argmax(scores) + 2

    def perform_clustering(self):
        """
        Perform K-means clustering on the RMSD matrix and print the cluster composition.

        This method applies the K-means clustering algorithm to the RMSD (Root Mean Square Deviation) matrix
        to partition the data into a specified number of clusters. It then prints the number of clusters found
        and the composition of each cluster in terms of residue names.

        Attributes:
            n_clusters (int): The number of clusters to form.
            rmsd_matrix (ndarray): The RMSD matrix used for clustering.
            golh (object): An object containing residue information.
            cluster_labels (ndarray): The labels of the clusters assigned to each residue.

        Prints:
            The number of clusters found and the composition of each cluster, including the total count of 
            'GOLO' and 'GOLH' residues, as well as their individual counts.
        """
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        self.cluster_labels = kmeans.fit_predict(self.rmsd_matrix)
        print(f"Found {self.n_clusters} clusters")
        for cluster_id in range(self.n_clusters):
            cluster_resname = [mol.resname for i, mol in enumerate(self.golh.residues) if self.cluster_labels[i] == cluster_id]
            counts = Counter(cluster_resname)
            print(f"Cluster {cluster_id+1}: TOT: {counts['GOLO']+counts['GOLH']}, GOLH: {counts['GOLH']}, GOLO: {counts['GOLO']}")

    def run(self):
        """
        Executes the sequence of methods to perform clustering on trajectory data.

        This method performs the following steps:
        1. Computes the frame-wise RMSD (Root Mean Square Deviation).
        2. Determines the optimal number of clusters.
        3. Performs clustering on the trajectory data.
        4. Writes the clustering results to output files.
        """
        self.compute_framewise_rmsd()
        self.optimal_clusters()
        self.perform_clustering()
        self.write_cluster_outputs()

if __name__ == "__main__":
    import argparse
# Suppress specific warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    parser = argparse.ArgumentParser(description="Cluster GOLH conformations using KMeans.")
    parser.add_argument("-t", "--topology", required=True, help="Input topology file (e.g., .tpr)")
    parser.add_argument("-x", "--trajectory", required=True, help="Input trajectory file (e.g., .xtc or .trr)")
    parser.add_argument("-o", "--output", default="clusters", help="Output directory")
    parser.add_argument("-u", "--zup", type=float, default=60.0, help="Upper limit for the membrane")
    parser.add_argument("-d", "--zdown", type=float, default=19.0, help="Lower limit for the membrane")
    parser.add_argument("-b", "--begin", type=int, default=None, help="Start frame for trajectory processing")
    parser.add_argument("-e", "--end", type=int, default=None, help="End frame for trajectory processing")
    parser.add_argument("-dt", "--step", type=int, default=1, help="End frame for trajectory processing")

    args = parser.parse_args()

    clusterer = GOLHClusterer(
        topology_file=args.topology,
        trajectory_file=args.trajectory,
        z_up=args.zup,
        z_down=args.zdown,
        output_dir=args.output,
        begin=args.begin,
        end=args.end,
        step=args.step
    )

    try:
        clusterer.run()
    except (RuntimeError, ValueError) as e:
        print(f"Error: {e}")
