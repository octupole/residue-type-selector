import MDAnalysis as mda
from MDAnalysis.analysis import align, rms

import numpy as np
from sklearn.cluster import KMeans
import os
from collections import Counter

SELECTION_C = "name C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 C14 C15 C16 C17 C18"

class ResidueSelector:
    """
    ResidueSelector class for selecting residues based on z-coordinate ranges.

    Attributes:
        universe (MDAnalysis.Universe): The molecular dynamics universe containing the residues.

    Methods:
        select_residues_in_z_range(selection, z_down, z_up):
            Selects residues whose atoms all have z-coordinates within the specified range.

            Parameters:
                selection (MDAnalysis.core.groups.AtomGroup): The group of atoms to select from.
                z_down (float): The lower bound of the z-coordinate range.
                z_up (float): The upper bound of the z-coordinate range.

            Returns:
                MDAnalysis.core.groups.AtomGroup: The atoms of the selected residues.
    Class to handle residue selection based on z-coordinate ranges.
    """
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
    """
    Attributes:
        topology_file (str): Path to the topology file.
        trajectory_file (str): Path to the trajectory file.
        z_up (float): Upper z-coordinate boundary for residue selection.
        z_down (float): Lower z-coordinate boundary for residue selection.
        n_clusters (int): Number of clusters to form. Default is 3.
        output_dir (str): Directory to save cluster output files. Default is "clusters".
        begin (int, optional): Starting frame for trajectory analysis. Default is None.
        end (int, optional): Ending frame for trajectory analysis. Default is None.
        universe (MDAnalysis.Universe): MDAnalysis Universe object for the given topology and trajectory.
        selector (ResidueSelector): ResidueSelector object for selecting residues within z range.
        golh (MDAnalysis.AtomGroup): AtomGroup of selected GOLH/GOLO residues.
        rmsd_matrix (np.ndarray): Average RMSD matrix across frames.
        average_positions (list): List of average positions for each atom group.
        cluster_labels (np.ndarray): Cluster labels for each molecule.
    Methods:
        compute_framewise_rmsd():
        perform_clustering():
            Cluster molecules using KMeans and print cluster information.
        write_cluster_outputs(output_dir='clusters'):
        run():
            Execute the full clustering workflow: compute RMSD, perform clustering, and write outputs.
    Class to handle clustering of GOLH/GOLO molecules based on RMSD.
    """
    def __init__(self, topology_file, trajectory_file, z_up, z_down, n_clusters=3, output_dir="clusters", begin=None, end=None):
        self.topology_file = topology_file
        self.trajectory_file = trajectory_file
        self.z_up = z_up
        self.z_down = z_down
        self.n_clusters = n_clusters
        self.output_dir = output_dir
        self.begin = begin
        self.end = end
        

        try:
            self.universe = mda.Universe(topology_file, trajectory_file)
        except Exception as e:
            raise RuntimeError(f"Error loading topology/trajectory: {e}")

        self.selector = ResidueSelector(self.universe)


    def compute_framewise_rmsd(self):
        """"
        This method performs the following steps:
        1. Initializes storage for RMSD matrices and accumulated positions.
        2. Selects GOLH or GOLO residues within a specified z-range.
        3. Initializes reference positions and centers of mass for carbon atom groups.
        4. Iterates over each frame in the trajectory:
            a. Selects molecules dynamically per frame.
            b. Translates and rotates mobile positions to align with reference positions.
            c. Accumulates positions for averaging.
            d. Computes the RMSD matrix for the current frame.
        5. Aggregates RMSD matrices across all frames.
        6. Computes the average RMSD matrix and average positions.
        Attributes:
            rmsd_matrix (np.ndarray): The average RMSD matrix across all frames.
            average_positions (list): The average positions of atoms across all frames.
        
        Compute RMSD matrices for each frame and aggregate data across frames.
        """
        rmsd_matrices = []
        carbons=[]
        N=0
        golh=0
        carbon_atomgroups=0
        accumulated_positions =0
        self.universe.trajectory[self.begin]  # Set to the first frame
        golh = self.universe.select_atoms("resname GOLH or resname GOLO")
        golh = self.selector.select_residues_in_z_range(golh, self.z_down, self.z_up)
        self.golh=golh
        carbon_atomgroups = [mol.atoms.select_atoms(SELECTION_C) for mol in golh.residues]
        n_atoms_per_group = [len(ag) for ag in carbon_atomgroups]

        # Initialize storage for accumulated positions
        accumulated_positions = [np.zeros((n_atoms, 3)) for n_atoms in n_atoms_per_group]

        ref_positions = [ag.positions.copy() for ag in carbon_atomgroups]
        ref_com=[ag.center_of_mass() for ag in carbon_atomgroups]
        for n, positions in enumerate(ref_positions):
            for m, pos in enumerate(positions):
                ref_positions[n][m] = pos-ref_com[n]
            ref_com[n]=np.zeros(3)

        for ts in self.universe.trajectory[self.begin:self.end+1]:
            # Select molecules dynamically per frame
            print(f"Processing step: {ts.frame}, timestep: {ts.time} fs")
            mobile_positions=[ag.positions.copy() for ag in carbon_atomgroups]
            mobile_com=[ag.center_of_mass() for ag in carbon_atomgroups]
            
            n_molecules = len(golh.residues)
            if n_molecules == 0:
                continue  # Skip frames with no molecules in the z range

            for m,ag in enumerate(ref_com):
                translation_vector = ref_com[m] - mobile_com[m]

                mobile_positions[m] += translation_vector

            # 2. Rotation using align.rotation_matrix
                rotation_matrix, rmsd = align.rotation_matrix(mobile_positions[m], ref_positions[m])
                mobile_positions[m] = mobile_positions[m] @ rotation_matrix.T

                
            for n, pos in enumerate(mobile_positions):
                accumulated_positions[n] += pos
                            
            # Compute RMSD matrix for this frame
            rmsd_matrix = np.zeros((n_molecules, n_molecules))
            for i in range(n_molecules):
                for j in range(i + 1, n_molecules):
                    rmsd_matrix[i, j] = rms.rmsd(
                        mobile_positions[i],
                        mobile_positions[j],
                        center=True,
                        superposition=True
                    )
                    rmsd_matrix[j, i] = rmsd_matrix[i, j]
            rmsd_matrices.append(rmsd_matrix)
    

            # Store frame-specific RMSD matrix and residue information
            N+=1
        self.rmsd_matrix = np.mean(rmsd_matrices, axis=0)
        self.average_positions = [pos / float(N) for pos in accumulated_positions]
        
    def perform_clustering(self):
        """
        Perform K-means clustering on the RMSD matrix and print the results.

        This method uses the KMeans algorithm from scikit-learn to cluster the RMSD matrix
        into a specified number of clusters. It then prints the number of clusters found
        and the counts of specific residue names ('GOLO' and 'GOLH') within each cluster.

        Attributes:
            n_clusters (int): The number of clusters to form.
            rmsd_matrix (ndarray): The RMSD matrix used for clustering.
            golh (object): An object containing residue information.
            cluster_labels (ndarray): The labels of the clusters assigned to each residue.

        Prints:
            The number of clusters found and the counts of 'GOLO' and 'GOLH' residues in each cluster.
        """
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        self.cluster_labels = kmeans.fit_predict(self.rmsd_matrix)
        print(f"Found {self.n_clusters} clusters")
        for cluster_id in range(self.n_clusters):
            cluster_resname = [mol.resname for i, mol in enumerate(self.golh.residues) if self.cluster_labels[i] == cluster_id]
            counts = Counter(cluster_resname)
            print(f"Cluster {cluster_id+1}: TOT: {counts['GOLO']+counts['GOLH']}, GOLH: {counts['GOLH']}, GOLO: {counts['GOLO']}") 
        

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



    def run(self):
        self.compute_framewise_rmsd()
        self.perform_clustering()
        self.write_cluster_outputs()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cluster GOLH conformations using KMeans.")
    parser.add_argument("-t", "--topology", required=True, help="Input topology file (e.g., .tpr)")
    parser.add_argument("-x", "--trajectory", required=True, help="Input trajectory file (e.g., .xtc or .trr)")
    parser.add_argument("-k", "--n_clusters", type=int, default=3, help="Number of clusters for KMeans")
    parser.add_argument("-o", "--output", default="clusters", help="Output directory")
    parser.add_argument("-u", "--zup", type=float, default=60.0, help="Upper limit for the membrane")
    parser.add_argument("-d", "--zdown", type=float, default=19.0, help="Lower limit for the membrane")
    parser.add_argument("-b", "--begin", type=int, default=None, help="Start frame for trajectory processing")
    parser.add_argument("-e", "--end", type=int, default=None, help="End frame for trajectory processing")

    args = parser.parse_args()

    clusterer = GOLHClusterer(
        topology_file=args.topology,
        trajectory_file=args.trajectory,
        z_up=args.zup,
        z_down=args.zdown,
        n_clusters=args.n_clusters,
        output_dir=args.output,
        begin=args.begin,
        end=args.end
    )

    try:
        clusterer.run()
    except (RuntimeError, ValueError) as e:
        print(f"Error: {e}")

