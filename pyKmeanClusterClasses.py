import MDAnalysis as mda
from MDAnalysis.analysis import align, rms
import numpy as np
from sklearn.cluster import KMeans
import os
from collections import Counter

SELECTION_C = "name C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 C14 C15 C16 C17 C18"

class ResidueSelector:
    """
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
    Class to handle clustering of GOLH/GOLO molecules based on RMSD.
    """
    def __init__(self, pdb_file, z_up, z_down, n_clusters=3, output_dir="clusters"):
        self.pdb_file = pdb_file
        self.z_up = z_up
        self.z_down = z_down
        self.n_clusters = n_clusters
        self.output_dir = output_dir

        try:
            self.universe = mda.Universe(pdb_file)
        except Exception as e:
            raise RuntimeError(f"Error loading PDB: {e}")

        self.selector = ResidueSelector(self.universe)

    def prepare_data(self):
        """Select and filter residues."""
        golh = self.universe.select_atoms("resname GOLH or resname GOLO")

        if not golh:
            raise ValueError("No GOLH residues found in the PDB.")

        self.golh = self.selector.select_residues_in_z_range(golh, self.z_down, self.z_up)

        self.carbon_atoms = self.golh.select_atoms(SELECTION_C)
        if not self.carbon_atoms:
            raise ValueError("No C1-C18 carbons found in GOLH residues.")

        self.n_molecules = len(self.golh.residues)
        self.n_carbons = len(self.carbon_atoms) // self.n_molecules
        self.carbon_atomgroups = [
            mol.atoms.select_atoms(SELECTION_C) for mol in self.golh.residues
        ]

    def compute_rmsd_matrix(self):
        """Compute pairwise RMSD matrix."""
        rmsd_matrix = np.zeros((self.n_molecules, self.n_molecules))

        for i in range(self.n_molecules):
            for j in range(i + 1, self.n_molecules):
                rmsd_matrix[i, j] = rms.rmsd(
                    self.carbon_atomgroups[i].positions,
                    self.carbon_atomgroups[j].positions,
                    superposition=True
                )
                rmsd_matrix[j, i] = rmsd_matrix[i, j]

        self.rmsd_matrix = rmsd_matrix

    def perform_clustering(self):
        """Cluster molecules using KMeans."""
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        self.cluster_labels = kmeans.fit_predict(self.rmsd_matrix)
        print(f"Found {self.n_clusters} clusters")

        for cluster_id in range(self.n_clusters):
            cluster_resname = [mol.resname for i, mol in enumerate(self.golh.residues) if self.cluster_labels[i] == cluster_id]
            counts = Counter(cluster_resname)

            print(f"Cluster {cluster_id}: TOT: {counts['GOLO']+counts['GOLH']}, GOLH: {counts['GOLH']}, GOLO: {counts['GOLO']}") 
        

    def write_cluster_outputs(self):
        """
        Align and write clustered molecules to PDB files, and calculate/write average structures.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        for cluster_id in range(self.n_clusters):
            cluster_members = [
                mol for i, mol in enumerate(self.golh.residues)
                if self.cluster_labels[i] == cluster_id
            ]

            # --- Transformation ---
            ref_molecule = cluster_members[0].atoms
            ref_carbon_atoms = ref_molecule.select_atoms(SELECTION_C)

            for molecule in cluster_members:
                mobile_carbon_atoms = molecule.atoms.select_atoms(SELECTION_C)
                translation_vector = ref_carbon_atoms.center_of_mass() - mobile_carbon_atoms.center_of_mass()
                molecule.atoms.translate(translation_vector)

                rotation_matrix, _ = align.rotation_matrix(
                    mobile_carbon_atoms.positions, ref_carbon_atoms.positions
                )
                molecule.atoms.rotate(rotation_matrix)

            # --- Write PDB file for the cluster ---
            output_filename = os.path.join(self.output_dir, f"cluster_{cluster_id + 1}.pdb")
            with open(output_filename, 'w') as f:
                for molecule in cluster_members:
                    for atom in molecule.atoms:
                        atom_line = (
                            f"ATOM  {atom.id:>5d}  {atom.name:<4s}{atom.resname:>3s} {atom.resid:>4d}    "
                            f"{atom.position[0]:8.3f}{atom.position[1]:8.3f}{atom.position[2]:8.3f}"
                            f"{1.00:6.2f}{0.00:6.2f}          {atom.element:>2s}  \n"
                        )
                        f.write(atom_line)
                    f.write("ENDMDL\n")

            # --- Calculate and Write Average Structure ---
            avg_positions = np.mean(
                [mol.atoms.select_atoms(SELECTION_C).positions for mol in cluster_members], axis=0
            )
            avg_universe = mda.Merge(cluster_members[0].atoms.select_atoms(SELECTION_C))
            avg_universe.dimensions=[100.0,100.0,100.0, 90.0, 90.0, 90.0]
            avg_universe.atoms.positions = avg_positions

            avg_output_filename = os.path.join(self.output_dir, f"cluster_{cluster_id + 1}_avg.pdb")
            avg_universe.atoms.write(avg_output_filename)

    def write_cluster_outputs_old(self):
        """Align and write clustered molecules to PDB files."""
        os.makedirs(self.output_dir, exist_ok=True)

        for cluster_id in range(self.n_clusters):
            cluster_members = [
                mol for i, mol in enumerate(self.golh.residues)
                if self.cluster_labels[i] == cluster_id
            ]

            ref_molecule = cluster_members[0].atoms
            ref_carbon_atoms = ref_molecule.select_atoms(SELECTION_C)

            for molecule in cluster_members:
                mobile_carbon_atoms = molecule.atoms.select_atoms(SELECTION_C)
                translation_vector = ref_carbon_atoms.center_of_mass() - mobile_carbon_atoms.center_of_mass()
                molecule.atoms.translate(translation_vector)

                rotation_matrix, _ = align.rotation_matrix(
                    mobile_carbon_atoms.positions, ref_carbon_atoms.positions
                )
                molecule.atoms.rotate(rotation_matrix)

            output_filename = os.path.join(self.output_dir, f"cluster_{cluster_id + 1}.pdb")
            with open(output_filename, 'w') as f:
                for molecule in cluster_members:
                    for atom in molecule.atoms:
                        atom_line = (
                            f"ATOM  {atom.id:>5d}  {atom.name:<4s}{atom.resname:>3s} {atom.resid:>4d}    "
                            f"{atom.position[0]:8.3f}{atom.position[1]:8.3f}{atom.position[2]:8.3f}"
                            f"{1.00:6.2f}{0.00:6.2f}          {atom.element:>2s}  \n"
                        )
                        f.write(atom_line)
                    f.write("ENDMDL\n")

    def run(self):
        self.prepare_data()
        self.compute_rmsd_matrix()
        self.perform_clustering()
        self.write_cluster_outputs()


if __name__ == "__main__":
    import argparse
    import warnings

    # Suppress all user warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="MDAnalysis")

    parser = argparse.ArgumentParser(description="Cluster GOLH conformations using KMeans.")
    parser.add_argument("-f", "--file", required=True, help="Input PDB file")
    parser.add_argument("-k", "--n_clusters", type=int, default=3, help="Number of clusters for KMeans")
    parser.add_argument("-o", "--output", default="clusters", help="Output directory")
    parser.add_argument("-u", "--zup", type=float, default=60.0, help="Upper limit for the membrane")
    parser.add_argument("-d", "--zdown", type=float, default=19.0, help="Lower limit for the membrane")

    args = parser.parse_args()

    clusterer = GOLHClusterer(
        pdb_file=args.file,
        z_up=args.zup,
        z_down=args.zdown,
        n_clusters=args.n_clusters,
        output_dir=args.output
    )

    try:
        clusterer.run()
    except (RuntimeError, ValueError) as e:
        print(f"Error: {e}")
