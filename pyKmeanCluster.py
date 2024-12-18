import MDAnalysis as mda
from MDAnalysis.analysis import align, rms

import numpy as np
from sklearn.cluster import KMeans
import math, os
from collections import Counter

SELECTION_C="name C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 C14 C15 C16 C17 C18"

def select_residues_in_z_range(selection, z_down, z_up):
    """
    Select residues where all atoms have z-coordinates between z_down and z_up.

    Parameters:
    ----------
    residue_selection : MDAnalysis.core.groups.ResidueGroup
        The input residue selection (e.g., u.select_atoms("your_selection").residues).
    z_down : float
        The lower z-coordinate limit.
    z_up : float
        The upper z-coordinate limit.

    Returns:
    -------
    MDAnalysis.core.groups.ResidueGroup
        ResidueGroup containing residues that meet the criteria.
    """
    z_down = float(z_down)  # Ensure z_down is a float
    z_up = float(z_up)      # Ensure z_up is a float

    # List to store residues that meet the condition
    selected_residues = []
    residue_selection=selection.residues
    # Iterate over the given residues
    for res in residue_selection:
        # Check if all atoms in the residue satisfy z_down < z < z_up
        if all(z_down < atom.position[2] < z_up for atom in res.atoms):
            selected_residues.append(res.resindex)

    # Convert the list of residues into a ResidueGroup
    selected_residue_group = residue_selection.universe.residues[selected_residues]

    return selected_residue_group.atoms

def cluster_golh_conformations(pdb_file, z_up, z_down, n_clusters=3, output_dir="clusters"):
    """
    Clusters conformations of GOLH based on RMSD of C1-C18 carbons using KMeans,
    translates and rotates all molecules in each cluster to align with the first 
    molecule, and writes them to separate PDB files.
    """
    try:
        universe = mda.Universe(pdb_file)
    except Exception as e:
        raise RuntimeError(f"Error loading PDB: {e}")

    golh = universe.select_atoms("resname GOLH or resname GOLO")

    if not golh:
        raise ValueError("No GOLH residues found in the PDB.")
    
    golh=select_residues_in_z_range(golh, z_down, z_up)

    carbon_atoms = golh.select_atoms(SELECTION_C)

    if not carbon_atoms:
        raise ValueError("No C1-C18 carbons found in GOLH residues.")

    n_molecules = len(golh.residues)
    n_carbons = len(carbon_atoms) // n_molecules

    all_carbon_positions = carbon_atoms.positions.reshape(n_molecules, n_carbons, 3)
    carbon_atomgroups = [mol.atoms.select_atoms(SELECTION_C) for mol in golh.residues]

    rmsd_matrix = np.zeros((n_molecules, n_molecules))

    for i in range(n_molecules):
        for j in range(i + 1, n_molecules):
            rmsd_matrix[i, j] = rms.rmsd(carbon_atomgroups[i].positions, carbon_atomgroups[j].positions,  superposition=True)
            rmsd_matrix[j, i] = rmsd_matrix[i, j]

    # --- KMeans Clustering ---
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(rmsd_matrix)

    print(f"Found {n_clusters} clusters")
    
    # unique_values, counts = np.unique(cluster_labels, return_counts=True)

    # # Create a dictionary for easy access
    # counts_dict = dict(zip(unique_values, counts))

    # # Print the counts
    # for value, count in counts_dict.items():
    #     print(f"Value {value}: {count} occurrences")
    for cluster_id in range(n_clusters):
        cluster_resname = [mol.resname for i, mol in enumerate(golh.residues) if cluster_labels[i] == cluster_id]
        counts = Counter(cluster_resname)

        print(f"Cluster {cluster_id}: TOT: {counts['GOLO']+counts['GOLH']}, GOLH: {counts['GOLH']}, GOLO: {counts['GOLO']}") 
        

    # --- Align and Write PDB files ---
    os.makedirs(output_dir, exist_ok=True)

    for cluster_id in range(n_clusters):
        cluster_members = [mol for i, mol in enumerate(golh.residues) if cluster_labels[i] == cluster_id]
                
        # --- Transformation ---
        ref_molecule = cluster_members[0].atoms  # Select atoms of the reference molecule
        ref_carbon_atoms = ref_molecule.select_atoms(SELECTION_C)

        for molecule in cluster_members:
            mobile_carbon_atoms = molecule.atoms.select_atoms(SELECTION_C)

            # 1. Translation to overlap centers of mass
            translation_vector = ref_carbon_atoms.center_of_mass() - mobile_carbon_atoms.center_of_mass()
            molecule.atoms.translate(translation_vector)

            # 2. Rotation using align.rotation_matrix
            rotation_matrix, rmsd = align.rotation_matrix(mobile_carbon_atoms.positions, ref_carbon_atoms.positions)
            molecule.atoms.rotate(rotation_matrix)
        # --- Write PDB file for the cluster ---
        output_filename = os.path.join(output_dir, f"cluster_{cluster_id + 1}.pdb")
        with open(output_filename, 'w') as f:
            for molecule in cluster_members:
                for atom in molecule.atoms:
                    atom_line = f"ATOM  {atom.id:>5d}  {atom.name:<4s}{atom.resname:>3s} {atom.resid:>4d}    " \
                                f"{atom.position[0]:8.3f}{atom.position[1]:8.3f}{atom.position[2]:8.3f}" \
                                f"{1.00:6.2f}{0.00:6.2f}          {atom.element:>2s}  \n"
                    f.write(atom_line)
                f.write("ENDMDL\n")


        # --- Calculate and Write Average Structure ---
        avg_positions = np.mean([mol.atoms.select_atoms(SELECTION_C).positions for mol in cluster_members], axis=0)
        avg_universe = mda.Merge(cluster_members[0].atoms.select_atoms(SELECTION_C))  # Use the first molecule as a template
        avg_universe.atoms.positions = avg_positions
        
        avg_output_filename = os.path.join(output_dir, f"cluster_{cluster_id + 1}_avg.pdb")
#        avg_universe.atoms.write(avg_output_filename)

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

if __name__ == "__main__":
    import argparse
    import warnings
    from Bio import BiopythonDeprecationWarning

    # Suppress BiopythonDeprecationWarning
    warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)


    parser = argparse.ArgumentParser(description="Cluster GOLH conformations using KMeans.")
    parser.add_argument("-f", "--file", required=True, help="Input PDB file")
    parser.add_argument("-k", "--n_clusters", type=int, default=3, help="Number of clusters for KMeans")
    parser.add_argument("-o", "--output", default="clusters", help="Output directory")

    parser.add_argument("-u", "--zup", type=float, default=60.0, help="Upper limit for the membrane")
    parser.add_argument("-d", "--zdown", type=float, default=19.0, help="Lower limit for the membrane")  
    args = parser.parse_args()

    try:
        cluster_golh_conformations(args.file, args.zup, args.zdown, args.n_clusters, args.output)
    except (RuntimeError, ValueError) as e:
        print(f"Error: {e}")
        