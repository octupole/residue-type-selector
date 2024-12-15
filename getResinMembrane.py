from pymol import cmd

@cmd.extend
def z_filter(selection="all", z_down=0.0, z_up=10.0):
    """
    Filters residues where all atoms are within a specified z-coordinate range.

    Parameters:
    selection (str): The PyMOL selection to filter (default: "all").
    z_down (float): The lower z-coordinate limit.
    z_up (float): The upper z-coordinate limit.
    """
    z_down = float(z_down)  # Ensure z_down is a float
    z_up = float(z_up)      # Ensure z_up is a float

    selected_residues = set()  # Use a set for efficient storage and lookup

    # Get all atoms in the input selection
    atoms = cmd.get_model(selection).atom

    # Group atoms by residue (using resi and chain as keys)
    residue_atoms = {}
    for atom in atoms:
        resi = atom.resi
        chain = atom.chain or ""  # Handle empty chains
        key = (resi, chain)
        if key not in residue_atoms:
            residue_atoms[key] = []
        residue_atoms[key].append(atom)

    # Filter residues based on the z-coordinate range
    for (resi, chain), atom_list in residue_atoms.items():
        if all(z_down < atom.coord[2] < z_up for atom in atom_list):
            chain_selector = f" and chain {chain}" if chain else ""
            selected_residues.add(f"resi {resi}{chain_selector}")

    # Create the selection if residues match the criteria
    if selected_residues:
        cmd.select("residues_in_z_range", " or ".join(selected_residues))
        cmd.show("sticks", "residues_in_z_range")
        cmd.color("cyan", "residues_in_z_range")
        print(f"Created selection: residues_in_z_range")
    else:
        print("No residues found with all atoms in the specified z range.")

# # Load your structure
# cmd.load("your_pdb_file.pdb")

# # Set the z-coordinate range
# z_down = 5.0
# z_up = 15.0

# # Call the function
# select_residues_within_z_range("all", z_down, z_up)

# # Show the selection (optional)
# cmd.show("sticks", "residues_within_z_range")

# # Save the selection (optional)
# cmd.save("residues_within_z_range.pdb", "residues_within_z_range")