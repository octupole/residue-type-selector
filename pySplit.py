import MDAnalysis as mda

def split_golh_residues(pdb_file, output_dir="golh_residues"):
    """
    Splits a PDB file into separate PDB files for each GOLH residue.
    """
    try:
        universe = mda.Universe(pdb_file)
    except Exception as e:
        raise RuntimeError(f"Error loading PDB: {e}")

    golh_residues = universe.select_atoms("resname GOLH").residues

    if not golh_residues:
        raise ValueError("No GOLH residues found in the PDB.")

    # Create output directory
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Write each GOLH residue to a separate PDB file
    for i, residue in enumerate(golh_residues):
        output_filename = os.path.join(output_dir, f"golh_{i+1}.pdb")
        with mda.Writer(output_filename) as W:
            W.write(residue)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split a PDB file into separate files for each GOLH residue.")
    parser.add_argument("-f", "--file", required=True, help="Input PDB file")
    parser.add_argument("-o", "--output", default="golh_residues", help="Output directory")
    args = parser.parse_args()

    try:
        split_golh_residues(args.file, args.output)
    except (RuntimeError, ValueError) as e:
        print(f"Error: {e}")
        