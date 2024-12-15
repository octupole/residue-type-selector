from pymol import cmd

@cmd.extend
def find_min_max_z_for_residue(residue, resname):
    """Finds the minimum and maximum z-coordinates for a given residue."""
    selection_string = f"resi {residue} and resn {resname}"
    cmd.select("temp_selection", selection_string)
    model = cmd.get_model("temp_selection")
    cmd.delete("temp_selection")

    if model.atom:
        z_coords = [atom.coord[2] for atom in model.atom]
        max_z = max(z_coords)
        min_z = min(z_coords)
        print(f"For residue {residue}:")
        print(f"  Maximum z-coordinate: {max_z}")
        print(f"  Minimum z-coordinate: {min_z}")
        cmd.select("max_z_atoms", f"resi {residue} and z=={max_z}")
        cmd.select("min_z_atoms", f"resi {residue} and z=={min_z}")
        return min_z, max_z
    else:
        print(f"Residue {residue} not found.")
        return None, None

# Example usage:
#residue_to_check = 10
#min_z_value, max_z_value = find_min_max_z_for_residue(residue_to_check)

#residue_to_check = "A 20"
#min_z_value, max_z_value = find_min_max_z_for_residue(residue_to_check)