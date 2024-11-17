# File: __init__.py
from pymol import cmd
from pymol.plugins import addmenuitemqt

def __init_plugin__(app=None):
    '''Add plugin to the PyMOL menu'''
    addmenuitemqt('Plugin', 'Residue Type Selector', create_selections_by_residue_type)

def create_selections_by_residue_type():
    """
    Automatically creates selections for each unique residue type in the system
    """
    # Get list of all unique residue names in the structure
    stored.residues = set()
    cmd.iterate("*", "stored.residues.add(resn)")
    
    # Clear any existing selections that might conflict
    cmd.delete("restype_*")
    
    # Create selections for each unique residue type
    for restype in stored.residues:
        selection_name = f"restype_{restype.lower()}"
        cmd.select(selection_name, f"resn {restype}")
        print(f"Created selection: {selection_name} for residue type {restype}")
    
    print(f"\nTotal number of unique residue types: {len(stored.residues)}")
    print("Unique residue types found:", ", ".join(sorted(stored.residues)))

# Add the function to PyMOL's command line
cmd.extend('create_residue_selections', create_selections_by_residue_type)
