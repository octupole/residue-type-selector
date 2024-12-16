import MDAnalysis as mda
from MDAnalysis.analysis import density
import argparse
import numpy as np
import matplotlib.pyplot as plt

class ResidueSelector:
    """
    Class to handle residue selection based on z-coordinate ranges.
    """
    def __init__(self, universe):
        self.universe = universe

    def select_residues_in_z_range(self, selection, z_down, z_up):
        """
        Selects residues within a specified z-coordinate range.

        Args:
            selection (MDAnalysis.core.groups.AtomGroup): An AtomGroup to select from.
            z_down (float): Lower z-coordinate boundary.
            z_up (float): Upper z-coordinate boundary.

        Returns:
            MDAnalysis.core.groups.AtomGroup: An AtomGroup containing the selected atoms.
        """
        z_down = float(z_down)
        z_up = float(z_up)

        selected_residues = [
            res.resindex for res in selection.residues
            if all(z_down < atom.position[2] < z_up for atom in res.atoms)
        ]

        return self.universe.residues[selected_residues].atoms


class MembraneDensity:
    """
    Calculates and analyzes the density profile of a lipid membrane along the z-axis.
    """

    def __init__(self, topology, trajectory, lipid_selection):
        """
        Initializes the MembraneDensity object.

        Args:
            topology (str): Path to the topology file (e.g., 'topol.tpr').
            trajectory (str): Path to the trajectory file (e.g., 'traj.xtc').
            lipid_selection (str): MDAnalysis selection string for the lipid atoms 
                                    (e.g., 'resname DOPC').
        """
        self.universe = mda.Universe(topology, trajectory)
        self.lipids = self.universe.select_atoms(lipid_selection)
        self.density_profile = None
        self.residue_selector = ResidueSelector(self.universe)  # Initialize ResidueSelector

    def calculate_density(self, atom_selection=None):
        """
        Calculates the density profile along the z-axis.

        Args:
            atom_selection (MDAnalysis.core.groups.AtomGroup, optional): 
                An AtomGroup to calculate density for. If None, uses self.lipids.
        """
        if atom_selection is None:
            atom_selection = self.lipids
        self.density_profile = density.DensityAnalysis(atom_selection, axis=2)
        self.density_profile.run()

    def get_density_data(self):
        """
        Returns the z-coordinates and density values.

        Returns:
            tuple: A tuple containing the z-coordinates and density values.
        """
        if self.density_profile is None:
            self.calculate_density()
        return self.density_profile.results.z, self.density_profile.results.density

    def save_density_data(self, output_filename):
        """
        Saves the density data to a file.

        Args:
            output_filename (str): Name of the output file.
        """
        z_coords, density_values = self.get_density_data()
        np.savetxt(output_filename, np.column_stack((z_coords, density_values)), 
                   header='z-coordinate (A) Density', comments='')

    def plot_density_profile(self, plot_filename):
        """
        Plots the density profile and saves it as an image.

        Args:
            plot_filename (str): Name of the output plot file.
        """
        z_coords, density_values = self.get_density_data()
        plt.plot(z_coords, density_values)
        plt.xlabel('z-coordinate (Å)')
        plt.ylabel('Density')
        plt.title('Density Profile of Lipid Membrane')
        plt.savefig(plot_filename)
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate density profile of a lipid membrane.')
    parser.add_argument('-t', '--topology', required=True, help='Topology file (e.g., topol.tpr)')
    parser.add_argument('-f', '--trajectory', required=True, help='Trajectory file (e.g., traj.xtc)')
    parser.add_argument('-l', '--lipid', default='resname DOPC', 
                        help='MDAnalysis selection string for lipid atoms (default: resname DOPC)')
    parser.add_argument('-o', '--output', default='density.dat', help='Output filename for density data')
    parser.add_argument('-p', '--plot', default='density.png', help='Output filename for the plot')
    parser.add_argument('-zd', '--zdown', help='Lower z-coordinate boundary for residue selection')
    parser.add_argument('-zu', '--zup', help='Upper z-coordinate boundary for residue selection')

    args = parser.parse_args()

    membrane_density = MembraneDensity(args.topology, args.trajectory, args.lipid)

    if args.zdown and args.zup:
        # Select residues within the specified z-range
        selected_atoms = membrane_density.residue_selector.select_residues_in_z_range(
            membrane_density.lipids, args.zdown, args.zup
        )
        # Calculate density for the selected atoms
        membrane_density.calculate_density(selected_atoms)  

    else:
        # Calculate density for all lipid atoms
        membrane_density.calculate_density()

    membrane_density.save_density_data(args.output)
    membrane_density.plot_density_profile(args.plot)
    