import argparse
import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt

class DensityProfileCalculator:
    def __init__(self, trajectory, tpr, axis='z', bins=100, start=0, end=None):
        """
        Initialize the density profile calculator.

        Parameters:
            trajectory (str): Path to the trajectory file.
            tpr (str): Path to the TPR file.
            axis (str): Axis along which to compute the density profile ('x', 'y', or 'z').
            bins (int): Number of bins for the histogram.
            start (int): Starting step of the trajectory.
            end (int): Ending step of the trajectory.
        """
        self.trajectory = trajectory
        self.tpr = tpr
        self.axis = axis
        self.bins = bins
        self.start = start
        self.end = end
        self.universe = mda.Universe(tpr, trajectory)

    def compute_density_profile(self, selection):
        """
        Compute the density profile for the specified atom selection.

        Parameters:
            selection (str): Atom selection string.

        Returns:
            bin_centers (numpy.ndarray): Centers of the histogram bins.
            density (numpy.ndarray): Density values.
        """
        positions = []
        for ts in self.universe.trajectory[self.start:self.end]:
            print(f"Processing step: {ts.frame}, timestep: {ts.time} fs")            
            atoms = selection
            positions.extend(atoms.positions[:, 'xyz'.index(self.axis)])
            
        print(len(positions))

        box_length = self.universe.trajectory.ts.dimensions['xyz'.index(self.axis)]
        hist, bin_edges = np.histogram(positions, bins=self.bins, range=(0, box_length), density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        return bin_centers, hist

    def plot_density_profile(self, bin_centers, density, output):
        """
        Plot and save the density profile.

        Parameters:
            bin_centers (numpy.ndarray): Centers of the histogram bins.
            density (numpy.ndarray): Density values.
            output (str): Path to save the density profile plot.
        """
        plt.figure()
        plt.plot(bin_centers, density, label="DOPC density")
        plt.xlabel(f"{self.axis}-coordinate (\u212B)")
        plt.ylabel("Density (a.u.)")
        plt.title("Density Profile of DOPC Membrane")
        plt.legend()
        plt.grid()
        plt.savefig(output)
        print(f"Density profile saved to {output}")

class ResidueSelector:
    """
    Class to handle residue selection based on z-coordinate ranges.
    """
    def __init__(self, universe):
        self.universe = universe

    def select_residues_in_z_range(self, selection, z_down, z_up):
        """
        Select residues based on z-coordinate range.

        Parameters:
            selection (MDAnalysis.AtomGroup): Atom selection to filter.
            z_down (float): Lower bound of the z-coordinate range.
            z_up (float): Upper bound of the z-coordinate range.

        Returns:
            MDAnalysis.AtomGroup: Atoms of residues within the z-coordinate range.
        """
        z_down = float(z_down)
        z_up = float(z_up)
        selected_residues = [
            res.resindex for res in selection.residues
            if all(z_down < atom.position[2] < z_up for atom in res.atoms)
        ]
        filtered_atoms = selection.select_atoms(f"resid {' '.join([str(res.resid) for res in selection.residues if res.resindex in selected_residues])}")

        return filtered_atoms

class CommandLineInterface:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Compute the density profile for a DOPC membrane.")
        self.parser.add_argument("-t", "--trajectory", required=True, help="Path to the trajectory file.")
        self.parser.add_argument("-p", "--tpr", required=True, help="Path to the TPR file.")
        self.parser.add_argument("-o", "--output", default='density.png',help="Path to save the density profile plot.")
        self.parser.add_argument("-sl", "--selection", default='resname DOPC',help="Path to save the density profile plot.")
        self.parser.add_argument("-a", "--axis", default="z", choices=["x", "y", "z"], help="Axis for density profile (default: z).")
        self.parser.add_argument("-b", "--bins", type=int, default=100, help="Number of bins for the histogram (default: 100).")
        self.parser.add_argument("--z_down", type=float, help="Lower bound of z-coordinate range for residue selection.")
        self.parser.add_argument("--z_up", type=float, help="Upper bound of z-coordinate range for residue selection.")
        self.parser.add_argument("-s", "--start", type=int, default=0, help="Starting step of the trajectory (default: 0).")
        self.parser.add_argument("-e", "--end", type=int, help="Ending step of the trajectory (default: end of trajectory).")

    def parse_args(self):
        return self.parser.parse_args()

def main():
    cli = CommandLineInterface()
    args = cli.parse_args()

    # Initialize the DensityProfileCalculator
    calculator = DensityProfileCalculator(
        trajectory=args.trajectory,
        tpr=args.tpr,
        axis=args.axis,
        bins=args.bins,
        start=args.start,
        end=args.end
    )

    # Selection for DOPC
    selection = calculator.universe.select_atoms(args.selection)


    # Optionally filter residues by z range
    if args.z_down is not None and args.z_up is not None:
        print("Selecting residues within z range...")
        selector = ResidueSelector(calculator.universe)
        selection = selector.select_residues_in_z_range(selection, args.z_down, args.z_up)

    # Compute density profile
    print("Computing density profile...")
    bin_centers, density = calculator.compute_density_profile(selection)

    # Plot and save the density profile
    print("Saving density profile plot...")
    calculator.plot_density_profile(bin_centers, density, args.output)

if __name__ == "__main__":
    main()
