import argparse
import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from MDAnalysis.transformations import unwrap


class DensityProfileCalculator:
    def __init__(self, trajectory, tpr, axis='z', bins=100, start=0, end=None, step=1):
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
        self.step = step
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
        self.universe.trajectory.add_transformations(
            unwrap(self.universe.atoms))

        for ts in self.universe.trajectory[self.start:self.end:self.step]:
            print(f"Processing step: {ts.frame}, timestep: {ts.time} fs")
            atoms = selection
            positions.extend(atoms.positions[:, 'xyz'.index(self.axis)])

        box_length = self.universe.trajectory.ts.dimensions['xyz'.index(
            self.axis)]
        hist, bin_edges = np.histogram(
            positions, bins=self.bins, range=(0, box_length), density=True)
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

    def write_xmgrace_file(self, bin_centers, density, output_file):
        """
        Write the density profile to a file compatible with xmgrace.

        Parameters:
            bin_centers (numpy.ndarray): Centers of the histogram bins (x values).
            density (numpy.ndarray): Density values (y values).
            output_file (str): Path to save the xmgrace-compatible file.
        """
        with open(output_file, 'w') as f:
            f.write("# GraceXY format\n")
            f.write("@    title \"Density Profile\"\n")
            f.write(f"@    xaxis label \"Z-coordinate (\\cE\\C)\"\n")
            f.write("@    yaxis label \"Density (a.u.)\"\n")
            f.write("@TYPE xy\n")
            for x, y in zip(bin_centers, density):
                f.write(f"{x:.6f} {y:.6f}\n")
        print(f"Density profile written to xmgrace file: {output_file}")


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
        filtered_atoms = selection.select_atoms(f"resid {' '.join(
            [str(res.resid) for res in selection.residues if res.resindex in selected_residues])}")

        return filtered_atoms


class CommandLineInterface:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Compute the density profile for a DOPC membrane.")
        self.parser.add_argument(
            "-x", "--trajectory", required=True, help="Path to the trajectory file.")
        self.parser.add_argument(
            "-t", "--tpr", required=True, help="Path to the TPR file.")
        self.parser.add_argument(
            "-p", "--plot", help="Path to save the density profile plot.")
        self.parser.add_argument("-sl", "--selection", default='resname DOPC',
                                 help="Path to save the density profile plot.")
        self.parser.add_argument("-a", "--axis", default="z", choices=[
                                 "x", "y", "z"], help="Axis for density profile (default: z).")
        self.parser.add_argument("-b", "--bins", type=int, default=100,
                                 help="Number of bins for the histogram (default: 100).")
        self.parser.add_argument(
            "--z_down", type=float, help="Lower bound of z-coordinate range for residue selection.")
        self.parser.add_argument(
            "--z_up", type=float, help="Upper bound of z-coordinate range for residue selection.")
        self.parser.add_argument("-s", "--start", type=int, default=0,
                                 help="Starting step of the trajectory (default: 0).")
        self.parser.add_argument(
            "-e", "--end", type=int, help="Ending step of the trajectory (default: end of trajectory).")
        self.parser.add_argument("-o", "--output", default="density.xvg",
                                 help="Path to save the xmgrace-compatible file.")
        self.parser.add_argument("-dt", "--step", type=int, default=1,
                                 help="End frame for trajectory processing")

    def parse_args(self):
        return self.parser.parse_args()


def check_file_exists(file_path, description):
    """
    Check if a file exists, if not, exit with an error message.

    Parameters:
        file_path (str): Path to the file to check.
        description (str): Description of the file being checked.
    """
    if not os.path.isfile(file_path):
        print(f"Error: {description} '{file_path}' does not exist.")
        sys.exit(1)


def main():
    cli = CommandLineInterface()
    args = cli.parse_args()
    # Check if input files exist
    check_file_exists(args.trajectory, "Trajectory file")
    check_file_exists(args.tpr, "TPR file")

    # Initialize the DensityProfileCalculator
    calculator = DensityProfileCalculator(
        trajectory=args.trajectory,
        tpr=args.tpr,
        axis=args.axis,
        bins=args.bins,
        start=args.start,
        end=args.end,
        step=args.step
    )

    # Selection for DOPC
    selection = calculator.universe.select_atoms(args.selection)

    # Optionally filter residues by z range
    if args.z_down is not None and args.z_up is not None:
        print("Selecting residues within z range...")
        selector = ResidueSelector(calculator.universe)
        selection = selector.select_residues_in_z_range(
            selection, args.z_down, args.z_up)

    # Compute density profile
    print("Computing density profile...")
    bin_centers, density = calculator.compute_density_profile(selection)

    # Plot and save the density profile
    if args.plot:
        print("Saving density profile plot...")
        calculator.plot_density_profile(bin_centers, density, args.plot)

    if args.output:
        # Write density profile to xmgrace file
        print("Writing density profile to xmgrace file...")
        calculator.write_xmgrace_file(bin_centers, density, args.output)


if __name__ == "__main__":
    main()
