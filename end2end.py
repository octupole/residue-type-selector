import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.stats import gaussian_kde
from itertools import chain
from MDAnalysis.transformations import unwrap


class MembranePlanes:
    def __init__(self, universe, selection="name P", cutoff=5.0):
        """
        Initialize the MembranePlanes class.

        Parameters:
        - universe: MDAnalysis Universe object containing the simulation data.
        - selection: Atom selection string to identify the membrane atoms (default "name P").
                     Typically, phosphate atoms (P) are used for phospholipids.
        - cutoff: Distance cutoff (in Å) to distinguish between the upper and lower membrane planes.
        """
        self.universe = universe
        self.selection = selection
        self.cutoff = cutoff
        self.upper_plane_z = []
        self.lower_plane_z = []

        # Prepare the atom selection once
        self.membrane_atoms = self.universe.select_atoms(self.selection)

    def calculate_planes_for_frame(self):
        """
        Calculate the z-coordinates of the upper and lower membrane planes for the current frame.

        Returns:
        - (upper_z, lower_z): Tuple containing the mean z-coordinates for the upper and lower planes.
        """
        z_coords = self.membrane_atoms.positions[:, 2]  # Extract z-coordinates

        # Determine the midpoint of the z-coordinates
        z_mid = np.median(z_coords)

        # Separate atoms into upper and lower planes based on their z-coordinate relative to the midpoint
        upper_plane = z_coords[z_coords >= z_mid + self.cutoff]
        lower_plane = z_coords[z_coords <= z_mid - self.cutoff]

        # Compute the mean z-coordinates for the upper and lower planes
        upper_z = np.mean(upper_plane) if len(upper_plane) > 0 else np.nan
        lower_z = np.mean(lower_plane) if len(lower_plane) > 0 else np.nan

        # Store the plane z-coordinates for this frame
        self.upper_plane_z.append(upper_z)
        self.lower_plane_z.append(lower_z)

        return upper_z, lower_z

    def get_planes(self):
        """
        Return the calculated z-coordinates for the upper and lower planes over all frames.

        Returns:
        - upper_plane_z: List of upper plane z-coordinates for each frame.
        - lower_plane_z: List of lower plane z-coordinates for each frame.
        """
        return self.upper_plane_z, self.lower_plane_z

# Example usage:
# u = mda.Universe("your_topology_file", "your_trajectory_file")
# membrane_planes = MembranePlanes(u)

# Loop over frames in the trajectory externally
# for ts in u.trajectory:
#     upper_z, lower_z = membrane_planes.calculate_planes_for_frame()

# Retrieve all plane data
# upper_z_all, lower_z_all = membrane_planes.get_planes()


class EndToEndDistanceCalculator:
    def __init__(self, topology_file, trajectory_file, start_frame, end_frame, step, output_file, bin_size, bandwidth):
        self.universe = mda.Universe(topology_file, trajectory_file)
        self.golh_selection = self.universe.select_atoms("resname GOLH")
        self.golo_selection = self.universe.select_atoms("resname GOLO")
        self.golh_distances_all_frames = []
        self.golo_distances_all_frames = []
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.step = step
        self.output_file = output_file
        self.bin_size = bin_size
        self.bandwidth = bandwidth

    def calculate_end_to_end_distance(self, molecule):
        """
        Calculate the end-to-end distance for a given molecule.
        Args:
            molecule: MDAnalysis AtomGroup representing the molecule.
        Returns:
            Distance between atom C1 and atom C18 of the molecule.
        """
        start_pos = molecule.select_atoms("name C1A").positions[0]
        end_pos = molecule.select_atoms("name C18").positions[0]
        distance = np.linalg.norm(end_pos - start_pos)
        return distance

    def compute_end_to_end_distribution(self, selection):
        distances = []
        for res in selection.residues:
            distances.append(self.calculate_end_to_end_distance(res.atoms))
        return distances

    def calculate_distances_for_all_frames(self, membrane_planes=None):
        with open(self.output_file, 'w') as f:
            # f.write("# Frame GOLH_avg_distance GOLO_avg_distance\n")
            self.universe.trajectory.add_transformations(
                unwrap(self.universe.atoms))

            for ts_index, ts in enumerate(self.universe.trajectory[self.start_frame:self.end_frame:self.step]):
                print(f"Processing step: {ts.frame}, timestep: {ts.time} fs")

                if membrane_planes != None:
                    upper_z, lower_z = membrane_planes.calculate_planes_for_frame()
                else:
                    upper_z, lower_z = 1e10, -1e10

                golh = self.golh_selection.select_atoms(
                    f"prop z < {upper_z} and prop z > {lower_z}")
                golo = self.golo_selection.select_atoms(
                    f"prop z < {upper_z} and prop z > {lower_z}")

                golh_distances = self.compute_end_to_end_distribution(golh)
                golo_distances = self.compute_end_to_end_distribution(golo)
                self.golh_distances_all_frames.append(golh_distances)
                self.golo_distances_all_frames.append(golo_distances)

            self.golh_distances_all_frames = list(
                chain.from_iterable(self.golh_distances_all_frames))
            self.golo_distances_all_frames = list(
                chain.from_iterable(self.golo_distances_all_frames))

    def calculate_average_distances(self):
        self.golh_distances_avg = np.array(self.golh_distances_all_frames)
        self.golo_distances_avg = np.array(self.golo_distances_all_frames)

    def plot_average_distribution(self):
        plt.figure()
        if self.golo_distances_avg.size != 0:
            bins = np.arange(0, max(max(self.golh_distances_avg), max(
                self.golo_distances_avg)) + self.bin_size, self.bin_size)
            plt.hist(self.golh_distances_avg, bins=bins,
                     density=True, alpha=0.5, label='GOLH')
            plt.hist(self.golo_distances_avg, bins=bins,
                     density=True, alpha=0.5, label='GOLO')
        else:
            bins = np.arange(0, max(self.golh_distances_avg) +
                             self.bin_size, self.bin_size)
            plt.hist(self.golh_distances_avg, bins=bins,
                     density=True, alpha=0.5, label='GOLH')

        plt.xlabel('End-to-End Distance (Angstroms)')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.title(
            'Average End-to-End Distance Probability Distribution for GOLH and GOLO')
        plt.show()

    def write_probability_distribution(self):
        with open(self.output_file, 'a') as f:
            f.write(
                "\n# Probability distribution data for GOLH and GOLO (using bins)\n")
            f.write("# Bin_center GOLH_density GOLO_density\n")
            if self.golo_distances_avg.size != 0:
                bins = np.arange(0, max(max(self.golh_distances_avg), max(
                    self.golo_distances_avg)) + self.bin_size, self.bin_size)
                golh_hist, bin_edges = np.histogram(
                    self.golh_distances_avg, bins=bins, density=True)
                golo_hist, _ = np.histogram(
                    self.golo_distances_avg, bins=bins, density=True)
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            # Write to the output file
                for center, golh_dens, golo_dens in zip(bin_centers, golh_hist, golo_hist):
                    f.write(f"{center:.3f} {golh_dens:.5f} {golo_dens:.5f}\n")
            else:
                bins = np.arange(
                    0, max(self.golh_distances_avg) + self.bin_size, self.bin_size)
                golh_hist, bin_edges = np.histogram(
                    self.golh_distances_avg, bins=bins, density=True)
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            # Write to the output file
                for center, golh_dens in zip(bin_centers, golh_hist):
                    f.write(f"{center:.3f} {golh_dens:.5f} \n")

    def write_smooth_probability_distribution(self):
        with open(self.output_file, 'a') as f:
            f.write("\n# Smoothed probability distribution data for GOLH and GOLO\n")
            f.write("# Distance GOLH_density GOLO_density\n")
            if self.golo_distances_avg.size != 0:

                # Calculate KDE for smoothing
                if self.bandwidth is None:
                    bandwidth_golh = self.bin_size / \
                        np.std(self.golh_distances_avg)
                    bandwidth_golo = self.bin_size / \
                        np.std(self.golo_distances_avg)
                else:
                    bandwidth_golh = self.bandwidth
                    bandwidth_golo = self.bandwidth

                kde_golh = gaussian_kde(
                    self.golh_distances_avg, bw_method=bandwidth_golh)
                kde_golo = gaussian_kde(
                    self.golo_distances_avg, bw_method=bandwidth_golo)

                # Define the range for KDE evaluation, extending to capture tail behavior going to zero
                distance_range = np.linspace(
                    0, max(max(self.golh_distances_avg), max(self.golo_distances_avg)) * 1.5, 1000)
                golh_density = kde_golh(distance_range)
                golo_density = kde_golo(distance_range)

                # Ensure the density goes smoothly to zero at large values
                golh_density[-1] = 0
                golo_density[-1] = 0

                # Write to the output file
                for dist, golh_dens, golo_dens in zip(distance_range, golh_density, golo_density):
                    f.write(f"{dist:.3f} {golh_dens:.5f} {golo_dens:.5f}\n")
            else:
                # Calculate KDE for smoothing
                if self.bandwidth is None:
                    bandwidth_golh = self.bin_size / \
                        np.std(self.golh_distances_avg)
                else:
                    bandwidth_golh = self.bandwidth

                kde_golh = gaussian_kde(
                    self.golh_distances_avg, bw_method=bandwidth_golh)

                # Define the range for KDE evaluation, extending to capture tail behavior going to zero
                distance_range = np.linspace(
                    0, max(self.golh_distances_avg) * 1.5, 1000)
                golh_density = kde_golh(distance_range)

                # Ensure the density goes smoothly to zero at large values
                golh_density[-1] = 0

                # Write to the output file
                for dist, golh_dens in zip(distance_range, golh_density):
                    f.write(f"{dist:.3f} {golh_dens:.5f}\n")

    def compute_histogram(self, data):
        """
        Compute the histogram for the given data without using matplotlib.
        Args:
            data: The data for which to compute the histogram.
        Returns:
            A tuple containing bin centers and histogram values.
        """
        bins = np.arange(0, max(data) + self.bin_size, self.bin_size)
        hist, bin_edges = np.histogram(data, bins=bins, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        return bin_centers, hist


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Calculate end-to-end distance probability distribution for GOLH and GOLO.")
    parser.add_argument('-t', '--topology', required=True,
                        help='Path to the topology file (.tpr)')
    parser.add_argument('-x', '--trajectory', required=True,
                        help='Path to the trajectory file (.xtc)')
    parser.add_argument('-b', '--start', type=int, default=0,
                        help='Start frame for analysis (default: 0)')
    parser.add_argument('-e', '--end', type=int, default=-1,
                        help='End frame for analysis (default: -1, meaning till the end)')
    parser.add_argument('--step', type=int, default=1,
                        help='Step size for trajectory frames (default: 1)')
    parser.add_argument('-o', '--output', default='output.xvg',
                        help='Output file for xmgrace (default: output.xvg)')
    parser.add_argument('--binsize', type=float, default=0.5,
                        help='Bin size for histogram in Angstroms (default: 0.5)')
    parser.add_argument('--bandwidth', type=float, default=None,
                        help='Bandwidth for KDE smoothing (default: None, automatic calculation)')
    parser.add_argument("--membrane", action="store_true",
                        help="Enable select gols if in a membrane")

    args = parser.parse_args()

    # Create an instance of the calculator and perform calculations
    calculator = EndToEndDistanceCalculator(
        args.topology, args.trajectory, args.start, args.end, args.step, args.output, args.binsize, args.bandwidth)
    if args.membrane:
        print("Calculating distances for membrane-bound GOLH and GOLO...")
        membrane_planes = MembranePlanes(
            universe=calculator.universe, cutoff=10.0, selection='name C5A')
        calculator.calculate_distances_for_all_frames(membrane_planes)
    else:
        calculator.calculate_distances_for_all_frames()

    calculator.calculate_average_distances()
    calculator.plot_average_distribution()
    calculator.write_smooth_probability_distribution()
