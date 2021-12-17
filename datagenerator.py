import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline
from tqdm import tqdm

import config


def get_knots(n: int, p: int = 3, cl: bool = False) -> np.ndarray:
    """
    :param n: Number of control points
    :param p: Curve degree, default is 3
    :param cl: Whether for a closed curve, default is false
    :return: The required knot vector for given curve characteristics
    """
    # Check whether the input condition is valid
    assert n > p, 'At least k+1 coefficients are required for a spline of degree k.'
    # Check whether the knots are for a closed curve
    if cl:
        # For a closed curve, create an uniform knot sequence
        knots = np.zeros(n + 2 * p + 1)
        knots[p:-p] = p1 = np.linspace(0, 1, num=(n + 1))
        step = 1 / n
        for ii in range(p):
            knots[p - 1 - ii] = knots[p] - (ii + 1) * step
            knots[p + n + 1 + ii] = knots[p + n] + (ii + 1) * step
    else:
        # For an open curve, the first p+1 and last p+1 knots must be identical
        knots = np.zeros(n + p + 1)
        knots[p:-p] = np.linspace(0, 1, num=(n + 1 - p))
        knots[-p:] = np.ones(p)
    return knots


def point_to_line(p: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> float:
    """
    :param p: Independent point
    :param q1: First point on the line
    :param q2: Second point on the line
    :return: The distance between the point and the line
    """
    # Compute the distance via square area
    vec1 = q1 - p
    vec2 = q2 - p
    vec3 = q1 - q2
    dis = np.abs(np.cross(vec1, vec2)) / np.sqrt(vec3.dot(vec3))
    return dis


def unwanted_features(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> bool:
    """
    :param p1: Start point of first line
    :param p2: End point of first line
    :param q1: Start point of second line
    :param q2: End point of second line
    :return: Whether the two lines have unwanted features
    """
    # Direction vector of each line
    v1 = p2 - p1
    v2 = q2 - q1
    # Length of each line
    l1 = np.sqrt(v1.dot(v1))
    l2 = np.sqrt(v2.dot(v2))
    # Compute the included angle between lines
    theta = np.arccos(v1.dot(v2) / (l1*l2))
    # If parallel, check the distances between points
    if np.abs(theta) <= 1e-3 or np.abs(theta-np.pi) <= 1e-3:
        # Compute the distances between points and lines
        dis1 = point_to_line(p1, q1, q2)
        dis2 = point_to_line(p2, q1, q2)
        # If too close, view the lines as collinear
        if min(dis1, dis2) <= 0.0015:
            return True
        # Otherwise, parallel lines will not have unwanted features
        else:
            return False
    # If no parallel, check intersections
    else:
        s = (- v1[1] * (p1[0] - q1[0]) + v1[0] * (p1[1] - q1[1])) / (
             - v2[0] * v1[1] + v1[0] * v2[1])
        t = (v2[0] * (p1[1] - q1[1]) - v2[1] * (p1[0] - q1[0])) / (
             - v2[0] * v1[1] + v1[0] * v2[1])
        if 0 <= s <= 1 and 0 <= t <= 1:
            # int = p1 + t * v1
            return True
        else:
            return False


def strange_shape(cofs: np.ndarray) -> bool:
    """
    :param cofs: A polyline consisting of control points
    :return: Check if the polyline form a strange shape
    """
    # Number of straight lines
    line_num = cofs.shape[0] - 1
    for ii in range(line_num-2):
        for jj in range(ii+2, line_num):
            if unwanted_features(cofs[ii, :], cofs[ii+1, :], cofs[jj, :], cofs[jj+1, :]):
                return True
    return False


def get_cofs(n: int, p: int = 3, cl: bool = False) -> np.ndarray:
    """
    :param n: Number of control points
    :param p: Curve degree, default is 3
    :param cl: Whether for a closed curve, default is false
    :return: Positions of a random created set of control points
    """
    exit_flag = False
    while not exit_flag:
        cofs = list()
        # Sample control points for closed curves
        if cl:
            # Check number of control points for closed curves
            assert n >= 3, 'At least three control points are needed for closed curves'
            # Sample by disturbing regular polygons
            theta_n = 2 * np.pi / n
            for ii in range(n):
                theta = (ii + 0.4 * np.random.rand(1) - 0.2) * theta_n
                length = 1 + 0.5 * np.random.randn(1)
                step = np.squeeze(np.array([length * np.cos(theta), length * np.sin(theta)]))
                cofs.append(step)
        # Sample control points for open curves
        else:
            # Start from the origin
            cofs.append(np.array([0, 0]))
            # Randomly select a direction and distance
            theta = (2 * np.random.rand(1) - 1) * np.pi
            length = 1 + 0.5 * np.random.randn(1)
            # Place the second control point
            step = np.squeeze(np.array([length * np.cos(theta), length * np.sin(theta)]))
            cofs.append(cofs[0] + step)
            # Loop for placing the rest control points
            for ii in range(1, n - 1):
                # Randomly select a direction by sampling the angle between the new direction and the old direction,
                # then randomly sample a distance between control points
                theta = theta + np.random.choice([-1, 1]) * (0.5 * np.random.rand(1) + 1 / 6) * np.pi
                length = 1 + 0.25 * np.random.randn(1)
                # Place the next control point
                step = np.squeeze(np.array([length * np.cos(theta), length * np.sin(theta)]))
                cofs.append(cofs[ii] + step)
        # Check whether the polyline form a strange shape
        if not strange_shape(np.array(cofs)):
            exit_flag = True
    # Wrap the first p and last p control points for closed curves
    if cl:
        for ii in range(p):
            cofs.append(cofs[ii])
    # Convert the list into numpy arrays
    return np.array(cofs)


def get_spline(n: int, p: int = 3, cl: bool = False) -> [BSpline, np.ndarray, np.ndarray]:
    """
    :param n: Number of control points
    :param p: Curve degree, default is 3
    :param cl: Whether to create a closed curve, default is False
    :return: Created BSpline object, control points and knots
    """
    # Create random control points
    cofs = get_cofs(n, p, cl)
    # Compute the corresponding knot vector
    knots = get_knots(n, p, cl)
    # Create the spline using control points, knot vector and curve degree
    spline = BSpline(knots, cofs, p, extrapolate=False)
    return spline, cofs, knots


class SplineGenerator:
    """
    A generator supports generating images of one random B-spline of variable number of control points
    """

    def __init__(self, h=512, w=512, dpi=100, lw=2.5):
        """
        :param h: Height of generated figures, default is 512
        :param w: Width of generated figures, default is 512
        :param dpi: Dpi of generated figures, default is 100
        :param lw: Line width of generated curves, default is 2.5
        """
        # Width and height of generated figures
        self.w = w
        self.h = h
        # Dpi of generated figures
        self.dpi = dpi
        # Line width of the generated curves
        self.lw = lw

    def __call__(self, n: int, p: int = 3, cl: bool = False, plot: bool = False, sample: int = 500,
                 folder: str = None, name: str = None):
        """
        :param n: Number of control points
        :param p: Curve degree
        :param cl: Whether to create a closed curve, default is False
        :param plot: Whether to show the generated curve with its control points
        :param sample: Number of samples to plot the created curve
        :param folder: Folder to place the created image
        :param name: Name of the image to be saved
        :return: All the related information about the created curve as a dictionary
        """
        # Get a randomly created spline and corresponding control points and knots
        spline, cofs, knots = get_spline(n, p, cl)
        # Show the spline with its control points if required
        if plot:
            xx = np.linspace(0, 1, num=sample)
            fig, ax = plt.subplots(figsize=(self.w / self.dpi, self.h / self.dpi), dpi=self.dpi)
            ax.plot(spline(xx)[:, 0], spline(xx)[:, 1], lw=2.5, label='Generated curve')
            if cl:
                ax.plot(cofs[:-p + 1, 0], cofs[:-p + 1, 1], '.r', markersize=12, label='Generated control points')
                ax.plot(cofs[:-p + 1, 0], cofs[:-p + 1, 1], '--r')
            else:
                ax.plot(cofs[:, 0], cofs[:, 1], '.r', markersize=12, label='Generated control points')
                ax.plot(cofs[:, 0], cofs[:, 1], '--r')
            plt.grid()
            plt.legend(loc='best')
            plt.show()
        # Save the figure to the assigned folder with assigned name if required
        if folder is not None and name is not None:
            xx = np.linspace(0, 1, num=sample)
            fig, ax = plt.subplots(figsize=(self.w / self.dpi, self.h / self.dpi), dpi=self.dpi)
            ax.plot(spline(xx)[:, 0], spline(xx)[:, 1], lw=2.5)
            plt.axis('off')
            if not os.path.exists(folder):
                os.makedirs(folder)
            path = folder + name + '.png'
            plt.savefig(path, bbox_inches=0)
            plt.close()
            dict_value = {'path': path, 'n': n, 'p': p, 'cl': cl, 'cofs': cofs.tolist(), 'knots': knots.tolist()}
            # Return the curve and image information
            return dict_value
        else:
            return None


if __name__ == '__main__':
    # Track the running time
    start = time.time()
    # Ensure reproducibility
    seed = 0
    np.random.seed(seed)
    # Read the settings in config.py
    N_min = config.N_min
    N_max = config.N_max
    Images_op = config.Images_op
    Images_cl = config.Images_cl
    Deg = config.Deg
    Width = config.Width
    Height = config.Height
    Dpi = config.Dpi
    Linewidth = config.Linewidth
    # Create the spline generator
    Spl = SplineGenerator(h=Height, w=Width, dpi=Dpi, lw=Linewidth)
    # Dictionaries saving all the related information
    Dict_list = [{}, {}]
    # List of number of control points
    N_list = range(N_min, N_max + 1)
    # List of training and validation folder names
    Folders = ['.\\Splines\\Train\\', '.\\Splines\\Val\\']
    # Calculate the total images for training and validation
    Images_total = [len(N_list) * (Images_op[0] + Images_cl[0]),
                    len(N_list) * (Images_op[1] + Images_cl[1])]
    # Progress bar messages list
    Msgs = ['Creating images for training', 'Creating images for validation']
    # Json file list
    if not os.path.exists('.\\Json\\'):
        os.makedirs('.\\Json\\')
    Json = ['.\\Json\\img_train.json', '.\\Json\\img_val.json']
    # Loop for creating training dataset and validation dataset
    Image_num = 0
    for mode in range(2):
        with tqdm(total=Images_total[mode], desc=Msgs[mode]) as pbar:
            # Loop over all the numbers of control points
            for N in N_list:
                # The folder for placing the created image
                Folder = Folders[mode] + str(N) + '\\'
                # Create the open curves
                for idx in range(Images_op[mode]):
                    Image_num += 1
                    Image_name = str(Image_num).rjust(7, '0')
                    Dict_value = Spl(N, p=Deg, cl=False, folder=Folder, name=Image_name)
                    # Store the information
                    Dict_list[mode][Image_name] = Dict_value
                    pbar.update(1)
                # Create the closed curves
                for idx in range(Images_cl[mode]):
                    Image_num += 1
                    Image_name = str(Image_num).rjust(7, '0')
                    Dict_value = Spl(N, p=Deg, cl=True, folder=Folder, name=Image_name)
                    # Store the information
                    Dict_list[mode][Image_name] = Dict_value
                    pbar.update(1)
            # Save the stored information into json files
            Json_cache = json.dumps(Dict_list[mode], indent=2)
            Json_file = open(Json[mode], 'w')
            Json_file.write(Json_cache)
            Json_file.close()
    # Send an end message and show the running time
    end = time.time()
    print('The images have been successfully created within %.2f seconds.' % (end - start))
