# -*- coding: utf-8 -*-
"""
- Time distribution histogram in each cell for each plane

@author: Sara Costa Faya

EDIT: Miguel Cruces

E-MAILs:
  - mcsquared.fz@gmail.com
  - miguel.cruces.fernandez@usc.es
"""

import sys
import os
from os.path import join as join_path
import numpy as np
import matplotlib.pyplot as plt
import copy
import time as tm


class CookingData:
    """
    Class which prepares and plots histograms
    """

    def __init__(self):
        # Root Directory of the Project
        self.root_dir = os.path.abspath("./")

        # Add ROOT_DIR to $PATH
        if self.root_dir not in sys.path:
            sys.path.append(self.root_dir)

        self.coord_ix = {1: [1, 1], 2: [2, 1], 3: [3, 1], 4: [4, 1], 5: [5, 1], 6: [6, 1],
                         7: [1, 2], 8: [2, 2], 9: [3, 2], 10: [4, 2], 11: [5, 2], 12: [6, 2],
                         13: [1, 3], 14: [2, 3], 15: [3, 3], 16: [4, 3], 17: [5, 3], 18: [6, 3],
                         19: [1, 4], 20: [2, 4], 21: [3, 4], 22: [4, 4], 23: [5, 4], 24: [6, 4],
                         25: [1, 5], 26: [2, 5], 27: [3, 5], 28: [4, 5], 29: [5, 5], 30: [6, 5]}

        time_0 = tm.time()

        # Read data
        data_path = join_path(self.root_dir, "dst_export.txt")
        data = np.loadtxt(data_path, delimiter=',')  # , max_rows=1000)  # , usecols=range(186))

        time_1 = tm.time()
        print(f"Time reading data: {time_1 - time_0:.3f} seconds")

        self.mdat = None
        # self.set_mdat(data)

        self.time_clusters = None
        self.char_clusters = None
        # self.set_matrices(data)

        self.diff_matrix = None
        self.different_multiplicities(data)

        time_2 = tm.time()
        print(f"Time setting matrices: {time_2 - time_1:.3f} seconds")

        # self.make_histograms(data)

        time_3 = tm.time()
        print(f"Time making histograms: {time_3 - time_2:.3f} seconds")
        print(f"Total time: {time_3 - time_0:.3f} seconds")

    def different_multiplicities(self, data):
        """
        Method which sets arrays by planes with anomalous multiplicities.
        :param data: Array with data taken from file.txt
        :return: Void function, but sets self.time_clusters and self.char_clusters
        """
        max_multi = 5  # Maximum multiplicity
        max_columns = 2 + 6*max_multi  # Maximum of columns we'll need
        # Void matrices of max_columns columns in a list for three planes:
        data_planes = [np.zeros([0, max_columns])] * 3
        for row in data:
            # Arrays with values of temperatures and charges of planes 1, 2 and 3 respectively
            times_list = [row[it:it + 30] for it in range(6, 127, 60)]  # len = 3
            chars_list = [row[iq:iq + 30] for iq in range(36, 157, 60)]  # len = 3

            # Arrays with indexes of non-zero values
            times_id_list = [np.nonzero(tim)[0] for tim in times_list]  # len = 3
            chars_id_list = [np.nonzero(cha)[0] for cha in chars_list]  # len = 3

            # MULTIPLICITY MATRICES [CLUSTERS]
            # print(f"row:{row[0]}")

            for p in range(3):
                if len(times_id_list[p]) > 5 or len(chars_id_list[p]) > 5:
                    continue
                if len(times_id_list[p]) != len(chars_id_list[p]):
                    print(f"* Plane {p+1}: time mult. {len(times_id_list[p])}, charge mult. {len(chars_id_list[p])}")

                    ary_times = [self.coord_ix[k + 1] for k in times_id_list[p]]
                    ary_chars = [self.coord_ix[k + 1] for k in chars_id_list[p]]

                    if len(ary_times) == 0:
                        coord_chars = np.hstack(ary_chars)
                        new_row = np.hstack((0, len(chars_id_list[p]),
                                             coord_chars, chars_list[p][chars_id_list[p]]))
                    elif len(ary_chars) == 0:
                        coord_times = np.hstack(ary_times)
                        new_row = np.hstack((len(times_id_list[p]), 0,
                                             coord_times, times_list[p][times_id_list[p]]))
                    else:
                        coord_times = np.hstack(ary_times)
                        coord_chars = np.hstack(ary_chars)
                        new_row = np.hstack((len(times_id_list[p]), len(chars_id_list[p]),
                                             coord_times, times_list[p][times_id_list[p]],
                                             coord_chars, chars_list[p][chars_id_list[p]]))
                    new_row = np.hstack((new_row, [0]*(max_columns - new_row.shape[0])))
                    print(f"New ROW Length: {new_row.shape[0]}")
                    data_planes[p] = np.vstack((data_planes[p], new_row))
        self.diff_matrix = data_planes

    def set_mdat(self, data, multi=None):
        """
        Function that returns an array with columns: [x coordinate, y coordinate, time, charge] x 3 planes
        :param data: Array with data taken from file.txt
        :param multi: Optional parameter which sets multiplicity of hits. Default [1, 1, 1]. Not implemented yet.
        :return: Array with all data specified above.
        """

        if multi is None:
            multi = [1, 1, 1]  # Default multiplicity

        # Cells position in Time & Charge matrices
        it1cel = 6
        it2cel = 66
        it3cel = 126

        iq1cel = 36
        iq2cel = 96
        iq3cel = 156

        m_dat = np.zeros([0, 4 * len(multi)])
        for row in data:
            # Arrays with values of temperatures and charges of planes 1, 2 and 3 respectively
            t1, t2, t3 = row[it1cel:it1cel + 30], row[it2cel:it2cel + 30], row[it3cel:it3cel + 30]
            q1, q2, q3 = row[iq1cel:iq1cel + 30], row[iq2cel:iq2cel + 30], row[iq3cel:iq3cel + 30]

            # Arrays with indexes of non-zero values
            t1_id = np.nonzero(t1)[0]
            t2_id = np.nonzero(t2)[0]
            t3_id = np.nonzero(t3)[0]
            q1_id = np.nonzero(q1)[0]
            q2_id = np.nonzero(q2)[0]
            q3_id = np.nonzero(q3)[0]

            # MDAT
            multi_list = [len(t1_id), len(t2_id), len(t3_id), len(q1_id), len(q2_id), len(q3_id)]

            # Only hits with multiplicity "multi" on time and charge (multi*2) will be stored on m_dat
            if multi_list == multi * 2:
                kx1, ky1 = self.coord_ix[t1_id[0] + 1]
                kx2, ky2 = self.coord_ix[t2_id[0] + 1]
                kx3, ky3 = self.coord_ix[t3_id[0] + 1]

                new_row = np.hstack((kx1, ky1, t1[t1_id[0]], q1[q1_id[0]],
                                     kx2, ky2, t2[t2_id[0]], q2[q2_id[0]],
                                     kx3, ky3, t3[t3_id[0]], q3[q3_id[0]]))
                m_dat = np.vstack((m_dat, new_row))

        self.mdat = m_dat

    def set_matrices(self, data):
        """
        Method which sets lists with arrays of hits sorted by multiplicities.
        :param data: Array with data taken from file.txt
        :return: Void function, but sets self.time_clusters and self.char_clusters
        """
        time_clusters = empty_list([3, 5])
        char_clusters = empty_list([3, 5])
        for row in data:
            # Arrays with values of temperatures and charges of planes 1, 2 and 3 respectively
            times_list = [row[it:it + 30] for it in range(6, 127, 60)]
            chars_list = [row[iq:iq + 30] for iq in range(36, 157, 60)]

            # Arrays with indexes of non-zero values
            times_id_list = [np.nonzero(tim)[0] for tim in times_list]
            chars_id_list = [np.nonzero(cha)[0] for cha in chars_list]

            # MULTIPLICITY MATRICES [CLUSTERS]

            for p in range(len(times_id_list)):  # Planes
                multiplicities = len(times_id_list[p])
                if multiplicities == 0 or multiplicities > 5:
                    continue
                ary = [self.coord_ix[k + 1] for k in times_id_list[p]]
                coords = np.hstack(ary)
                # common_ix = np.intersect1d(t1_id, q1_id)  # Con esto tenemos un array con los valores comunes
                time_clusters[p][multiplicities - 1].append(np.hstack((coords, times_list[p][times_id_list[p]])))

            for p in range(len(chars_id_list)):  # Planes
                multiplicities = len(chars_id_list[p])
                if multiplicities == 0 or multiplicities > 5:
                    continue
                ary = [self.coord_ix[k + 1] for k in chars_id_list[p]]
                coords = np.hstack(ary)
                # common_ix = np.intersect1d(t1_id, q1_id)  # Con esto tenemos un array con los valores comunes
                char_clusters[p][multiplicities - 1].append(np.hstack((coords, chars_list[p][chars_id_list[p]])))

        for p in range(len(time_clusters)):  # Planes
            for m in range(len(time_clusters[p])):  # Multiplicities
                time_clusters[p][m] = np.vstack(time_clusters[p][m])

        for p in range(len(char_clusters)):  # Planes
            for m in range(len(char_clusters[p])):  # Multiplicities
                char_clusters[p][m] = np.vstack(char_clusters[p][m])
        self.time_clusters = time_clusters
        self.char_clusters = char_clusters

    def make_histograms(self, data):
        """
        This function sorts all data and creates 180 histograms for
        time and charge, one for each cell in each plane. They are
        stored on ROOT_DIR/Histograms/
        :return: It is a void function.
        """
        self.set_mdat(data=data)
        # 3D array of shape (No. Planes, No. X cells, No. Y cells)
        data_iter = np.asarray(np.hsplit(self.mdat, 3))
        time_hist = empty_list([3, 6, 5])
        char_hist = empty_list([3, 6, 5])
        for p in range(data_iter.shape[0]):
            for row in data_iter[p]:
                x, y = int(row[0] - 1), int(row[1] - 1)
                time = row[2]
                charge = row[3]
                time_hist[p][x - 1][y - 1].append(time)
                char_hist[p][x - 1][y - 1].append(charge)

        histograms_dir = join_path(self.root_dir, "Histograms")
        if not os.path.exists(histograms_dir):
            os.mkdir(histograms_dir)

        # Time Histograms
        plot_hist(time_hist, kind="time")

        # Charge Histograms
        plot_hist(char_hist, kind="charge")


def plot_hist(data_hist, kind: str):
    """
    This function save all histograms on "./Histograms"
    :param data_hist: list of lists with all data
    :param kind: string ["time", "charge"]
    :return: It is a void function
    """
    for p in range(len(data_hist)):
        for x in range(len(data_hist[p])):
            for y in range(len(data_hist[p][x])):
                plt.figure(f"{kind}_p{p + 1}x{x + 1}y{y + 1}")
                plt.title(f"Plane {p + 1} - Cell ({x + 1}, {y + 1}) - {kind}")
                arr = np.asarray(data_hist[p][x][y])
                filtered = arr[~is_outlier(arr)]
                plt.hist(filtered, bins="auto")
                plt.xlabel(f"{kind} (T{p + 1})")
                plt.ylabel(f"# Counts")
                plt.savefig(f"./Histograms/{kind}_T{p + 1}_{x + 1}{y + 1}.png")
                plt.close(f"{kind}_p{p + 1}x{x + 1}y{y + 1}")


def empty_list(shape):
    """
    Function empty_list creates empty lists with a given shape
    :param shape: list with the wanted shape.
    :return: empty list of empty lists of empty lists...
    """
    if len(shape) == 1:
        return [[] for _ in range(shape[0])]
    items = shape[0]
    newshape = shape[1:]
    sublist = empty_list(newshape)
    return [copy.deepcopy(sublist) for _ in range(items)]


def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.
    :param points: An numobservations by numdimensions array of observations
    :param thresh: The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.
    :return: (mask) A numobservations-length boolean array.
    :references:
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


if __name__ == "__main__":
    CD = CookingData()
    # mdat = CD.mdat

    # # k_clusters = list: [Plane][Multiplicity] -> [array]
    # t_clusters = CD.time_clusters
    # c_clusters = CD.char_clusters
