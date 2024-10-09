'''
Code from
github repo population-stability-index ->https://github.com/mwburke/population-stability-index
'''

import numpy as np


def calculate_psi(expected, actual, buckettype='bins', buckets=100, axis=0):
    bin_edges = np.histogram_bin_edges(np.concatenate([expected, actual]), bins=buckets)

    # Calculate the proportions for each list in the same bins
    list_1_hist, _ = np.histogram(expected, bins=bin_edges)
    list_2_hist, _ = np.histogram(actual, bins=bin_edges)

    # Convert counts to proportions
    list_1_proportions = list_1_hist / len(expected)
    list_2_proportions = list_2_hist / len(actual)

    # PSI calculation: handle 0 counts by replacing with a small value to avoid division by zero
    epsilon = 1e-6
    psi_values = (list_1_proportions - list_2_proportions) * np.log(
        (list_1_proportions + epsilon) / (list_2_proportions + epsilon))

    # Total PSI
    psi_total = np.sum(psi_values)
    return psi_total
