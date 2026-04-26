import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


def plot_dendrogram(similarity, ax=None, method="average", title=None):
    """Render a hierarchical-clustering dendrogram from a similarity matrix.

    Similarity is converted to distance via ``1 - similarity``, the square
    matrix is condensed for SciPy, and ``linkage`` is run with the chosen
    method (default ``"average"``). If ``ax`` is given the dendrogram is
    drawn into it; otherwise a new figure and axes are created. Returns
    the (figure, axes) pair so callers can save or further customize.
    """
    distance = 1 - similarity.values
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    linked = linkage(condensed, method=method)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    dendrogram(linked, labels=list(similarity.index), ax=ax)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel("Distance")
    return fig, ax
