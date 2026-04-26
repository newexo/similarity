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


def plot_similarity_matrix(similarity, ax=None, title=None, cmap="viridis"):
    """Render a similarity matrix as a labeled heatmap.

    Cells are colored on a fixed [0, 1] scale so heatmaps from different
    datasets are directly comparable. Card labels from the DataFrame
    index/columns appear on the axes; a colorbar is added so absolute
    values are readable. Returns the (figure, axes) pair.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    image = ax.imshow(similarity.values, vmin=0.0, vmax=1.0, cmap=cmap)
    ax.set_xticks(range(len(similarity.columns)))
    ax.set_yticks(range(len(similarity.index)))
    ax.set_xticklabels(list(similarity.columns), rotation=45, ha="right")
    ax.set_yticklabels(list(similarity.index))
    fig.colorbar(image, ax=ax)
    if title is not None:
        ax.set_title(title)
    return fig, ax
