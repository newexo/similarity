def normalize(counts):
    """Convert a card-sort co-occurrence count matrix into a similarity matrix.

    Each cell is divided by the number of participants who sorted both
    cards, taken from the diagonal. The result is a similarity matrix in
    [0, 1] with 1.0 on the diagonal: similarity[i, j] is the proportion
    of participants who placed cards i and j in the same category.

    The diagonal of ``counts`` must be uniform — i.e. every card must
    have been sorted by the same number of participants. Datasets where
    participants only saw subsets of cards need a more general
    normalization that we can add when we encounter them.
    """
    diag = counts.values.diagonal()
    n = diag[0]
    if not (diag == n).all():
        raise ValueError(
            "Co-occurrence matrix has a non-uniform diagonal "
            f"({diag.tolist()}); normalize() currently requires every "
            "card to be sorted by the same number of participants."
        )
    if n == 0:
        raise ValueError("Co-occurrence matrix has zero participants on the diagonal.")
    return counts / n
