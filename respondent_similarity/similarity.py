import numpy as np


def normalize(counts, n_participants=None):
    """Convert a card-sort co-occurrence count matrix into a similarity matrix.

    The output is a ``pandas.DataFrame`` in [0, 1] with 1.0 on the
    diagonal: ``similarity[i, j]`` is the proportion of participants who
    placed cards i and j into the same category.

    Values along the input diagonal are ignored. The diagonal of a
    similarity matrix is 1.0 by definition, so it is filled in regardless
    of what the input contained — this lets the function accept matrices
    where the diagonal was left blank, recorded as zero, or filled with
    something else entirely.

    The number of participants is taken from ``n_participants`` when
    given. If omitted, it is inferred from the input diagonal, which
    must then carry a uniform non-zero value. When inference is not
    possible (the diagonal is blank, zero, or non-uniform) ``normalize``
    raises ``ValueError`` and the caller should pass ``n_participants``
    explicitly.
    """
    if n_participants is None:
        diag = counts.values.diagonal().astype(float)
        usable = diag[~np.isnan(diag) & (diag != 0)]
        if len(usable) != len(diag) or not (usable == usable[0]).all():
            raise ValueError(
                f"Cannot infer n_participants from the diagonal ({diag.tolist()}); "
                "pass n_participants explicitly."
            )
        n_participants = usable[0]

    if n_participants <= 0:
        raise ValueError("n_participants must be positive.")

    similarity = counts.astype(float) / n_participants
    np.fill_diagonal(similarity.values, 1.0)
    return similarity
