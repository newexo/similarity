def cooccurrence_from_responses(responses):
    """Build a card-to-card co-occurrence count matrix from raw card-sort responses.

    ``responses`` is the long-format table with one row per participant per
    card, with columns ``participant_id``, ``card_id``, ``card_label``, and
    ``category_label``. Cards are identified in the output by ``card_id``;
    ``card_label`` is read but otherwise unused here, since two participants
    can give the same conceptual category different names (open card sort)
    and only the within-participant grouping matters for co-occurrence.

    The output is a square ``pandas.DataFrame`` of integer counts indexed
    and columned by ``card_id``. ``cell[i, j]`` is the number of
    participants who placed cards ``i`` and ``j`` into the same category.
    The diagonal carries the number of participants who sorted each card.
    """
    pairs = responses.merge(
        responses,
        on=["participant_id", "category_label"],
        suffixes=("_a", "_b"),
    )
    counts = pairs.groupby(["card_id_a", "card_id_b"]).size().unstack(fill_value=0)
    counts.index.name = None
    counts.columns.name = None
    return counts
