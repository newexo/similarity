import pandas as pd


def load_cooccurrence_csv(path):
    """Load a card-sort co-occurrence count matrix from a CSV file.

    The CSV is expected to have card labels in both the header row and
    the first column. Body cells hold integer counts of how many
    participants placed each pair of cards into the same category;
    diagonal cells hold the number of participants who sorted each card.
    """
    return pd.read_csv(path, index_col=0)
