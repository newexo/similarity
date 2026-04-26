import pandas as pd

from respondent_similarity import directories
from respondent_similarity.loaders import load_cooccurrence_csv


CARDS = ["Shipping", "Returns", "Tracking", "Login", "Profile", "Account"]


class TestLoadCooccurrenceCsv:
    def test_returns_dataframe(self):
        df = load_cooccurrence_csv(directories.test_data("cooccurrence_counts.csv"))
        assert isinstance(df, pd.DataFrame)

    def test_labels_match_fixture(self):
        df = load_cooccurrence_csv(directories.test_data("cooccurrence_counts.csv"))
        assert list(df.index) == CARDS
        assert list(df.columns) == CARDS

    def test_matrix_is_symmetric(self):
        df = load_cooccurrence_csv(directories.test_data("cooccurrence_counts.csv"))
        assert (df.values == df.values.T).all()

    def test_diagonal_is_participant_count(self):
        df = load_cooccurrence_csv(directories.test_data("cooccurrence_counts.csv"))
        # Fixture has 20 participants and every participant sorted every card.
        assert (df.values.diagonal() == 20).all()

    def test_off_diagonal_at_most_diagonal(self):
        df = load_cooccurrence_csv(directories.test_data("cooccurrence_counts.csv"))
        n = df.values.diagonal().max()
        assert (df.values <= n).all()
