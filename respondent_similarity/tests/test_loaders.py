import pandas as pd


CARDS = ["Shipping", "Returns", "Tracking", "Login", "Profile", "Account"]


class TestLoadCooccurrenceCsv:
    def test_returns_dataframe(self, cooccurrence_counts):
        assert isinstance(cooccurrence_counts, pd.DataFrame)

    def test_labels_match_fixture(self, cooccurrence_counts):
        assert list(cooccurrence_counts.index) == CARDS
        assert list(cooccurrence_counts.columns) == CARDS

    def test_matrix_is_symmetric(self, cooccurrence_counts):
        values = cooccurrence_counts.values
        assert (values == values.T).all()

    def test_diagonal_is_participant_count(self, cooccurrence_counts):
        # Fixture has 20 participants and every participant sorted every card.
        assert (cooccurrence_counts.values.diagonal() == 20).all()

    def test_off_diagonal_at_most_diagonal(self, cooccurrence_counts):
        n = cooccurrence_counts.values.diagonal().max()
        assert (cooccurrence_counts.values <= n).all()
