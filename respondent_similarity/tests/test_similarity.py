import numpy as np
import pandas as pd
import pytest

from respondent_similarity import directories
from respondent_similarity.loaders import load_cooccurrence_csv
from respondent_similarity.similarity import normalize


class TestNormalize:
    def setup_method(self):
        self.counts = load_cooccurrence_csv(
            directories.test_data("cooccurrence_counts.csv")
        )
        self.similarity = normalize(self.counts)

    def test_returns_dataframe(self):
        assert isinstance(self.similarity, pd.DataFrame)

    def test_labels_preserved(self):
        assert list(self.similarity.index) == list(self.counts.index)
        assert list(self.similarity.columns) == list(self.counts.columns)

    def test_diagonal_is_one(self):
        np.testing.assert_allclose(self.similarity.values.diagonal(), 1.0)

    def test_values_in_unit_interval(self):
        values = self.similarity.values
        assert (values >= 0).all()
        assert (values <= 1).all()

    def test_symmetry_preserved(self):
        np.testing.assert_array_equal(self.similarity.values, self.similarity.values.T)

    def test_matches_counts_divided_by_n(self):
        np.testing.assert_allclose(self.similarity.values, self.counts.values / 20)

    def test_rejects_non_uniform_diagonal(self):
        ragged = pd.DataFrame([[20, 5], [5, 10]], index=["A", "B"], columns=["A", "B"])
        with pytest.raises(ValueError, match="non-uniform diagonal"):
            normalize(ragged)

    def test_rejects_zero_diagonal(self):
        empty = pd.DataFrame([[0, 0], [0, 0]], index=["A", "B"], columns=["A", "B"])
        with pytest.raises(ValueError, match="zero participants"):
            normalize(empty)
