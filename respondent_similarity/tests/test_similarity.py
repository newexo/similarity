import numpy as np
import pandas as pd
import pytest

from respondent_similarity.similarity import infer_n_participants, normalize


class TestInferNParticipants:
    def test_returns_uniform_diagonal_value(self, cooccurrence_counts):
        assert infer_n_participants(cooccurrence_counts) == 20

    def test_works_on_minimal_matrix(self):
        counts = pd.DataFrame([[6, 2], [2, 6]], index=["A", "B"], columns=["A", "B"])
        assert infer_n_participants(counts) == 6

    def test_rejects_empty_diagonal(self, cooccurrence_counts_empty_diagonal):
        with pytest.raises(ValueError, match="Cannot infer n_participants"):
            infer_n_participants(cooccurrence_counts_empty_diagonal)

    def test_rejects_zero_diagonal(self):
        zeros = pd.DataFrame([[0, 0], [0, 0]], index=["A", "B"], columns=["A", "B"])
        with pytest.raises(ValueError, match="Cannot infer n_participants"):
            infer_n_participants(zeros)

    def test_rejects_non_uniform_diagonal(self):
        ragged = pd.DataFrame([[20, 5], [5, 10]], index=["A", "B"], columns=["A", "B"])
        with pytest.raises(ValueError, match="Cannot infer n_participants"):
            infer_n_participants(ragged)


class TestNormalize:
    def test_returns_dataframe(self, similarity):
        assert isinstance(similarity, pd.DataFrame)

    def test_labels_preserved(self, cooccurrence_counts, similarity):
        assert list(similarity.index) == list(cooccurrence_counts.index)
        assert list(similarity.columns) == list(cooccurrence_counts.columns)

    def test_diagonal_is_one(self, similarity):
        np.testing.assert_allclose(similarity.values.diagonal(), 1.0)

    def test_values_in_unit_interval(self, similarity):
        values = similarity.values
        assert (values >= 0).all()
        assert (values <= 1).all()

    def test_symmetry_preserved(self, similarity):
        np.testing.assert_array_equal(similarity.values, similarity.values.T)

    def test_off_diagonal_matches_counts_divided_by_n(
        self, cooccurrence_counts, similarity
    ):
        mask = ~np.eye(len(cooccurrence_counts), dtype=bool)
        np.testing.assert_allclose(
            similarity.values[mask], cooccurrence_counts.values[mask] / 20
        )

    def test_input_diagonal_is_ignored(
        self, cooccurrence_counts, cooccurrence_counts_empty_diagonal
    ):
        full_sim = normalize(cooccurrence_counts, n_participants=20)
        empty_sim = normalize(cooccurrence_counts_empty_diagonal, n_participants=20)
        pd.testing.assert_frame_equal(full_sim, empty_sim)

    def test_explicit_n_participants_overrides_diagonal(self, cooccurrence_counts):
        explicit = normalize(cooccurrence_counts, n_participants=10)
        np.testing.assert_allclose(explicit.values.diagonal(), 1.0)
        # Off-diagonal is now /10 rather than /20.
        np.testing.assert_allclose(explicit.values[0, 1], 17 / 10)

    def test_inference_used_when_n_omitted(self, cooccurrence_counts):
        # Integration: normalize delegates to infer_n_participants when
        # n_participants is not supplied.
        np.testing.assert_allclose(
            normalize(cooccurrence_counts).values,
            normalize(cooccurrence_counts, n_participants=20).values,
        )

    def test_propagates_inference_failure(self, cooccurrence_counts_empty_diagonal):
        with pytest.raises(ValueError, match="Cannot infer n_participants"):
            normalize(cooccurrence_counts_empty_diagonal)

    def test_rejects_non_positive_n_participants(self, cooccurrence_counts):
        with pytest.raises(ValueError, match="n_participants must be positive"):
            normalize(cooccurrence_counts, n_participants=0)
