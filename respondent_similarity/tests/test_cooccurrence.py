import pandas as pd

from respondent_similarity.cooccurrence import cooccurrence_from_responses
from respondent_similarity.similarity import normalize


CARD_IDS = [1, 2, 3, 4, 5, 6]


class TestCooccurrenceFromResponses:
    def test_returns_dataframe(self, responses):
        counts = cooccurrence_from_responses(responses)
        assert isinstance(counts, pd.DataFrame)

    def test_indexed_by_card_id(self, responses):
        counts = cooccurrence_from_responses(responses)
        assert list(counts.index) == CARD_IDS
        assert list(counts.columns) == CARD_IDS

    def test_symmetric(self, responses):
        counts = cooccurrence_from_responses(responses)
        assert (counts.values == counts.values.T).all()

    def test_diagonal_is_participants_per_card(self, responses):
        counts = cooccurrence_from_responses(responses)
        # Every participant sorted every card in the fixture.
        assert (counts.values.diagonal() == 5).all()

    def test_full_expected_matrix(self, responses):
        # Hand-computed from the fixture. The two clusters
        # ({1, 2, 3} and {4, 5, 6}) should dominate, with participant 4
        # introducing one cross-cluster pair (card 6 sits in their
        # "Big bucket" category alongside cards 1, 2, 3).
        expected = pd.DataFrame(
            [
                [5, 5, 4, 0, 0, 1],
                [5, 5, 4, 0, 0, 1],
                [4, 4, 5, 0, 0, 1],
                [0, 0, 0, 5, 5, 4],
                [0, 0, 0, 5, 5, 4],
                [1, 1, 1, 4, 4, 5],
            ],
            index=CARD_IDS,
            columns=CARD_IDS,
        )
        counts = cooccurrence_from_responses(responses)
        pd.testing.assert_frame_equal(counts, expected, check_dtype=False)

    def test_responses_to_similarity_pipeline(self, responses):
        # Round-trip: raw long-format responses → co-occurrence counts →
        # similarity matrix. Each off-diagonal cell is the proportion of
        # participants who placed that pair into the same category;
        # diagonal is 1.0 by definition. The counts fixture has a
        # uniform diagonal of 5, so normalize infers n_participants
        # without needing it passed explicitly.
        expected = pd.DataFrame(
            [
                [1.0, 1.0, 0.8, 0.0, 0.0, 0.2],
                [1.0, 1.0, 0.8, 0.0, 0.0, 0.2],
                [0.8, 0.8, 1.0, 0.0, 0.0, 0.2],
                [0.0, 0.0, 0.0, 1.0, 1.0, 0.8],
                [0.0, 0.0, 0.0, 1.0, 1.0, 0.8],
                [0.2, 0.2, 0.2, 0.8, 0.8, 1.0],
            ],
            index=CARD_IDS,
            columns=CARD_IDS,
        )
        similarity = normalize(cooccurrence_from_responses(responses))
        pd.testing.assert_frame_equal(similarity, expected, check_dtype=False)

    def test_open_sort_category_names_do_not_matter(self, responses):
        # Renaming every category label to a unique string per row must
        # not change any count, because co-occurrence only depends on
        # whether two cards share a category *within a single
        # participant*.
        renamed = responses.copy()
        renamed["category_label"] = (
            renamed["participant_id"].astype(str) + "::" + renamed["category_label"]
        )
        pd.testing.assert_frame_equal(
            cooccurrence_from_responses(responses),
            cooccurrence_from_responses(renamed),
        )
