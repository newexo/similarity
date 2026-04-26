import matplotlib
import pytest

from respondent_similarity import directories
from respondent_similarity.loaders import load_cooccurrence_csv
from respondent_similarity.similarity import normalize

matplotlib.use("Agg")


@pytest.fixture
def cooccurrence_counts():
    return load_cooccurrence_csv(directories.test_data("cooccurrence_counts.csv"))


@pytest.fixture
def cooccurrence_counts_empty_diagonal():
    return load_cooccurrence_csv(
        directories.test_data("cooccurrence_counts_empty_diagonal.csv")
    )


@pytest.fixture
def similarity(cooccurrence_counts):
    return normalize(cooccurrence_counts)
