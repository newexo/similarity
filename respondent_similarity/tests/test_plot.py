import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from respondent_similarity import directories
from respondent_similarity.loaders import load_cooccurrence_csv
from respondent_similarity.plot import plot_dendrogram, plot_similarity_matrix
from respondent_similarity.similarity import normalize


class TestPlotDendrogram:
    def setup_method(self):
        counts = load_cooccurrence_csv(directories.test_data("cooccurrence_counts.csv"))
        self.similarity = normalize(counts)

    def teardown_method(self):
        plt.close("all")

    def test_returns_figure_and_axes(self):
        fig, ax = plot_dendrogram(self.similarity)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_uses_card_labels_as_tick_text(self):
        _, ax = plot_dendrogram(self.similarity)
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert set(tick_labels) == set(self.similarity.index)

    def test_accepts_existing_axes(self):
        fig, ax = plt.subplots()
        result_fig, result_ax = plot_dendrogram(self.similarity, ax=ax)
        assert result_ax is ax
        assert result_fig is fig

    def test_title_applied(self):
        _, ax = plot_dendrogram(self.similarity, title="Card Sort")
        assert ax.get_title() == "Card Sort"

    def test_ylabel_is_distance(self):
        _, ax = plot_dendrogram(self.similarity)
        assert ax.get_ylabel() == "Distance"


class TestPlotSimilarityMatrix:
    def setup_method(self):
        counts = load_cooccurrence_csv(directories.test_data("cooccurrence_counts.csv"))
        self.similarity = normalize(counts)

    def teardown_method(self):
        plt.close("all")

    def test_returns_figure_and_axes(self):
        fig, ax = plot_similarity_matrix(self.similarity)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_tick_labels_match_cards(self):
        _, ax = plot_similarity_matrix(self.similarity)
        x_labels = [t.get_text() for t in ax.get_xticklabels()]
        y_labels = [t.get_text() for t in ax.get_yticklabels()]
        assert x_labels == list(self.similarity.columns)
        assert y_labels == list(self.similarity.index)

    def test_image_data_matches_similarity(self):
        _, ax = plot_similarity_matrix(self.similarity)
        (image,) = ax.get_images()
        assert (image.get_array() == self.similarity.values).all()

    def test_color_scale_is_unit_interval(self):
        _, ax = plot_similarity_matrix(self.similarity)
        (image,) = ax.get_images()
        assert image.get_clim() == (0.0, 1.0)

    def test_accepts_existing_axes(self):
        fig, ax = plt.subplots()
        result_fig, result_ax = plot_similarity_matrix(self.similarity, ax=ax)
        assert result_ax is ax
        assert result_fig is fig

    def test_title_applied(self):
        _, ax = plot_similarity_matrix(self.similarity, title="Card Sort Heatmap")
        assert ax.get_title() == "Card Sort Heatmap"
