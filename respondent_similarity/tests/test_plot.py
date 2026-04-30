import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from respondent_similarity.plot import plot_dendrogram, plot_similarity_matrix


class TestPlotDendrogram:
    def teardown_method(self):
        plt.close("all")

    def test_returns_figure_and_axes(self, similarity):
        fig, ax = plot_dendrogram(similarity)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_uses_card_labels_as_tick_text(self, similarity):
        _, ax = plot_dendrogram(similarity)
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert set(tick_labels) == set(similarity.index)

    def test_accepts_existing_axes(self, similarity):
        fig, ax = plt.subplots()
        result_fig, result_ax = plot_dendrogram(similarity, ax=ax)
        assert result_ax is ax
        assert result_fig is fig

    def test_title_applied(self, similarity):
        _, ax = plot_dendrogram(similarity, title="Card Sort")
        assert ax.get_title() == "Card Sort"

    def test_ylabel_is_distance(self, similarity):
        _, ax = plot_dendrogram(similarity)
        assert ax.get_ylabel() == "Distance"


class TestPlotSimilarityMatrix:
    def teardown_method(self):
        plt.close("all")

    def test_returns_figure_and_axes(self, similarity):
        fig, ax = plot_similarity_matrix(similarity)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_tick_labels_match_cards(self, similarity):
        _, ax = plot_similarity_matrix(similarity)
        x_labels = [t.get_text() for t in ax.get_xticklabels()]
        y_labels = [t.get_text() for t in ax.get_yticklabels()]
        assert x_labels == list(similarity.columns)
        assert y_labels == list(similarity.index)

    def test_image_data_matches_similarity(self, similarity):
        _, ax = plot_similarity_matrix(similarity)
        (image,) = ax.get_images()
        assert (image.get_array() == similarity.values).all()

    def test_color_scale_is_unit_interval(self, similarity):
        _, ax = plot_similarity_matrix(similarity)
        (image,) = ax.get_images()
        assert image.get_clim() == (0.0, 1.0)

    def test_accepts_existing_axes(self, similarity):
        fig, ax = plt.subplots()
        result_fig, result_ax = plot_similarity_matrix(similarity, ax=ax)
        assert result_ax is ax
        assert result_fig is fig

    def test_title_applied(self, similarity):
        _, ax = plot_similarity_matrix(similarity, title="Card Sort Heatmap")
        assert ax.get_title() == "Card Sort Heatmap"
