

import numpy as np

from numpy.typing import ArrayLike
from typing import Callable
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import CenteredNorm, Normalize
from correlnet.segments import SegmentPlotter


class NetworkPlotter:
    """Class to plot networks with nodes and edges via matplotlib."""

    def __init__(
        self, annotation_kwargs: dict = None, edge_snorm: Callable = None, edge_cmap: str = "coolwarm", edge_cnorm: Callable = None, edge_cbar: bool = True, node_cmap: str = "coolwarm",
        node_cnorm: Callable = None, shorten_node_labels: bool = True, max_node_label_len: int = 5, annotate_labels: bool = True
    ):
        """
        Args:
            annotation_kwargs (dict): keyword arguments that are passed to ax.annotate for node labeling
            edge_snorm (Callable): used to normalize edge weights to line widths
            edge_cmap (str): Name of the matplotlib Colormap for the edges
            edge_cnorm (Callable): used to normalize edge weights to colors from edge_cmap
            edge_cbar (bool): Add a colorbar for the edge weights
            node_cmap (str): Name of the matplotlib Colormap used for nodes
            node_cnorm: (Callable): used to normalize node colors
            shorten_node_labels (bool): If true, will shorten node labels to max_node_label_len
            max_node_label_len (int): maximum length of node labels
        """
        self.annotation_kwargs = dict(annotation_clip=True, clip_on=True, ha="center", va="center")
        if annotation_kwargs is not None:
            self.annotation_kwargs.update(annotation_kwargs)
        self.edge_cmap = plt.cm.get_cmap(edge_cmap)
        self.edge_cnorm = CenteredNorm(vcenter=0) if edge_cnorm is None else edge_cnorm
        self.edge_cbar = edge_cbar
        self.edge_snorm = Normalize(vmin=1, vmax=3) if edge_snorm is None else edge_snorm
        self.node_cmap = plt.cm.get_cmap(node_cmap)
        self.node_cnorm = CenteredNorm(vcenter=0) if edge_cnorm is None else node_cnorm
        self.shorten_node_labels = shorten_node_labels
        self.max_node_label_len = max_node_label_len
        self.annotate_labels = annotate_labels

    def plot(
        self, pos, edge_list, node_labels: ArrayLike = None, node_sizes: ArrayLike = None, node_colors=None,
        edge_weights: ArrayLike = None, edge_colors=None, edge_c=None, edge_cbar_label: str = None, ax: plt.Axes = None
    ) -> plt.Axes:
        """
        Plots a network via matplotlib.

        Args:
            pos: sequence of xy node positions
            edge_list: sequence of node indices (i,j) corresponding to positions in pos
            node_labels: sequence of strings
            node_sizes: sequence of node sizes for plt.scatter
            node_colors: sequence of node colors for plt.scatter
            node_markers: sequence of node markers for plt.scatter
            edge_weights: if provided, will be mapped to line sizes for each edge
            edge_colors: if provided, will be used to color edges using this plotters edge_cmap and edge_cnorm
            edge_cbar_label (str): if edge_colors is provided, use this as label for the colorbar
            ax (plt.Axes): matplotlib Axes object to draw the plot on
        """
        if ax is None:
            fig, ax = plt.subplots()
        self.plot_edges(ax, pos, edge_list, edge_weights, edge_colors, edge_c, edge_cbar_label)
        self.plot_nodes(ax, pos, node_sizes, node_colors)
        if node_labels is not None and self.annotate_labels:
            self.label_nodes(ax, pos, node_labels)
        ax.autoscale()
        return ax

    def plot_edges(self, ax: plt.Axes, pos, edge_list, edge_weights=None, edge_colors=None, edge_c=None, edge_cbar_label: str = None):
        """Adds the edges to the plot."""
        segments = np.array([(pos[i], pos[j]) for i, j in edge_list])
        x, y = segments[:, 0].T
        x_end, y_end = segments[:, 1].T
        if edge_c is not None and self.edge_cbar:
            mappable = ScalarMappable(self.edge_cnorm, self.edge_cmap)
            plt.colorbar(mappable, ax=ax, label=edge_cbar_label)
        linewidths = edge_weights
        if linewidths is not None:
            linewidths = self.edge_snorm(linewidths)
        segment_plotter = SegmentPlotter()
        segment_plotter.plot(x, y, x_end, y_end, linewidths, colors=edge_colors, c=edge_c, cmap=self.edge_cmap, norm=self.edge_cnorm, ax=ax)

    def plot_nodes(self, ax: plt.Axes, pos, node_sizes=None, node_colors=None):
        """Adds the nodes to the plot."""
        x, y = np.array(pos).T
        if node_colors is None:
            ax.scatter(x, y, s=node_sizes)
        else:
            ax.scatter(x, y, s=node_sizes, c=node_colors, norm=self.node_cnorm, cmap=self.node_cmap)

    def label_nodes(self, ax: plt.Axes, pos, node_labels):
        """Annotates each node with the corresponding labels."""
        assert len(pos) == len(node_labels)
        for i, label in enumerate(node_labels):
            if self.shorten_node_labels and self.max_node_label_len >= 0 and len(label) > self.max_node_label_len:
                label = f"{label[:self.max_node_label_len]}."
            ax.annotate(label, pos[i], **self.annotation_kwargs)
