

import numpy as np

from typing import Sequence
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import CenteredNorm, Normalize


class NetworkPlotter:
    """Class to plot networks with nodes and edges via matplotlib."""

    def __init__(self, annotation_kwargs: dict = None, edge_snorm=None, edge_cmap: str = "coolwarm", edge_cnorm=None, edge_cbar: bool = True, node_cmap: str = "coolwarm", node_cnorm=None, max_node_label_len: int = 5):
        self.annotation_kwargs = dict(annotation_clip=True, clip_on=True)
        if annotation_kwargs is not None:
            self.annotation_kwargs.update(annotation_kwargs)
        self.edge_cmap = plt.cm.get_cmap(edge_cmap)
        self.edge_cnorm = CenteredNorm(vcenter=0) if edge_cnorm is None else edge_cnorm
        self.edge_cbar = edge_cbar
        self.edge_snorm = Normalize(vmin=1, vmax=3) if edge_snorm is None else edge_snorm
        self.node_cnorm = plt.cm.get_cmap(node_cmap)
        self.node_cmap = CenteredNorm(vcenter=0) if edge_cnorm is None else edge_cnorm
        self.max_node_label_len = max_node_label_len

    def plot(
        self, pos, edge_list, node_labels: Sequence[str] = None, node_sizes: Sequence[float] = None, node_colors=None, node_markers=None,
        edge_weights: Sequence[float] = None, edge_colors=None, edge_cbar_label: str = None, title: str = None, ax: plt.Axes = None
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
        self.plot_edges(ax, pos, edge_list, edge_weights, edge_colors, edge_cbar_label)
        self.plot_nodes(ax, pos, node_sizes, node_colors, node_markers)
        if node_labels is not None:
            self.label_nodes(ax, pos, node_labels)
        if title is not None:
            ax.set_title(title)
        return ax

    def plot_edges(self, ax: plt.Axes, pos, edge_list, edge_weights=None, edge_colors=None, edge_cbar_label: str = None):
        """Adds the edges to the plot."""
        segments = [(pos[i], pos[j]) for i, j in edge_list]
        colors = None
        if edge_colors is not None:
            colors = self.edge_cmap(self.edge_cnorm(edge_colors))
            if self.edge_cbar:
                mappable = ScalarMappable(self.edge_cnorm, self.edge_cmap)
                plt.colorbar(mappable, ax=ax, label=edge_cbar_label)
        linewidths = edge_weights
        if linewidths is not None:
            linewidths = self.edge_snorm(linewidths)
        edges = LineCollection(segments, linewidths=linewidths, colors=colors)
        ax.add_collection(edges)

    def plot_nodes(self, ax: plt.Axes, pos, node_sizes=None, node_colors=None, node_markers=None):
        """Adds the nodes to the plot."""
        for i, (x, y) in enumerate(pos):
            x, y = pos[i]
            marker = node_markers[i] if node_markers is not None else None
            s = node_sizes[i] if node_sizes is not None else None
            c = node_colors[i] if node_colors is not None else None
            ax.scatter(x, y, s=s, c=c, marker=marker, norm=self.node_cnorm, cmap=self.node_cmap)

    def label_nodes(self, ax: plt.Axes, pos, node_labels):
        """Annotates each node with the corresponding labels."""
        assert len(pos) == len(node_labels)
        for i, label in enumerate(node_labels):
            if self.max_node_label_len >= 0:
                label = f"{label[:self.max_node_label_len]}."
            ax.annotate(label, pos[i], **self.annotation_kwargs)
