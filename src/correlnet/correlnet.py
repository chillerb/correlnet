"""
Draw Correlation Networks
"""


import networkx as nx
import numpy as np
import pandas as pd
import logging

from typing import Callable, Literal
from matplotlib import pyplot as plt
from typing import Union

from correlnet.annotator import Annotator
from correlnet.correlater import Correlater
from correlnet.embedder import Embedder, CorrelTSNEEmbedder, VarTSNEEmbedder, RandomEmbedder
from correlnet.plotter import NetworkPlotter

logger = logging.getLogger(__name__)


def correlnet(
    df: pd.DataFrame, method: Literal["pearson", "spearman", "kendall"] = "pearson",
    embedding: Literal["random", "var_tsne", "correl_tsne"] = "var_tsne",
    correction: str = "bonferroni",
    alpha: float = 0.05, tsne_kwargs: dict = None, standardize: bool = True, use_abs=True, random_state=None
):
    """
    Helps to constructs a new CorrelNet object.

    Args:

    """
    tsne_kwargs = dict() if tsne_kwargs is None else tsne_kwargs
    if not "random_state" in tsne_kwargs:
        tsne_kwargs["random_state"] = random_state
    correlater = Correlater(method=method, correction=correction)

    if embedding == "random":
        embedder = RandomEmbedder(random_state=random_state)
    elif embedding == "var_tsne":
        embedder = VarTSNEEmbedder(standardize=standardize, **tsne_kwargs)
    elif embedding == "correl_tsne":
        embedder = CorrelTSNEEmbedder(use_abs=use_abs, **tsne_kwargs)
    elif isinstance(embedding, Embedder):
        embedder = embedding
    else:
        raise NotImplementedError(f"embedding {embedding} is not supported")

    return CorrelNet(df, correlater=correlater, embedder=embedder, alpha=alpha)


class CorrelNet:
    """
    Computes and plots the correlation graph between variables. 
    """

    def __init__(
        self, df: pd.DataFrame, correlater: Correlater,
        embedder: Embedder,
        annotator: Annotator = None,
        net_plotter: NetworkPlotter = None,
        alpha: float = 0.05
    ):
        """
        Args:
            alpha (float): Threshold for statistical significance of correlations.

        """
        self.df = df
        self.vars = df.columns.to_list()
        self.correlater = correlater
        self.embedder = embedder
        self.alpha = alpha
        # compute correlations
        self.correl_df = self.correlater.pairwise_correlations(self.df)
        # compute embedding positions
        self.pos_df = self.embedder.embed(self.df, self.correl_df)
        self.plotter = NetworkPlotter() if net_plotter is None else net_plotter

    def plot(self, ax=None) -> plt.Axes:
        """Plots the correlation graph using the computed positions."""
        if ax is None:
            fig, ax = plt.subplots()
        node_labels = self.vars.copy()
        pos = [self.pos_df.loc[var] for var in self.vars]
        df = self.correl_df[self.correl_df["pvalue"] <= self.alpha]
        edge_list = [(self.vars.index(var_1), self.vars.index(var_2)) for var_1, var_2 in df.index]
        edge_colors = df["statistic"]
        edge_weights = df["statistic"]
        edge_cbar_label = self.correlater.method
        self.plotter.plot(pos, edge_list, node_labels, edge_c=edge_colors, edge_weights=edge_weights, edge_cbar_label=edge_cbar_label)
        return ax

    def to_graph(self, apply_filter=True) -> nx.Graph:
        """Returns correlations as networkx graph with attributes [statistic, pvalue].

        Args:
            apply_filter (bool): If true, apply an edge filter to remove self loops and non-significant edges
        """
        df = self.correl_df.reset_index(names=["var_1", "var_2"])
        graph = nx.from_pandas_edgelist(df, source="var_1", target="var_2", edge_attr=True)

        if apply_filter:
            logger.debug(f"applying edge filter with alpha={self.alpha}")
            # filter graph by significance and discard self loops
            edge_filter = self.get_edge_filter(graph)
            graph = nx.subgraph_view(graph, filter_edge=edge_filter)
        return graph

    def get_edge_filter(self, graph: nx.Graph) -> Callable:
        """Returns a function to filter edges based on the pvalue"""
        def edge_filter(n1, n2): return (graph.get_edge_data(n1, n2)["pvalue"] <= self.alpha) and (n1 != n2)
        return edge_filter
