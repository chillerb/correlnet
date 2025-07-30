"""
Plot line segments.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typing import Union
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, to_rgba_array, Colormap
from matplotlib.typing import ColorType
from numpy.typing import ArrayLike


class SegmentPlotter:

    def plot(
        self, x: ArrayLike, y: ArrayLike, x_end: ArrayLike, y_end: ArrayLike, linewidths=1,
        colors: Union[ColorType | ArrayLike] = None, c: ArrayLike = None, alpha=1,
        cmap: Union[str, Colormap] = "viridis", norm: Normalize = None, ax: plt.Axes = None
    ) -> plt.Axes:
        """
        Plots line segments, similar to ggplot's geom_segments. 

        Args:
            x (ArrayLike): x values of the segment starting points
            y (ArrayLike): y values of the segment starting points
            x_end (ArrayLike): x values of the segment end points
            y_end (ArrayLike): y values of the segment end points 
            linewidths (ArrayLike): line strengths of the segments
            colors (ColorType|ArrayLike[ColorType]): matplotlib colors of the segments
            c (ArrayLike): floats to be mapped to colors via cmap and norm
            cmap (ColorMap): Colormap for mapping c
            norm (Normalize): Normalization function for mapping c
            ax (Axes): matplotlib Axes to add the segments to
        """

        if ax is None:
            fig, ax = plt.subplots()

        lines = self.line_collection(x, y, x_end, y_end, linewidths, colors, c=c, alpha=alpha, cmap=cmap, norm=norm)
        ax.add_collection(lines)
        ax.autoscale()
        return ax

    def plot_df(
            self, df: pd.DataFrame, x: str, y: str, x_end: str, y_end: str,
            linewidths: str = None, colors: str = None, c: str = None, alpha: str = None, cmap="viridis", norm=None, ax=None
    ) -> plt.Axes:
        """
        seaborn-like interface, similar to ggplot geom_segments.
        """
        x = df[x]
        y = df[y]
        x_end = df[x_end]
        y_end = df[y_end]
        alpha = 1 if alpha is None else df[alpha]
        linewidths = 1 if linewidths is None else df[linewidths]
        colors = "black" if colors is None else df[colors]
        c = None if c is None else df[c]
        return self.plot(x, y, x_end, y_end, linewidths=linewidths, colors=colors, alpha=alpha, cmap=cmap, norm=norm, ax=ax)

    def line_collection(self, x, y, x_end, y_end, linewidths=1, colors=None, c=None, alpha=1, cmap="viridis", norm=None) -> LineCollection:
        """Creates a new line collection."""
        assert len(x) == len(y)
        assert len(x) == len(x_end)
        assert len(y) == len(y_end)
        cmap = plt.cm.get_cmap(cmap)
        norm = Normalize() if norm is None else norm
        # a little silly, but LineCollection does not allow single ints/floats for colors, even if cmap and norm are provided
        if c is not None and colors is None:
            colors = cmap(norm(c))
        segments = [[(x0, y0), (x1, y1)] for x0, y0, x1, y1 in zip(x, y, x_end, y_end)]
        lines = LineCollection(segments, linewidths=linewidths, colors=colors, cmap=cmap, norm=norm, alpha=alpha)
        return lines
