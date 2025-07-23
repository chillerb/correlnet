"""
Embedders that compute the 2D positions of each variable in the correlation network.
"""

import abc
import numpy as np
import pandas as pd
import logging

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class Embedder(abc.ABC):
    @abc.abstractmethod
    def embed(self, df: pd.DataFrame, correl_df: pd.DataFrame) -> pd.DataFrame:
        """
        Embeds variables in a 2D space for plotting.

        Args:
            df (pd.DataFrame): original variable values
            cor_df(pd.DataFrame): pairwise correlations between variables
        """
        raise NotImplementedError()


class RandomEmbedder(Embedder):
    def __init__(self, random_state=None):
        super().__init__()
        self.random_state = random_state

    def embed(self, df, correl_df):
        rng = np.random.default_rng(self.random_state)
        x = rng.normal(size=(len(df.columns), 2))
        return pd.DataFrame(x, columns=["random_1", "random_2"], index=df.columns)


class TSNEEmbedder(Embedder):
    def __init__(self, **kwargs):
        super().__init__()
        self.tsne_kwargs = kwargs

    def fit_tsne(self, df: pd.DataFrame, index):
        if df.isna().any().any():
            logger.warning("replacing NaNs with 0 for t-SNE")
            df = df.fillna(0)
        logger.info("fitting TSNE...")
        tsne = TSNE(n_components=2, **self.tsne_kwargs)
        x_tsne = tsne.fit_transform(df)
        tsne_df = pd.DataFrame({"tsne_1": x_tsne[:, 0], "tsne_2": x_tsne[:, 1]}, index=index)
        return tsne_df


class VarTSNEEmbedder(TSNEEmbedder):
    """Uses the original values of the variables for a 2D TSNE embedding."""

    def __init__(self, standardize=True, **kwargs):
        super().__init__(**kwargs)
        self.standardize = standardize

    def embed(self, df: pd.DataFrame, correl_df: pd.DataFrame) -> pd.DataFrame:
        if self.standardize:
            logger.debug("standardizing data frame before TSNE")
            scaler = StandardScaler()
            df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

        tsne_df = self.fit_tsne(df.transpose(), df.columns)
        return tsne_df


class CorrelTSNEEmbedder(TSNEEmbedder):
    """Uses the correlation vectors of each variable for a 2D TSNE embedding."""

    def __init__(self, use_abs: bool = False, **kwargs):
        """ 
        Args:
            abs (bool): if true, uses the absolute correlation values
        """
        super().__init__(**kwargs)
        self.use_abs = use_abs

    def embed(self, df: pd.DataFrame, correl_df: pd.DataFrame) -> pd.DataFrame:
        x_df = pd.pivot(correl_df, index="var_1", columns="var_2", values="statistic")
        if self.use_abs:
            x_df = x_df.abs()
        tsne_df = self.fit_tsne(x_df, x_df.index)
        return tsne_df
