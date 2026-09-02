import abc

import numpy as np


class MetaTable(abc.ABC):
    @abc.abstractmethod
    def __array__(self, *args, **kwargs) -> np.ndarray:
        """Return array representation of the table"""

    @property
    @abc.abstractmethod
    def meta(self) -> dict:
        """Return metadata of the table (e.g. graph colors)"""

    @abc.abstractmethod
    def has_graphs(self) -> bool:
        """Return True when the table has key-based graphs"""

    @abc.abstractmethod
    def keys(self):
        """Return keys of the graphs, None if `not self.has_graphs()`"""

    @abc.abstractmethod
    def __getitem__(self, key) -> np.ndarray:
        """Return a graph or otherwise part of the data array"""
