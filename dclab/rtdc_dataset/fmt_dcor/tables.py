import numpy as np

from ..meta_table import MetaTable


class DCORTables:
    def __init__(self, api):
        self.api = api
        self._tables_cache = None

    def __contains__(self, key):
        return key in self.keys()

    def __getitem__(self, key):
        return self._tables[key]

    def __iter__(self):
        for key in self.keys():
            yield key

    def __len__(self):
        return len(self._tables)

    def keys(self):
        return self._tables.keys()

    @property
    def _tables(self):
        if self._tables_cache is None:
            table_data = self.api.get(query="tables", timeout=13)
            # assemble the tables
            tables = {}
            for key in table_data:
                tables[key] = DCORTable(table_data[key])

            self._tables_cache = tables
        return self._tables_cache


class DCORTable(MetaTable):
    def __init__(self, table_content):
        self._columns, data = table_content
        self._tab_data = np.asarray(data)
        if self._columns is not None:
            # We have a rec-array (named columns)

            ds_dt = np.dtype({'names': self._columns,
                              'formats': [np.float64] * len(self._columns)})
            self._tab_data = np.rec.array(self._tab_data, dtype=ds_dt)

    def __array__(self, *args, **kwargs):
        return self._tab_data.__array__(*args, **kwargs)

    @property
    def meta(self):
        # TODO: Implement metadata sending from DCOR.
        return {}

    def has_graphs(self):
        return self._columns is not None

    def keys(self):
        return self._columns

    def __getitem__(self, key):
        return self._tab_data[key]
