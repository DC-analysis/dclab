import functools

import numpy as np


class DCORTables:
    def __init__(self, api):
        self.api = api

    def __getitem__(self, key):
        return self._tables[key]

    def __len__(self):
        return len(self._tables)

    def keys(self):
        return self._tables.keys()

    @property
    @functools.lru_cache()
    def _tables(self):
        table_data = self.api.get(query="tables")
        # assemble the tables
        tables = {}
        for key in table_data:
            columns, data = table_data[key]
            ds_dt = np.dtype({'names': columns,
                              'formats': [np.float64] * len(columns)})
            tab_data = np.asarray(data)
            rec_arr = np.rec.array(tab_data, dtype=ds_dt)
            tables[key] = rec_arr

        return tables
