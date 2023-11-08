import numpy as np


class DCORTables:
    def __init__(self, api):
        self.api = api
        self._tables_cache = None

    def __contains__(self, key):
        return key in self.keys()

    def __getitem__(self, key):
        return self._tables[key]

    def __len__(self):
        return len(self._tables)

    def keys(self):
        return self._tables.keys()

    @property
    def _tables(self):
        if self._tables_cache is None:
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

            self._tables_cache = tables
        return self._tables_cache
