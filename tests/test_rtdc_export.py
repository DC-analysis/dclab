import collections

import numpy as np

from dclab.rtdc_dataset.export import yield_filtered_array_stacks


def test_yield_filtered_array_stacks_array():
    enum = np.arange(547)
    ebol = np.ones(547, dtype=bool)
    ebol[10] = False
    ebol[412] = False

    data = np.random.random((547, 80, 320))
    indices = enum[ebol]
    stacked = []
    for chunk in yield_filtered_array_stacks(data, indices):
        stacked.append(chunk)
    assert len(stacked) == 55
    assert len(stacked[0]) == len(stacked[1])
    assert len(stacked[-1]) == 547 - 2 - 54 * len(stacked[0])

    data2 = np.concatenate(stacked, axis=0)
    assert len(data2) == 547 - 2
    assert data2[0].shape == data[0].shape
    assert np.all(data[indices] == data2)


def test_yield_filtered_array_stacks_list():
    # custom list we will be using, which implements shape and dtype, but
    # no __array__.
    class ListFeat(collections.UserList):
        def __init__(self, data):
            super(ListFeat, self).__init__(data)

        @property
        def shape(self):
            return tuple([len(self.data)] + list(self.data[0].shape))

        @property
        def dtype(self):
            return self.data[0].dtype

    enum = np.arange(547, dtype=int)
    ebol = np.ones(547, dtype=bool)
    ebol[10] = False
    ebol[412] = False

    # instead of the above case, create a list of arrays here.
    datalist = []
    for _ in range(547):
        datalist.append(np.random.random((80, 320)))
    data = ListFeat(datalist)
    assert data.shape == (547, 80, 320)
    assert data.dtype == datalist[0].dtype
    assert np.all(data[0] == datalist[0])

    indices = enum[ebol]
    stacked = []
    for chunk in yield_filtered_array_stacks(data, indices):
        stacked.append(np.array(chunk, copy=True))
    assert len(stacked) == 55
    assert len(stacked[0]) == len(stacked[1])
    assert len(stacked[-1]) == 547 - 2 - 54 * len(stacked[0])
    assert np.all(stacked[0][0] == datalist[0])

    data2 = np.concatenate(stacked, axis=0)
    assert len(data2) == 547 - 2
    assert data2[0].shape == data[0].shape
    assert np.all(np.array(datalist)[indices] == data2)
