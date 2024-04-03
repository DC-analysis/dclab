import io
from os.path import join
import tempfile

import dclab

from helper_methods import example_data_dict


def test_tsv_export():
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=222, keys=keys)
    ds = dclab.new_dataset(ddict)

    edest = tempfile.mkdtemp()
    f1 = join(edest, "test.tsv")
    f2 = join(edest, "test_unicode.tsv")

    ds.export.tsv(f1, keys, override=True)
    ds.export.tsv(f2, [u"area_um", u"deform", u"time",
                       u"frame", u"fl3_width"], override=True)

    with io.open(f1) as fd:
        a1 = fd.read()

    with io.open(f2) as fd:
        a2 = fd.read()

    assert a1 == a2
    assert len(a1) != 0


def test_tsv_override():
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=212, keys=keys)
    ds = dclab.new_dataset(ddict)

    edest = tempfile.mkdtemp()
    f1 = join(edest, "test.tsv")
    ds.export.tsv(f1, keys, override=True)
    try:
        ds.export.tsv(f1[:-4], keys, override=False)
    except OSError:
        pass
    else:
        raise ValueError("Should append .tsv and not override!")


def test_tsv_not_filtered():
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=127, keys=keys)
    ds = dclab.new_dataset(ddict)

    edest = tempfile.mkdtemp()
    f1 = join(edest, "test.tsv")
    ds.export.tsv(f1, keys, filtered=False)
