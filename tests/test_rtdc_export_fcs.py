from os.path import join
import tempfile
from unittest import mock

import pytest

import dclab

from helper_methods import example_data_dict


def test_fcs_export():
    pytest.importorskip("fcswrite")
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=222, keys=keys)
    ds = dclab.new_dataset(ddict)

    edest = tempfile.mkdtemp()
    f1 = join(edest, "test.fcs")
    f2 = join(edest, "test_unicode.fcs")

    ds.export.fcs(f1, keys, override=True)
    ds.export.fcs(f2, ["area_um", "deform", "time",
                       "frame", "fl3_width"], override=True)

    with open(f1, mode="rb") as fd:
        a1 = fd.read()

    with open(f2, mode="rb") as fd:
        a2 = fd.read()

    assert a1 == a2
    assert len(a1) != 0


def test_fcs_export_progress(tmp_path):
    pytest.importorskip("fcswrite")
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=222, keys=keys)
    ds = dclab.new_dataset(ddict)

    f1 = tmp_path / "test.fcs"

    callback = mock.MagicMock()
    ds.export.fcs(f1, keys, progress_callback=callback)

    callback.assert_any_call(0.0, "collecting data")
    callback.assert_any_call(0.5, "exporting data")
    callback.assert_any_call(1.0, "export complete")


def test_fcs_override():
    pytest.importorskip("fcswrite")
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=212, keys=keys)
    ds = dclab.new_dataset(ddict)

    edest = tempfile.mkdtemp()
    f1 = join(edest, "test.fcs")
    ds.export.fcs(f1, keys, override=True)
    try:
        ds.export.fcs(f1[:-4], keys, override=False)
    except OSError:
        pass
    else:
        raise ValueError("Should append .fcs and not override!")


def test_fcs_not_filtered():
    pytest.importorskip("fcswrite")
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=127, keys=keys)
    ds = dclab.new_dataset(ddict)

    edest = tempfile.mkdtemp()
    f1 = join(edest, "test.tsv")
    ds.export.fcs(f1, keys, filtered=False)


def test_fcs_export_meta_data_not_modified(tmp_path):
    """The `meta_data` dictionary of the caller must not be modified"""
    pytest.importorskip("fcswrite")
    keys = ["area_um", "deform"]
    ddict = example_data_dict(size=222, keys=keys)
    ds = dclab.new_dataset(ddict)
    meta_data = {"sample": "test"}
    ds.export.fcs(tmp_path / "test.fcs", keys, meta_data=meta_data)
    assert meta_data == {"sample": "test"}
