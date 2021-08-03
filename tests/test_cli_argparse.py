import hashlib
import sys
from unittest import mock

import h5py
import numpy as np
import pytest

from dclab import cli, new_dataset, rtdc_dataset

from helper_methods import retrieve_data


def test_compress():
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("compressed.rtdc")

    with mock.patch("sys.argv", ["", str(path_in), str(path_out)]):
        cli.compress()

    with new_dataset(path_out) as dsj, new_dataset(path_in) as ds0:
        assert "dclab-compress" in dsj.logs
        assert len(dsj)
        assert len(dsj) == len(ds0)
        for feat in ds0.features:
            if feat in ["contour", "image", "mask"]:
                for ii in range(len(dsj)):
                    assert np.all(dsj[feat][ii] == ds0[feat][ii]), feat
            else:
                assert np.all(dsj[feat] == ds0[feat]), feat


def test_compress_already_compressed_no_force():
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("compressed.rtdc")
    path_out2 = path_in.with_name("compressed2.rtdc")

    with mock.patch("sys.argv", ["", str(path_in), str(path_out)]):
        cli.compress()

    with mock.patch("sys.argv", ["", str(path_out), str(path_out2)]):
        cli.compress()

    h1 = hashlib.md5(path_out.read_bytes()).hexdigest()
    h2 = hashlib.md5(path_out2.read_bytes()).hexdigest()
    assert h1 == h2


def test_compress_already_compressed_with_force():
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("compressed.rtdc")
    path_out2 = path_in.with_name("compressed2.rtdc")

    with mock.patch("sys.argv", ["", str(path_in), str(path_out)]):
        cli.compress()

    with mock.patch("sys.argv",
                    ["", "--force", str(path_out), str(path_out2)]):
        cli.compress()

    h1 = hashlib.md5(path_out.read_bytes()).hexdigest()
    h2 = hashlib.md5(path_out2.read_bytes()).hexdigest()
    assert h1 != h2


def test_condense():
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("condensed.rtdc")

    with mock.patch("sys.argv", ["", str(path_in), str(path_out)]):
        cli.condense()

    with new_dataset(path_out) as dsj, new_dataset(path_in) as ds0:
        assert "dclab-condense" in dsj.logs
        assert len(dsj)
        assert len(dsj) == len(ds0)
        for feat in dsj.features:
            assert np.all(dsj[feat] == ds0[feat])


def test_condense_missing_argument():
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")

    with mock.patch("sys.argv", ["", str(path_in)]):
        with pytest.raises(SystemExit):
            cli.condense()


def test_join_tdms():
    pytest.importorskip("nptdms")
    path_in = retrieve_data("fmt-tdms_shapein-2.0.1-no-image_2017.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("out.rtdc")

    with mock.patch("sys.argv",
                    ["", "-o", str(path_out), str(path_in), str(path_in)]):
        cli.join()

    with new_dataset(path_out) as dsj, new_dataset(path_in) as ds0:
        assert len(dsj)
        assert len(dsj) == 2*len(ds0)
        assert len(ds0) == ds0.config["experiment"]["event count"]
        assert len(dsj) == dsj.config["experiment"]["event count"]
        assert np.all(dsj["circ"][:100] == ds0["circ"][:100])
        assert np.all(dsj["circ"][len(ds0):len(ds0)+100] == ds0["circ"][:100])
        assert set(dsj.features) == set(ds0.features)


def test_repack_basic():
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("repacked.rtdc")

    with mock.patch("sys.argv", ["", str(path_in), str(path_out)]):
        cli.repack()

    with new_dataset(path_out) as dsj, new_dataset(path_in) as ds0:
        assert len(dsj)
        assert len(dsj) == len(ds0)
        for feat in ds0.features_innate:
            if feat in ds0.features_scalar:
                assert np.all(dsj[feat] == ds0[feat]), feat
        for ii in range(len(ds0)):
            assert np.all(dsj["contour"][ii] == ds0["contour"][ii])
            assert np.all(dsj["image"][ii] == ds0["image"][ii])
            assert np.all(dsj["mask"][ii] == ds0["mask"][ii])


def test_repack_strip_logs():
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("repacked.rtdc")

    # write some logs
    with h5py.File(path_in, "a") as h5:
        rtdc_dataset.write(h5,
                           logs={"test_log": ["peter", "hans"]},
                           mode="append")

    with mock.patch("sys.argv",
                    ["", str(path_in), str(path_out), "--strip-logs"]):
        cli.repack()

    with new_dataset(path_out) as dsj, new_dataset(path_in) as ds0:
        assert ds0.logs
        assert not dsj.logs


def test_repack_strip_logs_control():
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("repacked.rtdc")

    # write some logs
    with h5py.File(path_in, "a") as h5:
        rtdc_dataset.write(h5,
                           logs={"test_log": ["peter", "hans"]},
                           mode="append")

    with mock.patch("sys.argv", ["", str(path_in), str(path_out)]):
        cli.repack()

    with new_dataset(path_out) as dsj, new_dataset(path_in) as ds0:
        assert ds0.logs
        assert dsj.logs


def test_split():
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    path_out = path_in.parent / "out"
    path_out.mkdir()

    with mock.patch("sys.argv",
                    ["", str(path_in), "--path_out", str(path_out),
                     "--split-events", "3"]):
        paths = cli.split(ret_out_paths=True)
    with new_dataset(path_in) as ds:
        ecount = 0
        for pp in paths:
            with new_dataset(pp) as di:
                for feat in ds.features_scalar:
                    if feat == "index":
                        continue
                    assert np.all(
                        ds[feat][ecount:ecount+len(di)] == di[feat]), feat
                ecount += len(di)


def test_tdms2rtdc_features():
    pytest.importorskip("nptdms")
    path_in = retrieve_data("fmt-tdms_shapein-2.0.1-no-image_2017.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("out.rtdc")

    with mock.patch("sys.argv",
                    ["", str(path_in), str(path_out),
                     "--compute-ancillary-features"]):
        cli.tdms2rtdc()

    with new_dataset(path_out) as ds2, new_dataset(path_in) as ds1:
        assert len(ds2)
        assert set(ds1.features) == set(ds2.features)
        # features were computed
        assert set(ds2._events.keys()) == set(ds1.features)


def test_verify_dataset_exit_code_alert(monkeypatch):
    # get the exit status from the script
    def sys_exit(status):
        return status
    monkeypatch.setattr(sys, "exit", sys_exit)

    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")
    # provoke a warning
    with h5py.File(h5path, "r+") as h5:
        h5.attrs["experiment:unknown"] = ""
    with mock.patch("sys.argv", ["", str(h5path)]):
        exit_status = cli.verify_dataset(h5path)

    assert exit_status == 1  # unknown key leads to alert
