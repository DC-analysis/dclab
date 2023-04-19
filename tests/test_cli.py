"""Test command-line interface"""
import sys
import tempfile
import pathlib
import shutil

from dclab import cli, new_dataset, rtdc_dataset

import h5py
import numpy as np
import pytest

from helper_methods import retrieve_data


def test_check_suffix_disabled_repack():
    path_in_o = retrieve_data("fmt-hdf5_polygon_gate_2021.zip")
    path_in = path_in_o.with_suffix("")
    path_in_o.rename(path_in)
    assert path_in.suffix == ""
    with pytest.raises(ValueError, match="Unsupported file type"):
        cli.repack(path_in=path_in,
                   path_out=path_in.with_name("repacked.rtdc"))
    # but this should work:
    cli.repack(path_in=path_in,
               path_out=path_in.with_name("repacked2.rtdc"),
               check_suffix=False)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
@pytest.mark.parametrize("method", ["compress", "condense", "repack"])
def test_compressed(method):
    """Make sure the output data are compressed"""
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("out.rtdc")

    mcallable = getattr(cli, method)
    mcallable(path_in=path_in,
              path_out=path_out)

    ic = rtdc_dataset.check.IntegrityChecker(path_out)
    ccue = ic.check_compression()[0]
    assert ccue.data["uncompressed"] == 0


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_compressed_split():
    """Make sure the split output data are compressed"""
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.parent

    paths = cli.split(path_in=path_in,
                      path_out=path_out,
                      split_events=3,
                      ret_out_paths=True)

    for pp in paths:
        ic = rtdc_dataset.check.IntegrityChecker(pp)
        ccue = ic.check_compression()[0]
        assert ccue.data["uncompressed"] == 0


def test_method_available():
    # DCOR depotize needs this
    assert hasattr(cli, "get_job_info")
    assert hasattr(cli, "get_command_log")


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_repack_basic():
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("repacked.rtdc")

    cli.repack(path_out=path_out, path_in=path_in)

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


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_repack_remove_secrets():
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("repacked.rtdc")

    with h5py.File(path_in, "a") as h5:
        h5.attrs["experiment:sample"] = "my dirty secret"

    with h5py.File(path_in, "a") as h5:
        h5.attrs["experiment:sample"] = "sunshine"

    # test whether the dirty secret is still there
    with open(str(path_in), "rb") as fd:
        data = fd.read()
        assert str(data).count("my dirty secret")

    # now repack
    cli.repack(path_out=path_out, path_in=path_in)

    # clean?
    with open(str(path_out), "rb") as fd:
        data = fd.read()
        assert not str(data).count("my dirty secret")


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_repack_strip_logs():
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("repacked.rtdc")

    # write some logs
    with h5py.File(path_in, "a") as h5:
        hw = rtdc_dataset.RTDCWriter(h5)
        hw.store_log("test_log", ["peter", "hans"])

    cli.repack(path_out=path_out, path_in=path_in, strip_logs=True)

    with new_dataset(path_out) as dsj, new_dataset(path_in) as ds0:
        assert ds0.logs
        assert not dsj.logs


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_repack_user_metadata():
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    with h5py.File(path_in, "a") as h5:
        h5.attrs["user:peter"] = "hans"

    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("repacked.rtdc")

    cli.repack(path_out=path_out, path_in=path_in)

    with new_dataset(path_out) as ds:
        assert ds.config["user"]["peter"] == "hans"


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_split():
    path_in = retrieve_data("fmt-hdf5_mask-contour_2018.zip")
    paths = cli.split(path_in=path_in, split_events=3, ret_out_paths=True)
    with new_dataset(path_in) as ds:
        ecount = 0
        for pp in paths:
            with new_dataset(pp) as di:
                for feat in ds.features_scalar:
                    if feat in ["index",
                                "time",  # issue 204
                                ]:
                        continue
                    assert np.all(
                        ds[feat][ecount:ecount + len(di)] == di[feat]), feat
                ecount += len(di)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_split_traces():
    path_in = retrieve_data("fmt-hdf5_fl_2018.zip")
    paths = cli.split(path_in=path_in, split_events=3, ret_out_paths=True)
    with new_dataset(path_in) as ds:
        ecount = 0
        for pp in paths:
            with new_dataset(pp) as di:
                for flkey in ds["trace"].keys():
                    trace1 = ds["trace"][flkey][ecount:ecount + len(di)]
                    trace2 = di["trace"][flkey][:]
                    assert len(trace1) == len(trace2)
                    assert np.all(trace1 == trace2), flkey
                ecount += len(di)


def test_tdms2rtdc():
    pytest.importorskip("nptdms")
    path_in = retrieve_data("fmt-tdms_shapein-2.0.1-no-image_2017.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("out.rtdc")

    cli.tdms2rtdc(path_tdms=path_in,
                  path_rtdc=path_out,
                  compute_features=False)

    with new_dataset(path_out) as ds2, new_dataset(path_in) as ds1:
        assert len(ds2)
        assert set(ds1.features) == set(ds2.features)
        # not all features are computed
        assert set(ds2._events.keys()) < set(ds1.features)
        for feat in ds1:
            assert np.all(ds1[feat] == ds2[feat])


def test_tdms2rtdc_bulk():
    pytest.importorskip("nptdms")
    path_data = retrieve_data("fmt-tdms_shapein-2.0.1-no-image_2017.zip")
    path_wd = pathlib.Path(
        tempfile.mkdtemp(prefix="tdms2rtdc_bulk_")).resolve()
    path_in = path_wd / "input"
    path_in.mkdir()
    shutil.copytree(path_data.parent, path_in / "data_1")
    shutil.copytree(path_data.parent, path_in / "data_2")
    shutil.copytree(path_data.parent, path_in / "data_3")
    (path_in / "data_nested").mkdir()
    shutil.copytree(path_data.parent, path_in / "data_nested" / "data_4")
    # same directory (will be cleaned up with path_in)
    path_out = path_wd / "output"
    path_out.mkdir()

    cli.tdms2rtdc(path_tdms=path_in,
                  path_rtdc=path_out,
                  compute_features=False)

    for pp in [path_out / "data_1" / "M1_data.rtdc",
               path_out / "data_2" / "M1_data.rtdc",
               path_out / "data_3" / "M1_data.rtdc",
               path_out / "data_nested" / "data_4" / "M1_data.rtdc"]:
        assert pp.exists()

        with new_dataset(pp) as ds2, new_dataset(path_data) as ds1:
            assert len(ds2)
            assert set(ds1.features) == set(ds2.features)
            # not all features are computed
            assert set(ds2._events.keys()) < set(ds1.features)
            for feat in ds1:
                assert np.all(ds1[feat] == ds2[feat])


def test_tdms2rtdc_features():
    pytest.importorskip("nptdms")
    path_in = retrieve_data("fmt-tdms_shapein-2.0.1-no-image_2017.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("out.rtdc")

    cli.tdms2rtdc(path_tdms=path_in,
                  path_rtdc=path_out,
                  compute_features=True)

    with new_dataset(path_out) as ds2, new_dataset(path_in) as ds1:
        assert len(ds2)
        assert set(ds1.features) == set(ds2.features)
        # features were computed
        assert set(ds2.features_loaded) == set(ds1.features)


def test_tdms2rtdc_remove_nan_image():
    pytest.importorskip("nptdms")
    imageio = pytest.importorskip("imageio")
    path_in = retrieve_data("fmt-tdms_fl-image-bright_2017.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("out.rtdc")

    # generate fake video
    with new_dataset(path_in) as ds:
        video_length = len(ds) - 1
    vname = path_in.with_name("M4_0.040000ul_s_imaq.avi")
    # remove contour data (not necessary for this test)
    path_in.with_name("M4_0.040000ul_s_contours.txt").unlink()

    imgs = imageio.mimread(vname)
    with imageio.get_writer(vname) as writer:
        for ii in range(video_length):
            writer.append_data(imgs[ii % len(imgs)])

    # without removal
    cli.tdms2rtdc(path_tdms=path_in,
                  path_rtdc=path_out,
                  compute_features=False,
                  skip_initial_empty_image=False)

    with new_dataset(path_out) as ds2, new_dataset(path_in) as ds1:
        assert len(ds2) == len(ds1)
        assert np.all(ds2["image"][0] == 0)

    # with removal
    cli.tdms2rtdc(path_tdms=path_in,
                  path_rtdc=path_out,
                  compute_features=False,
                  skip_initial_empty_image=True)

    with new_dataset(path_out) as ds2, new_dataset(path_in) as ds1:
        assert len(ds2) == video_length
        assert not np.all(ds2["image"][0] == 0)
        assert ds2.config["experiment"]["event count"] == video_length


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_tdms2rtdc_update_roi_size():
    pytest.importorskip("nptdms")
    path_in = retrieve_data("fmt-tdms_fl-image_2016.zip")
    # set wrong roi sizes
    camin = path_in.with_name("M1_camera.ini")
    with camin.open("r") as fd:
        lines = fd.readlines()
    lines = lines[:-2]
    lines.append("width = 23\n")
    lines.append("height = 24\n")
    with camin.open("w") as fd:
        fd.writelines(lines)

    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("out.rtdc")

    cli.tdms2rtdc(path_tdms=path_in,
                  path_rtdc=path_out,
                  compute_features=False,
                  skip_initial_empty_image=True)

    with new_dataset(path_out) as dsj, new_dataset(path_in) as ds0:
        assert ds0.config["imaging"]["roi size x"] == 23
        assert ds0.config["imaging"]["roi size y"] == 24
        assert dsj.config["imaging"]["roi size x"] == 256
        assert dsj.config["imaging"]["roi size y"] == 96
        wlog = "dclab-tdms2rtdc-warnings"
        assert "LimitingExportSizeWarning" in dsj.logs[wlog]


def test_tdms2rtdc_update_sample_per_events():
    pytest.importorskip("nptdms")
    path_in = retrieve_data("fmt-tdms_2fl-no-image_2017.zip")
    # same directory (will be cleaned up with path_in)
    path_out = path_in.with_name("out.rtdc")

    # set wrong samples per event
    with path_in.with_name("M1_para.ini").open("a") as fd:
        fd.write("Samples Per Event = 1234")

    with new_dataset(path_in) as ds:
        assert ds.config["fluorescence"]["samples per event"] == 1234

    cli.tdms2rtdc(path_tdms=path_in,
                  path_rtdc=path_out,
                  compute_features=True)

    with new_dataset(path_out) as ds2:
        assert ds2.config["fluorescence"]["samples per event"] == 566


@pytest.mark.parametrize("dataset,exit_status_expected", [
    ["fmt-hdf5_fl_2017.zip", 3],
    ["fmt-hdf5_fl_2018.zip", 1],
    ["fmt-hdf5_polygon_gate_2021.zip", 0],
])
def test_verify_dataset_exit_codes(dataset, exit_status_expected, monkeypatch):
    # get the exit status from the script
    def sys_exit(status):
        return status

    monkeypatch.setattr(sys, "exit", sys_exit)

    h5path = retrieve_data(dataset)
    exit_status = cli.verify_dataset(h5path)
    assert exit_status == exit_status_expected


def test_verify_dataset_exit_code_alert(monkeypatch):
    # get the exit status from the script
    def sys_exit(status):
        return status

    monkeypatch.setattr(sys, "exit", sys_exit)

    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")

    # Since 0.35.0 we really check the configuration key types.
    # Just make sure that they are properly set:
    with h5py.File(h5path, "a") as h5:
        for key in ["fluorescence:sample rate",
                    "imaging:roi position x",
                    "imaging:roi position y"]:
            h5.attrs[key] = int(h5.attrs[key])

    # provoke a warning
    with h5py.File(h5path, "r+") as h5:
        h5.attrs["experiment:unknown"] = ""
    exit_status = cli.verify_dataset(h5path)
    assert exit_status == 1  # unknown key leads to alert


def test_verify_dataset_exit_code_error(monkeypatch):
    # get the exit status from the script
    def sys_exit(status):
        return status

    monkeypatch.setattr(sys, "exit", sys_exit)

    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")

    # Since 0.35.0 we really check the configuration key types.
    # Just make sure that they are properly set:
    with h5py.File(h5path, "a") as h5:
        for key in ["fluorescence:sample rate",
                    "imaging:roi position x",
                    "imaging:roi position y"]:
            h5.attrs[key] = int(h5.attrs[key])

    # provoke a warning
    with h5py.File(h5path, "r+") as h5:
        h5.attrs["setup:channel width"] = "peter"
    exit_status = cli.verify_dataset(h5path)
    assert exit_status == 4  # Cannot convert string to float error


def test_verify_dataset_exit_code_user_ok(monkeypatch):
    # get the exit status from the script
    def sys_exit(status):
        return status

    monkeypatch.setattr(sys, "exit", sys_exit)

    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")

    # Since 0.35.0 we really check the configuration key types.
    # Just make sure that they are properly set:
    with h5py.File(h5path, "a") as h5:
        for key in ["fluorescence:sample rate",
                    "imaging:roi position x",
                    "imaging:roi position y"]:
            h5.attrs[key] = int(h5.attrs[key])

    # provoke a warning
    with h5py.File(h5path, "r+") as h5:
        h5.attrs["user:pan"] = 2
    exit_status = cli.verify_dataset(h5path)
    assert exit_status == 0


def test_verify_dataset_exit_code_violation_1(monkeypatch):
    # get the exit status from the script
    def sys_exit(status):
        return status

    monkeypatch.setattr(sys, "exit", sys_exit)

    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")

    # Since 0.35.0 we really check the configuration key types.
    # Just make sure that they are properly set:
    with h5py.File(h5path, "a") as h5:
        for key in ["fluorescence:sample rate",
                    "imaging:roi position x",
                    "imaging:roi position y"]:
            h5.attrs[key] = int(h5.attrs[key])

    # provoke a warning
    with h5py.File(h5path, "r+") as h5:
        h5.attrs["setup:flow rate"] = 0
    exit_status = cli.verify_dataset(h5path)
    assert exit_status == 3  # zero flow rate and warnings about mismatch


def test_verify_dataset_exit_code_violation_2(monkeypatch):
    # get the exit status from the script
    def sys_exit(status):
        return status

    monkeypatch.setattr(sys, "exit", sys_exit)

    h5path = retrieve_data("fmt-hdf5_fl_2018.zip")

    # Since 0.35.0 we really check the configuration key types.
    # Just make sure that they are properly set:
    with h5py.File(h5path, "a") as h5:
        for key in ["fluorescence:sample rate",
                    "imaging:roi position x",
                    "imaging:roi position y"]:
            h5.attrs[key] = int(h5.attrs[key])

    # provoke a warning
    with h5py.File(h5path, "r+") as h5:
        h5.attrs["setup:flow rate"] = 0
        h5.attrs["setup:flow rate sample"] = 0
        h5.attrs["setup:flow rate sheath"] = 0
    exit_status = cli.verify_dataset(h5path)
    assert exit_status == 2  # zero flow rate
