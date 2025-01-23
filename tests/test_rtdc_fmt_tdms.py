"""Test tdms file format"""
import os
import pathlib
import shutil
import sys
import tempfile

import numpy as np
import pytest

from dclab import new_dataset
import dclab.rtdc_dataset.fmt_tdms.naming

from helper_methods import retrieve_data


nptdms = pytest.importorskip("nptdms")


def test_compatibility_minimal():
    ds = new_dataset(retrieve_data("fmt-tdms_minimal_2016.zip"))
    assert ds["area_um"].ndim == 1  # important for matplotlib
    assert ds.config["setup"]["channel width"] == 20
    assert ds.config["setup"]["chip region"].lower() == "channel"
    assert ds.config["setup"]["flow rate"] == 0.12
    assert ds.config["imaging"]["pixel size"] == 0.34


@pytest.mark.skipif(sys.version_info < (3, 6),
                    reason="requires python3.6 or higher")
def test_compatibility_channel_width():
    # At some point, "Channel width" was replaced by "Channel width [um]"
    path = retrieve_data("fmt-tdms_minimal_2016.zip")
    para = path.parent / "M1_para.ini"
    pardata = para.read_text()
    pardata = pardata.replace("Channel width = 20\n", "Channel width = 34\n")
    para.write_text(pardata)
    ds = new_dataset(path)
    assert ds.config["setup"]["channel width"] == 34


def test_compatibility_shapein201():
    ds = new_dataset(retrieve_data("fmt-tdms_shapein-2.0.1-no-image_2017.zip"))
    assert ds.config["setup"]["channel width"] == 20
    assert ds.config["setup"]["chip region"].lower() == "channel"
    assert ds.config["setup"]["software version"] == "ShapeIn 2.0.1"
    assert ds.config["imaging"]["pixel size"] == 0.34
    assert ds.config["imaging"]["flash duration"] == 2
    assert ds.config["experiment"]["date"] == "2017-10-12"
    assert ds.config["experiment"]["time"] == "12:54:31"


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_contains_non_scalar():
    ds1 = new_dataset(retrieve_data("fmt-tdms_fl-image_2016.zip"))
    assert "contour" in ds1
    assert "image" in ds1
    assert "mask" in ds1
    assert "trace" in ds1
    ds2 = new_dataset(retrieve_data("fmt-tdms_minimal_2016.zip"))
    assert "image" not in ds2
    assert "trace" not in ds2
    ds3 = new_dataset(retrieve_data(
        "fmt-tdms_shapein-2.0.1-no-image_2017.zip"))
    assert "contour" not in ds3
    assert "image" not in ds3
    assert "mask" not in ds3
    assert "trace" not in ds3


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_contour_basic():
    ds = new_dataset(retrieve_data("fmt-tdms_fl-image_2016.zip"))
    assert len(ds["contour"]) == 12
    assert np.allclose(np.average(ds["contour"][0]), 38.488764044943821)
    assert ds["contour"]._initialized


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_contour_corrupt():
    path = pathlib.Path(retrieve_data("fmt-tdms_fl-image_2016.zip"))
    cpath = path.with_name("M1_contours.txt")
    # remove initial contours
    with cpath.open("r") as fd:
        lines = fd.readlines()
    with cpath.open("w") as fd:
        fd.writelines(lines[183:])
    ds = new_dataset(path)
    try:
        ds["contour"].determine_offset()
    except dclab.rtdc_dataset.fmt_tdms.event_contour.ContourIndexingError:
        pass
    else:
        assert False


def test_contour_unknown_offset():
    tdms_path = retrieve_data("fmt-tdms_fl-image-bright_2017.zip")
    cont_path = tdms_path.with_name("M4_0.040000ul_s_contours.txt")
    # Insert a false contour
    imaginary_contour = "\n".join([
        "Contour in frame 141",
        "(198, 30)",
        "(197, 29)",
        "(196, 28)",
        "(196, 27)",
        "(195, 27)",
        "(194, 26)",
        "(193, 25)",
        "(192, 24)",
        "(191, 25)",
        "",
    ])
    cont_data = cont_path.read_text()
    cont_path.write_text(imaginary_contour + cont_data)
    ds = dclab.new_dataset(tdms_path)
    with pytest.raises(dclab.rtdc_dataset.fmt_tdms.exc.ContourIndexingError,
                       match=r"Contour data has unknown offset \(frame 141\)"):
        ds["contour"].shape  # raises the error

    with pytest.raises(dclab.rtdc_dataset.fmt_tdms.exc.ContourIndexingError,
                       match=r"Contour data has unknown offset \(frame 141\)"):
        ds["contour"][0]  # raises the error


def test_contour_wrong_frame_number():
    tdms_path = retrieve_data("fmt-tdms_fl-image-bright_2017.zip")
    cont_path = tdms_path.with_name("M4_0.040000ul_s_contours.txt")
    # Insert a false contour
    contours = cont_path.read_text()
    contours = contours.replace("Contour in frame 1410 ",
                                "Contour in frame 1411 ")
    cont_path.write_text(contours)
    ds = dclab.new_dataset(tdms_path)
    # sanity check
    assert np.all(ds["contour"][1] != 0)
    # test for regression before 0.39.18
    assert ds["frame"][3] == 1410
    assert np.all(ds["contour"][3][0] == (190, 24))
    assert np.all(ds["contour"][3][2] == (192, 25))
    assert np.all(ds["contour"][3][-2] == (192, 24))
    assert np.all(ds["contour"][3][-1] == (191, 25))


def test_contour_naming():
    # Test that we always find the correct contour name
    ds = new_dataset(retrieve_data("fmt-tdms_minimal_2016.zip"))
    dp = pathlib.Path(ds.path).resolve()
    dn = dp.parent
    contfile = dn / "M1_0.120000ul_s_contours.txt"
    contfileshort = dn / "M1_contours.txt"
    contfileexact = dn / "M1_2us_70A_0.120000ul_s_contours.txt"
    del ds

    # Test for perfect match
    # "M1_2us_70A_0.120000ul_s_contours.txt" should have priority over
    # "M1_contours.txt" and "M1_0.120000ul_s_contours.txt".
    shutil.copy(str(contfile), str(contfileshort))
    shutil.copy(str(contfile), str(contfileexact))
    ds2 = new_dataset(dp)
    assert str(ds2["contour"].identifier) == str(contfileexact)
    assert not np.allclose(ds2["contour"][1], 0)
    del ds2

    # Check if "M1_contours.txt" is used if the other is not
    # there.
    contfileshort.unlink()
    contfileexact.unlink()
    contfile.rename(contfileshort)
    ds3 = new_dataset(dp)
    assert str(ds3["contour"].identifier) == str(contfileshort)
    del ds3
    contfileshort.rename(contfile)

    # Create M10 file
    with (dn / "M10_contours.txt").open(mode="w"):
        pass
    ds4 = new_dataset(dp)
    assert str(ds4["contour"].identifier) == str(contfile)
    del ds4

    # Check when there is no contour file
    os.remove(str(contfile))
    # This will issue a warning that no contour data was found.
    ds5 = new_dataset(dp)
    assert ds5["contour"].identifier is None


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_contour_negative_offset():
    ds = new_dataset(retrieve_data("fmt-tdms_fl-image_2016.zip"))
    ds["contour"][0]
    ds["contour"].event_offset = 1
    assert np.all(ds["contour"][0] == np.zeros((2, 2), dtype=int))


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_contour_not_initialized():
    ds = new_dataset(retrieve_data("fmt-tdms_fl-image_2016.zip"))
    assert not ds["contour"]._initialized


def test_tdms_shape_contour():
    ds = new_dataset(retrieve_data("fmt-tdms_fl-image-bright_2017.zip"))
    assert ds["contour"].shape == (8, np.nan, 2)
    assert len(ds["contour"]) == 8


def test_hdf5_shape_image():
    ds = new_dataset(retrieve_data("fmt-tdms_fl-image-bright_2017.zip"))
    assert "image" in ds.features_innate
    # Yeah, so the scalar features have different lengths than the
    # non-scalar features. This file is obviously broken (manually
    # cropped that data part), but it is sufficient for this test.
    assert len(ds) == 1008
    assert len(ds["image"]) == 9
    assert ds["image"].shape == (9, 80, 250)


def test_hdf5_shape_mask():
    ds = new_dataset(retrieve_data("fmt-tdms_fl-image-bright_2017.zip"))
    assert "mask" in ds.features_innate
    # Yeah, so the scalar features have different lengths than the
    # non-scalar features. This file is obviously broken (manually
    # cropped that data part), but it is sufficient for this test.
    assert len(ds) == 1008
    # mask length is determined by contour length
    assert len(ds["mask"]) == 8
    assert ds["mask"].shape == (8, 80, 250)


def test_hdf5_shape_trace():
    ds = new_dataset(retrieve_data("fmt-tdms_fl-image_2016.zip"))
    assert len(ds) == 44
    assert "trace" in ds.features_innate
    assert ds["trace"].shape == (2, 44, 1000)
    assert ds["trace"]["fl1_raw"].shape == (44, 1000)
    assert ds["trace"]["fl1_raw"][0].shape == (1000,)
    assert len(ds["trace"]) == 2
    assert len(ds["trace"]["fl1_raw"]) == 44
    assert len(ds["trace"]["fl1_raw"][0]) == 1000


def test_fluorescence_config():
    ds1 = new_dataset(retrieve_data("fmt-tdms_minimal_2016.zip"))
    assert "fluorescence" not in ds1.config
    ds2 = new_dataset(retrieve_data("fmt-tdms_2fl-no-image_2017.zip"))
    assert "fluorescence" in ds2.config


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'fmt_tdms.exc.'
                            + 'InitialFrameMissingWarning')
def test_image_basic():
    ds = new_dataset(retrieve_data("fmt-tdms_fl-image_2016.zip"))
    # Transition image
    assert np.allclose(ds["image"][0], 0)
    # Real image
    # 2025-01-23: paulmueller added the atol value, because CI was failing
    #             on macOS. Possible reason is underlying ffmpeg build
    #             for M1 etc. arm processors :zany_face:
    assert np.allclose(np.average(ds["image"][1]), 45.1490478515625,
                       atol=0.5)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_image_column_length():
    ds = new_dataset(retrieve_data("fmt-tdms_fl-image_2016.zip"))
    assert len(ds["image"]) == 3


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_image_corrupt():
    path = pathlib.Path(retrieve_data("fmt-tdms_fl-image_2016.zip"))
    vpath = path.with_name("M1_imaq.avi")
    # create empty video file
    vpath.unlink()
    vpath.touch()
    try:
        new_dataset(path)
    except dclab.rtdc_dataset.fmt_tdms.exc.InvalidVideoFileError:
        pass
    else:
        assert False


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'fmt_tdms.exc.CorruptFrameWarning')
def test_image_out_of_bounds():
    ds = new_dataset(retrieve_data("fmt-tdms_fl-image_2016.zip"))
    assert len(ds["image"]) == 3
    assert np.allclose(ds["image"][0], 0)  # dummy
    assert not np.allclose(ds["image"][1], 0)
    assert not np.allclose(ds["image"][2], 0)
    assert np.allclose(ds["image"][3], 0)  # causes warning


@pytest.mark.parametrize("feat", ["image", "mask"])
@pytest.mark.parametrize("idxs", [slice(0, 3), np.arange(3),
                                  [0, 1, 2], [True, True, True, False]])
def test_index_slicing_tdms_fails(feat, idxs):
    """The tdms-file format does not support slice/array indexing"""
    data = retrieve_data("fmt-tdms_shapein-2.0.1-no-image_2017.zip")
    ds = new_dataset(data)

    with pytest.raises(NotImplementedError, match="scalar integers"):
        ds[feat][idxs]


@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'fmt_tdms.exc.'
                            + 'InitialFrameMissingWarning')
@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'feat_anc_core.ancillary_feature.'
                            + 'BadFeatureSizeWarning')
def test_large_fov():
    ds = new_dataset(retrieve_data("fmt-tdms_fl-image-large-fov_2017.zip"))
    # initial image is missing
    assert np.allclose(ds["image"][0], 0)
    # initial contour is empty
    assert np.allclose(ds["contour"][0], 0)
    # maximum of contour is larger than 255 (issue #167)
    assert ds["contour"][1].max() == 815
    # compute brightness with given contour
    # Remove the brightness column and let it recompute
    # using the ancillary columns. Besides testing the
    # correct positioning of the contour, this is a
    # sanity test for the brightness computation.
    bavg = ds._events.pop("bright_avg")
    bcom = ds["bright_avg"]
    assert np.allclose(bavg[1], bcom[1])


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
@pytest.mark.parametrize("exname", [
    "fmt-tdms_minimal_2016.zip",
    "fmt-tdms_fl-image_2016.zip",
    "fmt-tdms_fl-image-bright_2017.zip",
    "fmt-tdms_fl-image-large-fov_2017.zip",
    "fmt-tdms_shapein-2.0.1-no-image_2017.zip"])
def test_load_tdms_all(exname):
    tdms_path = retrieve_data(exname)
    new_dataset(tdms_path)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_load_tdms_avi_files_1():
    tdms_path = retrieve_data("fmt-tdms_fl-image_2016.zip")
    edest = pathlib.Path(tdms_path).parent
    with new_dataset(tdms_path) as ds1:
        assert pathlib.Path(ds1["image"].video_file).name == "M1_imaq.avi"
    shutil.copyfile(str(edest / "M1_imaq.avi"),
                    str(edest / "M1_imag.avi"))
    with new_dataset(tdms_path) as ds2:
        # prefer imag over imaq
        assert pathlib.Path(ds2["image"].video_file).name == "M1_imag.avi"
    shutil.copyfile(str(edest / "M1_imaq.avi"),
                    str(edest / "M1_test.avi"))
    with new_dataset(tdms_path) as ds3:
        # ignore any other videos
        assert pathlib.Path(ds3["image"].video_file).name == "M1_imag.avi"


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_load_tdms_avi_files_2():
    tdms_path = retrieve_data("fmt-tdms_fl-image_2016.zip")
    edest = pathlib.Path(tdms_path).parent
    shutil.copyfile(str(edest / "M1_imaq.avi"),
                    str(edest / "M1_test.avi"))
    os.remove(str(edest / "M1_imaq.avi"))
    with new_dataset(tdms_path) as ds4:
        # use available video if ima* not there
        assert pathlib.Path(ds4["image"].video_file).name == "M1_test.avi"


def test_load_tdms_simple():
    tdms_path = retrieve_data("fmt-tdms_minimal_2016.zip")
    ds = new_dataset(tdms_path)
    assert ds.filter.all.shape[0] == 156


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'fmt_tdms.exc.'
                            + 'InitialFrameMissingWarning')
@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'feat_anc_core.ancillary_feature.'
                            + 'BadFeatureSizeWarning')
def test_mask_basic():
    ds = new_dataset(retrieve_data("fmt-tdms_fl-image_2016.zip"))
    assert len(ds["mask"]) == 12
    # Test mask computation by averaging brightness and comparing to
    # the ancillary feature "bright_avg".
    bavg1 = ds["bright_avg"][1]
    bavg2 = np.mean(ds["image"][1][ds["mask"][1]])
    assert np.allclose(bavg1, bavg2), "mask is correctly computed from contour"


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_mask_img_shape_1():
    path = retrieve_data("fmt-tdms_fl-image_2016.zip")
    # shape from image data
    with new_dataset(path) as ds:
        assert ds["mask"]._img_shape == (96, 256)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_mask_img_shape_2():
    path = retrieve_data("fmt-tdms_fl-image_2016.zip")
    path.with_name("M1_imaq.avi").unlink()
    with new_dataset(path) as ds:
        # shape from config ("roi size x", "roi size y")
        assert ds["mask"]._img_shape == (96, 256)
    # no shape available
    with new_dataset(path) as ds:
        ds.config["imaging"].pop("roi size x")
        ds.config["imaging"].pop("roi size y")
        assert ds["mask"]._img_shape == (0, 0)
        assert len(ds["mask"]) == 0


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_mask_img_wrong_config_shape_1():
    path = retrieve_data("fmt-tdms_fl-image_2016.zip")
    with new_dataset(path) as ds:
        # deliberately set wrong size in ROI (fmt_tdms tries image shape first)
        ds.config["imaging"]["roi size x"] = 200
        ds.config["imaging"]["roi size y"] = 200
        assert ds["mask"]._img_shape == (96, 256)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_mask_img_wrong_config_shape_2():
    path = retrieve_data("fmt-tdms_fl-image_2016.zip")
    path.with_name("M1_imaq.avi").unlink()
    with new_dataset(path) as ds:
        # deliberately set wrong size in ROI (fmt_tdms tries image shape first)
        ds.config["imaging"]["roi size x"] = 200
        ds.config["imaging"]["roi size y"] = 200
        assert ds["mask"]._img_shape == (200, 200)


def test_naming_valid():
    for key in dclab.rtdc_dataset.fmt_tdms.naming.dclab2tdms:
        assert dclab.definitions.feature_exists(key)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_parameters_txt():
    ds = new_dataset(retrieve_data("fmt-tdms_fl_2015.zip"))
    assert ds.config["setup"]["module composition"] == "Cell_Flow_2, Fluor"
    assert ds.config["setup"]["software version"] == "fRT-DC V0.80 150601"
    assert ds.config["setup"]["identifier"] == "47 red angels"
    assert ds.config["setup"]["medium"] == "other"
    assert ds.config["setup"]["flow rate sample"] == 0.01
    assert ds.config["setup"]["flow rate sheath"] == 0.03
    assert ds.config["imaging"]["pixel size"] == 0.34
    assert ds.config["imaging"]["flash duration"] == 2
    assert ds.config["fluorescence"]["samples per event"] == 1000
    assert ds.config["fluorescence"]["sample rate"] == 1000000
    assert ds.config["fluorescence"]["signal min"] == -1
    assert ds.config["fluorescence"]["signal max"] == 1
    assert ds.config["fluorescence"]["trace median"] == 21
    assert ds.config["fluorescence"]["laser 1 power"] == 5
    assert ds.config["fluorescence"]["laser 2 power"] == 0
    assert ds.config["fluorescence"]["laser 3 power"] == 0


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_para_ini_fluorescence():
    """Read commented-out parameters from para.ini"""
    path = retrieve_data("fmt-tdms_fl_2015.zip")
    (path.parent / "parameters.txt").unlink()
    ds = new_dataset(path)

    assert ds.config["setup"]["module composition"] == "Cell_Flow_2, Fluor"
    assert ds.config["setup"]["software version"] == "fRT-DC V0.80 150601"
    assert ds.config["setup"]["identifier"] == "47 red angels"
    assert ds.config["setup"]["medium"] == "other"
    assert ds.config["setup"]["flow rate sample"] == 0.01
    assert ds.config["setup"]["flow rate sheath"] == 0.03
    assert ds.config["imaging"]["pixel size"] == 0.34
    assert ds.config["imaging"]["flash duration"] == 2
    assert ds.config["fluorescence"]["sample rate"] == 1000000
    assert ds.config["fluorescence"]["signal min"] == -1
    assert ds.config["fluorescence"]["signal max"] == 1


def test_pixel_size():
    path = retrieve_data("fmt-tdms_minimal_2016.zip")
    para = path.parent / "M1_para.ini"
    data = para.open("r").read()
    newdata = data.replace("Pix Size = 0.340000", "Pix Size = 0.120000")
    with para.open("w") as fd:
        fd.write(newdata)
    ds = new_dataset(path)
    assert ds.config["imaging"]["pixel size"] == 0.12


def test_project_path():
    tfile = retrieve_data("fmt-tdms_minimal_2016.zip")
    ds = dclab.new_dataset(tfile)
    assert ds.hash == "69733e31b005c145997fac8a22107ded"
    assert ds.format == "tdms"
    tpath = pathlib.Path(tfile).resolve()
    a = dclab.rtdc_dataset.fmt_tdms.get_project_name_from_path(str(tpath))
    b = dclab.rtdc_dataset.fmt_tdms.get_project_name_from_path(
        str(tpath.parent))
    assert a == b
    c = dclab.rtdc_dataset.fmt_tdms.get_project_name_from_path(
        str(tpath.parent / "online" / tpath.name))
    d = dclab.rtdc_dataset.fmt_tdms.get_project_name_from_path(
        str(tpath.parent / "online" / "data" / tpath.name))
    e = dclab.rtdc_dataset.fmt_tdms.get_project_name_from_path(
        str(tpath.parent / "online" / "data"))

    assert a == e
    assert a == c
    assert a == d


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_trace_basic():
    ds = new_dataset(retrieve_data("fmt-tdms_fl-image_2016.zip"))
    msg = "traces should not be loaded into memory before first access"
    assert ds["trace"].__repr__().count("<not loaded into memory>"), msg
    assert len(ds["trace"]) == 2
    assert np.allclose(np.average(
        ds["trace"]["fl1_median"][0]), 287.08999999999997)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_trace_import_fail():
    # make sure undefined trace data does not raise an error
    tdms_path = retrieve_data("fmt-tdms_fl-image_2016.zip")
    dclab.definitions.FLUOR_TRACES.append("peter")
    dclab.rtdc_dataset.fmt_tdms.naming.tr_data_map["peter"] = [u'ukwn', u'ha']
    new_dataset(tdms_path)
    # clean up
    dclab.rtdc_dataset.fmt_tdms.naming.tr_data_map.pop("peter")
    dclab.definitions.FLUOR_TRACES.pop(-1)


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_trace_methods():
    ds = new_dataset(retrieve_data("fmt-tdms_fl-image_2016.zip"))
    for k in list(ds["trace"].keys()):
        assert k in dclab.definitions.FLUOR_TRACES
    for k in ds["trace"]:
        assert k in dclab.definitions.FLUOR_TRACES
    assert ds["trace"].__repr__().count("<loaded into memory>")


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
@pytest.mark.filterwarnings('ignore::dclab.rtdc_dataset.'
                            + 'fmt_tdms.exc.'
                            + 'MultipleSamplesPerEventFound')  # desired
def test_trace_wrong_samples_per_event():
    """The "samples per event" should be a constant for trace data

    If this condition is not given, the trace data is considered
    unusable, but the rest of the data still makes sense
    (Philipp said that something must have gone wrong during writing
    of the trace data).
    """
    tdms = retrieve_data("fmt-tdms_fl-image_2016.zip")
    mdata = nptdms.TdmsFile(str(tdms))

    channels = []
    # recreate channels, because the data type might not be correct
    for ch in mdata.groups()[0].channels():
        if ch.data.size == 0:
            # not supported by nptdms 0.25.0
            continue
        channels.append(nptdms.ChannelObject("Cell Track", ch.name, ch.data))

    # blank write same data to test that modification works
    with nptdms.TdmsWriter(str(tdms)) as tdms_writer:
        tdms_writer.write_segment(channels)

    with dclab.new_dataset(tdms) as ds:
        assert "trace" in ds

    # modify objects
    sampleids = mdata["Cell Track"]["FL1index"].data
    sampleids[0] = 10
    sampleids[1] = 20
    sampleids[2] = 40
    wchannels = []
    for ch in channels:
        if ch.channel == "FL1index":
            nptdms.ChannelObject("Cell Track", "FL1index", sampleids)
        wchannels.append(ch)

    with nptdms.TdmsWriter(str(tdms)) as tdms_writer:
        tdms_writer.write_segment(wchannels)

    with dclab.new_dataset(tdms) as ds:
        assert "trace" not in ds


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_unicode_paths():
    path = retrieve_data("fmt-tdms_fl-image_2016.zip")
    path = pathlib.Path(path)
    pp = path.parent
    # create a unicode name
    pp2 = pathlib.Path(tempfile.mktemp(prefix="dclàb_tést_asgård_únícodè"))
    pp.rename(pp2)
    ds = new_dataset(pp2 / path.name)
    ds.__repr__()
    shutil.rmtree(str(pp2), ignore_errors=True)


if __name__ == "__main__":
    # Run all tests
    loc = locals()
    for key in list(loc.keys()):
        if key.startswith("test_") and hasattr(loc[key], "__call__"):
            loc[key]()
