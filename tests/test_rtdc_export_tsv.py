import dclab

from helper_methods import example_data_dict, retrieve_data

import pytest


def test_tsv_export(tmp_path):
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=222, keys=keys)
    ds = dclab.new_dataset(ddict)

    f1 = tmp_path / "test.tsv"
    f2 = tmp_path / "test_unicode.tsv"

    ds.export.tsv(f1, keys, override=True)
    ds.export.tsv(f2, [u"area_um", u"deform", u"time",
                       u"frame", u"fl3_width"], override=True)

    with f1.open("r", encoding="utf-8") as fd:
        a1 = fd.read()

    with f2.open("r", encoding="utf-8") as fd:
        a2 = fd.read()

    assert a1 == a2
    assert len(a1) != 0


@pytest.mark.filterwarnings(
    "ignore::dclab.rtdc_dataset.config.WrongConfigurationTypeWarning")
def test_tsv_export_ds_metadata(tmp_path):
    path = retrieve_data("fmt-hdf5_image-bg_2020.zip")
    ds = dclab.new_dataset(path)

    f1 = tmp_path / "test.tsv"

    ds.export.tsv(f1, ["area_um", "bright_avg"], override=True)

    lines = f1.read_text(encoding="utf-8").split("\n")

    for ll in [
        "# dc:experiment:date = 2020-10-23",
        "# dc:experiment:event count = 5",
        "# dc:experiment:run index = 1",
        "# dc:experiment:sample = background image example",
        "# dc:experiment:time = 10:37:15",
        "# dc:filtering:enable filters = True",
        "# dc:filtering:hierarchy parent = none",
        "# dc:filtering:limit events = 0",
        "# dc:filtering:polygon filters = []",
        "# dc:filtering:remove invalid events = False",
        "# dc:imaging:flash device = LED (ZMD L1)",
        "# dc:imaging:flash duration = 2.0",
        "# dc:imaging:frame rate = 2000.0",
        "# dc:imaging:pixel size = 0.34",
        "# dc:imaging:roi position x = 720",
        "# dc:imaging:roi position y = 512",
        "# dc:imaging:roi size x = 250",
        "# dc:imaging:roi size y = 80",
        "# dc:online_contour:bin area min = 10",
        "# dc:online_contour:bin kernel = 5",
        "# dc:online_contour:bin threshold = -6",
        "# dc:online_contour:image blur = 0",
        "# dc:online_contour:no absdiff = True",
        "# dc:online_filter:target event count = 5",
        "# dc:setup:channel width = 20.0",
        "# dc:setup:chip region = channel",
        "# dc:setup:flow rate = 0.16",
        "# dc:setup:flow rate sample = 0.04",
        "# dc:setup:flow rate sheath = 0.12",
        "# dc:setup:identifier = ZMDD-AcC-478cb7-dc924c",
    ]:
        assert ll in lines


def test_tsv_override(tmp_path):
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=212, keys=keys)
    ds = dclab.new_dataset(ddict)

    f1 = tmp_path / "test.tsv"
    ds.export.tsv(f1, keys, override=True)
    try:
        ds.export.tsv(f1.with_name(f1.stem), keys, override=False)
    except OSError:
        pass
    else:
        raise ValueError("Should append .tsv and not override!")


def test_tsv_not_filtered(tmp_path):
    keys = ["area_um", "deform", "time", "frame", "fl3_width"]
    ddict = example_data_dict(size=127, keys=keys)
    ds = dclab.new_dataset(ddict)

    f1 = tmp_path / "test.tsv"
    ds.export.tsv(f1, keys, filtered=False)
