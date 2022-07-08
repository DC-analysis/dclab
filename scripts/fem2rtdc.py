"""Generate in-silico .rtdc files from FEM simulation data

The simulation dataset must be an HDF5 file with a specific structure
that contains the results of FEM simulations. The original HDF5 files
were provided by Lucas Wittwer.

This script generates multiple (default: 5) .rtdc datasets using the
shape data from the FEM simulations. The datasets are generated from
all simulations, but each with a random pixel offset when casting the
contour data onto a regular grid. In addition, an "original" .rtdc
dataset is generated, which consists of the original simulation
parameters.

Supported simulation modalities:
- 2D axis-symmetric (used for the first emodulus LUT in dclab)

Testing: `pytest fem2rtdc.py`
"""
import argparse
import copy
import pathlib

import cv2
import h5py
import numpy as np

import dclab
from dclab.features.contour import get_contour
from dclab.features.volume import get_volume
from dclab.rtdc_dataset import write_hdf5
from dclab.polygon_filter import PolygonFilter


def features_from_2daxis_simulation(fem_cont_2daxis, pixel_size=0.34,
                                    shape=(80, 250), seed=42):
    """Compute dclab features from the FEM contour

    Map the FEM contour onto a regular grid and displace it
    randomly by up to +/- 0.5 pixels (used for investigating
    pixelation effects).

    Parameters
    ----------
    fem_cont_2daxis: ndarray of shape (2, N)
        The 2D axis-symmetric FEM contour. It should only cover
        180 degrees of the entire contour.
    pixel_size: float
        Detector pixel size [um]
    shape: tuple of int
        Detector shape / ROI in pixels
    seed: int
        Seed for numpy.random which is used for the random
        displacements.
    """
    np.random.seed(seed)
    offx = np.random.uniform(0, 1)
    offy = np.random.uniform(0, 1)
    # make contour circular
    simsym = fem_cont_2daxis
    sh = simsym.shape[1]
    cont_sim = np.zeros((simsym.shape[1] * 2 - 2, 2), dtype=np.float32)
    cont_sim[:sh, 1] = simsym[0]
    cont_sim[:sh, 0] = simsym[1]
    cont_sim[sh:, 1] = simsym[0, 1:-1][::-1]
    cont_sim[sh:, 0] = -simsym[1, 1:-1][::-1]
    # convert um to pixels
    cont_sim /= pixel_size
    # center around center of channel
    cont_sim[:, 0] += shape[0] // 2 - np.mean(cont_sim[:, 0]) / 2 + offx
    cont_sim[:, 1] += shape[1] // 2 - np.mean(cont_sim[:, 1]) / 2 + offy
    # put on grid
    mask = np.zeros(shape)
    for x in np.arange(shape[0]):
        for y in np.arange(shape[1]):
            # mask[x, y] = cv2.pointPolygonTest(cont_sim, (x, y), False) >= 0
            mask[x, y] = PolygonFilter.point_in_poly((x, y), cont_sim)

    return features_from_mask(mask, pixel_size=pixel_size)


def features_from_mask(mask, pixel_size=0.34):
    """Compute dclab features from a binary mask image

    This essentially mimicks what Shape-In does with OpenCV.

    Parameters
    ----------
    mask: 2d boolean ndarray
        binary mask image
    pixel_size: float
        Detector pixel size [um]
    """
    cont_raw = get_contour(mask)
    # this is how Shape-In does things
    cont = cv2.convexHull(cont_raw)
    arc = cv2.arcLength(cont, True)
    mu = cv2.moments(cont, False)
    pos_x = (mu["m10"] / mu["m00"]) * pixel_size
    pos_y = (mu["m01"] / mu["m00"]) * pixel_size
    feats = {
        # scalar features
        "area_um": mu["m00"] * pixel_size ** 2,
        "deform": 1 - 2.0 * np.sqrt(np.pi * mu["m00"]) / arc,
        "inert_ratio_cvx": np.sqrt(mu["mu20"] / mu["mu02"]),
        "pos_x": pos_x,
        "pos_y": pos_y,
        "volume": get_volume(cont_raw, pos_x, pos_y, pixel_size),
        # image data
        "image": np.array(254 * mask, dtype=np.uint8),
        "mask": mask,
    }
    return feats


def generate_rtdc_from_simulation(path, repetitions=5, pixel_size=0.34):
    """Generate in-silico .rtdc files from a FEM simulation file

    In addition to the in-silico datasets, an "original" .rtdc
    dataset is generated, which consists of the original simulation
    parameters.

    Parameters
    ----------
    path: pathlib.Path or str
        The simulation dataset must be an HDF5 file with a specific
        structure that contains the results of FEM simulations. The
        original HDF5 files were provided by Lucas Wittwer.
    repetitions: int
        Number of in-silico .rtdc files to generate. The datasets are
        generated from all simulations, but each with a random pixel
        offset when casting the contour data onto a regular grid.
    pixel_size: float
        Detector pixel size [um]
    """
    path = pathlib.Path(path)

    h5mapped = []
    for ii in range(repetitions):
        pi = path.with_suffix(".map{}.rtdc".format(ii + 1))
        if pi.exists():
            pi.unlink()
        h5mapped.append(write_hdf5.write(path_or_h5file=pi, mode="append"))

    # Exact data from simulations
    po = path.with_suffix(".orig.rtdc")
    if po.exists():
        po.unlink()
    h5orig = write_hdf5.write(path_or_h5file=po, mode="append")

    # Extract dataset from simulation contour
    with h5py.File(path, "r") as h5:
        # determine total size
        ntot = 0
        for dk in h5.keys():
            for sk in h5[dk]:
                ntot += 1

        # metadata
        meta = {
            "setup": {
                "channel width": h5.attrs["channel_width"],
                "flow rate": h5.attrs["flow_rate"],
                "chip region": "channel",
                "medium": "other",
                "software version": "dclab {}".format(dclab.__version__)
            },
            "imaging": {
                "pixel size": pixel_size,
                "roi size x": 250,
                "roi size y": 80,
            },
            "experiment": {
                "date": h5.attrs["date"],
                "time": "00:00:00",
                "event count": ntot,
            },
        }

        for ii, wrt in enumerate(h5mapped):
            mi = copy.deepcopy(meta)
            mi["experiment"]["run index"] = ii + 1
            mi["experiment"]["sample"] = "Contour mapping {}".format(ii + 1)
            write_hdf5.write(wrt, meta=mi, mode="append")

        mo = copy.deepcopy(meta)
        mo["experiment"]["run index"] = 1
        mo["experiment"]["sample"] = "Simulation reference"
        write_hdf5.write(h5orig, meta=mo, mode="append")

        dkeys = sorted(h5.keys(), key=lambda x: int(x))
        ss = 0
        for dk in dkeys:
            for sk in h5[dk]:
                ss += 1
                print("Converting simulation data: {:.1f}%".format(
                    ss / ntot * 100), end="\r")
                sim = h5[dk][sk]
                simsym = sim["coords"][:]
                for jj, wrt in enumerate(h5mapped):
                    feats = features_from_2daxis_simulation(
                        simsym, pixel_size=pixel_size, seed=jj)
                    write_hdf5.write(wrt, data=feats, mode="append",
                                     compression="gzip")

                feats_orig = {
                    "area_um": sim.attrs["area"],
                    "deform": sim.attrs["deformation"],
                    "volume": sim.attrs["volume"],
                    "emodulus": sim.parent.attrs["emodulus"] / 1000,
                }
                write_hdf5.write(h5orig, data=feats_orig, mode="append",
                                 compression="gzip")

        for ho in h5mapped + [h5orig]:
            ho.close()


def test_features_from_2daxis_simulation():
    feats = features_from_2daxis_simulation(
        fem_cont_2daxis=np.copy(test_coords),
        pixel_size=0.34,
        shape=(80, 250),
        seed=42)
    assert np.allclose(feats['area_um'], 33.69740000000001)
    assert np.allclose(feats['deform'], 0.1168877846976153)
    assert np.allclose(feats['inert_ratio_cvx'], 1.285950706011422)
    assert np.allclose(feats['pos_x'], 43.4609033733562)
    assert np.allclose(feats['pos_y'], 13.717804459691251)
    assert np.allclose(feats['volume'], 125.581725903289)


#: The first contour from https://doi.org/10.6084/m9.figshare.12155064.v3
test_coords = np.array(
    [[-2.647751, -2.647751, -2.647751, -2.647141, -2.646541,
      -2.645251, -2.643971, -2.642191, -2.640421, -2.638001,
      -2.635591, -2.632401, -2.629221, -2.625081, -2.620941,
      -2.6156511, -2.610371, -2.6037111, -2.597061, -2.588711,
      -2.5803611, -2.5700011, -2.5596411, -2.5470111, -2.5343711,
      -2.5190911, -2.5038111, -2.485591, -2.467371, -2.445271,
      -2.423161, -2.397831, -2.3724911, -2.343981, -2.315471,
      -2.284131, -2.252801, -2.216171, -2.179531, -2.134901,
      -2.090281, -2.032001, -1.973711, -1.900521, -1.827321,
      -1.744361, -1.661401, -1.568581, -1.4757711, -1.3715811,
      -1.267401, -1.152081, -1.036771, -0.90868104, -0.780591,
      -0.639801, -0.49901104, -0.34514102, -0.19128104, -0.02494103,
      0.14138897, 0.31868896, 0.49598897, 0.68322897, 0.870469,
      1.064479, 1.258499, 1.4578589, 1.6572189, 1.8587589,
      2.060299, 2.262629, 2.464949, 2.665009, 2.865059,
      3.0620189, 3.258979, 3.449929, 3.640879, 3.824039,
      4.007209, 4.180559, 4.353919, 4.513099, 4.672279,
      4.815419, 4.958549, 5.078679, 5.198809, 5.294079,
      5.389349, 5.455509, 5.521669, 5.5633388, 5.605009,
      5.622299, 5.639589],
     [0., 0.09835, 0.19869, 0.29785, 0.39701,
      0.49532, 0.59363, 0.69125, 0.78887, 0.88545,
      0.98203, 1.07731, 1.17258, 1.2662, 1.35981,
      1.4515, 1.54319, 1.63265, 1.7221, 1.809,
      1.89591, 1.97986, 2.06381, 2.14455, 2.2253,
      2.30242, 2.37953, 2.45276, 2.52598, 2.59443,
      2.66287, 2.72659, 2.79032, 2.84945, 2.90857,
      2.96377, 3.01897, 3.0693, 3.11963, 3.16366,
      3.2077, 3.24223, 3.27676, 3.29975, 3.32275,
      3.33584, 3.34893, 3.35274, 3.35654, 3.35024,
      3.34393, 3.32765, 3.31137, 3.28401, 3.25665,
      3.21855, 3.18045, 3.13136, 3.08227, 3.02287,
      2.96347, 2.89481, 2.82615, 2.74929, 2.67243,
      2.58959, 2.50675, 2.41919, 2.33163, 2.24136,
      2.15109, 2.05934, 1.96759, 1.87593, 1.78428,
      1.69324, 1.60219, 1.51283, 1.42347, 1.33594,
      1.24841, 1.16308, 1.07775, 0.99495, 0.91216,
      0.83188, 0.7516, 0.6742, 0.59681, 0.52143,
      0.44606, 0.37198, 0.29791, 0.22335, 0.1488,
      0.07659, 0.]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('input', type=str, help='Input path (.hdf5 file)')
    parser.add_argument('--repetitions', type=int, default=5,
                        help='Number of repetitions (grid mapping)')
    parser.add_argument('--pixelsize', type=float, default=0.34,
                        help='Detector pixel size [um]')

    args = parser.parse_args()

    generate_rtdc_from_simulation(path=pathlib.Path(args.input),
                                  repetitions=args.repetitions,
                                  pixel_size=args.pixelsize)
