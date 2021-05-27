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
    cont_sim = np.zeros((simsym.shape[1]*2-2, 2), dtype=np.float32)
    cont_sim[:sh, 1] = simsym[0]
    cont_sim[:sh, 0] = simsym[1]
    cont_sim[sh:, 1] = simsym[0, 1:-1][::-1]
    cont_sim[sh:, 0] = -simsym[1, 1:-1][::-1]
    # convert um to pixels
    cont_sim /= pixel_size
    # center around center of channel
    cont_sim[:, 0] += shape[0] // 2 - np.mean(cont_sim[:, 0])/2 + offx
    cont_sim[:, 1] += shape[1] // 2 - np.mean(cont_sim[:, 1])/2 + offy
    # put on grid
    mask = np.zeros(shape)
    for x in np.arange(shape[0]):
        for y in np.arange(shape[1]):
            mask[x, y] = cv2.pointPolygonTest(cont_sim, (x, y), False) >= 0

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
    pos_x = (mu["m10"]/mu["m00"])*pixel_size
    pos_y = (mu["m01"]/mu["m00"])*pixel_size
    feats = {
        # scalar features
        "area_um": mu["m00"]*pixel_size**2,
        "deform": 1 - 2.0 * np.sqrt(np.pi * mu["m00"]) / arc,
        "inert_ratio_cvx": np.sqrt(mu["mu20"] / mu["mu02"]),
        "pos_x": pos_x,
        "pos_y": pos_y,
        "volume": get_volume(cont_raw, pos_x, pos_y, pixel_size),
        # image data
        "image": np.array(254*mask, dtype=np.uint8),
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
        pi = path.with_suffix(".map{}.rtdc".format(ii+1))
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
            mi["experiment"]["run index"] = ii+1
            mi["experiment"]["sample"] = "Contour mapping {}".format(ii+1)
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
                    ss/ntot*100), end="\r")
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
                    "emodulus": sim.parent.attrs["emodulus"]/1000,
                }
                write_hdf5.write(h5orig, data=feats_orig, mode="append",
                                 compression="gzip")

        for ho in h5mapped + [h5orig]:
            ho.close()


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
