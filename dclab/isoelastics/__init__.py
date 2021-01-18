"""Isoelastics management"""
import functools
import warnings

from pkg_resources import resource_filename

import numpy as np

from .. import definitions as dfn
from ..features import emodulus as feat_emod

ISOFILES = ["isoel-linear-2Daxis-analyt-area_um-deform.txt",
            "isoel-linear-2Daxis-FEM-area_um-deform.txt",
            "isoel-linear-2Daxis-FEM-volume-deform.txt",
            ]
ISOFILES = [resource_filename("dclab.isoelastics", _if) for _if in ISOFILES]


class IsoelasticsEmodulusMeaninglessWarning(UserWarning):
    pass


class Isoelastics(object):
    def __init__(self, paths=None):
        """Isoelasticity line management

        .. versionchanged:: 0.24.0
            The isoelasticity lines of the analytical model
            :cite:`Mietke2015` and the linear-elastic numerical
            model :cite:`Mokbel2017` were recomputed with an
            equidistant spacing. The metadata section of the text
            file format was restructured.
        """
        if paths is None:
            paths = []
        self._data = AutoRecursiveDict()

        for path in paths:
            self.load_data(path)

    def _add(self, isoel, col1, col2, lut_identifier, meta):
        """Convenience method for population self._data"""
        self._data[lut_identifier][col1][col2]["isoelastics"] = isoel
        self._data[lut_identifier][col1][col2]["meta"] = meta

        # Use advanced slicing to flip the data columns
        isoel_flip = [iso[:, [1, 0, 2]] for iso in isoel]
        self._data[lut_identifier][col2][col1]["isoelastics"] = isoel_flip
        self._data[lut_identifier][col2][col1]["meta"] = meta

    def add(self, isoel, col1, col2, channel_width,
            flow_rate, viscosity, method=None,
            lut_identifier=None):
        """Add isoelastics

        Parameters
        ----------
        isoel: list of ndarrays
            Each list item resembles one isoelastic line stored
            as an array of shape (N,3). The last column contains
            the emodulus data.
        col1: str
            Name of the first feature of all isoelastics
            (e.g. isoel[0][:,0])
        col2: str
            Name of the second feature of all isoelastics
            (e.g. isoel[0][:,1])
        channel_width: float
            Channel width in µm
        flow_rate: float
            Flow rate through the channel in µL/s
        viscosity: float
            Viscosity of the medium in mPa*s
        method: str
            The method used to compute the isoelastics
            DEPRECATED since 0.32.0. Please use
            `lut_identifier` instead.
        lut_identifier: str:
            Look-up table identifier used to identify which
            isoelasticity lines to show. The function
            :func:`get_available_identifiers` returns a list
            of available identifiers.

        Notes
        -----
        The following isoelastics are automatically added for
        user convenience:

        - isoelastics with `col1` and `col2` interchanged
        - isoelastics for circularity if deformation was given
        """
        lut_identifier = check_lut_identifier(lut_identifier, method)

        for col in [col1, col2]:
            if not dfn.scalar_feature_exists(col):
                raise ValueError("Not a valid feature name: {}".format(col))

        meta = [channel_width, flow_rate, viscosity]

        # Add the feature data
        self._add(isoel, col1, col2, lut_identifier, meta)

        # Also add the feature data for circularity
        if "deform" in [col1, col2]:
            col1c, col2c = col1, col2
            if col1c == "deform":
                deform_ax = 0
                col1c = "circ"
            else:
                deform_ax = 1
                col2c = "circ"
            iso_circ = []
            for iso in isoel:
                iso = iso.copy()
                iso[:, deform_ax] = 1 - iso[:, deform_ax]
                iso_circ.append(iso)
            self._add(iso_circ, col1c, col2c, lut_identifier, meta)

    @staticmethod
    def add_px_err(isoel, col1, col2, px_um, inplace=False):
        """Undo pixelation correction

        Since isoelasticity lines are usually computed directly from
        the simulation data (e.g. the contour data are not discretized
        on a grid but are extracted from FEM simulations), they are
        not affected by pixelation effects as described in
        :cite:`Herold2017`.

        If the isoelasticity lines are displayed alongside experimental
        data (which are affected by pixelation effects), then the lines
        must be "un"-corrected, i.e. the pixelation error must be added
        to the lines to match the experimental data.

        Parameters
        ----------
        isoel: list of 2d ndarrays of shape (N, 3)
            Each item in the list corresponds to one isoelasticity
            line. The first column is defined by `col1`, the second
            by `col2`, and the third column is the emodulus.
        col1, col2: str
            Define the fist two columns of each isoelasticity line.
        px_um: float
            Pixel size [µm]
        inplace: bool
            If True, do not create a copy of the data in `isoel`
        """
        new_isoel = []
        for iso in isoel:
            iso = np.array(iso, copy=not inplace)
            delt1, delt2 = feat_emod.get_pixelation_delta_pair(
                feat1=col1, feat2=col2, data1=iso[:, 0], data2=iso[:, 1],
                px_um=px_um)
            iso[:, 0] += delt1
            iso[:, 1] += delt2
            new_isoel.append(iso)
        return new_isoel

    @staticmethod
    def convert(isoel, col1, col2,
                channel_width_in, channel_width_out,
                flow_rate_in, flow_rate_out,
                viscosity_in, viscosity_out,
                inplace=False):
        """Perform isoelastics scale conversion

        Parameters
        ----------
        isoel: list of 2d ndarrays of shape (N, 3)
            Each item in the list corresponds to one isoelasticity
            line. The first column is defined by `col1`, the second
            by `col2`, and the third column is the emodulus.
        col1, col2: str
            Define the fist to columns of each isoelasticity line.
            One of ["area_um", "circ", "deform"]
        channel_width_in: float
            Original channel width [µm]
        channel_width_out: float
            Target channel width [µm]
        flow_rate_in: float
            Original flow rate [µL/s]
        flow_rate_out: float
            Target flow rate [µL/s]
        viscosity_in: float
            Original viscosity [mPa*s]
        viscosity_out: float
            Target viscosity [mPa*s]
        inplace: bool
            If True, do not create a copy of the data in `isoel`

        Returns
        -------
        isoel_scale: list of 2d ndarrays of shape (N, 3)
            The scale-converted isoelasticity lines.

        Notes
        -----
        If only the positions of the isoelastics are of interest and
        not the value of the elastic modulus, then it is sufficient
        to supply values for the channel width and set the values
        for flow rate and viscosity to a constant (e.g. 1).

        See Also
        --------
        dclab.features.emodulus.scale_linear.scale_feature: scale
            conversion method used
        """
        new_isoel = []

        for iso in isoel:
            iso = np.array(iso, copy=not inplace)
            scale_kw = {"channel_width_in": channel_width_in,
                        "channel_width_out": channel_width_out,
                        "flow_rate_in": flow_rate_in,
                        "flow_rate_out": flow_rate_out,
                        "viscosity_in": viscosity_in,
                        "viscosity_out": viscosity_out,
                        "inplace": True}
            feat_emod.scale_feature(col1, data=iso[:, 0], **scale_kw)
            feat_emod.scale_feature(col2, data=iso[:, 1], **scale_kw)
            feat_emod.scale_emodulus(emodulus=iso[:, 2], **scale_kw)
            new_isoel.append(iso)
        return new_isoel

    def get(self, col1, col2, channel_width, method=None, lut_identifier=None,
            flow_rate=None, viscosity=None, add_px_err=False, px_um=None):
        """Get isoelastics

        Parameters
        ----------
        col1: str
            Name of the first feature of all isoelastics
            (e.g. isoel[0][:,0])
        col2: str
            Name of the second feature of all isoelastics
            (e.g. isoel[0][:,1])
        channel_width: float
            Channel width in µm
        method: str
            The method used to compute the isoelastics
            DEPRECATED since 0.32.0. Please use
            `lut_identifier` instead.
        lut_identifier: str:
            Look-up table identifier used to identify which
            isoelasticity lines to show. The function
            :func:`get_available_identifiers` returns a list
            of available identifiers.
        flow_rate: float or `None`
            Flow rate through the channel in µL/s. If set to
            `None`, the flow rate of the imported data will
            be used (only do this if you do not need the
            correct values for elastic moduli).
        viscosity: float or `None`
            Viscosity of the medium in mPa*s. If set to
            `None`, the flow rate of the imported data will
            be used (only do this if you do not need the
            correct values for elastic moduli).
        add_px_err: bool
            If True, add pixelation errors according to
            C. Herold (2017), https://arxiv.org/abs/1704.00572
            and scripts/pixelation_correction.py
        px_um: float
            Pixel size [µm], used for pixelation error computation

        See Also
        --------
        dclab.features.emodulus.scale_linear.scale_feature: scale
            conversion method used
        dclab.features.emodulus.pxcorr.get_pixelation_delta:
            pixelation correction (applied to the feature data)
        """
        lut_identifier = check_lut_identifier(lut_identifier, method)

        for col in [col1, col2]:
            if not dfn.scalar_feature_exists(col):
                raise ValueError("Not a valid feature name: {}".format(col))

        if "isoelastics" not in self._data[lut_identifier][col2][col1]:
            msg = "No isoelastics matching {}, {}, {}".format(col1, col2,
                                                              lut_identifier)
            raise KeyError(msg)

        isoel = self._data[lut_identifier][col1][col2]["isoelastics"]
        meta = self._data[lut_identifier][col1][col2]["meta"]

        if flow_rate is None:
            flow_rate = meta[1]

        if viscosity is None:
            viscosity = meta[2]

        isoel_ret = self.convert(isoel, col1, col2,
                                 channel_width_in=meta[0],
                                 channel_width_out=channel_width,
                                 flow_rate_in=meta[1],
                                 flow_rate_out=flow_rate,
                                 viscosity_in=meta[2],
                                 viscosity_out=viscosity,
                                 inplace=False)

        if add_px_err:
            self.add_px_err(isoel=isoel_ret,
                            col1=col1,
                            col2=col2,
                            px_um=px_um,
                            inplace=True)

        return isoel_ret

    def get_with_rtdcbase(self, col1, col2, dataset, method=None,
                          lut_identifier=None, viscosity=None,
                          add_px_err=False):
        """Convenience method that extracts the metadata from RTDCBase

        Parameters
        ----------
        col1: str
            Name of the first feature of all isoelastics
            (e.g. isoel[0][:,0])
        col2: str
            Name of the second feature of all isoelastics
            (e.g. isoel[0][:,1])
        method: str
            The method used to compute the isoelastics
            DEPRECATED since 0.32.0. Please use
            `lut_identifier` instead.
        lut_identifier: str:
            Look-up table identifier used to identify which
            isoelasticity lines to show. The function
            :func:`get_available_identifiers` returns a list
            of available identifiers.
        dataset: dclab.rtdc_dataset.RTDCBase
            The dataset from which to obtain the metadata.
        viscosity: float, `None`, or False
            Viscosity of the medium in mPa*s. If set to
            `None`, the viscosity is computed from the meta
            data (medium, flow rate, channel width, temperature)
            in the [setup] config section. If this is not possible,
            the flow rate of the imported data is used and a warning
            will be issued.
        add_px_err: bool
            If True, add pixelation errors according to
            C. Herold (2017), https://arxiv.org/abs/1704.00572
            and scripts/pixelation_correction.py
        """
        lut_identifier = check_lut_identifier(lut_identifier, method)

        cfg = dataset.config
        if viscosity is None:
            if "temperature" in cfg["setup"] and "medium" in cfg["setup"]:
                viscosity = feat_emod.get_viscosity(
                    medium=cfg["setup"]["medium"],
                    channel_width=cfg["setup"]["channel width"],
                    flow_rate=cfg["setup"]["flow rate"],
                    temperature=cfg["setup"]["temperature"])
            else:
                warnings.warn("Computing emodulus data for isoelastics from "
                              + "RTDCBase is not possible. Isoelastics will "
                              + "not have correct emodulus values (this is "
                              + "not relevant for plotting).",
                              IsoelasticsEmodulusMeaninglessWarning)
        return self.get(col1=col1,
                        col2=col2,
                        lut_identifier=lut_identifier,
                        channel_width=cfg["setup"]["channel width"],
                        flow_rate=cfg["setup"]["flow rate"],
                        viscosity=viscosity,
                        add_px_err=add_px_err,
                        px_um=cfg["imaging"]["pixel size"])

    def load_data(self, path):
        """Load isoelastics from a text file

        Parameters
        ----------
        path: str
            Path to an isoelasticity lines text file
        """
        isodata, meta = feat_emod.load_mtext(path)
        assert len(meta["column features"]) == 3
        assert meta["column features"][2] == "emodulus"
        assert meta["lut identifier"]

        # Slice out individual isoelastics
        emoduli = np.unique(isodata[:, 2])
        isoel = []
        for emod in emoduli:
            where = isodata[:, 2] == emod
            isoel.append(isodata[where])

        # Add isoelastics to instance
        self.add(isoel=isoel,
                 col1=meta["column features"][0],
                 col2=meta["column features"][1],
                 channel_width=meta["channel_width"],
                 flow_rate=meta["flow_rate"],
                 viscosity=meta["fluid_viscosity"],
                 lut_identifier=meta["lut identifier"])


class AutoRecursiveDict(dict):
    def __getitem__(self, key):
        if key not in self:
            self[key] = AutoRecursiveDict()
        return super(AutoRecursiveDict, self).__getitem__(key)


def check_lut_identifier(lut_identifier, method):
    """Transitional function that can be removed once `method` is removed"""
    if lut_identifier is None:
        if method is not None:
            warnings.warn("The `method` argument is deprecated "
                          + "please use `lut_identifier` instead!",
                          DeprecationWarning)
            if method == "analytical":
                lut_identifier = "LE-2D-ana-18"
            elif method == "numerical":
                lut_identifier = "LE-2D-FEM-19"
            else:
                raise ValueError("Please read the docstring")
    # Now check again (this can be removed once lut_identifier becomes
    # a non-keyword argument)
    if lut_identifier is None:
        raise ValueError("Please specify `lut_identifier`!")
    return lut_identifier


@functools.lru_cache()
def get_available_identifiers():
    """Return a list of available LUT identifiers"""
    ids = []
    for ff in ISOFILES:
        _, meta = feat_emod.load_mtext(ff)
        ids.append(meta["lut identifier"])
    return sorted(set(ids))


def get_default():
    """Return default isoelasticity lines"""
    return Isoelastics(ISOFILES)
