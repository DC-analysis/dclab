"""Class for efficiently handling contour data"""
import numbers
import sys
import warnings

import numpy as np

from ...features import inert_ratio

from .exc import ContourIndexingError


class ContourVerificationWarning(UserWarning):
    pass


class ContourColumn(object):
    def __init__(self, rtdc_dataset):
        """A wrapper for ContourData that takes into account event offsets

        Event offsets appear when the first event that is recorded in the
        tdms files does not have a corresponding contour in the contour
        text file.
        """
        fname = self.find_contour_file(rtdc_dataset)
        if fname is None:
            self.identifier = None
        else:
            if sys.version_info[0] == 2:
                self.identifier = str(fname).decode("utf-8")
            else:
                self.identifier = str(fname)
        if fname is not None:
            self._contour_data = ContourData(fname)
            self._initialized = False
        else:
            self._contour_data = []
            # prevent `determine_offset` to be called
            self._initialized = True
        self.frame = rtdc_dataset["frame"]
        # if they are set, these features are used for verifying the contour
        self.pxfeat = {}

        if "area_msd" in rtdc_dataset:
            self.pxfeat["area_msd"] = rtdc_dataset["area_msd"]
        if "pixel size" in rtdc_dataset.config["imaging"]:
            px_size = rtdc_dataset.config["imaging"]["pixel size"]
            for key in ["pos_x", "pos_y", "size_x", "size_y"]:
                if key not in rtdc_dataset.features_innate:
                    # abort
                    self.pxfeat.clear()
                    break
                self.pxfeat[key] = rtdc_dataset[key] / px_size
        if "image" in rtdc_dataset:
            self.shape = rtdc_dataset["image"].shape
        else:
            self.shape = None
        self.event_offset = 0

    def __getitem__(self, idx):
        if not isinstance(idx, numbers.Integral):
            raise NotImplementedError(
                "The RTDC_TDMS data handler does not support indexing with "
                "anything else than scalar integers. Please convert your data "
                "to the .rtdc file format!")

        if not self._initialized:
            self.determine_offset()
        idnew = idx-self.event_offset
        cdata = None
        if idnew < 0:
            # No contour data
            cdata = np.zeros((2, 2), dtype=int)
        else:
            # Assign contour based on stored frame index
            frame_ist = self.frame[idx]
            # Do not only check the exact frame, but +/- 2 events around it
            for idn in [idnew, idnew-1, idnew+1, idnew-2, idnew+2]:
                # check frame
                try:
                    frame_soll = self._contour_data.get_frame(idn)
                except IndexError:
                    # reached end of file
                    continue
                if np.allclose(frame_soll, frame_ist, rtol=0):
                    cdata = self._contour_data[idn]
                    break
        if cdata is None and self.shape and self.pxfeat:  #
            # The frame is wrong, but the contour might be correct.
            # We check that by verifying several features.
            cdata2 = self._contour_data[idnew]
            cont = np.zeros((self.shape[1], self.shape[0]))
            cont[cdata2[:, 0], cdata2[:, 1]] = True
            mm = inert_ratio.cont_moments_cv(cdata2)

            if (np.allclose(self.pxfeat["size_x"][idx],
                            np.ptp(cdata2[:, 0]) + 1,
                            rtol=0, atol=1e-5)
                and np.allclose(self.pxfeat["size_y"][idx],
                                np.ptp(cdata2[:, 1]) + 1,
                                rtol=0, atol=1e-5)
                and np.allclose(mm["m00"],
                                self.pxfeat["area_msd"][idx],
                                rtol=0, atol=1e-5)
                # atol=6 for positions, because the original positions
                # are computed from the convex contour, which would be
                # computed using cv2.convexHull(cdata2).
                and np.allclose(self.pxfeat["pos_x"][idx],
                                mm["m10"]/mm["m00"],
                                rtol=0, atol=6)
                and np.allclose(self.pxfeat["pos_y"][idx],
                                mm["m01"]/mm["m00"],
                                rtol=0, atol=6)):
                cdata = cdata2

        if cdata is None:
            # No idea what went wrong, but we make the beste guess and
            # issue a warning.
            cdata = self._contour_data[idnew]
            frame_c = self._contour_data.get_frame(idnew)
            warnings.warn(
                "Couldn't verify contour {} in {}".format(idx, self.identifier)
                + " (frame index {})!".format(frame_c),
                ContourVerificationWarning
            )
        return cdata

    def __len__(self):
        length = len(self._contour_data)
        if length:
            if not self._initialized:
                self.determine_offset()
            length += self.event_offset
        return length

    def determine_offset(self):
        """Determines the offset of the contours w.r.t. other data columns

        Notes
        -----
        - the "frame" column of `rtdc_dataset` is compared to
          the first contour in the contour text file to determine an
          offset by one event
        - modifies the property `event_offset` and sets `_initialized`
          to `True`
        """
        # In case of regular RTDC, the first contour is
        # missing. In case of fRTDC, it is there, so we
        # might have an offset. We find out if the first
        # contour frame is missing by comparing it to
        # the "frame" column of the rtdc dataset.
        fref = self._contour_data.get_frame(0)
        f0 = self.frame[0]
        f1 = self.frame[1]
        # Use allclose to avoid float/integer comparison problems
        if np.allclose(fref, f0, rtol=0):
            self.event_offset = 0
        elif np.allclose(fref, f1, rtol=0):
            self.event_offset = 1
        else:
            msg = "Contour data has unknown offset (frame {})!".format(fref)
            raise ContourIndexingError(msg)
        self._initialized = True

    @staticmethod
    def find_contour_file(rtdc_dataset):
        """Tries to find a contour file that belongs to an RTDC dataset

        Returns None if no contour file is found.
        """
        cont_id = rtdc_dataset.path.stem
        cands = [c.name for c in rtdc_dataset._fdir.glob("*_contours.txt")]
        cands = sorted(cands)
        # Search for perfect matches, e.g.
        # - M1_0.240000ul_s.tdms
        # - M1_0.240000ul_s_contours.txt
        for c1 in cands:
            if c1.startswith(cont_id):
                cfile = rtdc_dataset._fdir / c1
                break
        else:
            # Search for M* matches with most overlap, e.g.
            # - M1_0.240000ul_s.tdms
            # - M1_contours.txt
            for c2 in cands:
                if (c2.split("_")[0] == rtdc_dataset._mid):
                    # Do not confuse with M10_contours.txt
                    cfile = rtdc_dataset._fdir / c2
                    break
            else:
                cfile = None
        return cfile


class ContourData(object):
    def __init__(self, fname):
        """Access an MX_contour.txt as a dictionary

        Initialize this class with a *_contour.txt file.
        The individual contours can be accessed like a
        list (enumerated from 0 on).
        """
        self._initialized = False
        self.filename = fname

    def __getitem__(self, idx):
        cont = self.data[idx]
        cont = cont.strip()
        cont = cont.replace(")", "")
        cont = cont.replace("(", "")
        cont = cont.replace("(", "")
        cont = cont.replace("\n", ",")
        cont = cont.replace("   ", " ")
        cont = cont.replace("  ", " ")
        if len(cont) > 1:
            _frame, cont = cont.split(" ", 1)
            cont = cont.strip(" ,")
            data = np.fromstring(cont, sep=",", dtype=np.uint16).reshape(-1, 2)
            return data

    def __len__(self):
        return len(self.data)

    def _index_file(self):
        """Open and index the contour file

        This function populates the internal list of contours
        as strings which will be available as `self.data`.
        """
        with self.filename.open() as fd:
            data = fd.read()

        ident = "Contour in frame"
        self._data = data.split(ident)[1:]
        self._initialized = True

    @property
    def data(self):
        """Access self.data
        If `self._index_file` has not been computed before, this
        property will cause it to do so.
        """
        if not self._initialized:
            self._index_file()
        return self._data

    def get_frame(self, idx):
        """Return the frame number of a contour"""
        cont = self.data[idx]
        # previously was split using " ", but "(" is more general
        frame = int(cont.strip().split("(", 1)[0])
        return frame
