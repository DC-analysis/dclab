import numbers
import warnings

from dclab.polygon_filter import points_in_poly
import h5py
import matplotlib.pylab as plt
import numpy as np
import scipy.interpolate as spint
from scipy.spatial import ConvexHull
import skimage


from .hooks import lut_hooks


NORM = {
    "area_um": 290,
    "deform": 0.2,
    "volume": 3400,
}


class LutProcessor:
    def __init__(self, hdf5_path, featx="area_um", featy="deform",
                 use_hooks=True, verbose=False):
        """Process simulation data in the HDF5 file format

        The simulation dataset must be an HDF5 file with a specific structure
        that contains the results of FEM simulations. The original HDF5 files
        were provided by Lucas Wittwer. Any LUT added to dclab after version
        0.23.0 was extracted and created using this script.

        The following data post-processing is performed for the LUT:

        - Redundant values in the LUT are removed by checking whether they
          can be reproduced through linear interpolation of the remaining
          values.
        - The LUT is complemented with linearly interpolated isoelasticity
          lines to the convex hull of the simulation data in the area_um
          vs. deform plot.

        Additional post-processing hooks for LUT or isoelastics generation
        are defined in the Python files named according to the LUT identifier
        in the "fem_hooks" subdirectory.

        An example HDF5 file can be found on figshare
        (LE-2D-FEM-19, https://doi.org/10.6084/m9.figshare.12155064.v4).

        Parameters
        ----------
        hdf5_path: str or pathlib.Path
            path to the simulation file
        featx: str
            feature along the x-direction of the LUT
        featy: str
            feature along the y-direction of the LUT
        use_hooks: bool
            whether to perform custom LUT postprocessing using LUT hooks
        verbose: bool
            set to True to display informative plots
        """
        self.verbose = verbose
        self.path = hdf5_path
        self.data, self.meta = self.get_lut_raw()
        self.meta["column features"] = [featx, featy, "emodulus"]
        if use_hooks:
            self.hook = lut_hooks.get(self.meta["identifier"])
            if not self.hook:
                warnings.warn(f"No hook defined for {self.meta['identifier']}")
        else:
            self.hook = None

        # compute the raw LUT as stored in the simulation data
        lutsize = len(self.data["emodulus"])
        lut = np.zeros((lutsize, 3), dtype=float)
        lut[:, 0] = self.data[featx]
        lut[:, 1] = self.data[featy]
        lut[:, 2] = self.data["emodulus"]

        self.lut_raw = lut

        self.featx = featx
        self.featy = featy

        self.xmin = lut[:, 0].min()
        self.xmax = lut[:, 0].max()
        self.xptp = self.xmax - self.xmin

        self.ymin = lut[:, 1].min()
        self.ymax = lut[:, 1].max()
        self.yptp = self.ymax - self.ymin

        # Interpolation grid sizes
        xsize = int(250 * (self.xmax - self.xmin) / NORM[self.featx])
        ysize = int(250 * (self.ymax - self.ymin) / NORM[self.featy])
        self.size = int(np.max([xsize, ysize]))

        # Interpolation grid coordinates
        x = np.linspace(0, 1, self.size, endpoint=True)
        y = np.linspace(0, 1, self.size, endpoint=True)
        self.xitp, self.yitp = np.meshgrid(x, y, indexing="ij")

        # compute the convex hull
        hull = ConvexHull(lut[:, :2])

        self.convex_hull = hull.points[hull.vertices, :]

        if self.hook:
            self.hook.lut_preprocess(self)

    def points_in_convex_hull(self, points):
        convex_hull = np.array(self.convex_hull, copy=True)
        mx, my = np.mean(convex_hull, axis=0)
        for ii in range(len(convex_hull)):
            xi, yi = convex_hull[ii]

            dx = self.xptp / 1000
            if xi < mx:
                dx *= -1
            convex_hull[ii][0] += dx

            dy = self.yptp / 1000
            if yi < my:
                dy *= -1
            convex_hull[ii, 1] += dy
        return points_in_poly(points, convex_hull)

    def normalize_lut(self, lut):
        """Normalize an input LUT to the unit cube"""
        wlut = np.array(lut, copy=True)
        wlut[:, 0] = (wlut[:, 0] - self.xmin) / self.xptp
        wlut[:, 1] = (wlut[:, 1] - self.ymin) / self.yptp
        return wlut

    def denormalize_contour(self, contour):
        """Convert a contour obtained from the pixelated grid to real units"""
        ccu = np.zeros_like(contour)
        ccu[:, 0] = contour[:, 0] / self.size * self.xptp + self.xmin
        ccu[:, 1] = contour[:, 1] / self.size * self.yptp + self.ymin
        return ccu

    def map_lut_to_grid(self, lut=None):
        """Convert LUT data to a 2D image"""
        if lut is None:
            lut = self.lut_raw
        lut = np.array(lut, copy=True)
        lut = lut[lut[:, 0] <= self.xmax, :]
        lut = lut[lut[:, 1] <= self.ymax, :]

        # normalize
        wlut = self.normalize_lut(lut)

        emod = spint.griddata((wlut[:, 0], wlut[:, 1]), wlut[:, 2],
                              (self.xitp, self.yitp), method="linear")

        mask_sim = np.zeros_like(emod, dtype=bool)
        x = self.xitp[:, 0]
        y = self.yitp[0, :]
        for xi, yi, _ in wlut:
            dx = np.abs(x - xi)
            dy = np.abs(y - yi)
            xidx = np.argmin(dx)
            yidx = np.argmin(dy)
            if dx[xidx] + dy[yidx] < 1:
                mask_sim[xidx, yidx] = True

        return emod, mask_sim

    def extract_isoelastics_from_grid(self, emod, lut=None, num=13):
        """Extract isoelasitcs from gridded data

        Parameters
        ----------
        emod: 2d np.ndarray
            array containing Young's modulus gridded according to
            `self.xitp` and `self.yitp`.
        lut: 2d np.ndarray
            LUT (like `self.lut_raw`).
        num: int
            number of isoelastics to compute. They will be evenly
            distributed across the LUT using the coordinates
            `self.xitp` and `self.yitp`. That means that if you pass
            e.g. a smaller LUT, you will not get as many isoealstics
            back, since they will be nan-valued.
        """
        if lut is None:
            lut = self.lut_raw
        wlut = self.normalize_lut(lut)
        # Determine the levels via a line plot through the
        # given LUT.

        # These are the original levels (by Christoph Herold):
        # levels = [0.9 1.2 1.5 1.8 2.1 2.55 3. 3.6 4.2 5.4 6.9]
        # These are the new levels (by the following algorithm for
        # the LE-2D-FEM-19 dataset):
        # levels = [0.91 1.14 1.37 1.63 1.94 2.33 2.83 3.55 4.66 6.64 11.44]
        l0 = np.nanpercentile(emod, 1)
        l1 = np.nanpercentile(emod, 99)
        dl = (l1 - l0) / 200

        blob0 = np.where(np.abs(emod - l0) < .5 * dl)
        p0 = 0, blob0[1][-1]

        blob1 = np.where(np.abs(emod - l1) < .5 * dl)
        p1 = blob1[0][-1], 0

        xlev = np.linspace(self.xitp[p0[0], 0],
                           self.xitp[p1[0], 0], num, endpoint=True)
        ylev = np.linspace(self.yitp[0, p0[1]],
                           self.yitp[0, p1[1]], num, endpoint=True)

        elev = spint.griddata((wlut[:, 0], wlut[:, 1]), wlut[:, 2],
                              (xlev, ylev), method="linear")
        levels = elev[1:-1]

        levels = np.round(levels, 2)

        contours_px = []
        contours = []
        for level in levels:
            conts = skimage.measure.find_contours(emod, level=level)
            if not conts:
                continue
            # get the longest one
            idx = np.argmax([len(cc) for cc in conts])
            cc = conts[idx]
            # remove nan values
            cc = cc[~np.isnan(np.sum(cc, axis=1))]
            # downsample contour on normalized pixel grid
            cc_new = [cc[0]]
            last_used = 0
            for ii in range(1, len(cc)-1):
                if np.sqrt(np.sum((cc[ii] - cc[last_used])**2)) > 1:
                    cc_new.append(cc[ii])
                    last_used = ii
            cc_new.append(cc[-1])
            cc = np.array(cc_new)

            contours_px.append(cc)
            # convert pixel to absolute area_um and deform
            ccu = self.denormalize_contour(cc)
            contours.append(ccu)
        return levels, contours, contours_px

    def get_lut_raw(self):
        """Extract features from a FEM simulation HDF5 file (Lucas Wittwer)

        Returns
        -------
        data: dict
            Each key is a dclab feature, the value is a 1D ndarray
        meta: dict
            Metadata extracted from the HDF5 file
        """
        area_um = []
        deform = []
        emod = []
        volume = []
        with h5py.File(self.path, "r") as h5:
            assert h5.attrs["flow_rate_unit"].decode("utf-8") == "uL/s"
            assert h5.attrs["channel_width_unit"].decode("utf-8") == "um"
            assert h5.attrs["fluid_viscosity_unit"].decode("utf-8") == "Pa s"
            meta = dict(h5.attrs)
            # convert viscosity to mPa*s
            meta["fluid_viscosity_unit"] = "mPa s"
            meta["fluid_viscosity"] = meta["fluid_viscosity"] * 1000
            for key in meta:
                if isinstance(meta[key], bytes):
                    meta[key] = meta[key].decode("utf-8")
                elif isinstance(meta[key], str):
                    pass
                elif isinstance(meta[key], numbers.Integral):
                    meta[key] = int(meta[key])
                else:
                    meta[key] = float(meta[key])
            for Ek in h5.keys():
                assert h5[Ek].attrs["emodulus_unit"].decode("utf-8") == "Pa"
                for simk in h5[Ek].keys():
                    sim = h5[Ek][simk]
                    area_um.append(sim.attrs["area"])
                    assert sim.attrs["area_unit"].decode("utf-8") == "um^2"
                    deform.append(sim.attrs["deformation"])
                    assert sim.attrs["deformation_unit"].decode("utf-8") == ""
                    volume.append(sim.attrs["volume"])
                    assert sim.attrs["volume_unit"].decode("utf-8") == "um^3"
                    emod.append(h5[Ek].attrs["emodulus"] / 1000)
        data = {"area_um": np.array(area_um),
                "deform": np.array(deform),
                "emodulus": np.array(emod),
                "volume": np.array(volume),
                }
        return data, meta

    def compute_isoelastics(self, lut=None, num=13):
        """Compute isoelastics for a give LUT

        The gridded support of the LUT is reduced with a binary_closing
        of size 10.
        """
        if lut is None:
            lut = self.lut_raw

        lut = np.array(lut, copy=True)
        # Map the simulation data to a 2d grid
        emod, _ = self.map_lut_to_grid(lut)

        levels, contours, contours_px = self.extract_isoelastics_from_grid(
            emod=emod, lut=lut, num=num)

        return levels, contours, contours_px

    def assemble_lut_and_isoelastics(self):
        """Apply pipeline, return the LUT, isoelastics, and isoelastics level

        If `self.verbose` is True, then display informative plots.
        """
        emod, mask_sim = self.map_lut_to_grid()
        levels, contours_ip, contours_px = self.compute_isoelastics(num=13)

        if self.hook:
            contours_ip = self.hook.isoelastics_postprocess(self, contours_ip)

        contours = []
        for cc in contours_ip:
            inside = self.points_in_convex_hull(cc[:, :2])
            contours.append(cc[inside])

        if self.verbose:
            plt.figure(figsize=(13, 5))
            ax0 = plt.subplot(131,
                              title="simulation density and mask generation")
            ax0.imshow(mask_sim)

            ax1 = plt.subplot(132, title="isoelastics interpolation on grid")
            ax1.imshow(emod)
            for cont in contours_px:
                ax1.plot(cont[:, 1], cont[:, 0], "x")

            ax2 = plt.subplot(133, title="extracted isoelastics")
            for cont in contours:
                ax2.plot(cont[:, 0], cont[:, 1], "-")
            ax2.set_ylim(0, self.lut_raw[:, 1].max()*1.05)
            ax2.set_xlim(0, self.lut_raw[:, 0].max()*1.05)

            plt.tight_layout()
            plt.show()

        assert len(contours) == len(levels)

        lut = self.shrink_lut()

        return lut, contours, levels

    def shrink_lut(self, lut=None):
        """Remove redundant values in a LUT and enforce the convex hull

        This is achieved by checking whether they can be reproduced
        through linear interpolation from the remaining values.
        """
        print("...Shrinking LUT")
        if lut is None:
            lut = self.lut_raw

        inside = self.points_in_convex_hull(lut[:, :2])
        lut = lut[inside]

        wlut = self.normalize_lut(lut)

        keep = np.ones(wlut.shape[0], dtype=bool)

        checked = np.zeros(wlut.shape[0], dtype=int)

        # Take out a few points and see whether linear interpolation would
        # be enough to recover it.
        for ii in range(wlut.shape[0]):
            if checked[ii] >= 4:
                continue
            ids = np.arange(ii, wlut.shape[0], step=47)
            cur = np.array(keep, copy=True)
            cur[ids] = False
            emod = spint.griddata((wlut[cur, 0], wlut[cur, 1]), wlut[cur, 2],
                                  (wlut[ids, 0], wlut[ids, 1]),
                                  method='linear')

            for em, im in zip(emod, ids):
                checked[im] += 1
                if np.isnan(em):
                    continue
                elif np.allclose(em, wlut[im, 2], atol=1e-5, rtol=1e-4):
                    keep[im] = False
                    checked[im] = 4
        print("... -> Removed {} out of {} data points.".format(
            np.sum(~keep), keep.size))

        lut_new = 1*lut[keep, :]

        return lut_new
