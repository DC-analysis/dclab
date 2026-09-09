from __future__ import annotations

import numbers
import warnings
import weakref

import numpy as np

from ...definitions.feat_const import FEATURES_IMAGE_ROI
from ...util import copy_if_needed

from .basin_base import Basin
from .basin_common import BasinFeatureMissingWarning


class InternalH5DatasetChoppedImageBasin(Basin):
    basin_format = "h5datasetchop"
    basin_type = "internal"

    def __init__(self, *args, **kwargs):
        """Chopped image basin

        The chopped image basin is an internal basin that stores image
        data. Allowed features are listed in `FEATURES_IMAGE_ROI`. Only
        relevant image data are stored. This addresses the problem that
        in the .rtdc file format the full frame is stored for each event.
        If there is only one event in a frame, then ~80% of that frame are
        unused data that lead to larger file sizes. The chopped image
        approach stores smaller event images which are reassembled into
        the full frame images when read.

        The basin information must be registered with the correct basin format
        "h5datasetchop" and the "same" basin mapping. This means that for each
        event in the referring dataset, one image is stored in the basin.

        The basin data are organized in a group with an `index` dataset and
        multiple chunked datasets of different shapes for the image data.
        The `index` dataset contains the information how to read and position
        each event image in the event's frame. Multiple events per frame are
        supported. The other datasets, enumerated with "0", "1", etc. contain
        the event images sorted to optimize shape and chunking for storage.
        """
        super().__init__(*args, **kwargs)
        if self.mapping != "same":
            raise ValueError(
                "'internal_chopped_image' basins must be instantiated with "
                "the same mapping as the referring dataset.")
        if self._features is None:
            raise ValueError("You must specify features when defining "
                             "internal chopped image basins.")

        ref = self._get_ref()
        h5root = ref.h5file
        # available features
        available_features = []
        features_image_roi = [f[0] for f in FEATURES_IMAGE_ROI]
        for feat in self._features:
            if feat not in features_image_roi:
                raise NotImplementedError(
                    f"Feature '{feat}' is not defined in `FEATURES_IMAGE_ROI` "
                    f"and thus should not be used in {type(self)}; referring "
                    f"dataset is '{ref}'"
                )
            if self.location in h5root and feat in h5root[self.location]:
                available_features.append(feat)
            else:
                warnings.warn(
                    f"Feature '{feat}' is defined as an internal basin, "
                    f"but it cannot be found in '{self.location}'.",
                    BasinFeatureMissingWarning)
        self._features.clear()
        self._features += available_features

    def _get_ref(self):
        # to avoid circular imports...
        from ..fmt_hdf5 import RTDC_HDF5
        assert self._basinmap_referrer is not None
        ref = self._basinmap_referrer()
        assert isinstance(ref, RTDC_HDF5)
        return ref

    def _load_dataset(self, location, **kwargs):
        # to avoid circular imports...
        from ..fmt_dict import RTDC_Dict
        # get the h5file object
        ref = self._get_ref()
        h5root = ref.h5file
        assert self.location in h5root
        assert self._basinmap_referrer is not None
        # fetch metadata
        cfg = ref.config
        roi_y = cfg["imaging"]["roi size y"]
        roi_x = cfg["imaging"]["roi size x"]
        # fetch data
        ds_dict = {}
        for feat in self.features:
            feat_obj = InternalImageChoppedFeatureProxy(
                feat_obj=h5root[self.location][feat],
                roi_shape=(roi_y, roi_x),
                basinmap_referrer=self._basinmap_referrer
            )
            ds_dict[feat] = feat_obj
        return RTDC_Dict(ds_dict)

    def is_available(self):
        return bool(self._features)

    def verify_basin(self, *args, **kwargs):
        """It's not necessary to verify internal basins"""
        return True


class InternalImageChoppedFeatureProxy(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self,
                 feat_obj,
                 roi_shape: tuple[int, int],
                 basinmap_referrer: weakref.ref,
                 ) -> None:
        self.feat_obj = feat_obj
        # for `util.hashobj`
        self.identifier = feat_obj["index"]
        self.inverse_bg = feat_obj.attrs.get("inverse_background_feature")
        if feat_obj.attrs.get("is_boolean", False):
            self.dtype = np.dtype(np.bool_)
        else:
            self.dtype = self.feat_obj["0"].dtype
        self.roi_shape = roi_shape
        self._basinmap_referrer = basinmap_referrer
        self._chopper_index = None
        self._length = None
        self._shape = None

    def __len__(self) -> int:
        return self.shape[0]

    def __array__(self, dtype=None, copy=copy_if_needed, *args, **kwargs):
        # This is dangerous territory in terms of memory usage
        out_arr = np.empty(self.shape,
                           *args,
                           dtype=dtype or self.dtype,
                           **kwargs)

        for idx in range(len(self)):
            out_arr[idx] = self._get_image_frame(idx)
        return out_arr

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Convert all instances of `BasinProxyFeature` to arrays.
        inputs = [ip.__array__()
                  if isinstance(ip, InternalImageChoppedFeatureProxy) else ip
                  for ip in inputs]
        return getattr(ufunc, method)(*inputs, **kwargs)

    def __getitem__(self, index):
        if isinstance(index, numbers.Integral):
            return self._get_image_frame(index)
        elif isinstance(index, (slice, range, np.ndarray, list)):
            if isinstance(index, slice):
                indices = np.arange(self.chopper_index.shape[0])
                indices = indices[index]
            else:
                # range, or indexing array
                indices = index
            data = np.zeros((len(indices),) + self.roi_shape, dtype=self.dtype)
            for ii, idx in enumerate(indices):
                data[ii] = self._get_image_frame(idx)
            return data
        else:
            return self.__array__()[index]

    def _get_image_event(self, index) -> tuple[np.ndarray, int]:
        """Return the image from one event

        Returns
        -------
        image:
            Event image data placed in the frame ROI
        other:
            Event index of the next image that belongs into the same frame
        """
        dataset, sub_idx, offy, offx, other = self.chopper_index[index]
        image = np.asarray(self.feat_obj[str(dataset)][sub_idx],
                           dtype=self.dtype)

        # crop event image to maximum allowed size
        max_y = self.roi_shape[0] - offy
        max_x = self.roi_shape[1] - offx
        image_cropped = image[:max_y, :max_x]

        # place event image in full frame ROI
        imsh = image_cropped.shape
        full_image = np.zeros(self.roi_shape, dtype=image.dtype)
        full_image[offy:offy+imsh[0], offx:offx+imsh[1]] = image_cropped

        return full_image, other

    def _get_image_frame(self, index):
        """Reconstruct the frame for event `index`

        If the frame contains multiple events, all available image data
        from the other events is put into the image.
        """
        image, other = self._get_image_event(index)

        needs_frame_update = other != index
        needs_bg_update = self.inverse_bg is not None

        if needs_frame_update or needs_bg_update:
            ref = self._basinmap_referrer()
            assert ref is not None
            # If the 'other' index is identical to 'index', then there
            # is only one event in this frame.
            # If the 'other' index is different from 'index', then
            # add the image from the 'other' event to the current image.
            # The 'other' column is always defined in a circular manner.
            # This means that independent of which event you select, you
            # will always get all images from the other events.
            while other != index:
                # There are more events in this image
                imo, other = self._get_image_event(other)
                support = imo != 0
                image[support] = imo[support]

            if needs_bg_update:
                # Perform inverse background correction
                image += ref[self.inverse_bg][index]
        return image

    @property
    def chopper_index(self):
        """Access the chopped "index" information"""
        if self._chopper_index is None:
            self._chopper_index = self.feat_obj["index"][:]
        return self._chopper_index

    @property
    def shape(self):
        if self._shape is None:
            self._shape = (len(self.feat_obj["index"]),) + self.roi_shape
        return self._shape
