# this script overrides the methods defined in PlugInFeature

import numpy as np

from .plugin_feature import PlugInFeature


class MedianImagePixel(PlugInFeature):
    def __init__(self, *args, **kwargs):
        super(MedianImagePixel, self).__init__(*args, **kwargs)


    def get_feature_name(self):
        ''' Compute the feature you want

        Returns
        -------
        str
        '''

        # Just change the name in quotes
        feature_name = "Awesome Median"

        return feature_name

    def compute_feature(mm):
        ''' Compute the feature you want

        Returns
        -------
        ndarray of length len(mm)
        '''

        # calculate your feature here
        image_median = np.median(mm["image"])

        # check that the length is good
        # is this desired, or is warning in
        # AncillaryFeature.is_available sufficient
        # assert len(image_median) == len(mm)
        return image_median
