
# Create PlugInFeature class which is child of AncillaryFeature
# it should have methods that can be overridden by a script
# The script should create a child class of PlugInFeature
# The methods defined in the script class will override the
# PlugInFeature class.

import abc

from ..ancillaries import AncillaryFeature


class PlugInFeature(AncillaryFeature, abc.ABC):
    def __init__(self, *args, **kwargs):
        super(PlugInFeature, self).__init__(*args, **kwargs)

        self.feature_name = self.get_feature_name
        self.method = self.compute_feature

    @abc.abstractmethod
    def get_feature_name(self):
        pass

    @abc.abstractmethod
    def compute_feature(self):
        pass

