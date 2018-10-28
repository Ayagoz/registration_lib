import torch
from registration_lib.utils import bilinear_grid_sample


class SequentialScalarImages(object):

    def __init__(self, img, deformation_step):
        """
        container for sequentially deforming images

        Parameters
        ----------
        img : Image
            original fixed of moving image
        """
        self.deformation_step = deformation_step
        self.ndim = img.ndim
        self.shape = img.shape

        self.original = img.data.clone()

        self.data = [img.data for _ in range(deformation_step + 1)]

    def __getitem__(self, index):
        return self.data[index]

    def apply_transforms(self, mappings):
        self.data = [bilinear_grid_sample(self.original, self.ndim, mapping) for mapping in mappings]
