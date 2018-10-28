import numpy as np
import torch
import nibabel as nib
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import zoom
from .image import Image
from skimage.transform import warp


class ScalarImage(Image):

    def __init__(self, filename=None, data=None, affine=None, dtype=torch.cuda.DoubleTensor):
        if filename:
            img = nib.load(filename)
            data = torch.from_numpy(img.get_data()).type(dtype)
            data = data.squeeze()
            self.data = data.clone()
            self.ndim = self.data.ndimension() - 1
            self.shape = self.data.shape
            self.affine = img.get_affine()
        elif data is not None:
            self.dtype = dtype
            if type(data) != torch.Tensor:
                self.data = torch.from_numpy(data).type(self.dtype)
            else:
                self.data = data
            self.ndim = self.data.ndimension()
            self.shape = self.data.shape
            if affine is None:
                self.affine = np.identity(4)
            else:
                self.affine = affine

    def change_resolution(self, resolution, sigma, order=1):
        if resolution != 1:
            blurred_data = gaussian_filter(self.data.cpu().detach().numpy(), sigma)
            ratio = [1 / float(resolution)] * self.ndim
            data = torch.from_numpy(zoom(blurred_data, ratio, order=order)).type(self.dtype)
        elif resolution == 1:
            data = gaussian_filter(self.data.cpu().detach().numpy(), sigma)
        img = ScalarImage(data=data)
        return img

    def change_scale(self, maximum_value):

        data = maximum_value * self.data / torch.max(self.data)
        img = ScalarImage(data=data, affine=self.affine)
        return img

    def apply_transform(self, deformation, order=1):

        warped_data = warp(self.data.cpu().detach().numpy(),
                           deformation.grid.cpu().detach().numpy(), order=order)
        warped_img = ScalarImage(data=warped_data, affine=self.affine)
        return warped_img

    def show(self, x=None, y=None, z=None, show_axis=False, **kwargs):
        import matplotlib.pyplot as plt
        if self.ndim == 2:
            if show_axis is False:
                plt.axis('off')
            plt.imshow(self.data, cmap='gray', **kwargs)
            plt.show()
        if self.ndim == 3:
            if show_axis is False:
                plt.axis("off")
            if x is not None:
                plt.imshow(self.data[x, :, :], cmap="gray", **kwargs)
            elif y is not None:
                plt.imshow(self.data[:, y, :], cmap='gray', **kwargs)
            elif z is not None:
                plt.imshow(self.data[:, :, z], cmap="gray", **kwargs)
            else:
                raise ValueError("x, y, or z has to be not None")
            plt.show()
