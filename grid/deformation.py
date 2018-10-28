import numpy as np
import torch
import nibabel as nib
from scipy.ndimage.interpolation import map_coordinates
from .utils import identity_mapping


class Deformation(object):

    def __init__(self,
                 filename=None,
                 grid=None,
                 displacement=None,
                 shape=None,
                 dtype=torch.cuda.DoubleTensor):
        self.dtype = dtype
        if filename is not None:
            t = nib.load(filename)
            self.ndim = t.shape[-1]
            self.shape = t.shape[:-1]
            data = torch.from_numpy(t.get_data()).type(self.dtype).clone()
            transpose_axis = (self.ndim,)
            for i in range(self.ndim):
                transpose_axis = transpose_axis + (i,)
            data = torch.transpose(data, transpose_axis)
            self.grid = data + identity_mapping(self.shape)
        elif grid is not None:
            self.shape = grid.shape[1:]
            self.ndim = grid.shape[0]
            self.grid = grid.clone().type(dtype)
        elif displacement is not None:
            self.shape = displacement.shape[1:]
            self.ndim = displacement.shape[0]
            self.grid = displacement + identity_mapping(self.shape)
        elif shape is not None:
            self.ndim = len(shape)
            self.shape = shape
            self.grid = identity_mapping(self.shape)

    def __add__(self, deformation):
        grid = warp_grid(self.grid, deformation.grid)
        return Deformation(grid=grid)

    def __iadd__(self, deformation):
        self.grid = warp_grid(self.grid, deformation.grid)
        return self

    def show(self, interval=1, show_axis=False):
        import matplotlib.pyplot as plt
        if self.ndim == 2:

            if show_axis is False:
                plt.axis('off')
            ax = plt.gca()
            ax.invert_yaxis()
            ax.set_aspect('equal')

            for x in range(0, self.shape[0], interval):
                plt.plot(self.grid[1, x, :], self.grid[0, x, :], 'k')
            for y in range(0, self.shape[1], interval):
                plt.plot(self.grid[1, :, y], self.grid[0, :, y], 'k')
            plt.show()

    def save(self, filename, affine=torch.eye(4)):
        displacement = self.grid - identity_mapping(self.shape)

        transpose_axis = ()
        for i in range(1, self.ndim + 1):
            transpose_axis = transpose_axis + (i,)
        transpose_axis += (0,)

        displacement = torch.transpose(displacement, transpose_axis)

        nib.save(nib.Nifti1Image(displacement, affine), filename)

        print("saved transformation: %s" % filename)

    def save_as_img(self, filename, interval, show_axis=False):
        import matplotlib.pyplot as plt
        if self.ndim == 2:
            if show_axis is False:
                plt.axis('off')
            ax = plt.gca()
            ax.invert_yaxis()
            ax.set_aspect('equal')

            for x in range(0, self.shape[0], interval):
                plt.plot(self.grid[1, x, :], self.grid[0, x, :], 'k')
            for y in range(0, self.shape[1], interval):
                plt.plot(self.grid[1, :, y], self.grid[0, :, y], 'k')
            plt.savefig(filename)
            plt.clf()


def warp_grid(grid, mapping_function, order=3, mode='nearest', dtype=torch.cuda.DoubleTensor):
    """
    warp grid with a mapping function
    phi_1 = grid
    phi_2 = mapping_function
    the result is
    phi_1(phi_2)

    Parameters
    ----------
    grid : ndarray
        a grid which is going to be warped
    mapping_function : ndarray
        the grid will be deformed by this mapping function

    Returns
    -------
    warped_grid : ndarray
        grid deformed by the mapping function
    """
    if len(grid) != len(mapping_function):
        raise ValueError('the dimension of the two inputs are the same')

    warped_grid = np.zeros_like(grid)
    grid = np.array(grid)
    for i, lines in enumerate(grid):
        warped_grid[i] = map_coordinates(
            lines, mapping_function, order=order, mode=mode)

    return torch.from_numpy(warped_grid).type(dtype)
