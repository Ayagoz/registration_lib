import torch
import numpy as np
import torch.nn.functional as F
from registration_lib.grid.utils import identity_mapping



def gaussian_filter(img, ndim, kernel_size=3, sigma=1., dtype=torch.cuda.DoubleTensor):
    if ndim != 2 and ndim != 3:
        raise Exception("Input error, invalid dimension of image")

    if type(kernel_size) == float or type(kernel_size) == int:
        kernel = (kernel_size,) * ndim
    else:
        kernel = kernel_size

    grid = torch.from_numpy(identity_mapping(kernel).cpu().numpy().T).type(dtype)

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.
    gaussian_kernel = (1. / (2. * np.pi * variance)) * torch.exp(
        -torch.sum((grid - mean) ** 2., dim=-1) / (2 * variance))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    if ndim == 2:
        g_filter = torch.nn.Conv2d(in_channels=1, out_channels=1,
                                   kernel_size=kernel, bias=False,
                                   padding=1)
        g_filter.weight.data = gaussian_kernel.unsqueeze(0).unsqueeze(0).type(dtype)
        g_filter.weight.requires_grad = False

        return g_filter(img.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

    if ndim == 3:
        g_filter = torch.nn.Conv3d(in_channels=1, out_channels=1,
                                   kernel_size=kernel, bias=False,
                                   padding=1)
        g_filter.weight.data = gaussian_kernel.unsqueeze(0).unsqueeze(0).type(dtype)
        g_filter.weight.requires_grad = False

        return g_filter(img.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

# def gradient(img, model_x, model_y, model_z=None):
#     ndim = img.ndimension()
#     img = img.unsqueeze(0).unsqueeze(0)
#     #    assert ndim < 3, "Not invalid dimension, minimum supported (1,1, xlen)"
#
#     if ndim == 1:
#         return torch.cat(((img[:, :, 1] - img[:, :, 0]).unsqueeze(0),
#                           0.5 * (img[:, :, 2:] - img[:, :, :-2]),
#                           (img[:, :, -1] - img[:, :, -2]).unsqueeze(0)), dim=2)
#     if ndim == 2:
#         img_pad = F.pad(img, (1, 1, 1, 1), mode='replicate')
#         img_pad.cuda()
#
#         img_x = model_x(img_pad)[:, :, :, 1:-1]
#         img_y = model_y(img_pad)[:, :, 1:-1, :]
#
#         img_y[:, :, :, 0] *= 2
#         img_y[:, :, :, -1] *= 2
#
#         img_x[:, :, 0, :] *= 2
#         img_x[:, :, -1, :] *= 2
#
#         return torch.cat((img_x, img_y), dim=1).squeeze(0)
#     if ndim == 3:
#         img_pad = F.pad(img, (1, 1, 1, 1, 1, 1), mode='replicate')
#         img_pad.cuda()
#
#         img_x = model_x(img_pad)[:, :, :, 1:-1, 1:-1]
#         img_y = model_y(img_pad)[:, :, 1:-1, :, 1:-1]
#         img_z = model_z(img_pad)[:, :, 1:-1, 1:-1, :]
#
#         img_x[:, :, 0, :, :] *= 2
#         img_x[:, :, -1, :, :] *= 2
#
#         img_y[:, :, :, 0, :] *= 2
#         img_y[:, :, :, -1, :] *= 2
#
#         img_z[:, :, :, :, 0] *= 2
#         img_z[:, :, :, :, -1] *= 2
#
#         return torch.cat((img_x, img_y, img_z), dim=1).squeeze(0)


def interpolate_mapping(img, size):
    if img.ndimension() == 3:
        return F.interpolate(img.unsqueeze(0).unsqueeze(0), size=size, mode='trilinear', align_corners=True)
    if img.ndimension() == 2:
        return F.interpolate(img.unsqueeze(0).unsqueeze(0), size=size, mode='bilinear', align_corners=True)


def bilinear_grid_sample(image, ndim, grid, dtype=torch.cuda.DoubleTensor):
    samples = grid.type(dtype)

    if ndim == 2:
        image = image.type(dtype).permute(0, 1)
        W, H = image.shape
        samples = samples.permute(2, 1, 0).unsqueeze(0)

        samples[:, :, :, 0] /= (W - 1)
        samples[:, :, :, 1] /= (H - 1)

    elif ndim == 3:
        image = image.type(dtype).permute(2, 0, 1)
        D, H, W = image.shape
        samples = samples.permute(1, 2, 3, 0).unsqueeze(0)
        samples[:, :, :, :, 0] /= (W - 1)  # normalize to between  0 and 1
        samples[:, :, :, :, 1] /= (H - 1)  # normalize to between  0 and 1
        samples[:, :, :, :, 2] /= (D - 1)

    image = image.unsqueeze(0).unsqueeze(0)

    samples = 2 * samples - 1

    return torch.nn.functional.grid_sample(image, samples).squeeze(0).squeeze(0)
