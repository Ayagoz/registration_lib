import numpy as np
import torch
import torch.nn.functional as F


def identity_mapping(shape, dtype=torch.cuda.DoubleTensor):
    return torch.from_numpy(np.mgrid[tuple(map(slice, shape))]).type(dtype)


def jacobian_matrix(grid, model_x, model_y, model_z=None):
    dimension = grid.ndimension() - 1

    if dimension == 2:
        return torch.stack([gradient(grid[0], model_x, model_y),
                            gradient(grid[1], model_x, model_y)], 0)
    elif dimension == 3:
        if model_z != None:
            return torch.stack(
                [gradient(grid[0], model_x, model_y, model_z),
                 gradient(grid[1], model_x, model_y, model_z),
                 gradient(grid[2], model_x, model_y, model_z)], 0)
        else:
            print('No model input for z coordinate')


def determinant(J):
    dimension = J.ndimension() - 2

    if dimension == 2:
        return J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    elif dimension == 3:

        return (J[0, 0] * J[1, 1] * J[2, 2]
                + J[1, 0] * J[2, 1] * J[0, 2]
                + J[0, 1] * J[1, 2] * J[2, 0]
                - J[0, 0] * J[1, 2] * J[2, 1]
                - J[2, 2] * J[1, 0] * J[0, 1]
                - J[0, 2] * J[1, 1] * J[2, 0])


def gradient(img, model_x, model_y, model_z=None):
    ndim = img.ndimension()
    img = img.unsqueeze(0).unsqueeze(0)
    #    assert ndim < 3, "Not invalid dimension, minimum supported (1,1, xlen)"

    if ndim == 1:
        return torch.cat(((img[:, :, 1] - img[:, :, 0]).unsqueeze(0),
                          0.5 * (img[:, :, 2:] - img[:, :, :-2]),
                          (img[:, :, -1] - img[:, :, -2]).unsqueeze(0)), dim=2)
    if ndim == 2:
        img_pad = F.pad(img, (1, 1, 1, 1), mode='replicate')
        img_pad.cuda()

        img_x = model_x(img_pad)[:, :, :, 1:-1]
        img_y = model_y(img_pad)[:, :, 1:-1, :]

        img_y[:, :, :, 0] *= 2
        img_y[:, :, :, -1] *= 2

        img_x[:, :, 0, :] *= 2
        img_x[:, :, -1, :] *= 2

        return torch.cat((img_x, img_y), dim=1).squeeze(0)
    if ndim == 3:
        img_pad = F.pad(img, (1, 1, 1, 1, 1, 1), mode='replicate')
        img_pad.cuda()

        img_x = model_x(img_pad)[:, :, :, 1:-1, 1:-1]
        img_y = model_y(img_pad)[:, :, 1:-1, :, 1:-1]
        img_z = model_z(img_pad)[:, :, 1:-1, 1:-1, :]

        img_x[:, :, 0, :, :] *= 2
        img_x[:, :, -1, :, :] *= 2

        img_y[:, :, :, 0, :] *= 2
        img_y[:, :, :, -1, :] *= 2

        img_z[:, :, :, :, 0] *= 2
        img_z[:, :, :, :, -1] *= 2

        return torch.cat((img_x, img_y, img_z), dim=1).squeeze(0)

def kernel(ndim):

    if ndim == 2:
        conv_y = torch.nn.Conv2d(1, 1, kernel_size=(1, 3), bias=False)
        conv_y.weight.data = torch.from_numpy(np.array([[[[-1 / 2., 0, 1 / 2.]]]], dtype='float64')).cuda()
        conv_y.weight.require_grads = False

        conv_x = torch.nn.Conv2d(1, 1, kernel_size=(3, 1), bias=False)
        conv_x.weight.data = torch.from_numpy(np.array([[[[-1 / 2.], [0.], [1 / 2.]]]], dtype='float64')).cuda()
        conv_x.weight.require_grads = False

        model_x = torch.nn.Sequential(conv_x)
        model_y = torch.nn.Sequential(conv_y)

        model_x.cuda()
        model_y.cuda()

        return [model_x, model_y, None]

    if ndim == 3:
        conv_x = torch.nn.Conv3d(1, 1, kernel_size=(3, 1, 1), bias=False)
        conv_x.weight.data = torch.from_numpy(np.array([[[[[-1 / 2.]], [[0.]], [[1 / 2.]]]]], dtype='float64')).cuda()
        conv_x.weight.require_grads = False

        conv_y = torch.nn.Conv3d(1, 1, kernel_size=(1, 3, 1), bias=False)
        conv_y.weight.data = torch.from_numpy(np.array([[[[[-1 / 2.], [0.], [1 / 2.]]]]], dtype='float64')).cuda()
        conv_y.weight.require_grads = False

        conv_z = torch.nn.Conv3d(1, 1, kernel_size=(1, 1, 3), bias=False)
        conv_z.weight.data = torch.from_numpy(np.array([[[[[-1 / 2., 0., 1 / 2.]]]]], dtype='float64')).cuda()
        conv_z.weight.require_grads = False

        model_x = torch.nn.Sequential(conv_x)
        model_y = torch.nn.Sequential(conv_y)
        model_z = torch.nn.Sequential(conv_z)

        model_x.cuda()
        model_y.cuda()
        model_z.cuda()

        return [model_x, model_y, model_z]