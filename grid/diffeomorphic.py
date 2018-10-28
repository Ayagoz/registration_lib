import numpy as np
import torch
from .utils import identity_mapping, jacobian_matrix, determinant, kernel

class DiffeomorphicDeformation(object):

    def __init__(self, n_step, shape=None, time_interval=1., dtype=torch.cuda.DoubleTensor):
        self.n_step = n_step
        self.time_interval = time_interval
        self.delta_time = time_interval / n_step
        self.dtype = dtype
        if shape is not None:
            self.set_shape()

    def set_shape(self, shape):
        self.shape = shape
        self.ndim = len(shape)
        self.init_mappings()
        self.kernel_grads = kernel(self.ndim)

    def init_mappings(self):
        self.initial_grid = identity_mapping(self.shape)
        self.forward_mappings = torch.from_numpy(np.ones(
            (self.n_step + 1, self.ndim) + self.shape)).type(self.dtype)* self.initial_grid
        self.backward_mappings = self.forward_mappings.clone()

        # jacobian determinants of forward mappings
        self.forward_dets = torch.from_numpy(np.ones(
            (self.n_step + 1,) + self.shape)).type(self.dtype)
        # jacobian determinants of backward mappings
        self.backward_dets = torch.from_numpy(np.ones(
            (self.n_step + 1,) + self.shape)).type(self.dtype)

    def euler_integration(self, grid, J, vector_fields):
        return grid - torch.from_numpy(np.einsum('ij...,j...->i...',
                                J.cpu().detach().numpy(),
                                vector_fields.cpu().detach().numpy()) * self.delta_time).type(self.dtype)

    def update_mappings(self, vector_fields):
        assert len(vector_fields) == self.n_step

        forward_jacobian_matrix = jacobian_matrix(self.initial_grid, *self.kernel_grads)
        backward_jacobian_matrix = forward_jacobian_matrix.clone()


        for i in range(self.n_step):

            self.forward_mappings[i + 1] = self.euler_integration(
                self.forward_mappings[i],
                forward_jacobian_matrix,
                vector_fields[i])

            self.backward_mappings[i + 1] = self.euler_integration(
                self.backward_mappings[i],
                backward_jacobian_matrix,
                -vector_fields[-i - 1])

            forward_jacobian_matrix = jacobian_matrix(
                self.forward_mappings[i + 1], *self.kernel_grads)

            backward_jacobian_matrix = jacobian_matrix(
                self.backward_mappings[i + 1], *self.kernel_grads)

            self.forward_dets[i + 1] = determinant(
                forward_jacobian_matrix)
            # print('forward det(jac matrix)')
            # print(self.forward_dets[i+1])
            self.backward_dets[i + 1] = determinant(
                backward_jacobian_matrix)

