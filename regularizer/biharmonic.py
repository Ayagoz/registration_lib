import torch
import numpy as np

from scipy.fftpack import fftn, ifftn
from registration_lib.grid.utils import identity_mapping


class BiharmonicRegularizer(object):

    def __init__(self, convexity_penalty=1., norm_penalty=1.):
        self.convexity_penalty = convexity_penalty
        self.norm_penalty = norm_penalty

    def set_operator(self, shape, resolution=1):
        dx_sqinv = 1. / resolution ** 2

        self.A = self.norm_penalty * np.ones(shape)

        grid = identity_mapping(shape)

        for frequencies, length in zip(grid, shape):
            self.A += 2 * self.convexity_penalty * (
                    1 - np.cos(2 * np.pi * frequencies / length)) * dx_sqinv

        # Since this is biharmonic, the exponent is 2.
        self.operator = 1 / (self.A ** 2)

    def __call__(self, momentum):
        if (hasattr(self, 'operator')
                and momentum.shape[1:] == self.operator.shape):
            momentum = momentum.cpu().detach().numpy()
            G = np.zeros(momentum.shape, dtype=np.complex128)
            for i in range(len(momentum)):
                G[i] = fftn(momentum[i])

            F = G * self.operator

            vector_field = np.zeros_like(momentum)
            for i in range(len(momentum)):
                vector_field[i] = np.real(ifftn(F[i]))

            return torch.from_numpy(vector_field).type(torch.cuda.DoubleTensor)
        else:
            self.set_operator(momentum.shape[1:])
            return self.__call__(momentum)


    def norm(self, momentum):
        Lmomentum = self.__call__(momentum.clone())
        return torch.sum(Lmomentum * momentum)
