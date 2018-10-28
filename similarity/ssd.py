import torch
from registration_lib.grid.utils import gradient


class SSD(object):

    def __init__(self, variance):
        self.variance = variance

    def __str__(self):
        return ("Sum of Squared Difference, variance=%f" % self.variance)

    def cost(self, fixed, moving):
        return torch.sum(self.local_cost(fixed, moving))

    def local_cost(self, fixed, moving):
        return torch.pow((fixed - moving), 2)

    def derivative(self, fixed, moving, model_x, model_y, model_z):

        return 2 * gradient(moving.data, model_x, model_y, model_z) * (fixed.data - moving.data) / self.variance
