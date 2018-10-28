import registration_lib
from registration_lib.grid import DiffeomorphicDeformation
from registration_lib.image import ScalarImage
import torch
from registration_lib.grid.utils import kernel


class Registration(object):

    def __init__(self,
                 n_step,
                 regularizer,
                 similarity,
                 n_iters=(50, 20, 10),
                 resolutions=(4, 2, 1),
                 smoothing_sigmas=(2, 1, 0),
                 delta_phi_threshold=1.,
                 unit_threshold=0.1,
                 learning_rate=0.1,
                 init_vf=[None],
                 dtype=torch.cuda.DoubleTensor):
        self.n_step = n_step
        self.deformation = DiffeomorphicDeformation(
            n_step=n_step)
        self.regularizer = regularizer
        self.dtype = dtype

        self.similarity = similarity
        self.init_vf = init_vf*len(n_iters)

        try:
            self.n_iters = tuple(n_iters)
        except:
            self.n_iters = (n_iters,)

        try:
            self.resolutions = tuple(resolutions)
        except:
            self.resolutions = (resolutions,)
        while len(self.resolutions) < len(self.n_iters):
            self.resolutions += (self.resolutions[-1],)

        try:
            self.smoothing_sigmas = tuple(smoothing_sigmas)
        except:
            self.smoothing_sigmas = (smoothing_sigmas,)
        while len(self.smoothing_sigmas) < len(self.n_iters):
            self.smoothing_sigmas += (self.smoothing_sigmas[-1],)

        self.delta_phi_threshold = delta_phi_threshold
        self.unit_threshold = unit_threshold
        self.learning_rate = learning_rate


    def print_settings(self):
        print(self.__class__.__name__)
        print(self.similarity)
        print("regularization", self.regularizer.__class__.__name__)
        print("iterations", self.n_iters)
        print("resolutions", self.resolutions)
        print("smoothing sigmas", self.smoothing_sigmas)
        print("threshold of displacement update", self.delta_phi_threshold)
        print("threshold of grid unit", self.unit_threshold)
        print("learning rate", self.learning_rate)


    def set_images(self, fixed, moving):
        assert fixed.ndim == moving.ndim
        assert fixed.shape == moving.shape

        self.ndim = fixed.ndim
        self.shape = fixed.shape
        fixed = ScalarImage(data=fixed)
        moving = ScalarImage(data=moving)

        self.fixed = fixed.change_scale(255)
        self.moving = moving.change_scale(255)



    def zoom_grid(self, grid, resolution):
        shape = grid.shape[1:]
        if resolution != 1:
            interpolated_grid = torch.zeros((self.ndim,) + self.shape)
            for i in range(self.ndim):
                interpolated_grid[i] = registration_lib.utils.interpolate_mapping(
                    grid[i], self.shape) * (self.shape[i] - 1) / (shape[i] - 1)
            return interpolated_grid
        else:
            return grid

    def check_injectivity(self):
        self.min_unit = torch.min(
            self.deformation.forward_dets[-1])
        if self.min_unit < self.unit_threshold:
            self.vector_fields.back_to_previous()
            self.integrate_vector_fields()
            print("reached limit of jacobian determinant %f < %f" % (
                self.min_unit, self.unit_threshold))
        return self.min_unit > self.unit_threshold

