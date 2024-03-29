import torch
from .registration import Registration
from registration_lib.grid import Deformation, VectorFields
from registration_lib.image import SequentialScalarImages


class LDDMM(Registration):

    def set_vector_fields(self, shape, init_vf=None):
        self.vector_fields = VectorFields(self.n_step, shape, init_vf)

    def update_sequential(self, fixed, moving):

        for i in range(0, self.n_step + 1):
            j = - i - 1
            momentum = (self.similarity.derivative(fixed.data[j], moving.data[i], *self.deformation.kernel_grads)
                        * self.deformation.backward_dets[j])

            grad = 2 * self.vector_fields[i] + self.regularizer(momentum)

            delta = self.learning_rate * grad
            self.vector_fields.delta_vector_fields[i] = delta.clone()

        self.vector_fields.update()
        self.integrate_vector_fields()
        self.metric = torch.sum(torch.stack(
            [self.vector_fields[i] ** 2 * self.As[-1]
                                          for i in range(self.n_step + 1)], 0))/self.n_step


    def integrate_vector_fields(self):
        v = 0.5 * (self.vector_fields[:-1] + self.vector_fields[1:])

        forward_mapping_before = self.deformation.forward_mappings[-1].clone()
        self.deformation.update_mappings(v)
        forward_mapping_after = self.deformation.forward_mappings[-1].clone()
        self.delta_phi = torch.max(
            torch.abs(forward_mapping_after - forward_mapping_before))


    def execute(self):
        warp = Deformation(shape=self.shape)
        self.resulting_vector_fields = []
        self.resulting_metric = []
        self.As = []
        for n_iter, resolution, sigma, vf in zip(self.n_iters,
                                                 self.resolutions,
                                                 self.smoothing_sigmas,
                                                 self.init_vf):
            print("=======================================")
            print("resolution", resolution)


            warped_moving = self.moving.apply_transform(warp)

            moving = warped_moving.change_resolution(resolution, sigma)
            fixed = self.fixed.change_resolution(resolution, sigma)
            shape = moving.get_shape()
            self.regularizer.set_operator(shape)
            self.As += [torch.from_numpy(self.regularizer.A).type(self.dtype)]
            self.deformation.set_shape(shape)
            self.set_vector_fields(shape, vf)

            grid = self.optimization(fixed, moving, n_iter, resolution)
            self.resulting_vector_fields += [self.vector_fields]
            self.resulting_metric += [self.metric*resolution]
            warp += Deformation(grid=grid)

        return warp

    def optimization(self, fixed, moving, max_iter, resolution):
        moving_images = SequentialScalarImages(moving, self.n_step + 1)
        fixed_images = SequentialScalarImages(fixed, self.n_step + 1)

        # print("iteration   0, Energy %f" % (
        #     self.similarity.cost(fixed.data, moving.data)))

        for i in range(0, max_iter):


            self.update_sequential(fixed_images, moving_images)

            if not self.check_injectivity():
                break


            moving_images.apply_transforms(
                self.deformation.forward_mappings.clone())


            fixed_images.apply_transforms(
                self.deformation.backward_mappings.clone())

            max_delta_phi = self.delta_phi * (max_iter - i)
            # print("iteration%4d, Energy %f" % (
            #     i + 1,
            #     self.similarity.cost(fixed_images[0], moving_images[-1])))
            # print(14 * ' ', "minimum unit", np.float64(self.min_unit))
            # print(14 * ' ', "delta phi", np.float64(self.delta_phi))
            # print(14 * ' ', "maximum delta phi", np.float64(max_delta_phi))
            if max_delta_phi < self.delta_phi_threshold / resolution:
                # print("|L_inf norm of displacement| x iter < %f voxel" % (
                #         self.delta_phi_threshold / resolution))
                break

        return self.zoom_grid(self.deformation.forward_mappings[-1],
                              resolution)

    def execute_coarse_to_fine(self):
        vector_fields = VectorFields(self.n_step, shape=self.shape)

        for n_iter, resolution, sigma in zip(self.n_iters,
                                             self.resolutions,
                                             self.smoothing_sigmas):
            print("=======================================")
            print("resolution", resolution)
            fixed = self.fixed.change_resolution(resolution, sigma)
            moving = self.moving.change_resolution(resolution, sigma)
            shape = fixed.get_shape()
            self.vector_fields = vector_fields.change_resolution(resolution)
            self.deformation.set_shape(shape)
            v = 0.5 * (self.vector_fields[:-1] + self.vector_fields[1:])
            self.deformation.update_mappings(v)

            vector_fields = self.optimization_coarse_to_fine(
                fixed, moving, n_iter, resolution)

        return self.deformation

    def optimization_coarse_to_fine(self, fixed, moving, max_iter, resolution):
        fixed_images = SequentialScalarImages(fixed, self.n_step)
        moving_images = SequentialScalarImages(moving, self.n_step)
        fixed_images.apply_transforms(self.deformation.backward_mappings)
        moving_images.apply_transforms(self.deformation.forward_mappings)
        print("iteration   0, Energy %f" % (
            self.similarity.cost(fixed_images[0], moving_images[-1])))

        for i in range(0, max_iter):
            self.update_sequential(fixed_images, moving_images)

            if not self.check_injectivity():
                break


            moving_images.apply_transforms(
                self.deformation.forward_mappings)
            fixed_images.apply_transforms(
                self.deformation.backward_mappings)

            max_delta_phi = self.delta_phi * (max_iter - i)
            print("iteration%4d, Energy %f" % (
                i + 1,
                self.similarity.cost(fixed_images[0], moving_images[-1])))
            print(14 * ' ', "minimum unit", self.min_unit)
            print(14 * ' ', "delta phi", self.delta_phi)
            print(14 * ' ', "maximum delta phi {0}".format(max_delta_phi))
            if max_delta_phi < self.delta_phi_threshold / resolution:
                print("|L_inf norm of displacement| x iter < %f voxel" % (
                    self.delta_phi_threshold / resolution))
                break

        return self.vector_fields.change_resolution(resolution=1. / resolution)


def derivative(similarity,
               grads_model,
               fixed,
               moving,
               Dphi,
               vector_field,
               regularizer,
               learning_rate):
    momentum = similarity.derivative(fixed, moving, *grads_model) * Dphi
    grad = 2 * vector_field + regularizer(momentum)

    return learning_rate * grad
