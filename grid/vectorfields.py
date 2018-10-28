import torch
from torch.nn import functional as F


class VectorFields(object):

    def __init__(self, n_step, shape=None, vector_fields=None, dtype=torch.cuda.DoubleTensor):
        self.n_step = n_step
        self.dtype = dtype
        if shape is not None:
            self.set_shape(shape)

        if vector_fields is not None:
            assert n_step + 1 == len(vector_fields)
            self.vector_fields = vector_fields
            self.delta_vector_fields = torch.zeros_like(vector_fields).type(self.dtype)
            self.shape = vector_fields.shape[2:]
            self.ndim = vector_fields.shape[1]

    def __getitem__(self, index):
        return self.vector_fields[index]

    def set_shape(self, shape):
        self.shape = shape
        self.ndim = len(shape)
        self.init_vector_fields()

    def init_vector_fields(self):
        self.vector_fields = torch.zeros(
            (self.n_step + 1, self.ndim) + self.shape).type(self.dtype)
        self.delta_vector_fields = torch.zeros_like(self.vector_fields).type(self.dtype)

    def update(self):
        """
        update vector fields

        v_next = v_now - learning_rate * nabla(Energy)
        """
        self.vector_fields -= self.delta_vector_fields

    def back_to_previous(self):
        """
        get back to previous vector field

        v^(k+1) = v^(k) - learning_rate * nabla(Energy)

        v_before = v_now + learning_rate * nabla(Energy_before)
        """
        self.vector_fields += self.delta_vector_fields

    def change_resolution(self, resolution, order='bilinear'):

        if resolution == 1:
            return VectorFields(self.n_step,
                                vector_fields=self.vector_fields.data.clone())
        ratio = (1 / float(resolution),) * self.ndim
        data = torch.stack([torch.stack([F.interpolate(self.vector_fields[i][0].unsqueeze(0).unsqueeze(0),
                                                       scale_factor=ratio, mode=order,
                                                       align_corners=True).squeeze(
            0).squeeze(0),
                                         F.interpolate(self.vector_fields[i][1].unsqueeze(0).unsqueeze(0),
                                                       scale_factor=ratio, mode=order,
                                                       align_corners=True).squeeze(
                                             0).squeeze(0)], 0)
                            for i in range(self.n_step + 1)], 0)
        return VectorFields(self.n_step, vector_fields=data)
