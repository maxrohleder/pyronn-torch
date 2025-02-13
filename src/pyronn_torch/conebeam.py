#
# Copyright © 2020 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.
"""

"""
import numpy as np
import scipy.linalg
import torch
from typing import Union
import pyronn_torch


class State:
    def __init__(self,
                 projection_shape,
                 volume_shape,
                 source_points,
                 inverse_matrices,
                 projection_matrices,
                 volume_origin,
                 volume_spacing,
                 projection_multiplier,
                 step_size=1.,
                 with_texture=True):
        self.projection_shape = projection_shape
        self.volume_shape = volume_shape
        self.source_points = source_points
        self.inverse_matrices = inverse_matrices
        self.projection_matrices = projection_matrices
        self.volume_origin = volume_origin
        self.volume_spacing = volume_spacing
        self.projection_multiplier = projection_multiplier
        self.with_texture = with_texture
        self.step_size = step_size


class _ForwardProjection(torch.autograd.Function):
    @staticmethod
    def forward(self, volume, state=None):
        if state is None:
            state = self.state
            return_none = True
        else:
            return_none = False

        # strided/sliced tensor memory layout is not considered by custom kernel
        assert volume.is_contiguous(), "Given volume must be in contiguous memory layout. Call '.contiguous()' first."

        projection = torch.zeros(state.projection_shape,
                                 device='cuda',
                                 requires_grad=volume.requires_grad)

        assert pyronn_torch.cpp_extension
        if state.with_texture:
            pyronn_torch.cpp_extension.call_Cone_Projection_Kernel_Tex_Interp_Launcher(
                inv_matrices=state.inverse_matrices.cuda().contiguous(),
                projection=projection,
                source_points=state.source_points.cuda().contiguous(),
                step_size=state.step_size,
                volume=volume,
                volume_spacing_x=state.volume_spacing[0],
                volume_spacing_y=state.volume_spacing[1],
                volume_spacing_z=state.volume_spacing[2])
        else:
            pyronn_torch.cpp_extension.call_Cone_Projection_Kernel_Launcher(
                inv_matrices=state.inverse_matrices.cuda().contiguous(),
                projection=projection,
                source_points=state.source_points.cuda().contiguous(),
                step_size=state.step_size,
                volume=volume,
                volume_spacing_x=state.volume_spacing[0],
                volume_spacing_y=state.volume_spacing[1],
                volume_spacing_z=state.volume_spacing[2])

        self.state = state
        if return_none:
            return projection, None
        else:
            return projection,

    @staticmethod
    def backward(self, projection_grad, state=None, *args):
        if state is None:
            state = self.state
            return_none = True
        else:
            return_none = False

        # strided/sliced tensor memory layout is not considered by custom kernel
        assert projection_grad.is_contiguous(), "Given data must be contiguous in memory. Call '.contiguous()' first."

        volume_grad = torch.zeros(state.volume_shape,
                                  device='cuda',
                                  requires_grad=projection_grad.requires_grad)

        assert pyronn_torch.cpp_extension
        pyronn_torch.cpp_extension.call_Cone_Backprojection3D_Kernel_Launcher(
            state.projection_matrices.cuda().contiguous(), projection_grad,
            state.projection_multiplier, volume_grad, *state.volume_origin,
            *state.volume_spacing)

        self.state = state
        if return_none:
            return volume_grad, None
        else:
            return volume_grad,


class _BackwardProjection(torch.autograd.Function):
    backward = staticmethod(_ForwardProjection.forward)
    forward = staticmethod(_ForwardProjection.backward)


class ConeBeamProjector:
    def __init__(self,
                 volume_shape: tuple,
                 projection_shape: tuple,
                 volume_spacing=np.ones(3),
                 volume_origin=np.zeros(3),
                 projection_spacing=np.ones(2),
                 projection_matrices=None,
                 source_isocenter_distance=1,
                 source_detector_distance=1):
        assert len(volume_shape) == 3, f"volume shape must be (d, h, w), instead: {volume_shape}"
        assert len(projection_shape) == 3, f"projection shape must be (#p, h, w). instead: {projection_shape}"
        self._volume_shape = volume_shape
        self._projection_shape = projection_shape

        self._volume_origin = volume_origin
        self._volume_origin_tensor = torch.tensor(volume_origin, dtype=torch.float32)

        self._volume_spacing = volume_spacing
        self._volume_spacing_tensor = torch.tensor(volume_spacing, dtype=torch.float32)

        self._projection_matrices_numpy = projection_matrices
        self._projection_spacing = projection_spacing
        self._source_isocenter_distance = source_isocenter_distance
        self._source_detector_distance = source_detector_distance

        self._projection_multiplier = self._source_isocenter_distance * \
                                      self._source_detector_distance * \
                                      self._projection_spacing[-1] * \
                                      np.pi / self._projection_shape[0]
        if projection_matrices is not None:
            self._calc_inverse_matrices()

    @classmethod
    def from_conrad_config(cls):
        import pyconrad.autoinit
        import pyconrad.config
        volume_shape = pyconrad.config.get_reco_shape()
        volume_spacing = pyconrad.config.get_reco_spacing()
        volume_origin = pyconrad.config.get_reco_origin()
        projection_shape = pyconrad.config.get_sino_shape()
        projection_spacing = [
            pyconrad.config.get_geometry().getPixelDimensionX(),
            pyconrad.config.get_geometry().getPixelDimensionY(),
        ]
        projection_matrices = pyconrad.config.get_projection_matrices()

        obj = cls(volume_shape=volume_shape,
                  volume_spacing=volume_spacing,
                  volume_origin=volume_origin,
                  projection_shape=projection_shape,
                  projection_spacing=projection_spacing,
                  projection_matrices=projection_matrices)
        return obj

    def new_volume_tensor(self, requires_grad=False):
        return torch.zeros(self._volume_shape,
                           requires_grad=requires_grad,
                           device='cuda')

    def new_projection_tensor(self, requires_grad=False):
        return torch.zeros(self._projection_shape,
                           requires_grad=requires_grad,
                           device='cuda')

    def project_forward(self, volume, step_size=1., use_texture=True):
        return _ForwardProjection.apply(
            volume,
            State(projection_shape=self._projection_shape,
                  volume_shape=self._volume_shape,
                  source_points=self._source_points,
                  inverse_matrices=self._inverse_matrices,
                  projection_matrices=self._projection_matrices,
                  volume_origin=self._volume_origin,
                  volume_spacing=self._volume_spacing,
                  projection_multiplier=self._projection_multiplier,
                  step_size=step_size,
                  with_texture=use_texture))[0]

    def project_backward(self,
                         projection_stack,
                         step_size=1.,
                         use_texture=True):
        return _BackwardProjection.apply(
            projection_stack,
            State(projection_shape=self._projection_shape,
                  volume_shape=self._volume_shape,
                  source_points=self._source_points,
                  inverse_matrices=self._inverse_matrices,
                  projection_matrices=self._projection_matrices,
                  volume_origin=self._volume_origin,
                  volume_spacing=self._volume_spacing,
                  projection_multiplier=self._projection_multiplier,
                  step_size=step_size,
                  with_texture=use_texture))[0]

    def _calc_inverse_matrices(self):
        if self._projection_matrices_numpy is None:
            return
        self._projection_matrices = torch.stack(
            tuple(
                torch.from_numpy(p.astype(np.float32))
                for p in self._projection_matrices_numpy))

        inv_spacing = np.array([1 / s for s in self._volume_spacing],
                               np.float32)

        camera_centers = list(map(
            lambda x: np.array(np.expand_dims(scipy.linalg.null_space(x), 0), np.float32),
            self._projection_matrices_numpy))

        source_points = list(map(
            lambda x: (x[0, :3, 0] / x[0, 3, 0] * inv_spacing
                       - np.array(list(self._volume_origin)) * inv_spacing).astype(np.float32), camera_centers))

        scaling_matrix = np.array([[inv_spacing[0], 0, 0], [0, inv_spacing[1], 0], [0, 0, inv_spacing[2]]])
        inv_matrices = list(map(
            lambda x:
            (scaling_matrix @ np.linalg.inv(x[:3, :3])).astype(np.float32),
            self._projection_matrices_numpy))

        self._inverse_matrices = torch.stack(
            tuple(map(torch.from_numpy, inv_matrices))).float()
        self._source_points = torch.stack(
            tuple(map(torch.from_numpy, source_points))).float()

    def _calc_inverse_matrices_tensor(self, matrices: torch.Tensor) -> None:
        """
        Explanation
        The projection matrix P consists of a camera intrinsic K, and the extrinsics
        rotation R and translation t as P = K @ [R|t]. An alternative form uses the
        camera center C in world coordinates as P = K @ [R|-RC] = [KR|-KRC].

        Given P, we can obtain C = (KR)^-1 @ -(-KRC) = P[:3, :3]^-1 @ -P[:, 3].
        This is equivalent to the nullspace form used above C = ker(P).

        Furthermore, the inverse matrix M maps a point u = (u, v, 1) on the detector
        onto a 3D ray direction r as r = Mu. It is defined as M = -(KR)^-1

        The projector starts at the camera position C and steps along the ray
        direction r for either forward or back projection. All points along the
        line L = C + s*r, where s is in (0-sdd) are integrated over for the line
        integral at detector position u.

        For details see here https://ksimek.github.io/2012/08/22/extrinsic/

        :param matrices: maps a homogenous voxel index x to a detector index u through u = Px. Shaped (p, 3, 4)
        :return: None
        """
        with torch.no_grad():
            p, _, _, = matrices.shape
            assert matrices.dtype == torch.float32
            assert not matrices.requires_grad
            self._volume_origin_tensor = self._volume_origin_tensor.to(matrices.device)

            self._projection_matrices = matrices
            self._source_points = torch.zeros((p, 3), dtype=torch.float32, device=matrices.device)
            self._inverse_matrices = torch.zeros((p, 3, 3), dtype=torch.float32, device=matrices.device)

            # if the projection matrices are in pixel-from-voxel form, this has no effect
            inv_scale = torch.diag(self._volume_spacing_tensor).to(matrices.device)

            M = torch.linalg.inv(matrices[:, :3, :3])
            for i in range(p):
                self._source_points[i] = -M[i] @ matrices[i, :, 3] @ inv_scale - self._volume_origin_tensor @ inv_scale
                self._inverse_matrices[i] = inv_scale @ M[i]

    @property
    def projection_matrices(self):
        return self._projection_matrices_numpy

    @projection_matrices.setter
    def projection_matrices(self, matrices: Union[np.ndarray, torch.Tensor]):
        if type(matrices) == np.ndarray:
            self._projection_matrices_numpy = matrices
            self._calc_inverse_matrices()
        else:
            self._projection_matrices_numpy = None
            self._calc_inverse_matrices_tensor(matrices)

    @property
    def volume_shape(self):
        return self._volume_shape
