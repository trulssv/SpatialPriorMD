import odl
from odl.contrib.torch import OperatorModule
import numpy as np
import torch
from typing import Literal, Tuple


class RayTrafoModule(torch.nn.Module):
    """2-D parallel-beam ray transform retained for calibration workflows.

    ``src_radius`` and ``det_radius`` are accepted for compatibility with the
    tomography configuration. They are not used by parallel-beam geometry.
    """

    def __init__(
        self,
        physical_size: Tuple[float, float] = (200.0, 200.0),
        shape: Tuple[int, int] = (512, 512),
        n_angles: int = 512,
        n_detectors: int = 512,
        geometry: Literal["parallel"] = "parallel",
        src_radius: float | None = None,
        det_radius: float | None = None,
    ):
        super().__init__()
        if geometry != "parallel":
            raise ValueError(f"Only parallel geometry is supported, got {geometry!r}.")

        self.physical_size = physical_size
        self.shape = shape
        self.n_angles = n_angles
        self.n_detectors = n_detectors
        self.geometry = geometry
        self.src_radius = src_radius
        self.det_radius = det_radius

        min_pt = tuple(-size / 2 for size in physical_size)
        max_pt = tuple(size / 2 for size in physical_size)
        reco_space = odl.uniform_discr(min_pt=min_pt, max_pt=max_pt, shape=shape, dtype="float32")
        angle_partition = odl.uniform_partition(0, np.pi, n_angles)
        detector_half_width = float(np.hypot(*physical_size) / 2)
        detector_partition = odl.uniform_partition(
            -detector_half_width,
            detector_half_width,
            n_detectors,
        )
        ray_trafo = odl.tomo.RayTransform(reco_space, odl.tomo.Parallel2dGeometry(angle_partition, detector_partition))
        self.forward_module = OperatorModule(ray_trafo)
        self.backward_module = OperatorModule(
            odl.tomo.fbp_op(ray_trafo, filter_type="Ram-Lak", frequency_scaling=1.0)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Project an image or image batch with trailing shape ``[height, width]``."""
        if image.ndim < 2 or tuple(image.shape[-2:]) != self.shape:
            raise ValueError(f"Expected trailing image shape {self.shape}, got {tuple(image.shape)}.")
        return self.forward_module(image)

    def backproject(self, sinogram: torch.Tensor) -> torch.Tensor:
        """Filtered-backproject sinograms with trailing shape ``[angle, detector]``."""
        if (
            sinogram.ndim < 2
            or sinogram.shape[-2] != self.n_angles
            or sinogram.shape[-1] != self.n_detectors
        ):
            raise ValueError(
                f"Expected trailing sinogram shape ({self.n_angles}, {self.n_detectors}), "
                f"got {tuple(sinogram.shape)}."
            )
        return self.backward_module(sinogram)

    def backward(self, sinogram: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`backproject` used by torch-style call sites."""
        return self.backproject(sinogram)


class SliceWiseRayTrafoModule(torch.nn.Module):
    """Apply one 2-D parallel-beam transform independently to each axial slice."""

    def __init__(
        self,
        physical_size: Tuple[float, float],
        shape: Tuple[int, int],
        n_angles: int,
        n_detectors: int,
    ):
        super().__init__()
        self.physical_size = physical_size
        self.shape = shape
        self.n_angles = n_angles
        self.n_detectors = n_detectors

        min_pt = tuple(-size / 2 for size in physical_size)
        max_pt = tuple(size / 2 for size in physical_size)
        reco_space = odl.uniform_discr(min_pt=min_pt, max_pt=max_pt, shape=shape, dtype="float32")
        angle_partition = odl.uniform_partition(0, np.pi, n_angles)
        detector_half_width = float(np.hypot(*physical_size) / 2)
        detector_partition = odl.uniform_partition(
            -detector_half_width,
            detector_half_width,
            n_detectors,
        )
        geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
        ray_trafo = odl.tomo.RayTransform(reco_space, geometry)

        self.forward_module = OperatorModule(ray_trafo)
        self.backward_module = OperatorModule(
            odl.tomo.fbp_op(ray_trafo, filter_type="Ram-Lak", frequency_scaling=1.0)
        )

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """Project ``[material, slice, height, width]`` to ``[material, angle, slice, detector]``."""
        if volume.ndim != 4 or tuple(volume.shape[-2:]) != self.shape:
            raise ValueError(
                f"Expected [material, slice, {self.shape[0]}, {self.shape[1]}], got {tuple(volume.shape)}."
            )

        material_count, slice_count, height, width = volume.shape
        batch = volume.reshape(material_count * slice_count, height, width)
        sinograms = self.forward_module(batch)
        return sinograms.reshape(
            material_count, slice_count, self.n_angles, self.n_detectors
        ).permute(0, 2, 1, 3)

    def backward(self, sinograms: torch.Tensor) -> torch.Tensor:
        """Reconstruct ``[material, angle, slice, detector]`` to ``[material, slice, height, width]``."""
        if (
            sinograms.ndim != 4
            or sinograms.shape[1] != self.n_angles
            or sinograms.shape[3] != self.n_detectors
        ):
            raise ValueError(
                f"Expected [material, {self.n_angles}, slice, {self.n_detectors}], got {tuple(sinograms.shape)}."
            )

        material_count, _, slice_count, _ = sinograms.shape
        batch = sinograms.permute(0, 2, 1, 3).reshape(
            material_count * slice_count, self.n_angles, self.n_detectors
        )
        images = self.backward_module(batch)
        return images.reshape(material_count, slice_count, *self.shape)
