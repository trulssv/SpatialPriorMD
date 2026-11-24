from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
import torch


class IndependentDRUNetDenoiser:
    """Apply a pretrained RGB DRUNet independently to scalar image channels."""

    def __init__(
        self,
        sigmas: float | Sequence[float] = 0.35,
        model: torch.nn.Module | None = None,
    ) -> None:
        self.sigmas = sigmas
        self.model = model

    def __call__(self, images: np.ndarray) -> np.ndarray:
        images = self._validate_images(images)
        model = self._get_model()
        sigmas = self._channel_sigmas(images.shape[0])
        return np.stack(
            [self._denoise_scalar(model, image, sigma) for image, sigma in zip(images, sigmas)],
            axis=0,
        )

    def _get_model(self) -> torch.nn.Module:
        if self.model is None:
            import deepinv as dinv

            self.model = dinv.models.DRUNet()
        self.model.eval()
        return self.model

    def _channel_sigmas(self, n_channels: int) -> list[float]:
        if isinstance(self.sigmas, (float, int)):
            return [float(self.sigmas)] * n_channels
        if len(self.sigmas) != n_channels:
            raise ValueError(f"Expected {n_channels} DRUNet sigmas, got {len(self.sigmas)}.")
        return [float(sigma) for sigma in self.sigmas]

    @staticmethod
    def _validate_images(images: np.ndarray) -> np.ndarray:
        images = np.asarray(images, dtype=np.float32)
        if images.ndim != 3:
            raise ValueError(f"Expected images with shape [channels, height, width], got {images.shape}.")
        if images.shape[0] != 2:
            raise ValueError(f"Expected two material channels, got {images.shape[0]}.")
        return images

    @staticmethod
    def _denoise_scalar(model: torch.nn.Module, image: np.ndarray, sigma: float) -> np.ndarray:
        image_min = float(image.min())
        image_max = float(image.max())
        image_range = image_max - image_min
        if image_range <= np.finfo(np.float32).eps:
            return image.copy()

        scaled = (image - image_min) / image_range
        model_input = torch.from_numpy(scaled).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        with torch.inference_mode():
            denoised = model(model_input, sigma=sigma)[0, 0].detach().cpu().numpy()
        return denoised * image_range + image_min


class CovarianceDRUNetDenoiser(IndependentDRUNetDenoiser):
    """Denoise two correlated material images in locally estimated PCA coordinates.

    Covariance is estimated from high-pass residuals inside each supplied region.
    The PCA rotation decorrelates anti-correlated material noise before the scalar
    DRUNet calls. High-residual samples are trimmed to limit bias from anatomy.
    """

    def __init__(
        self,
        sigmas: float | Sequence[float] = 0.35,
        model: torch.nn.Module | None = None,
        blur_sigma: float = 2.0,
        trim_fraction: float = 0.2,
        min_region_pixels: int = 256,
        covariance_floor: float = 1e-8,
    ) -> None:
        super().__init__(sigmas=sigmas, model=model)
        if not 0.0 <= trim_fraction < 1.0:
            raise ValueError("trim_fraction must be in [0, 1).")
        self.blur_sigma = float(blur_sigma)
        self.trim_fraction = float(trim_fraction)
        self.min_region_pixels = int(min_region_pixels)
        self.covariance_floor = float(covariance_floor)
        self.covariances: dict[int, np.ndarray] = {}

    def __call__(
        self,
        images: np.ndarray,
        regions: np.ndarray | None = None,
        valid_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        images = self._validate_images(images)
        if regions is None:
            regions = np.zeros(images.shape[1:], dtype=np.int32)
        regions = np.asarray(regions)
        if regions.shape != images.shape[1:]:
            raise ValueError("regions must have shape [height, width].")

        if valid_mask is None:
            valid_mask = np.ones(images.shape[1:], dtype=bool)
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != images.shape[1:]:
            raise ValueError("valid_mask must have shape [height, width].")

        model = self._get_model()
        sigmas = self._channel_sigmas(2)
        output = images.copy()
        self.covariances = {}
        region_ids = np.unique(regions[valid_mask])
        for region_id in region_ids:
            region_mask = valid_mask & (regions == region_id)
            if np.count_nonzero(region_mask) < self.min_region_pixels:
                continue
            covariance = self._estimate_covariance(images, region_mask)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            eigenvalues = np.maximum(eigenvalues, self.covariance_floor)
            transformed = np.einsum("ij,jhw->ihw", eigenvectors.T, images)
            denoised = np.stack(
                [self._denoise_scalar(model, component, sigma) for component, sigma in zip(transformed, sigmas)],
                axis=0,
            )
            reconstructed = np.einsum("ij,jhw->ihw", eigenvectors, denoised)
            output[:, region_mask] = reconstructed[:, region_mask]
            self.covariances[int(region_id)] = covariance
        return output

    def _estimate_covariance(self, images: np.ndarray, region_mask: np.ndarray) -> np.ndarray:
        residuals = np.stack(
            [image - cv2.GaussianBlur(image, (0, 0), self.blur_sigma) for image in images],
            axis=0,
        )
        samples = residuals[:, region_mask].T
        centered = samples - np.median(samples, axis=0, keepdims=True)
        if self.trim_fraction > 0.0:
            squared_norms = np.sum(centered**2, axis=1)
            threshold = np.quantile(squared_norms, 1.0 - self.trim_fraction)
            centered = centered[squared_norms <= threshold]
        if len(centered) < 2:
            return np.eye(2, dtype=np.float32) * self.covariance_floor
        covariance = np.cov(centered, rowvar=False).astype(np.float32)
        return covariance + np.eye(2, dtype=np.float32) * self.covariance_floor


class PairedContrastCovarianceDRUNetDenoiser(CovarianceDRUNetDenoiser):
    """Denoise a contrast material pair using a paired no-contrast acquisition.

    The paired difference contains a low-frequency K-edge bias in addition to
    noise. A Gaussian low-pass estimate is subtracted from that difference
    before covariance estimation, so only its high-frequency residual enters
    the trimmed covariance. PCA is performed on channels normalized by their
    estimated noise standard deviations, preventing the material with larger
    numeric units from dominating the rotation.
    """

    def __call__(self, contrast_images: np.ndarray, reference_images: np.ndarray) -> np.ndarray:
        contrast_images = self._validate_images(contrast_images)
        reference_images = self._validate_images(reference_images)
        if reference_images.shape != contrast_images.shape:
            raise ValueError("reference_images must have the same shape as contrast_images.")

        model = self._get_model()
        paired_difference = contrast_images - reference_images
        self.noise_residual = self._high_pass(paired_difference)
        covariance = self._estimate_covariance(self.noise_residual)
        channel_scales = np.sqrt(np.maximum(np.diag(covariance), self.covariance_floor))
        normalized_covariance = covariance / np.outer(channel_scales, channel_scales)
        _, eigenvectors = np.linalg.eigh(normalized_covariance)
        normalized_contrast = contrast_images / channel_scales[:, np.newaxis, np.newaxis]
        transformed = np.einsum("ij,jhw->ihw", eigenvectors.T, normalized_contrast)
        sigmas = self._channel_sigmas(2)
        denoised = np.stack(
            [self._denoise_scalar(model, component, sigma) for component, sigma in zip(transformed, sigmas)],
            axis=0,
        )
        self.covariance = covariance
        self.correlation = normalized_covariance
        normalized_output = np.einsum("ij,jhw->ihw", eigenvectors, denoised)
        return normalized_output * channel_scales[:, np.newaxis, np.newaxis]

    def _estimate_covariance(self, images: np.ndarray) -> np.ndarray:
        samples = images.reshape(2, -1).T
        centered = samples - np.median(samples, axis=0, keepdims=True)
        if self.trim_fraction > 0.0:
            squared_norms = np.sum(centered**2, axis=1)
            threshold = np.quantile(squared_norms, 1.0 - self.trim_fraction)
            centered = centered[squared_norms <= threshold]
        if len(centered) < 2:
            return np.eye(2, dtype=np.float32) * self.covariance_floor
        covariance = np.cov(centered, rowvar=False).astype(np.float32)
        return covariance + np.eye(2, dtype=np.float32) * self.covariance_floor

    def _high_pass(self, images: np.ndarray) -> np.ndarray:
        return np.stack(
            [image - cv2.GaussianBlur(image, (0, 0), self.blur_sigma) for image in images],
            axis=0,
        )
