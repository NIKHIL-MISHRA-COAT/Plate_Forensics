"""
Deblurring (motion blur / defocus blur).

Design choice
-------------
Blind deblurring of arbitrary real-world blur is a hard, open research
problem (learned models like DeblurGAN-v2 / NAFNet are the SOTA route, and
are pluggable here -- see `LearnedDeblurrer` below, wired to a pretrained
torch.hub / local checkpoint). For a forensic pipeline where GPU access
during the interview may be constrained, we implement three classical,
dependency-light options with very different risk profiles:

  - Unsharp masking: boosts local edge contrast. Cannot "invert" blur the
    way deconvolution can, but it is self-limiting and safe -- it has no
    failure mode that makes an image *worse* than the input in a
    structural sense, only over/under-sharpened. This is the DEFAULT.

  - Wiener deconvolution with an assumed motion PSF: can meaningfully
    reverse blur, but ONLY if the assumed kernel (size/angle) is close to
    the real one. If it isn't -- e.g. the crop wasn't actually blurred
    that way, or wasn't blurred much at all -- inverse filtering amplifies
    frequencies near the kernel's zero-crossings and produces periodic
    ringing/ghosting (repeated, shifted copies of edges). This is *worse*
    than doing nothing, and it is not reliably caught by simple sharpness
    metrics like Laplacian variance, because ringing is itself full of
    sharp fake edges and can score as "sharper" than the clean original.
    We therefore gate this path behind a structural-similarity (SSIM)
    check against the input, not just a variance check, and opt-in only.

  - Richardson-Lucy deconvolution: same PSF-mismatch risk as Wiener,
    same gating applies.

Given that risk profile, `deblur()` now defaults to unsharp masking, and
only runs PSF-based deconvolution when the caller explicitly asks for it
(method="wiener" / "richardson_lucy") -- and even then, the result is
validated against the input before being accepted.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.restoration import richardson_lucy
from skimage.metrics import structural_similarity as ssim


def _to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def _laplacian_var(image: np.ndarray) -> float:
    return cv2.Laplacian(_to_gray(image), cv2.CV_64F).var()


def unsharp_mask(image: np.ndarray, sigma: float = 1.0, amount: float = 1.0) -> np.ndarray:
    """
    Safe default sharpening. Boosts high-frequency detail by subtracting a
    blurred copy from the original, without the kernel-mismatch risk of
    deconvolution. Cannot produce the periodic ghosting/ringing that a
    wrong PSF causes in Wiener/Richardson-Lucy -- worst case it looks
    slightly over- or under-sharpened, never structurally corrupted.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def _motion_psf(size: int, angle: float = 0.0) -> np.ndarray:
    psf = np.zeros((size, size))
    psf[size // 2, :] = 1.0
    center = (size / 2, size / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    psf = cv2.warpAffine(psf, M, (size, size))
    psf /= psf.sum() + 1e-8
    return psf


def _wiener_channel(channel: np.ndarray, psf: np.ndarray, snr: float) -> np.ndarray:
    channel = channel.astype(np.float32) / 255.0
    psf_padded = np.zeros_like(channel)
    kh, kw = psf.shape
    psf_padded[:kh, :kw] = psf
    psf_padded = np.roll(psf_padded, -kh // 2, axis=0)
    psf_padded = np.roll(psf_padded, -kw // 2, axis=1)

    H = np.fft.fft2(psf_padded)
    G = np.fft.fft2(channel)
    H_conj = np.conj(H)
    denom = (H * H_conj) + (1.0 / max(snr, 1e-3))
    F_hat = (H_conj / denom) * G
    restored = np.abs(np.fft.ifft2(F_hat))
    return np.clip(restored * 255.0, 0, 255).astype(np.uint8)


def wiener_deblur(image: np.ndarray, psf_size: int = 15, snr: float = 25, angle: float = 0.0) -> np.ndarray:
    psf = _motion_psf(psf_size, angle)
    if image.ndim == 3:
        channels = cv2.split(image)
        restored = [_wiener_channel(c, psf, snr) for c in channels]
        return cv2.merge(restored)
    return _wiener_channel(image, psf, snr)


def richardson_lucy_deblur(image: np.ndarray, psf_size: int = 15, iterations: int = 30, angle: float = 0.0) -> np.ndarray:
    psf = _motion_psf(psf_size, angle)

    def _rl_channel(c):
        c = c.astype(np.float64) / 255.0
        restored = richardson_lucy(c, psf, num_iter=iterations, clip=False)
        return np.clip(restored * 255.0, 0, 255).astype(np.uint8)

    if image.ndim == 3:
        channels = cv2.split(image)
        return cv2.merge([_rl_channel(c) for c in channels])
    return _rl_channel(image)


def _accept_deconvolution(original: np.ndarray, restored: np.ndarray,
                           min_ssim: float = 0.6) -> bool:
    """
    Structural-similarity gate for PSF-based deconvolution output.

    Laplacian variance alone is NOT a reliable accept/reject signal here:
    ringing artifacts from a mismatched PSF are full of sharp fake edges
    and often score *higher* on variance than the clean original, so a
    variance-only check lets exactly the failure case we're worried about
    straight through. SSIM instead measures whether the restored image is
    still structurally the same image as the input (same layout of
    strokes/edges, not just "has more high-frequency energy"), which is
    what breaks down under ghosting/ringing.
    """
    original_gray = _to_gray(original)
    restored_gray = _to_gray(restored)
    score = ssim(original_gray, restored_gray, data_range=255)
    return score >= min_ssim


def deblur(image: np.ndarray, method: str = "unsharp", psf_size: int = 15,
           snr: float = 25, rl_iterations: int = 30, angle: float = 0.0,
           sharpness_threshold: float = 150.0,
           unsharp_sigma: float = 1.0, unsharp_amount: float = 1.0,
           min_ssim: float = 0.6) -> np.ndarray:
    if method == "none":
        return image

    if method == "unsharp":
        # Self-limiting -- no PSF-mismatch failure mode, so no gating needed.
        return unsharp_mask(image, unsharp_sigma, unsharp_amount)

    if method not in ("wiener", "richardson_lucy"):
        raise ValueError(f"Unknown deblur method: {method}")

    original_var = _laplacian_var(image)
    if original_var >= sharpness_threshold:
        return image  # already sharp -- deconvolving a mismatched PSF only hurts it

    if method == "wiener":
        restored = wiener_deblur(image, psf_size, snr, angle)
    else:
        restored = richardson_lucy_deblur(image, psf_size, rl_iterations, angle)

    if not _accept_deconvolution(image, restored, min_ssim):
        # PSF assumption didn't match this crop -- reject to avoid
        # ringing/ghosting rather than pass corrupted pixels downstream.
        return image

    return restored


class LearnedDeblurrer:
    """
    Optional pluggable slot for a learned deblurring model (e.g. a
    fine-tuned NAFNet / DeblurGAN-v2 checkpoint). Kept separate from the
    classical path so it can be swapped in without touching pipeline.py --
    just point config `enhancement.deblur.method` at a new value and branch
    to this class. Not wired to a hosted checkpoint by default since it
    would require a network fetch at pipeline setup time.
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        import torch
        self.device = device
        self.model = torch.load(checkpoint_path, map_location=device)
        self.model.eval()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        import torch
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
        out = out.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        return (out * 255).astype(np.uint8)