"""Turn a clean render into something that looks scanned or photographed.

Severity 0 is a pristine digital print; severity 1 is a bad phone photo of a faxed copy.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


@dataclass
class Degradation:
    """What was actually done to one document, recorded so results can be sliced by condition."""

    severity: float
    grayscale: bool = False
    rotation_deg: float = 0.0
    perspective: float = 0.0
    blur_radius: float = 0.0
    brightness: float = 1.0
    contrast: float = 1.0
    noise_sigma: float = 0.0
    jpeg_quality: int = 100
    shadow: bool = False
    speckles: int = 0

    def as_row(self) -> dict[str, object]:
        return {"severity": round(self.severity, 3), "grayscale": self.grayscale,
                "rotation_deg": round(self.rotation_deg, 2), "perspective": round(self.perspective, 4),
                "blur_radius": round(self.blur_radius, 2), "brightness": round(self.brightness, 3),
                "contrast": round(self.contrast, 3), "noise_sigma": round(self.noise_sigma, 2),
                "jpeg_quality": self.jpeg_quality, "shadow": self.shadow, "speckles": self.speckles}


def _perspective_coeffs(src: list[tuple[float, float]], dst: list[tuple[float, float]]) -> tuple[float, ...]:
    matrix = []
    for (sx, sy), (dx, dy) in zip(src, dst):
        matrix.append([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy])
        matrix.append([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy])
    a = np.array(matrix, dtype=float)
    b = np.array(src, dtype=float).reshape(8)
    return tuple(np.linalg.solve(a, b))


def _warp(img: Image.Image, amount: float, rng: random.Random) -> Image.Image:
    w, h = img.size
    jitter = lambda span: rng.uniform(-span, span)  # noqa: E731
    span = amount * min(w, h)
    dst = [(jitter(span), jitter(span)), (w + jitter(span), jitter(span)),
           (w + jitter(span), h + jitter(span)), (jitter(span), h + jitter(span))]
    coeffs = _perspective_coeffs([(0, 0), (w, 0), (w, h), (0, h)], dst)
    return img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC, fillcolor=(255, 255, 255))


def _shadow(img: Image.Image, rng: random.Random) -> Image.Image:
    w, h = img.size
    gradient = np.linspace(rng.uniform(0.72, 0.9), 1.0, w, dtype=np.float32)
    if rng.random() < 0.5:
        gradient = gradient[::-1]
    mask = np.tile(gradient, (h, 1))[..., None]
    return Image.fromarray(np.clip(np.asarray(img, dtype=np.float32) * mask, 0, 255).astype(np.uint8))


def _speckle(img: Image.Image, count: int, rng: random.Random) -> Image.Image:
    arr = np.asarray(img).copy()
    h, w = arr.shape[:2]
    for _ in range(count):
        x, y = rng.randrange(w), rng.randrange(h)
        size = rng.randint(1, 3)
        value = 0 if rng.random() < 0.7 else 255
        arr[y:y + size, x:x + size] = value
    return Image.fromarray(arr)


def plan(severity: float, rng: random.Random) -> Degradation:
    """Choose degradation parameters for one document; magnitudes scale with severity."""
    s = max(0.0, min(1.0, severity))
    return Degradation(
        severity=s,
        grayscale=rng.random() < 0.55 + 0.35 * s,
        rotation_deg=rng.gauss(0, 0.35 + 2.2 * s),
        perspective=abs(rng.gauss(0, 0.004 + 0.02 * s)),
        blur_radius=max(0.0, rng.gauss(0.25 + 1.3 * s, 0.15)),
        brightness=1.0 + rng.uniform(-0.12 - 0.2 * s, 0.08 + 0.12 * s),
        contrast=1.0 + rng.uniform(-0.15 - 0.25 * s, 0.1 + 0.15 * s),
        noise_sigma=max(0.0, rng.gauss(1.5 + 13 * s, 1.5)),
        jpeg_quality=int(round(96 - 56 * s + rng.uniform(-4, 4))),
        shadow=rng.random() < 0.15 + 0.4 * s,
        speckles=int(rng.random() * 90 * s),
    )


def apply(img: Image.Image, spec: Degradation, rng: random.Random) -> Image.Image:
    out = img.convert("RGB")
    if spec.perspective > 0:
        out = _warp(out, spec.perspective, rng)
    if abs(spec.rotation_deg) > 0.01:
        out = out.rotate(spec.rotation_deg, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    if spec.shadow:
        out = _shadow(out, rng)
    out = ImageEnhance.Brightness(out).enhance(spec.brightness)
    out = ImageEnhance.Contrast(out).enhance(spec.contrast)
    if spec.blur_radius > 0.05:
        out = out.filter(ImageFilter.GaussianBlur(spec.blur_radius))
    if spec.noise_sigma > 0.2:
        arr = np.asarray(out, dtype=np.float32)
        noise = np.random.default_rng(rng.randrange(2**32)).normal(0, spec.noise_sigma, arr.shape)
        out = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
    if spec.speckles:
        out = _speckle(out, spec.speckles, rng)
    if spec.grayscale:
        out = out.convert("L").convert("RGB")
    quality = max(15, min(100, spec.jpeg_quality))
    buffer = io.BytesIO()
    out.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).copy()


def degrade(img: Image.Image, severity: float, rng: random.Random) -> tuple[Image.Image, Degradation]:
    spec = plan(severity, rng)
    return apply(img, spec, rng), spec
