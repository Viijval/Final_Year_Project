"""Producing a Page of positioned words, either from OCR or from a PDF's own text layer."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pymupdf
from PIL import Image

from gstpipe.extract.page import Page, Word

_WINDOWS_DEFAULT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
DEFAULT_CONFIG = "--oem 3 --psm 6"


def tesseract_path() -> str | None:
    """Locate the tesseract binary: PATH, the env override, or the default Windows install."""
    if override := os.environ.get("TESSERACT_EXE"):
        return override
    if found := shutil.which("tesseract"):
        return found
    return str(_WINDOWS_DEFAULT) if _WINDOWS_DEFAULT.exists() else None


def tesseract_available() -> bool:
    return tesseract_path() is not None


def from_pdf(pdf_bytes: bytes) -> Page:
    """The document's own text layer: a perfect reader, used as the upper bound for extraction."""
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        page = doc[0]
        words = [Word(text=w[4], x0=w[0], y0=w[1], x1=w[2], y1=w[3]) for w in page.get_text("words")]
        return Page(words=words, width=page.rect.width, height=page.rect.height)


def from_image(image: Image.Image, config: str = DEFAULT_CONFIG, lang: str = "eng") -> Page:
    """Run OCR over a rendered document image."""
    import pytesseract

    if (exe := tesseract_path()) is None:
        raise RuntimeError("tesseract not found; install it or set TESSERACT_EXE")
    pytesseract.pytesseract.tesseract_cmd = exe

    data = pytesseract.image_to_data(image, lang=lang, config=config,
                                     output_type=pytesseract.Output.DICT)
    words = []
    for text, left, top, width, height, conf in zip(
            data["text"], data["left"], data["top"], data["width"], data["height"], data["conf"]):
        if not text.strip():
            continue
        confidence = float(conf) / 100 if float(conf) >= 0 else 0.0
        words.append(Word(text=text.strip(), x0=float(left), y0=float(top),
                          x1=float(left + width), y1=float(top + height), conf=confidence))
    return Page(words=words, width=float(image.width), height=float(image.height))
