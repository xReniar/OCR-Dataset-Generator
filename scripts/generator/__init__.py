from .doctr import DoctrGenerator
from .mmocr import MMOCRGenerator
from .paddleocr import PaddleOCRGenerator
from .generator import Generator

OCR_SYSTEMS = {
    "doctr": DoctrGenerator,
    "mmocr": MMOCRGenerator,
    "paddleocr": PaddleOCRGenerator
}

__all__ = ["OCR_SYSTEMS", "Generator"]