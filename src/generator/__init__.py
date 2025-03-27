from .doctr import DoctrGenerator
from .easyocr import EasyOCRGenerator
from .mmocr import MMOCRGenerator
from .paddleocr import PaddleOCRGenerator
from .yolo_trocr import YOLOTrOCRGenerator
from .generator import Generator

OCR_SYSTEMS = {
    "doctr": DoctrGenerator,
    "easyocr": EasyOCRGenerator,
    "mmocr": MMOCRGenerator,
    "paddleocr": PaddleOCRGenerator,
    "yolo+trocr": YOLOTrOCRGenerator
}

__all__ = ["OCR_SYSTEMS", "Generator"]