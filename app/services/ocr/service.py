import asyncio
import time
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import structlog
from PIL import Image

from app.domain.models import OcrBlock, OcrResult

logger = structlog.get_logger(__name__)


class OcrService:
    def __init__(self) -> None:
        from paddleocr import PaddleOCR

        # PaddleOCR 3.x / PaddleX:
        #   • text_detection_model_name / text_recognition_model_name — explicit model selection
        #     (when set, `lang` and `ocr_version` are ignored)
        #   • PP-OCRv4_mobile_det  — lightweight det (~5 MB), fast on CPU
        #   • eslav_PP-OCRv5_mobile_rec — Russian/Slavic rec; already cached from first run
        #   • doc-preprocessor and textline-orientation disabled to avoid heavy auxiliary models
        self._ocr = PaddleOCR(
            device="cpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_detection_model_name="PP-OCRv4_mobile_det",
            text_recognition_model_name="eslav_PP-OCRv5_mobile_rec",
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, rgb_array: np.ndarray) -> np.ndarray:
        """Grayscale → adaptive binarisation → deskew."""
        gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)

        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Deskew: find contours on inverted image (text = white on black)
        contours, _ = cv2.findContours(
            cv2.bitwise_not(binary), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            all_pts = np.concatenate(contours)
            _, _, angle = cv2.minAreaRect(all_pts)
            # Normalise to (−45, 45]: OpenCV angle is in [−90, 0)
            if angle < -45:
                angle += 90
            if abs(angle) > 0.5:
                h, w = binary.shape
                M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                binary = cv2.warpAffine(
                    binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
                )

        # PaddleOCR 3.x predict() requires 3-channel input
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

    # ------------------------------------------------------------------
    # Page loading
    # ------------------------------------------------------------------

    def _load_pages(self, file_bytes: bytes, ext: str) -> list[np.ndarray]:
        if ext == ".pdf":
            import pypdfium2 as pdfium

            doc = pdfium.PdfDocument(file_bytes)
            pages: list[np.ndarray] = []
            for i in range(len(doc)):
                bitmap = doc[i].render(scale=300 / 72, rev_byteorder=False)
                pages.append(np.array(bitmap.to_pil().convert("RGB")))
            return pages
        else:
            pil_img = Image.open(BytesIO(file_bytes)).convert("RGB")
            return [np.array(pil_img)]

    # ------------------------------------------------------------------
    # Sync OCR (runs in thread)
    # ------------------------------------------------------------------

    def _run_ocr(self, pages: list[np.ndarray]) -> list:
        # PaddleOCR 3.x: predict() replaces ocr(); returns a generator per call
        return [list(self._ocr.predict(page)) for page in pages]

    # ------------------------------------------------------------------
    # Block extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_blocks(raw_results: list) -> list[OcrBlock]:
        """Parse PaddleOCR 3.x OCRResult objects into OcrBlock list.

        raw_results shape: [page][predict_call] → OCRResult
        OCRResult keys: rec_texts, rec_scores, rec_polys (list of 4-point polygons)
        """
        blocks: list[OcrBlock] = []
        for page_results in raw_results:
            for r in page_results:
                texts: list[str] = r["rec_texts"]
                scores: list[float] = r["rec_scores"]
                polys = r["rec_polys"]  # list of (4, 2) arrays

                if not texts:
                    continue

                # Sort top-to-bottom by y of first polygon point (top-left)
                items = sorted(zip(polys, texts, scores), key=lambda t: float(t[0][0][1]))
                for poly, text, conf in items:
                    xs = [float(pt[0]) for pt in poly]
                    ys = [float(pt[1]) for pt in poly]
                    blocks.append(
                        OcrBlock(
                            text=text,
                            confidence=float(conf),
                            bbox=[min(xs), min(ys), max(xs), max(ys)],
                        )
                    )
        return blocks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def recognize(self, file_bytes: bytes, filename: str) -> OcrResult:
        start = time.monotonic()
        ext = Path(filename).suffix.lower()

        pages = self._load_pages(file_bytes, ext)
        processed = [self._preprocess(p) for p in pages]

        raw_results = await asyncio.to_thread(self._run_ocr, processed)

        blocks = self._extract_blocks(raw_results)
        avg_conf = sum(b.confidence for b in blocks) / len(blocks) if blocks else 0.0
        elapsed = time.monotonic() - start

        logger.info(
            "OCR completed",
            filename=filename,
            page_count=len(pages),
            blocks_count=len(blocks),
            avg_confidence=round(avg_conf, 4),
            processing_time_sec=round(elapsed, 3),
        )

        return OcrResult(
            blocks=blocks,
            avg_confidence=avg_conf,
            page_count=len(pages),
            processing_time_sec=elapsed,
        )
