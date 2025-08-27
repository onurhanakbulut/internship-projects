import re
import cv2
import numpy as np
import os

TR_PLATE_REGEX = re.compile(r"^\s*\d{2}\s*[A-Z]{1,3}\s*\d{2,4}\s*$", re.I)

def preprocess_plate(roi_bgr: np.ndarray, *,
                     scale: float = 2.2,
                     use_clahe: bool = True,
                     do_denoise: bool = True,
                     do_sharpen: bool = True,
                     threshold: str = "adaptive",
                     morph: bool = True,
                     return_stages: bool = False):
    if roi_bgr is None or roi_bgr.size == 0:
        return None if not return_stages else (None, {})

    stages = {}

    # 1) gray
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    stages["01_gray"] = gray.copy()
    stages["gray"] = stages["01_gray"]

    # 2) scaled
    if scale and scale != 1.0:
        nh = max(1, int(gray.shape[0] * scale))
        nw = max(1, int(gray.shape[1] * scale))
        gray = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_CUBIC)
    stages["02_scaled"] = gray.copy()
    stages["scaled"] = stages["02_scaled"]

    # 3) clahe
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
    stages["03_clahe"] = gray.copy()
    stages["clahe"] = stages["03_clahe"]

    # 4) denoise
    if do_denoise:
        gray = cv2.GaussianBlur(gray, (3,3), 0)
    stages["04_denoise"] = gray.copy()
    stages["denoise"] = stages["04_denoise"]

    # 5) sharpen
    if do_sharpen:
        blur = cv2.GaussianBlur(gray, (0,0), 1.2)
        gray = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
    stages["05_sharpen"] = gray.copy()
    stages["sharpen"] = stages["05_sharpen"]

    # 6) threshold
    if threshold == "adaptive":
        bw = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 35, 10)
    elif threshold == "otsu":
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        bw = gray.copy()
    stages["06_thresh"] = bw.copy()
    stages["thresh"] = stages["06_thresh"]

    # 7) morph
    if morph:
        kern = np.ones((2,2), np.uint8)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kern, iterations=1)
    stages["07_morph"] = bw.copy()
    stages["morph"] = stages["07_morph"]

    # 8) final (alias)
    stages["08_final"] = bw.copy()
    stages["final"] = stages["08_final"]

    return (bw, stages) if return_stages else bw




