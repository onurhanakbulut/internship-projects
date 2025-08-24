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
        return None
    
    stages = {}
    
    #gray filter
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    stages["01_gray"] = gray.copy()
    
    #scaling
    if scale  and scale != 1.0:
        nh = max(1, int(gray.shape[0] * scale))
        nw = max(1, int(gray.shape[1] * scale))
        gray = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_CUBIC)
    stages["02_scaled"] = gray.copy()    
        
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
    stages["03_clahe"] = gray.copy()
    
    if do_denoise:
        gray = cv2.GaussianBlur(gray, (3,3), 0)
    stages["04_denoise"] = gray.copy()
        
        
    if do_sharpen:
        blur = cv2.GaussianBlur(gray, (0,0), 1.2)
        gray = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
    stages["05_sharpen"] = gray.copy()
    
    if threshold == "adaptive":
        bw = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 35, 10)
    else:
        bw = gray.copy()
    stages["06_thresh"] = bw.copy()
    
    if morph:
        kern = np.ones((2,2), np.uint8)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kern, iterations=1)
    stages["07_morph"] = bw.copy()
    
    return (bw, stages) if return_stages else bw

    
    
    
def save_preprocess_stages(stages: dict, out_dir: str, prefix: str):
    
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, img in stages.items():
        
        path = os.path.join(out_dir, f"{prefix}_{name}.png")
        cv2.imwrite(path, img)
        paths[name] = path
    return paths


