import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  

import cv2
import easyocr
import glob
import csv
import re
import numpy as np

# --- Ayarlar ---
IN_DIR  = "plate_snaps"                 
OUT_DIR = "plate_snaps_annotated"       
CSV_OUT = "plate_snaps_results.csv"     
LANGS   = ['en']                  
USE_GPU = True                          

os.makedirs(OUT_DIR, exist_ok=True)


re_tr_plate = re.compile(r'^[0-9]{2}[A-ZÇĞİÖŞÜ]{1,3}[0-9]{2,4}$', re.IGNORECASE)


def preprocess(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape[:2]
    if max(h, w) < 200:
        g = cv2.resize(g, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(g)
    g = cv2.medianBlur(g, 3)
    return g

def draw_poly(img, bbox, color=(0,255,0), thick=2):
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and isinstance(bbox[0], (list, tuple)):
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(img, [pts], True, color, thick)

reader = easyocr.Reader(LANGS, gpu=USE_GPU)


paths = sorted(glob.glob(os.path.join(IN_DIR, "*.jpg")) + 
               glob.glob(os.path.join(IN_DIR, "*.png")) + 
               glob.glob(os.path.join(IN_DIR, "*.jpeg")))

with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "text", "confidence"])

    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue

        gray = preprocess(img)
        ocr_res = reader.readtext(gray, detail=1, paragraph=False)

        best_txt, best_conf, best_bbox = None, -1.0, None

        for item in ocr_res:
            
            bbox, txt, conf = None, None, None
            if isinstance(item, (list, tuple)):
                if len(item) == 3:
                    bbox, txt, conf = item
                elif len(item) == 2:
                    txt, conf = item
                elif len(item) == 1:
                    txt = item[0]
            elif isinstance(item, str):
                txt = item

            if txt is None:
                continue
            clean = "".join(ch for ch in txt if ch.isalnum())
            if not clean:
                continue
            if conf is None:
                conf = 0.0

            
            score = conf + (0.15 if re_tr_plate.match(clean) else 0.0)
            if score > best_conf:
                best_conf = score
                best_txt = clean
                best_bbox = bbox

        
        writer.writerow([os.path.basename(p), best_txt or "", f"{best_conf:.3f}" if best_conf >= 0 else ""])

        
        annotated = img.copy()
        if best_txt:
            if best_bbox:
                draw_poly(annotated, best_bbox, (0,255,0), 2)
            cv2.putText(annotated, f"{best_txt} ({best_conf:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
        out_path = os.path.join(OUT_DIR, os.path.basename(p))
        cv2.imwrite(out_path, annotated)
        print(f"[OK] {os.path.basename(p)} -> {best_txt} (conf~{best_conf:.2f})")
