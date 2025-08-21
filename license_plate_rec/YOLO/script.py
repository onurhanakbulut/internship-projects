#import argparse
from pathlib import Path
import shutil
import cv2
import numpy as np



IMAGES_ROOT = Path("plaka.yolo/train/images")
LABELS_ROOT = Path("plaka.yolo/train/labels")
OUTPUT_ROOT = Path("DST")

INCLUDE_LABELS = True
MOVE_FILES = False



def parse(line: str):
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    try:
        return int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    
    except:
        return None
    
    



def classify_aspect(aspect: float) -> int:
    
    if 0.90 <= aspect < 2:
        return 1
    elif 2.00 <= aspect <= 5:
        return 0
    else:
        return -1
    


def move_or_copy(src: Path, dst: Path, move: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))
        






def main():

    out_sq = OUTPUT_ROOT / "square"
    out_rc = OUTPUT_ROOT / "rectangle"
    out_sq.mkdir(parents=True, exist_ok=True)
    out_rc.mkdir(parents=True, exist_ok=True)
    
    total, sq_count, rc_count, skipped = 0, 0, 0, 0
    
    for lbl in LABELS_ROOT.rglob("*.txt"):
        total += 1
        stem = lbl.stem
        
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            candidate = IMAGES_ROOT / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        
        if img_path is None:
            skipped += 1
            print(f"[WARN] Görsel yok: {lbl}")
            continue
        
        
        im = cv2.imread(str(img_path))
        if im is None:
            skipped += 1
            print(f"[WARN] Görüntü açılamadı: {img_path}")
            continue
        
        H, W = im.shape[:2]
        
        
        lines = lbl.read_text(encoding="utf-8").splitlines()
        labels=[]
        
        
        debug = True
        for line in lines:
            parsed = parse(line)
            if not parsed:
                continue
            
            c, x, y, w, h = parsed
            aspect = (w*W) / (h*H)
            if debug:
                print(f"[DBG] {lbl.name} aspect={aspect:.3f}")
            cls = classify_aspect(aspect)
            
            if cls != -1:
                labels.append(cls)
                
                
        if not labels:
            skipped += 1
            print(f"[WARN] Geçerli bbox yok: {lbl}")
            continue
        
        img_cls = 0 if any(l == 0 for l in labels) else 1
        
        target_dir = out_sq if img_cls == 1 else out_rc
        target_img = target_dir / img_path.name
        target_lab = target_dir / lbl.name
        
        move_or_copy(img_path, target_img, MOVE_FILES)
        if INCLUDE_LABELS:
            move_or_copy(lbl, target_lab, MOVE_FILES)
            
        if img_cls == 1:
            sq_count += 1
        else:
            rc_count += 1
            
            
    print("\n=== ÖZET ===")
    print(f"Toplam label   : {total}")
    print(f"Kare (square)  : {sq_count}")
    print(f"Dikdörtgen     : {rc_count}")
    print(f"Atlanan        : {skipped}")
    print(f"Hedef klasörler: {out_sq.resolve()} / {out_rc.resolve()}")
    print("=======================")    
    
    
    
if __name__ == "__main__":
    main()


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


