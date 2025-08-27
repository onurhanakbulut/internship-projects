import cv2
from plate_cls import deskew

from pathlib import Path

IN_DIR = "data/ocr/ocr/"
IMG_EXTS = {".jpg"}

def iter_image_paths(root: str, recursive: bool = True):
    rootp = Path(root)
    it = rootp.rglob("*") if recursive else rootp.glob("*")
    for p in it:
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield str(p)

if __name__ == "__main__":
    paths = sorted(iter_image_paths(IN_DIR, recursive=True))
    print(f"{len(paths)} görsel bulundu.")
    for p in paths:
        
        pass


