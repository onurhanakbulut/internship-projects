from ultralytics import YOLO
import cv2
import os
import numpy as np


model = YOLO('models/yolov8m.pt')
img = cv2.imread('data/car1.jpg')
os.makedirs('img_crops', exist_ok=True)








def crop_bottom_center(x1, y1, x2, y2, H, W, *,
                       bottom_frac=0.4, center_frac=0.8, pad=2):
    w = max(1, x2 - x1); h = max(1, y2 - y1)
    ny1 = y2 - int(bottom_frac * h); ny2 = y2
    keep_w = int(center_frac * w)
    dx = (w - keep_w) // 2
    nx1 = x1 + dx; nx2 = x2 - dx
    nx1 -= pad; ny1 -= pad; nx2 += pad; ny2 += pad
    nx1 = max(0, nx1); ny1 = max(0, ny1); nx2 = min(W, nx2); ny2 = min(H, ny2)
    if nx2 <= nx1 or ny2 <= ny1: return None
    return (nx1, ny1, nx2, ny2)





results = model(img, classes=[2, 5, 7], conf=0.50, verbose=False)
res = results[0] if isinstance(results, list) else results






H, W = img.shape[:2]
detections = []
if res.boxes is not None and len(res.boxes) > 0:
    for i, box in enumerate(res.boxes):
        
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        
        nb = crop_bottom_center(x1, y1, x2, y2, H, W,
                                bottom_frac=0.4, center_frac=0.8, pad=2)
        if not nb:
            continue
  
            

        


     


##############################
cv2.namedWindow('License Detect', cv2.WINDOW_NORMAL)
cv2.resizeWindow('License Detect', 960, 540)
annotated = res.plot()
cv2.imshow('License Detect', annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
