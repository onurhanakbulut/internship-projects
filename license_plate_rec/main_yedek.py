import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO
import cv2
import numpy as np


class StableLabel:
    def __init__(self, K=3):
        self.K = K
        self.current = None     #kabul edilen etiket
        self.buffer = []    #ardisik etiketler
        self.consec = 0     #ardisik sayaci
        
        
        
    def update(self, new_label):
        if new_label == self.current:
            self.consec = 0 
            return self.current
        
        if not self.current:
            self.buffer.append(new_label)
            if len(self.buffer) >= self.K and all(l == new_label for l in self.buffer[-self.K:]):   #son 3 etiket new_labela eşit mi diye bakar
                self.current = new_label
                self.buffer.clear()
            return self.current
        
        
    
        if new_label == (self.buffer[-1] if self.buffer else None):
            self.consec += 1 
        else:
            self.consec = 1 
        self.buffer.append(new_label)
    
        
        if self.consec >= self.K:
            self.current = new_label
            self.consec = 0
        return self.current





video = 'data/fullvideo/192.168.1.130_ch5_20250507110001_20250507113001.avi'
cap = cv2.VideoCapture(video)
model = YOLO('models/license_plate_detector.pt')


raw = np.load("roi_groups.npy", allow_pickle=True)

SNAP_EVERY = 10
OCR_EVERY_N = 10
OUT_DIR = "plate_snaps"
os.makedirs(OUT_DIR, exist_ok=True)

def safe_crop(img, x1, y1, x2, y2, pad=6):  
    H, W = img.shape[:2]
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad); y2 = min(H, y2 + pad)
    if x2 > x1 and y2 > y1:
        return img[y1:y2, x1:x2].copy()
    return None



def to_single_poly(raw):
    arr = np.asarray(raw)
    if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[2] == 2:
        arr = arr[0]
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr.astype(np.int32)
    

poly = to_single_poly(raw)
assert poly.ndim == 2 and poly.shape[1] == 2, f"Beklenen (N,2), gelen: {raw.shape}"
pts_poly = poly.reshape(-1, 1, 2)

mask = None

def build_mask(h, w, pts):
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m, [pts], 255)
    return m

def classify_plate(bbox):

    w, h = bbox
    
    
    if h <= 0 or w <= 0:
        print("Geçersiz bbox boyutu")
        return "unknown"
    
    aspect = w / h
    label = "unknown"
    
    if 0.90 <= aspect < 1.80:
        label = 'square'
    elif 1.80 <= aspect <= 5:
        label = 'rectangle'
    
    if label == "unknown":
        print("*******************Tanınamadı!!!!!*******************")
        
        
# =============================================================================
#     print(f"aspect={aspect:.2f} -> {label}")
# =============================================================================
    return label






cv2.namedWindow('GARAGE', cv2.WINDOW_NORMAL); cv2.resizeWindow('GARAGE', 960, 540)


stabilizer = StableLabel()
frame_id = 0
# =============================================================================
# ocr_done = False
# cooldown = 0
# =============================================================================

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    
    
    
    
    H, W = frame.shape[:2]
    if mask is None:
        mask = build_mask(H, W, pts_poly)
        
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    res = model(masked, imgsz=640, verbose=False)[0]
    annotated = res.plot()
    
    
    
    
    boxes = []
    for box in res.boxes:
        conf = float(box.conf[0])
        if conf < 0.60:
            continue
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        boxes.append((x1, y1, x2, y2, conf))
        w = x2-x1
        h = y2-y1
        
        if boxes:
            
            x1, y1, x2, y2, conf = max(boxes, key=lambda t: t[4])
            w, h = x2 - x1, y2 - y1
        
            label = classify_plate((w, h))
            stable_label = stabilizer.update(label)
        
            
            if (stable_label in ("square", "rectangle")) and (frame_id % SNAP_EVERY == 0) and (w >= 80 and h >= 25):
                crop = safe_crop(frame, x1, y1, x2, y2, pad=6)
                if crop is not None:
                    fname = os.path.join(OUT_DIR, f"plate_{frame_id}.jpg")
                    cv2.imwrite(fname, crop)
                    print("Saved:", fname)
        
        label = classify_plate((w, h))
        stable_label = stabilizer.update(label)
        print("Stable: ", stable_label)
        

        






        
    


    cv2.imshow('GARAGE', annotated)
    #print(f"Frame = {frame_id}")
    frame_id += 1  
       
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
    



cap.release()
cv2.destroyAllWindows()