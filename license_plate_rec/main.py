import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO
import cv2
import numpy as np
import time
import glob
import csv
import easyocr
import re
from preprocess_ocr2 import  preprocess_plate
from deep_sort_realtime.deepsort_tracker import DeepSort



###STABLE LABEL
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

# =============================================================================
# reader = easyocr.Reader(['en'], gpu=True)
# ALLOW = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
# =============================================================================
# =============================================================================
# TR_PLATE_REGEX = re.compile(r"^\s*\d{2}[A-Z]{1,3}\d{2,4}\s*$", re.I)
# =============================================================================

# =============================================================================
# def ocr_easy(img):
#     import cv2
#     
#     if img.ndim == 2:
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     out = reader.readtext(img, detail=1, paragraph=False, allowlist=ALLOW)
#     if not out:
#         return "", 0.0
#     best = max(out, key=lambda x: float(x[2]))
#     return best[1].upper().replace(" ", "").replace("-", "").replace("_",""), float(best[2])
# =============================================================================




# =============================================================================
# video = 'data/fullvideo/output.avi'
# cap = cv2.VideoCapture(video)
# =============================================================================
model = YOLO('models/license_plate_detector.pt')


raw = np.load("roi_groups.npy", allow_pickle=True)


OCR_EVERY_N = 6
VIDEO_DIR = "data/fullvideo/test"
OUT_DIR = "plate_snaps"
PREP_DEBUG_DIR = "prep_debug"
SHAPE_CSV = "shape_summary.csv"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PREP_DEBUG_DIR, exist_ok=True)



def safe_crop(img, x1, y1, x2, y2, pad=6):  
    H, W = img.shape[:2]
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad); y2 = min(H, y2 + pad)
    if x2 > x1 and y2 > y1:
        return img[y1:y2, x1:x2].copy()
    return None


#POLYNOM SHAPE FIX
def to_single_poly(raw):
    arr = np.asarray(raw)
    if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[2] == 2:
        arr = arr[0]
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr.astype(np.int32)
    

poly = to_single_poly(raw)
assert poly.ndim == 2 and poly.shape[1] == 2, f"Beklenen (N,2), gelen: {raw.shape}"
pts_poly = poly.reshape(-1, 1, 2)



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
    
    if 0.90 <= aspect < 1.60:
        label = 'square'
    elif 1.60 <= aspect <= 5:
        label = 'rectangle'
    
    if label == "unknown":
        print("*******************Tanınamadı!!!!!*******************")
        
        
# =============================================================================
#     print(f"aspect={aspect:.2f} -> {label}")
# =============================================================================
    return label






cv2.namedWindow('GARAGE', cv2.WINDOW_NORMAL); cv2.resizeWindow('GARAGE', 960, 540)


video_files = glob.glob(os.path.join(VIDEO_DIR, "*.avi")) 
print(f"{len(video_files)} video bulundu.")



# =============================================================================
# ocr_done = False
# cooldown = 0
# =============================================================================



for video_path in video_files:
    print(f"\n========Isleniyor : {video_path}========")
    cap = cv2.VideoCapture(video_path)
    stabilizer = StableLabel()
    frame_id = 0
    rect_count = 0  
    square_count = 0
    mask = None
    
    tracker = DeepSort(
    max_age=20,        
    n_init=3,          
    nn_budget=100,     
    max_cosine_distance=0.3
    )
    unique_ids = set()     
    

    
    vehicle_log = []        
    
    
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    from collections import defaultdict
    
    counted_ids = set()     # sayılan ID'ler
    rect_cnt = square_cnt = unknown_cnt = 0

    id2stab = defaultdict(lambda: StableLabel(K=3))
    
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
        
        detections = []
        for box in res.boxes:
            conf = float(box.conf[0])
            if conf < 0.60:    
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(([x1, y1, x2, y2], conf, 0)) 
        
        
        tracks = tracker.update_tracks(detections, frame=frame)
        
        
        
        
        
        for t in tracks:
            if not t.is_confirmed():
                continue
            tid = int(t.track_id)
            track_id = int(t.track_id)
            ltrb = t.to_ltrb()  
            x1, y1, x2, y2 = map(int, ltrb)
            unique_ids.add(track_id)
            w, h = x2 - x1, y2 - y1
            
            
            shape_label = classify_plate((w, h))          # 'square' / 'rectangle' / 'unknown'
            stab = id2stab[tid]                           # StableLabel per track
            stable_label = stab.update(shape_label)   # kararlı etiketi güncelle
            
# =============================================================================
#             if tid not in counted_ids and stable_label is not None:
#                 counted_ids.add(tid)
#                 if stable_label == "rectangle":
#                     rect_cnt += 1
#                 elif stable_label == "square":
#                     square_cnt += 1
#                 else:
#                     unknown_cnt += 1
# =============================================================================
        
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(annotated, f"ID:{tid} {stable_label}",
                        (x1, max(0, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
# =============================================================================
#             vehicle_log.append({
#                 "id": tid,
#                 "label": stable_label if stable_label else "unknown",
#                 "frame": frame_id,
#             })
# =============================================================================
        
        
        
        
        
        if frame_id % OCR_EVERY_N == 0:
            now = time.time()
            detections = []
            for box in res.boxes:
                conf = float(box.conf[0])
                if conf < 0.45:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                w, h = x2 - x1, y2 - y1
                if w < 80 or h < 25:
                    continue
                
                
                crop = safe_crop(frame, x1, y1, x2, y2)
                if crop is None or crop.size == 0:
                    continue


                
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                fname = os.path.join(OUT_DIR, f"{base_name}_f{frame_id}.jpg")
                cv2.imwrite(fname, crop)
                
                
                label = classify_plate((w, h))
                stable_label = stabilizer.update(label)
                
# =============================================================================
#                 if stable_label == "rectangle":
#                     rect_count += 1
#                 elif stable_label == "square":
#                     square_count += 1
# =============================================================================
                
# =============================================================================
#                 pre, stages = preprocess_plate(
#                     crop,
#                     scale=2.2,
#                     use_clahe=True,
#                     do_denoise=True,
#                     do_sharpen=True,
#                     threshold="adaptive",
#                     morph=True,
#                     return_stages=True,        
#                 )
# =============================================================================
# =============================================================================
#                 pre, stages = preprocess_plate(crop, return_stages=True)
#                 TRY_KEYS = ["final", "thresh", "sharpen", "clahe", "scaled", "gray", "denoise", "morph"]
#                 
#                 results = []
#                 
#                 for k in TRY_KEYS:
#                     img = stages.get(k)
#                     if img is None:
#                         continue
#                     t, c = ocr_easy(img)
#                     results.append((k, t, c))
#                     print(f"[{k:7}] '{t}' conf={c:.2f}")
#                 
#                 
#                 if results:
#                     best_stage, best_text, best_conf = max(results, key=lambda x: x[2])
#                     print(f"[BEST  {best_stage}] '{best_text}' conf={best_conf:.2f}")
#                 
#                 
#                 
#                 best_text, best_conf, best_key = "", 0.0, None
#                 for k in TRY_KEYS:
#                     img = stages.get(k)
#                     if img is None: 
#                         continue
#                     t, c = ocr_easy(img)
# 
#                     #     continue
#                     if c > best_conf:
#                         best_text, best_conf, best_key = t, c, k
#                 
#                 print(f"[OCR {best_key}] '{best_text}' conf={best_conf:.2f}")
# =============================================================================
                                
                
                
# =============================================================================
#                 prefix = f"f{frame_id}_x{x1}y{y1}x{x2}y{y2}_{'unknown'}"
#                 stage_paths = save_preprocess_stages(stages, PREP_DEBUG_DIR, prefix)
# =============================================================================
                
                
            
        


        cv2.imshow('GARAGE', annotated)
        #print(f"Frame = {frame_id}")
        frame_id += 1  
           
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print(f"[SUMMARY] {base_name} icin benzersiz plaka/araç sayisi: {len(unique_ids)}")
 
# =============================================================================
#     ###########CSV
#     base_name = os.path.splitext(os.path.basename(video_path))[0]
#     newfile = not os.path.exists(SHAPE_CSV)
#     with open(SHAPE_CSV, "a", newline="", encoding="utf-8") as f:
#         w = csv.writer(f)
#         if newfile:
#             w.writerow(["video", "rectangle_count", "square_count", "total"])
#         w.writerow([base_name, rect_count, square_count, rect_count + square_count])
# =============================================================================
    

# =============================================================================
#     log_csv = "vehicle_log.csv"
#     newfile_log = not os.path.exists(log_csv)
#     with open(log_csv, "a", newline="", encoding="utf-8") as f:
#         w = csv.DictWriter(f, fieldnames=["video","id","label","frame"])
#         if newfile_log: w.writeheader()
#         for r in vehicle_log:
#             w.writerow({"video": base_name, **r})
# =============================================================================

    
    



cap.release()
cv2.destroyAllWindows()




# =============================================================================
# 20 video bulundu.
# 
# ========Isleniyor : data/fullvideo\192.168.1.130_ch5_20250507083002_20250507090001.avi========
# [SUMMARY] 192.168.1.130_ch5_20250507083002_20250507090001 icin benzersiz plaka/araç sayisi: 2
# 
# ========Isleniyor : data/fullvideo\192.168.1.130_ch5_20250507090001_20250507093001.avi========
# [SUMMARY] 192.168.1.130_ch5_20250507090001_20250507093001 icin benzersiz plaka/araç sayisi: 3
# 
# ========Isleniyor : data/fullvideo\192.168.1.130_ch5_20250507093001_20250507100001.avi========
# [SUMMARY] 192.168.1.130_ch5_20250507093001_20250507100001 icin benzersiz plaka/araç sayisi: 5
# 
# ========Isleniyor : data/fullvideo\192.168.1.130_ch5_20250507100001_20250507103001.avi========
# [SUMMARY] 192.168.1.130_ch5_20250507100001_20250507103001 icin benzersiz plaka/araç sayisi: 18
# 
# ========Isleniyor : data/fullvideo\192.168.1.130_ch5_20250507103001_20250507110001.avi========
# *******************Tanınamadı!!!!!*******************
# [SUMMARY] 192.168.1.130_ch5_20250507103001_20250507110001 icin benzersiz plaka/araç sayisi: 25
# 
# ========Isleniyor : data/fullvideo\192.168.1.130_ch5_20250507110001_20250507113001.avi========
# [SUMMARY] 192.168.1.130_ch5_20250507110001_20250507113001 icin benzersiz plaka/araç sayisi: 16
# 
# ========Isleniyor : data/fullvideo\192.168.1.130_ch5_20250507113001_20250507120003.avi========
# [SUMMARY] 192.168.1.130_ch5_20250507113001_20250507120003 icin benzersiz plaka/araç sayisi: 25
# 
# ========Isleniyor : data/fullvideo\192.168.1.130_ch5_20250507120003_20250507123002.avi========
# [SUMMARY] 192.168.1.130_ch5_20250507120003_20250507123002 icin benzersiz plaka/araç sayisi: 18
# 
# ========Isleniyor : data/fullvideo\192.168.1.130_ch5_20250507123002_20250507130002.avi========
# [SUMMARY] 192.168.1.130_ch5_20250507123002_20250507130002 icin benzersiz plaka/araç sayisi: 43
# 
# ========Isleniyor : data/fullvideo\192.168.1.130_ch5_20250507130002_20250507133001.avi========
# [SUMMARY] 192.168.1.130_ch5_20250507130002_20250507133001 icin benzersiz plaka/araç sayisi: 34
# 
# ========Isleniyor : data/fullvideo\192.168.1.130_ch5_20250507133001_20250507140002.avi========
# [SUMMARY] 192.168.1.130_ch5_20250507133001_20250507140002 icin benzersiz plaka/araç sayisi: 47
# 
# ========Isleniyor : data/fullvideo\192.168.1.130_ch5_20250507140002_20250507143002.avi========
# [SUMMARY] 192.168.1.130_ch5_20250507140002_20250507143002 icin benzersiz plaka/araç sayisi: 52
# 
# ========Isleniyor : data/fullvideo\192.168.1.130_ch5_20250507143002_20250507150002.avi========
# [SUMMARY] 192.168.1.130_ch5_20250507143002_20250507150002 icin benzersiz plaka/araç sayisi: 30
# 
# ========Isleniyor : data/fullvideo\192.168.1.130_ch5_20250507150002_20250507153002.avi========
# [SUMMARY] 192.168.1.130_ch5_20250507150002_20250507153002 icin benzersiz plaka/araç sayisi: 32
# 
# ========Isleniyor : data/fullvideo\192.168.1.130_ch5_20250507153002_20250507160002.avi========
# *******************Tanınamadı!!!!!*******************
# [SUMMARY] 192.168.1.130_ch5_20250507153002_20250507160002 icin benzersiz plaka/araç sayisi: 47
# 
# ========Isleniyor : data/fullvideo\192.168.1.130_ch5_20250507160002_20250507163002.avi========
# [SUMMARY] 192.168.1.130_ch5_20250507160002_20250507163002 icin benzersiz plaka/araç sayisi: 50
# 
# ========Isleniyor : data/fullvideo\192.168.1.130_ch5_20250507163002_20250507170001.avi========
# [SUMMARY] 192.168.1.130_ch5_20250507163002_20250507170001 icin benzersiz plaka/araç sayisi: 40
# 
# ========Isleniyor : data/fullvideo\192.168.1.130_ch5_20250507170001_20250507173001.avi========
# [SUMMARY] 192.168.1.130_ch5_20250507170001_20250507173001 icin benzersiz plaka/araç sayisi: 58
# 
# ========Isleniyor : data/fullvideo\192.168.1.130_ch5_20250507173001_20250507180001.avi========
# *******************Tanınamadı!!!!!*******************
# [SUMMARY] 192.168.1.130_ch5_20250507173001_20250507180001 icin benzersiz plaka/araç sayisi: 58
# 
# ========Isleniyor : data/fullvideo\192.168.1.130_ch5_20250507180001_20250507183001.avi========
# [SUMMARY] 192.168.1.130_ch5_20250507180001_20250507183001 icin benzersiz plaka/araç sayisi: 41
# =============================================================================


#################################################################################644-1 = 643 rectangle 
