import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO
import cv2
import numpy as np
import glob
#from preprocess_ocr import  preprocess_plate, save_preprocess_stages




model = YOLO('models/license_plate_detector.pt')


raw = np.load("roi_groups.npy", allow_pickle=True)


SNAP_EVERY = 10
VIDEO_DIR = "data/fullvideo/test2"
OUT_DIR = "plate_snaps"
PREP_DEBUG_DIR = "prep_debug"


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





cv2.namedWindow('GARAGE', cv2.WINDOW_NORMAL); cv2.resizeWindow('GARAGE', 960, 540)


video_files = glob.glob(os.path.join(VIDEO_DIR, "*.avi")) 
print(f"{len(video_files)} video bulundu.")






for video_path in video_files:
    print(f"\n========Isleniyor : {video_path}========")
    cap = cv2.VideoCapture(video_path)
    mask = None
    frame_id = 0


    
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    
    
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
        
        

        
        if frame_id % SNAP_EVERY == 0:
            for det_idx, box in enumerate(res.boxes):
                conf = float(box.conf[0])
                if conf < 0.65:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                w, h = x2 - x1, y2 - y1
                if w < 80 or h < 25:
                    continue
                
                
                crop = safe_crop(frame, x1, y1, x2, y2)
                if crop is None or crop.size == 0:
                    continue

                
                
                
                unique = f"{base_name}_f{frame_id:06d}_i{det_idx:02d}_x{x1}y{y1}x{x2}y{y2}"
                fname = os.path.join(OUT_DIR, f"{unique}.jpg")
                cv2.imwrite(fname, crop)
                
                
                


                
            
        


        cv2.imshow('GARAGE', annotated)
        #print(f"Frame = {frame_id}")
        frame_id += 1  
           
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
 




cap.release()
cv2.destroyAllWindows()



