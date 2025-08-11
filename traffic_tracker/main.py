from ultralytics import YOLO
import cv2
from utils import get_tracker
import math
import numpy as np
import torch
import time



# =============================================================================
# YOLO ile araçları tespit etmek----------
# 
# ROI ile 4 şerit bölgesi tanımlamak---------
# 
# DeepSORT ile araçları takip edip aynı ID'nin tekrar sayılmasını önlemek------------
# 
# Her şeritte geçen araçları ayrı sayaçlarla saymak---------
#
# optimizasyon



# =============================================================================
# Daha hafif model kullan
# yolov8m.pt → yolov8n.pt veya yolov8s.pt
# 
# img size küçült
# 1280p kareyi tam beslemek yerine imgsz=640 (veya 512).
# 
# Half precision (FP16)
# GPU’da ciddi hız kazandırır: half=True
# 
# Gösterim boyutunu küçült
# cv2.resizeWindow(...) yaptın, iyi; gerekirse frame’i de küçült.
#
# deepsort optimizasyonu ve embedder = gpu
# =============================================================================
# =============================================================================




roi_groups = list(np.load("roi_groups.npy", allow_pickle=True))


FRAME_TTL = 300
CLEAN_EVERY = 120

frame_idx = 0
last_seen = {}
counted_ids = [dict() for _ in range(len(roi_groups))]

prev_time = time.time()
fps=0


def points_in_polygon(pt, poly):
    return cv2.pointPolygonTest(np.array(poly, np.int32), pt, False) >= 0



lane_count = [0] * len(roi_groups)      # lane_count = [0,0,0,0]




##ROI KONUMLARI PRINT
# =============================================================================
# for i, group in enumerate(roi_groups):
#     print(f"ROI {i+1}: {group}")
# =============================================================================



model = YOLO('yolov8s.pt')
half = True if torch.cuda.is_available() else False

tracker = get_tracker()

cap = cv2.VideoCapture('data/traffic.mp4')



cv2.namedWindow("YOLO + DEEPSORT", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO + DEEPSORT", 960, 540)



allowed_labels = ['car', 'truck', 'bus', 'motorcycle']

color_map = {
    'car' : (0, 255, 0),
    'truck' : (255, 0, 0),
    'bus' : (0, 0, 255),
    'motorcycle' : (255, 255, 0),
    'heavy' : (0, 128, 255)
    }






while cap.isOpened():
    t0 = time.perf_counter()    #DELAY
    ret, frame = cap.read()
    t1 = time.perf_counter()    #DELAY
    if not ret:
        break
    
    
    
#####---------------ROI SINIRLARI------------------------------  
    for group in roi_groups:
        for i in range(len(group)):
            #pt1 = group[i]
            #pt2 = group[(i+1) % len(group)]

            pt1 = tuple(group[i])
            pt2 = tuple(group[(i + 1) % len(group)])
            cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
            
            
            
            
    results = model(frame, imgsz=640, half=half, verbose = False, classes=[2, 3, 5, 7])
    t2 = time.perf_counter()        #DELAY
    
    
    
    
    
#############--------------------YOLO OUTPUT --->>> DEEPSORT INPUT
    detections =  []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[class_id]
    
        if label not in allowed_labels:
            continue
        
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        
        detections.append(([x1, y1, w, h], conf, label))
        
    
        
    tracks = tracker.update_tracks(detections, frame=frame)
    t3 = time.perf_counter()        ##DELAY
    
    
#####------------------TRACK DONGUSU------------------------
    frame_centers = []  
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        
        track_id = track.track_id
        ltrb = track.to_ltrb()      #left top right bottom
        x1, y1, x2, y2 = map(int, ltrb)
        
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        
        
        
######--------------------ROI GİRDİ ÇIKTI KONTROLÜ VE SAYAÇ----------------------
        
        for i, poly in enumerate(roi_groups):
            inside = points_in_polygon((cx, cy), poly)
            
            
            if inside:
                if track_id not in counted_ids[i]:
                    lane_count[i] += 1
                    counted_ids[i][track_id] =  frame_idx
                    
            else:
                if track_id in counted_ids[i]:
                    counted_ids[i].pop(track_id, None)
        
        total_car_count = sum(lane_count)
# =============================================================================
#         inside_idx = [i for i, poly in enumerate(roi_groups)    #list comprehension 
#                       if points_in_polygon((cx, cy), poly)]
# =============================================================================
        


        
#########----------------------100 PİKSELDEN YAKINSA SİL-----------------------     
      
        too_close = False
        
        for (px, py) in frame_centers:
            if math.hypot(cx - px, cy - py) < 100:      ###öklid
                too_close = True
                break
            
            
        if too_close:
            continue
        
        frame_centers.append((cx, cy))
        
        
        
        last_seen[track_id] = frame_idx
        
        

                
##########----------------------OPENCV YAZI İŞLEMLERİ-----------------------
        
        label = track.get_det_class() or 'Vehicle'
        
        if label in ['truck', 'bus']:
            label = 'heavy'
        
        color = color_map.get(label, (255, 255, 255))
        text = f"{label.upper()} #{track_id}"
        
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
        
        
        cv2.putText(frame, text, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        cv2.circle(frame, (cx - 40, cy + 50), 12, (0, 0, 255), -1)
        
        for i, c in enumerate(lane_count):
            cv2.putText(frame, f"Serit {i+1}: {c}",(20, 40+30*i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50,255,50),2)
            
        cv2.putText(frame, f"Toplam Arac: {total_car_count}", (20, 40 + 30*len(lane_count)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)    
            
        
        
        
        
#####-------------------120 FRAMEDE BİR ESKİ IDLERİ TEMİZLE------------------
        if frame_idx % CLEAN_EVERY == 0:
            stale_ids = []
            for tid, f in last_seen.items():
                if frame_idx - f > FRAME_TTL:
                    stale_ids.append(tid)
                    
            for tid in stale_ids:
                last_seen.pop(tid, None)
                
    
        t4 = time.perf_counter()        #DELAY
        print(f"R:{(t1-t0)*1000:.1f}ms  D:{(t2-t1)*1000:.1f}ms  T:{(t3-t2)*1000:.1f}ms  Draw:{(t4-t3)*1000:.1f}ms")       
                    
    
            
########---------------------------------FPS-------------------------------
    current_time = time.time()
    elapsed = current_time - prev_time
    if elapsed > 0:
        fps=1 / elapsed
    else:
        fps = 0
    prev_time = current_time
        
        
########------------------------FPS YAZILARI-------------------------------
    frame_width = frame.shape[1]
    fps_text = f"FPS: {fps:.2f}"
    (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        
    cv2.rectangle(frame, (frame_width-tw-20, 10), (frame_width - 10, 10+th+10), (0,0,0), -1)
    cv2.putText(frame, fps_text, (frame_width - tw - 15, 10 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)        
#*******************************    
    frame_idx += 1  
    
    
    
    
    
    
    
    
    
    cv2.imshow("YOLO + DEEPSORT", frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    



#=================================================================
print("\n==== ŞERİT BAZLI SONUÇLAR ====")
for i, c in enumerate(lane_count):
    print(f"Şerit {i+1}: {c}")

total_car_count = sum(lane_count)
print(f"Toplam Araç: {total_car_count}")


with open("sonuclar.txt", "w", encoding="utf-8") as f:
    f.write("==== ŞERİT BAZLI SONUÇLAR ====\n")
    for i, c in enumerate(lane_count):
        f.write(f"Şerit {i+1}: {c}\n")
    f.write(f"Toplam Araç: {total_car_count}\n")    

#=================================================================    
    
cap.release()
cv2.destroyAllWindows()



