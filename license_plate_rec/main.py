from ultralytics import YOLO
import cv2
import numpy as np




cap = cv2.VideoCapture('data/plt.mp4')
model = YOLO('models/yolov8n.pt')



def img_process(frame, detections):         #gray filter->
    
    for (x1, y1, x2, y2) in detections:
        pro_res = []
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        
        
        
        
        
        
        
        pro_res.append({'bbox': (x1, y1, x2, y2), 'gray': gray})
        
        
        
    
    
    
    return pro_res
        
        





cv2.namedWindow('PLATE', cv2.WINDOW_NORMAL)
cv2.resizeWindow('PLATE', 960, 540)

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    results = model(frame, classes = [2, 5, 7], verbose=False)[0]
    annotated = results.plot()
    
    
    
    detections = []
    if hasattr(results, 'boxes') and results.boxes is not None and len(results.boxes) >  0:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append((x1, y1, x2, y2))
    
    
    #print(detections)
    
    

    
    
    
    
    
    
    cv2.imshow('PLATE', annotated)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    


cap.release()
cv2.destroyAllWindows()
    
    
    
    
    

    
    
    
    

