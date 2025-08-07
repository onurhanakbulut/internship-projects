from ultralytics import YOLO
import cv2


# =============================================================================
# 15 dakikalık sabit kamera görüntüsünde
# 
# YOLO ile araçları tespit etmek
# 
# ROI ile 4 şerit bölgesi tanımlamak
# 
# DeepSORT ile araçları takip edip aynı ID'nin tekrar sayılmasını önlemek
# 
# Her şeritte geçen araçları ayrı sayaçlarla saymak
# =============================================================================




model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture('data/traffic.mp4')


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    
    
    
    results = model(frame)
    
    annotated_frame = results[0].plot()
    
    cv2.imshow("Araç Tespiti", annotated_frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()