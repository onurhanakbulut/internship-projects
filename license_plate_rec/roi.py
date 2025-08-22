import cv2
import numpy as np


roi_points = []
roi_groups = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points.append((x,y))
        print(f'point added : [{x},{y}]')


        if len(roi_points) == 4:
            roi_groups.append(roi_points.copy())
            print(f'ROI tamamlandı: {roi_points}')
            roi_points.clear()




video_path =  'data/fullvideo/192.168.1.130_ch5_20250507083002_20250507090001.avi'

cap = cv2.VideoCapture(video_path)



ret, frame = cap.read()
cap.release()



cv2.namedWindow('DRAW ROI', cv2.WINDOW_NORMAL)
cv2.resizeWindow("DRAW ROI", 1280, 720)
cv2.setMouseCallback('DRAW ROI', mouse_callback)


while True:
    temp_frame = frame.copy()
    
    
    
    for group in roi_groups:
        for i in range (len(group)):
            pt1 = group[i]
            pt2 = group[(i + 1) % len(group)]
            cv2.line(temp_frame, pt1, pt2, (255, 0, 0), 2)
            
            
            
    for i in range(len(roi_points) - 1):
        pt1 = roi_points[i]
        pt2 = roi_points[i + 1]
        cv2.line(temp_frame, pt1, pt2, (0, 255, 255), 2)
    
    
    
    
    for point in roi_points:
        cv2.circle(temp_frame, point, 5, (0, 0, 255), -1)
        
    
    cv2.imshow('DRAW ROI', temp_frame)
    
    key = cv2.waitKey(1) & 0XFF
    if key == ord("q"):
        break
    
    elif key == ord("s") and roi_groups:
        np.save("roi_groups.npy", np.array(roi_groups, dtype=object))
        print("ROI groups saved (roi_groups.npy)")
        break
    
cv2.destroyAllWindows()
            
            
        















