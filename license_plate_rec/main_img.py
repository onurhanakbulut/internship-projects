from ultralytics import YOLO
import cv2
import os


model = YOLO('models/yolov8m.pt')
img = cv2.imread('data/car1.jpg')

os.makedirs('img_crops', exist_ok=True)



############################################### FUNCTİONS
def gray_filter(img, detections):
    results = []
    for (x1, y1, x2, y2) in detections:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        roi = img[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        results.append({'bbox': (x1, y1, x2, y2), 'roi': roi, 'gray': gray})

    return results


def save_img(items, prefix, out_dir='img_crops'):
    os.makedirs(out_dir, exist_ok=True)
    
    for i, it in enumerate(items):
        path = os.path.join(out_dir, f'{prefix}_{i:02d}.png')  
        ok = cv2.imwrite(path, it[prefix])
        if not ok:
            print('Yazılamadı:', path)


















#############################
results = model(img, classes=[2, 5, 7], conf=0.50, verbose=False)
res = results[0] if isinstance(results, list) else results



detections = []
if res.boxes is not None and len(res.boxes) > 0:
    for i, box in enumerate(res.boxes):
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        h = max(1, y2 - y1)
        new_y1 = y2 - int(0.5 * h)
        new_y1 = max(y1, min(new_y1, y2 - 1))
        detections.append((x1, new_y1, x2, y2))

        crop = img[new_y1:y2, x1:x2]
        name = res.names[cls] if hasattr(res, 'names') else str(cls)
        out_path = f'img_crops/{i:02d}_{name}_{conf:.2f}.jpg'
        cv2.imwrite(out_path, crop)
else:
    print('Arac Tespiti Yok')


items = gray_filter(img, detections)
save_img(items, 'gray')




##############################
cv2.namedWindow('License Detect', cv2.WINDOW_NORMAL)
cv2.resizeWindow('License Detect', 960, 540)
annotated = res.plot()
cv2.imshow('License Detect', annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

