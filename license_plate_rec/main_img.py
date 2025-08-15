from ultralytics import YOLO
import cv2
import os
import numpy as np
from plate_select import extract_and_save_plate_crops





model = YOLO('models/yolov8m.pt')
img = cv2.imread('data/car9.jpg')

os.makedirs('img_crops', exist_ok=True)



def remove_small_white(items, min_area=150, min_w=0, min_h=0):
    for it in items:
        m = it['morph']
   
        

        num, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        out = np.zeros_like(m)
        for i in range(1, num):  
            x, y, w, h, area = stats[i]
            if area >= int(min_area) and w >= int(min_w) and h >= int(min_h):
                out[lab == i] = 255  # sadece yeterince büyük beyazları koru
    
        it['cleaned_morph'] = out
  
    return items








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


def median_gray(items, ksize: int=5):
    if ksize % 2 == 0:
        ksize += 1
    
    for it in items:
        g = it['gray']
        it['median'] = cv2.medianBlur(g, ksize)
    return items

def median_filter(items, source1, source2, ksize: int=3):
    if ksize % 2 == 0:
        ksize += 1
    
    for it in items:
        g = it[source1]
        it[source2] = cv2.medianBlur(g, ksize)
    return items




# =============================================================================
# def otsu(items):
#     flag = cv2.THRESH_BINARY
#     for it in items:
#         src = it['median']
#         
#         _, binimg = cv2.threshold(src, 0, 255, flag | cv2.THRESH_OTSU)
#         it['bin'] = binimg
#     return items
# =============================================================================

# =============================================================================
# def binarize_adaptive(items, block_ratio=0.10, C=6, invert=False, use_clahe=True):
#     
#     for it in items:
#         src = it.get('median', it['gray'])
# 
#         if use_clahe:
#             clh = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8,8))
#             src = clh.apply(src)
# 
#         h = src.shape[0]
#         block = max(3, int(round(block_ratio * h)))
#         if block % 2 == 0: block += 1
# 
#         th_flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
#         it['bin'] = cv2.adaptiveThreshold(
#             src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, th_flag, block, C
#         )
#     return items
# =============================================================================



    

# =============================================================================
# def binary(items):
#     for it in items:
#         src = it.get('median')
#         
#         
#         _, binimg = cv2.threshold(src, 120, 255, cv2.THRESH_BINARY)
#         
#         it['bin'] = binimg
#         
#     return items
# =============================================================================
        


def binarize_adaptive(items, block_ratio=0.10, C=12,
                      method='gaussian',      # 'gaussian' | 'mean'
                      invert=None,            # None => otomatik, True/False => sabit
                      use_clahe=True,
                      post_median=3,
                      post_close_iter=1):
    """
    items[i]['bin'] üretir. invert=None ise otomatik (BINARY/BINARY_INV) seçer.
    """
    for it in items:
        src = it.get('median', it['gray'])   # median yoksa gray kullan
        # güvenli uint8
        if src.dtype != np.uint8:
            src = cv2.normalize(src, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # (opsiyonel) yerel kontrast artır
        if use_clahe:
            clh = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
            src = clh.apply(src)

        # pencere boyutu: ROI yüksekliğinin ~%10'u (tek sayı)
        h = src.shape[0]
        block = max(3, int(round(block_ratio * h)))
        if block % 2 == 0: block += 1

        # yöntem seçimi
        adapt = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method.lower().startswith('g') \
                else cv2.ADAPTIVE_THRESH_MEAN_C

        def _th(flag):
            b = cv2.adaptiveThreshold(src, 255, adapt, flag, block, C)
            if post_median and post_median >= 3:
                k = post_median + (1 - post_median % 2)  # tek yap
                b = cv2.medianBlur(b, k)
            if post_close_iter and post_close_iter > 0:
                b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8),
                                     iterations=int(post_close_iter))
            return b

        if invert is None:
            # Otomatik kutup: iki sonucu da hesapla, beyaz oranı 0.25–0.75 aralığa yakın olanı seç
            b1 = _th(cv2.THRESH_BINARY)
            b2 = _th(cv2.THRESH_BINARY_INV)
            wr1 = (b1 > 0).mean()
            wr2 = (b2 > 0).mean()
            # 0.5'e yakın olanı tercih et
            binimg = b1 if abs(wr1 - 0.5) <= abs(wr2 - 0.5) else b2
            info = {'mode': 'adaptive', 'auto_invert': True, 'block': block, 'C': C,
                    'wr': (b1 > 0).mean() if binimg is b1 else (b2 > 0).mean()}
        else:
            flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
            binimg = _th(flag)
            info = {'mode': 'adaptive', 'auto_invert': False, 'invert': bool(invert),
                    'block': block, 'C': C, 'wr': (binimg > 0).mean()}

        it['bin'] = binimg
        it['bin_info'] = info  # debug istersen bakarsın

    return items








def morph(items):
    
    def _kernel(sz):
        w, h = sz
        if w % 2 == 0: w += 1
        if h % 2 == 0: h += 1
        return cv2.getStructuringElement(cv2.MORPH_RECT, (w, h))
    
    

    for it in items:
        b = it['bin'].copy()
        
        
# =============================================================================
#         b = cv2.erode(b, _kernel((5,5)), iterations=1)
# =============================================================================
        
        
# =============================================================================
#         b = cv2.dilate(b, _kernel((5,5)), iterations=1)
# =============================================================================
        
        b = cv2.morphologyEx(b, cv2.MORPH_OPEN, _kernel((3,3)), iterations=1) 
    
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, _kernel((3,3)), iterations=1) 
        
        
            
        it['morph'] = b
    return items
    


###***********************************************************************
def save_img(items, prefix, out_dir='img_crops'):
    os.makedirs(out_dir, exist_ok=True)
    
    for i, it in enumerate(items):
        path = os.path.join(out_dir, f'{prefix}_{i:02d}.png')  
        ok = cv2.imwrite(path, it[prefix])
        if not ok:
            print('Yazılamadı:', path)




def crop_bottom_center(x1, y1, x2, y2, H, W, *, 
                       bottom_frac=0.6,   
                       center_frac=0.6,   
                       pad=0):            
    # boyutlar
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    # alt bant
    ny1 = y2 - int(bottom_frac * h)
    ny2 = y2

    # yatayda orta kısım
    keep_w = int(center_frac * w)
    dx = (w - keep_w) // 2
    nx1 = x1 + dx
    nx2 = x2 - dx

    # pad ve kadraja sıkıştır
    nx1 -= pad; ny1 -= pad; nx2 += pad; ny2 += pad
    nx1 = max(0, nx1); ny1 = max(0, ny1)
    nx2 = min(W, nx2); ny2 = min(H, ny2)

    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return (nx1, ny1, nx2, ny2)














#############################***************************************
results = model(img, classes=[2, 5, 7], conf=0.50, verbose=False)
res = results[0] if isinstance(results, list) else results


H, W = img.shape[:2]
detections = []
if res.boxes is not None and len(res.boxes) > 0:
    for i, box in enumerate(res.boxes):
        
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        
        nb = crop_bottom_center(x1, y1, x2, y2, H, W,
                                bottom_frac=0.4,   # alt %60
                                center_frac=0.8,  # orta %75
                                pad=2)             # ufak güvenlik payı
        
        if nb:
            detections.append(nb)
            
        
        conf = float(box.conf[0])
        cls = int(box.cls[0])

# =============================================================================
#         h = max(1, y2 - y1)
#         new_y1 = y2 - int(0.5 * h)
#         new_y1 = max(y1, min(new_y1, y2 - 1))
#         detections.append((x1, new_y1, x2, y2))
# =============================================================================

        crop = img[y1:y2, x1:x2]
        name = res.names[cls] if hasattr(res, 'names') else str(cls)
        out_path = f'img_crops/{i:02d}_{name}_{conf:.2f}.jpg'
        cv2.imwrite(out_path, crop)
else:
    print('Arac Tespiti Yok')

##################
items = gray_filter(img, detections)
save_img(items, 'gray')

items = median_gray(items)
save_img(items, 'median')

items = binarize_adaptive(items)
save_img(items, 'bin')

items = median_filter(items, 'bin', 'blurred_bin')
save_img(items, 'blurred_bin')

items = morph(items)
save_img(items, 'morph')



clean = remove_small_white(items, min_area=300)   # eşiği görüntüne göre ayarla


save_img(items, 'cleaned_morph')



n_saved = extract_and_save_plate_crops(items, img, mask_key='cleaned_morph',
                                       out_dir='img_crops/plates', prefix='plate')
print(f"Kaydedilen plaka sayısı: {n_saved}")








##############################
cv2.namedWindow('License Detect', cv2.WINDOW_NORMAL)
cv2.resizeWindow('License Detect', 960, 540)
annotated = res.plot()
cv2.imshow('License Detect', annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

