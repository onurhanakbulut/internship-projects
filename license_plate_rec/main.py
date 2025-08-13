from ultralytics import YOLO
import cv2
import numpy as np


from save_debug import save_steps



cap = cv2.VideoCapture('data/plt.mp4')
model = YOLO('models/yolov8n.pt')



# =============================================================================
# def img_process(frame, detections, clip_limit=3.0, tile_grid_size=(8, 8),
#                 k_w_ratio=0.18, k_h_ratio=0.06, block_ratio=0.08, C=10, method='mean', invert=False, auto_fix_polarity=True,
#                 close_w_ratio=0.30, close_h_ratio=0.06, dilate_iter=1):          #gray filter->blur->clahe->
#     
# # =============================================================================
# #     dets = np.asarray(detections).reshape(-1, 4).astype(int)
# # =============================================================================
#     clh = cv2.createCLAHE(clip_limit, tile_grid_size)
#     
#     
#     results = []
#     for (x1, y1, x2, y2) in detections:
#         
#         roi = frame[y1:y2, x1:x2]
#         
# # =============================================================================
# #         if roi.size == 0:
# #             continue
# # =============================================================================
#         
#         #---gray
#         gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#         #---blur
#         blur = cv2.GaussianBlur(gray, (3,3), 0)
#         #---clahe
#         clahe = clh.apply(blur)
#         
#         #---blackhat
#         h1 = clahe.shape[0]
#         kw = max(9, int(round(k_w_ratio * h1)))
#         kh = max(3, int(round(k_h_ratio * h1)))
#         
#         if kh % 2 == 0: kh += 1
#         if kw % 2 == 0: kw += 1
#         
#         se = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
#         bh = cv2.morphologyEx(clahe, cv2.MORPH_BLACKHAT, se)
#         
#         #---threshold
#         h2 = bh.shape[0]
#         block = max(3, int(round(block_ratio * h2)))
#         if block % 2 == 0:
#             block += 1
#             
#         th_flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
#         if method.lower() == 'gaussian':
#             binimg = cv2.adaptiveThreshold(bh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, th_flag, block, C)
#             
#         else:
#             binimg = cv2.adaptiveThreshold(bh, 255, cv2.ADAPTIVE_THRESH_MEAN_C, th_flag, block, C)
#             
#             
#         
# # =============================================================================
# #         if auto_fix_polarity:
# #             if (binimg > 0).mean() > 0.5:
# #                 binimg = cv2.bitwise_not(binimg)
# # =============================================================================
#         #####*********burayı sonradna ekledim belki kaldırırım
#         white_ratio = (binimg > 0).mean()
#         if white_ratio > 0.5:
#             binimg = cv2.bitwise_not(binimg)
#             
#         binimg = cv2.morphologyEx(binimg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)
# 
# 
# 
# 
# 
# 
#     
#         #---horizontal closing
#         h3 = binimg.shape[0]
#         ch = max(1, int(round(0.008 * h3)))
#         cw = max(5, int(round(0.015 * h3)))
#         
#         if ch % 2 ==0: ch += 1
#         if cw % 2 == 0: cw += 1
# 
#         ch = 1
#         cw = 5
#         
#         se_close = cv2.getStructuringElement(cv2.MORPH_RECT, (cw, ch))
#         merged = cv2.morphologyEx(binimg, cv2.MORPH_CLOSE, se_close, iterations=1)
#         
#         if dilate_iter > 0:
#             se_d = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#             merged = cv2.dilate(merged, se_d, iterations=int(dilate_iter))
#             
#             
# # =============================================================================
# #         print("white_ratio bin   :", (binimg  > 0).mean())
# #         print("white_ratio merged:", (merged > 0).mean())
# # =============================================================================
# 
#         
# 
#         
#         
#         results.append({'bbox': (x1, y1, x2, y2),
#                         '1gray': gray,
#                         '2blur': blur,
#                         '3clahe':clahe,
#                         '4blackhat':bh,
#                         '5binary':binimg,
#                         '6merged':merged
#                         })
#         
#         
#         
#     
#     
#     
#     return results
# =============================================================================
        



def gray_filter(frame, detections):
    results = []
    for(x1, y1, x2, y2) in detections:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            continue
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        results.append({'bbox': (x1, y1, x2, y2), 'roi': roi, 'gray': gray})
        
    return results



def median_filter(items, ksize: int=7):
    if ksize % 2 == 0:
        ksize += 1
    
    for it in items:
        g = it['gray']
        it['median'] = cv2.medianBlur(g, ksize)
    return items




# =============================================================================
# def sobel_filter(items, ksize: int = 3):
#     for it in items:
#         m = it.get('median')
#         sx = cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=ksize)
#         sy = cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=ksize)
#         mag = cv2.magnitude(sx, sy)
#         it['sobel_mag'] = cv2.convertScaleAbs(mag)
#     return items
# =============================================================================


# =============================================================================
# def sobel_filter(items, *,
#                  mode='x',
#                  ksize=5,
#                  pre_blur=True,
#                  blur_ks=5,
#                  blur_sigma=1.2,
#                  q_low=90.0,
#                  q_high=99.8,
#                  gamma=2.2,
#                  keep_top_pct=5.0,
#                  min_cc_area=40):
#     
#     for it in items:
#         src = it.get('median')
#         
#         
#         if pre_blur:
#             src = cv2.GaussianBlur(src, (blur_ks, blur_ks), blur_sigma)
#             
#         
#         gx = gy = None
#         if mode in('x', 'mag'):
#             gx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=ksize)
#         if mode in ('y', 'mag'):
#             gy = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=ksize)
#             
#             
#         if mode == 'x':
#             g = np.abs(gx)
#         elif mode == 'y':
#             g = np.abs(gy)
#         else:
#             g = cv2.magnitude(gx, gy)
#         
#         
#         lo = np.percentile(g, q_low)
#         hi = np.percentile(g, q_high)
#         g = np.maximum(g - lo, 0.0)
#         scale = max(hi - lo, 1e-6)
#         g = g / scale
#         g = np.clip(g, 0.0, 1.0)
#         g = g ** gamma
#         sobel_u8 = (g * 255.0).astype(np.uint8)
#         
#         
#         
#         
#         if keep_top_pct and keep_top_pct > 0:
#             t = np.percentile(sobel_u8, 100.0 - keep_top_pct)
#             edge = (sobel_u8 >= t).astype(np.uint8) * 255
#             # küçük parçaları temizle
#             num, labels, stats, _ = cv2.connectedComponentsWithStats(edge, 8)
#             mask = np.zeros_like(edge)
#             for i in range(1, num):
#                 if stats[i, cv2.CC_STAT_AREA] >= int(min_cc_area):
#                     mask[labels == i] = 255
#             sobel_u8 = mask
# 
#         it['sobel_mag'] = sobel_u8
#     
#     
#     return items
# =============================================================================
    
    
    


def otsu(items):
    flag = cv2.THRESH_BINARY
    for it in items:
        src = it['median']
        # Otsu global eşik
        _, binimg = cv2.threshold(src, 0, 255, flag | cv2.THRESH_OTSU)
        it['bin'] = binimg
    return items




# =============================================================================
# def morph(items,
#          erode_ks=(3,3), erode_iter=2,
#          dilate_ks=(5,5), dilate_iter=2,
#          open_ks=(5,5), open_iter=2,
#          close_ks=(5,5), close_iter=3):
# =============================================================================

# =============================================================================
# def morph(items,
#          erode_ks=(5,5), erode_iter=1,
#          dilate_ks=(5,5), dilate_iter=2,
#          open_ks=(5,5), open_iter=2,
#          close_ks=(5,5), close_iter=5):
# =============================================================================

      
# =============================================================================
# def morph(items,
#          erode_ks=(5,5), erode_iter=1,
#          dilate_ks=(7,7), dilate_iter=2,
#          open_ks=(9,9), open_iter=4,
#          close_ks=(5,5), close_iter=5):
# =============================================================================
    

# =============================================================================
# def morph(items,
#          erode_ks=(3,3), erode_iter=1,
#          dilate_ks=(5,5), dilate_iter=2,
#          open_ks=(5,5), open_iter=3,
#          close_ks=(5,5), close_iter=5):
# =============================================================================


# =============================================================================
# def morph(items,
#          erode_ks=(3,3), erode_iter=1,
#          dilate_ks=(5,5), dilate_iter=2,
#          open_ks=(5,5), open_iter=3,
#          close_ks=(5,5), close_iter=5):
#     
#     def _kernel(sz):
#         w, h = sz
#         if w % 2 == 0: w += 1
#         if h % 2 == 0: h += 1
#         return cv2.getStructuringElement(cv2.MORPH_RECT, (w, h))
#     
#     
#     k_er = _kernel(erode_ks)
#     k_di = _kernel(dilate_ks)
#     k_op = _kernel(open_ks)
#     k_cl = _kernel(close_ks)
#     
#     for it in items:
#         b = it['bin'].copy()
#         
#         if erode_iter > 0:
#             b = cv2.erode(b, k_er, iterations=erode_iter)
#         if dilate_iter > 0:
#             b = cv2.dilate(b, k_di, iterations=dilate_iter)
#         if open_iter > 0:
#             b = cv2.morphologyEx(b, cv2.MORPH_OPEN, k_op, iterations=open_iter)
#         if close_iter > 0:
#             b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k_cl, iterations=close_iter)
#         if erode_iter > 0:
#             b = cv2.erode(b, _kernel((5,5)), iterations=6)
#         if dilate_iter > 0:
#             b = cv2.dilate(b, _kernel((7,7)), iterations=3)    
#          
#         it['morph'] = b
#     return items
# =============================================================================






def morph(items,
         erode_ks=(5,5), erode_iter=2,
         dilate_ks=(3,3), dilate_iter=2,
         open_ks=(3,3), open_iter=1,
         close_ks=(3,3), close_iter=1):
    
    def _kernel(sz):
        w, h = sz
        if w % 2 == 0: w += 1
        if h % 2 == 0: h += 1
        return cv2.getStructuringElement(cv2.MORPH_RECT, (w, h))
    
    
    k_er = _kernel(erode_ks)
    k_di = _kernel(dilate_ks)
    k_op = _kernel(open_ks)
    k_cl = _kernel(close_ks)
    
    for it in items:
        b = it['bin'].copy()
        
        if erode_iter > 0:
            b = cv2.erode(b, k_er, iterations=erode_iter)
        if dilate_iter > 0:
            b = cv2.dilate(b, k_di, iterations=dilate_iter)
        if open_iter > 0:
            b = cv2.morphologyEx(b, cv2.MORPH_OPEN, k_op, iterations=open_iter) 
        if close_iter > 0:
            b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k_cl, iterations=close_iter) 
        if dilate_iter > 0:
            b = cv2.dilate(b, k_di, iterations=1)    
        
        it['morph'] = b
    return items




    

cv2.namedWindow('PLATE', cv2.WINDOW_NORMAL)
cv2.resizeWindow('PLATE', 960, 540)



def process_and_save_every_10(frame, detections, frame_idx):
   
    items = gray_filter(frame, detections)
    if not items:
        return

    
    items = median_filter(items)

    
# =============================================================================
#     items = sobel_filter(items, ksize=3)
# =============================================================================

    
    items = otsu(items)
    
    items = morph(items)

    
    if frame_idx % SAVE_EVERY_N == 0:
        step_order = ['gray', 'median', 'sobel_mag', 'bin', 'morph']  # sırayla kaydeder
        save_steps(items, out_root='debug_out', frame_idx=frame_idx, step_order=step_order)



frame_idx=0
SAVE_EVERY_N = 10







########################
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    results = model(frame, classes = [2, 5, 7], conf=0.50, verbose=False)[0]
    annotated = results.plot()
    
    
    
    detections = []
    if hasattr(results, 'boxes') and results.boxes is not None and len(results.boxes) >  0:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            h = max(1, y2 - y1)
            new_y1 = y2 - int(0.5 * h)
            new_y1 = max(y1, min(new_y1, y2 -1))
            detections.append((x1, new_y1, x2, y2))
    
    
    #print(detections)
    process_and_save_every_10(frame, detections, frame_idx)
    

    
    

    
    
    
    
    
    
    cv2.imshow('PLATE', annotated)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_idx += 1 
    


cap.release()
cv2.destroyAllWindows()
    
    
    
    
    

    
    
    
    

