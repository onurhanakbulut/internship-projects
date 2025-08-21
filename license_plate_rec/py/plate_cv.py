import cv2
import os
import numpy as np


# =================== DEBUG VE YARDIMCI FONKSİYONLAR ===================

#main fonksiyonu
# =============================================================================
#     test_on_roi(
#             img, nb, idx=idx, debug_root="debug_out",
#             clahe_clip=1.6, clahe_tile=(8,8),
#             block_ratio=0.10, C=10,
#             h_ratio=0.30, v_ratio=0.06  # True yaparsan aşamaları pencerede de görürsün
#         )
# =============================================================================



def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p



def _to_bgr(img: np.ndarray) -> np.ndarray | None:
    if img is None:
        return None
    if img.ndim == 3:
        return img
    
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)




def _is_binary(m: np.ndarray) -> bool:
    if m is None or m.ndim !=2:
        return False
    vals = np.unique(m)
    return (len(vals) <= 2) and (0 in vals) and (255 in vals)


def _save_dbg(root: str, roi_idx: int, step: str, img:np.ndarray, txt: str | None = None) -> None:
    d = _ensure_dir(os.path.join(root, f"roi_{roi_idx:02d}"))
    
    if img is None:
        return
    
    im = _to_bgr(img).copy()
    if txt:
        cv2.putText(im, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
        
    path = os.path.join(d, f"{step}.png")
    cv2.imwrite(path, im)
    
    
def white_ratio(mask: np.ndarray) -> float:
    if mask is None or mask.size == 0:
        return np.nan
    m = mask
    
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        
    if not _is_binary(m):
        _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    
    return float((m > 0).mean())



def cc_count(mask: np.ndarray, min_area_frac=0.002) -> int:
    if mask is None or mask.size == 0:
        return 0
    m = mask
    if not _is_binary(m):
        _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    
    H, W = m.shape[:2]
    min_area = int(min_area_frac * H * W)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)

    cnt = 0
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cnt += 1
    return cnt



######################################3

def prep_for_threshold(gray, use_gauss=True, use_bilateral=False, use_tophat=False):
    g = gray
    if use_gauss:
        g=cv2.GaussianBlur(g, (5,5), 0)
    if use_bilateral:
        g =cv2.bilateralFilter(g, d=5, sigmaColor=30, sigmaSpace=5)
    if use_tophat:
        H = g.shape[0]
        k = max(9, int(0.08 * H))                        # ROI yüksekliğinin ~%8'i
        if k % 2 == 0: k += 1
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        g = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, se)
    return g






def gray_clahe(roi, clipLimit=1.6, tile=(8,8)):
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clh = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tile)       ###--düzensiz aydınlatma için
    g = clh.apply(g)
    g = prep_for_threshold(g, use_gauss=True, use_bilateral=False, use_tophat=False)
    
    return g

# =============================================================================
# def adaptive_threshold(gray, block_ratio=0.10, C=10):
#     h = gray.shape[0]
#     block = max(3, int(round(block_ratio * h)))                          ###---pencere boyutuna oranla çalışmasını sağlar.
#     if block % 2 == 0:
#         block += 1
#         
#     b1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, C)      ###--- pikseller arası farka bakar
#     b2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, C)  ###--- C = eşik
#     wr1, wr2 = (b1>0).mean(), (b2>0).mean()     ##white ratio = wr
#     
#     select_bin = b1 if abs(wr1-0.5) <= abs(wr2-0.5) else b2                ###---0.5e yakın olanı seç
#     
#     return select_bin, {'block':block, 'C':C, 'wr': (select_bin>0).mean()}
# =============================================================================
    
    
    

def adaptive_threshold_auto(gray, block_ratio=0.14, C_candidates=(6,8,10,12,14), target_wr=0.55, method='gaussian'):
    
    h = gray.shape[0]
    block = max(3, int(round(block_ratio * h)))
    if block % 2 == 0: block += 1
    adapt = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method=='gaussian' else cv2.ADAPTIVE_THRESH_MEAN_C

    best = None
    for C in C_candidates:
        b1 = cv2.adaptiveThreshold(gray, 255, adapt, cv2.THRESH_BINARY,     block, C)
        b2 = cv2.adaptiveThreshold(gray, 255, adapt, cv2.THRESH_BINARY_INV, block, C)
        for b in (b1, b2):
            wr = (b > 0).mean()
            sc = abs(wr - target_wr)
            if (best is None) or (sc < best[0]):   
                best = (sc, wr, C, b)
    _, wr, C_best, bin_sel = best
    return bin_sel, {'block': block, 'C': C_best, 'wr': wr}



    
    
# =============================================================================
# def morph_bridge(bin_img, base_open=1, base_close=1, h_ratio=0.30, v_ratio=0.06):
#     b = bin_img.copy()
#     b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=base_open)
#     b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=base_close)
#     
#     H = b.shape[0]
#     cw = max(3, int(round(h_ratio*H))); cw += (1 - cw%2)
#     ch = max(1, int(round(v_ratio*H))); ch += (1 - ch%2)
#     
#     k = cv2.getStructuringElement(cv2.MORPH_RECT, (cw, ch))
#     bridged = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k, iterations=1)
#     
#     return bridged, {'cw':cw, 'ch':ch}
# =============================================================================



def morph_bridge_adaptive(bin_img, wr_bin,
                          base_open=1, base_close=1,
                          h_ratio_range=(0.18, 0.28),
                          v_ratio_range=(0.04, 0.07),
                          max_cw_frac=0.22,
                          cw_hard_cap=41):
    """
    wr_bin yüksekse → kernel küçülür; düşükse büyür. Ayrıca cw üst sınırlarla kısıtlanır.
    Zaten iyi durumdaysa (wr_base ~0.45–0.85 ve cc<=2) köprüleme yapılmaz.
    """
    b = bin_img.copy()
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8), iterations=base_open)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=base_close)

    def _wr(m):
        if m.ndim == 3: m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        if not _is_binary(m): _, m = cv2.threshold(m,127,255,cv2.THRESH_BINARY)
        return float((m>0).mean())
    def _cc(m, min_area_frac=0.002):
        if not _is_binary(m): _, m = cv2.threshold(m,127,255,cv2.THRESH_BINARY)
        H,W = m.shape[:2]; min_area = int(min_area_frac*H*W)
        num, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        return sum(1 for i in range(1,num) if stats[i, cv2.CC_STAT_AREA] >= min_area)

    wr_base, cc_base = _wr(b), _cc(b)
    if 0.45 <= wr_base <= 0.85 and cc_base <= 2:
        return b, {"mode":"base_only","cw":0,"ch":0,"wr_base":wr_base,"cc_base":cc_base}

    h_min,h_max = h_ratio_range; v_min,v_max = v_ratio_range
    h_ratio = np.interp(wr_bin, [0.30,0.55,0.80], [h_max,(h_min+h_max)/2,h_min])
    v_ratio = np.interp(wr_bin, [0.30,0.55,0.80], [v_max,(v_min+v_max)/2,v_min])

    H = b.shape[0]
    cw = max(3, int(round(h_ratio*H)));  cw += (1 - cw%2)
    ch = max(1, int(round(v_ratio*H)));  ch += (1 - ch%2)

    cw = min(cw, int(max_cw_frac*H))
    cw = min(cw, cw_hard_cap)

    k  = cv2.getStructuringElement(cv2.MORPH_RECT, (cw, ch))
    bridged = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k, iterations=1)
    return bridged, {"mode":"bridged","cw":cw,"ch":ch,"wr_base":wr_base,"cc_base":cc_base}

    


def test_on_roi(img_bgr, roi_xyxy, idx=0, debug_root="debug_out",
                clahe_clip=1.6, clahe_tile=(8,8),
                block_ratio=0.10, C=10,
                h_ratio=0.30, v_ratio=0.06):
   
    x1, y1, x2, y2 = map(int, roi_xyxy)
    H, W = img_bgr.shape[:2]
    x1 = max(0, min(W-1, x1)); x2 = max(0, min(W, x2))
    y1 = max(0, min(H-1, y1)); y2 = max(0, min(H, y2))
    if x2 <= x1 or y2 <= y1:
        print(f"[ROI {idx}] geçersiz kutu:", roi_xyxy)
        return

    roi = img_bgr[y1:y2, x1:x2]
    _save_dbg(debug_root, idx, "00_roi", roi)


    gray = gray_clahe(roi, clipLimit=clahe_clip, tile=clahe_tile)
    _save_dbg(debug_root, idx, "01_gray", gray, f"CLAHE clip={clahe_clip} tile={clahe_tile}")


    bin_sel, info = adaptive_threshold_auto(gray, block_ratio=0.14, C_candidates=(6,8,10,12,14), target_wr=0.55)
    wr = info["wr"]
    _save_dbg(debug_root, idx, "02_bin", bin_sel, f"wr={wr:.2f} block={info['block']} C={info['C']}")


    morph, minfo = morph_bridge_adaptive(bin_sel, wr, base_open=1, base_close=1)
    wr_m = white_ratio(morph); cc_m = cc_count(morph)
    _save_dbg(debug_root, idx, "03_morph", morph,
          f"mode={minfo.get('mode')} cw={minfo['cw']} ch={minfo['ch']} "
          f"wr_base={minfo['wr_base']:.2f} cc_base={minfo['cc_base']} wr={wr_m:.2f} cc={cc_m}")


    print(f"[ROI {idx}] "
          f"CLAHE clip={clahe_clip} tile={clahe_tile} | "
          f"block={info['block']} C={info['C']} wr={wr:.2f} | "
          f"cw={minfo['cw']} ch={minfo['ch']} wr_m={wr_m:.2f} cc_m={cc_m}")










