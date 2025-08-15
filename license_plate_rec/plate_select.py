# plate_select.py (OPTIMIZED)
import cv2, os, numpy as np

# ---------- Yardımcılar: ikili garanti + temizlik ----------
def _ensure_binary(m):
    if m is None or m.size == 0:
        return None
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    if m.dtype != np.uint8:
        m = cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return m

def _drop_small_white(m, min_area_frac=0.0008, min_w_frac=0.02, min_h_frac=0.02):
    H, W = m.shape[:2]
    min_area = int(min_area_frac * H * W)
    min_w    = int(min_w_frac * W)
    min_h    = int(min_h_frac * H)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m)
    for i in range(1, num):
        x,y,w,h,area = stats[i]
        if area >= min_area and w >= min_w and h >= min_h:
            out[lab == i] = 255
    return out

def _fill_small_black_holes(m, max_hole_frac=0.001):
    H, W = m.shape[:2]
    inv = cv2.bitwise_not(m)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(inv, 8)
    out = m.copy()
    for i in range(1, num):
        x,y,w,h,area = stats[i]
        touches = (x==0 or y==0 or x+w==W or y+h==H)
        if (not touches) and area <= int(max_hole_frac * H * W):
            out[lab == i] = 255
    return out

def _bridge_horizontal(m, h_ratios=(0.26,0.32,0.36), v_ratios=(0.05,0.07), iters=(1,)):
    """Yatay geniş closing kernel’leri ile harfleri plaka bloğuna köprüle."""
    H, W = m.shape[:2]
    outs = [m]
    for cw_r in h_ratios:
        for ch_r in v_ratios:
            cw = max(3, int(round(cw_r * H)))
            ch = max(1, int(round(ch_r * H)))
            if cw % 2 == 0: cw += 1
            if ch % 2 == 0: ch += 1
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (cw, ch))
            for it in iters:
                outs.append(cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=it))
    return outs

def _get_mask_from_item(it, keys=('cleaned_morph', 'morph', 'bin')):
    for k in keys:
        v = it.get(k, None)
        if isinstance(v, np.ndarray) and v.size > 0:
            return v, k
    return None, None

# ---------- Zor karelerde: ROI gri’den sağlam maske üret (fallback) ----------
def _fallback_mask_from_gray(gray):
    g = gray
    if g.ndim == 3: g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    if g.dtype != np.uint8:
        g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clh = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8,8))
    g = clh.apply(g)
    h = g.shape[0]
    block = max(3, int(round(0.10 * h)))
    if block % 2 == 0: block += 1
    b1 = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY,     block, 8)
    b2 = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, block, 8)
    wr1, wr2 = (b1>0).mean(), (b2>0).mean()
    m = b1 if abs(wr1-0.5) <= abs(wr2-0.5) else b2
    m = cv2.medianBlur(m, 3)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)
    return m

# ---------- Aday seçim (genişletilmiş tolerans + ağırlıklı skor) ----------
def _best_component_in_mask(m,
                            area_frac=(0.002, 0.30),   # %0.2–%30 (geniş aralık)
                            ar_long=(3.4, 6.0),        # uzun plaka
                            ar_square=(1.15, 2.0),     # kare plaka
                            rect_min=0.50,             # contourArea/(w*h)
                            prefer_bottom=True,
                            prefer_center=True):
    H, W = m.shape[:2]
    roiA = float(H*W)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None, -1.0

    best, best_score = None, -1.0
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w == 0 or h == 0: 
            continue
        frac = (w*h) / roiA
        if not (area_frac[0] <= frac <= area_frac[1]): 
            continue

        ar = (w/h) if w >= h else (h/w)
        ar_ok = (ar_long[0] <= ar <= ar_long[1]) or (ar_square[0] <= ar <= ar_square[1])
        if not ar_ok: 
            continue

        rect = cv2.contourArea(c) / float(w*h)
        if rect < rect_min:
            continue

        # konumsal puan
        pos_bottom = (y + h) / H if prefer_bottom else 0.5
        cx = x + 0.5*w
        pos_center = 1.0 - abs(cx - (W*0.5)) / (W*0.5) if prefer_center else 0.5

        # doluluk puanı (aday içi beyaz oranı)
        wr_local = (m[y:y+h, x:x+w] > 0).mean()
        wr_score = max(0.0, 1.0 - abs(wr_local - 0.75)/0.75)

        target_ar = 4.5 if ar >= 3.0 else 1.5
        ar_score  = max(0.0, 1.0 - abs(ar - target_ar)/target_ar)

        score = 0.40*ar_score + 0.30*rect + 0.15*pos_bottom + 0.10*pos_center + 0.05*wr_score
        if score > best_score:
            best_score, best = score, (x,y,w,h)
    return best, best_score

# ---------- Ana: maskeden varyant üret → en iyi adayı seç ----------
def _find_plate_in_mask(mask, debug=False):
    m = _ensure_binary(mask)
    if m is None: 
        return None

    # Maske aşırı beyaz/siyah ise önce temizle
    wr = (m>0).mean()
    if wr < 0.02 or wr > 0.98:
        # bu durumda bile şansımızı deneyeceğiz ama varyantlara daha çok bel bağlayacağız
        pass

    # temel temizlik + delik doldurma
    base = _drop_small_white(m, 0.0006, 0.015, 0.015)
    base = _fill_small_black_holes(base, 0.001)

    variants = [base] + _bridge_horizontal(base, (0.24,0.30,0.36), (0.05,0.08), (1,))

    best, best_score = None, -1.0
    for v in variants:
        cand, score = _best_component_in_mask(v)
        if cand is not None and score > best_score:
            best, best_score = cand, score

    if debug and best is None:
        print("[plate_select] uyarı: maskede uygun aday bulunamadı.")

    return best  # None veya (x,y,w,h)

# ---------- Dış API: ROI -> global crop kaydet ----------
def extract_and_save_plate_crops(items, full_image, mask_key='cleaned_morph',
                                 out_dir='img_crops', prefix='plate',
                                 debug=False):
    os.makedirs(out_dir, exist_ok=True)
    H, W = full_image.shape[:2]
    saved = 0

    for i, it in enumerate(items):
        # 1) Maske önceliği: cleaned_morph -> morph -> bin
        mask, used_key = _get_mask_from_item(it, keys=(mask_key, 'morph', 'bin'))
        if debug:
            print(f"\n[ROI {i}] bbox={it.get('bbox')} mask_key={used_key}")

        # 2) Bu maskede ara
        local = _find_plate_in_mask(mask, debug=debug)

        # 3) Bulamadıysan: ROI gri’den fallback maske üret ve tekrar dene
        if local is None and 'gray' in it:
            if debug: print(f"[ROI {i}] fallback: gray'den maske üretiyorum")
            m2 = _fallback_mask_from_gray(it['gray'])
            local = _find_plate_in_mask(m2, debug=debug)

        if local is None:
            if debug: print(f"[ROI {i}] aday yok, atlandı.")
            continue

        x,y,w,h = local
        px, py = int(0.05*w), int(0.08*h)

        x1g, y1g, x2g, y2g = it['bbox']
        gx1 = max(0, x1g + x - px); gy1 = max(0, y1g + y - py)
        gx2 = min(W, x1g + x + w + px); gy2 = min(H, y1g + y + h + py)
        if gx2 <= gx1 or gy2 <= gy1:
            if debug: print(f"[ROI {i}] global kutu geçersiz: {(gx1,gy1,gx2,gy2)}")
            continue

        crop = full_image[gy1:gy2, gx1:gx2]
        out_path = os.path.join(out_dir, f"{prefix}_{i:02d}.png")
        ok = cv2.imwrite(out_path, crop)
        if debug:
            print(f"[ROI {i}] kaydet: {out_path} => {ok}")
        if ok:
            it['plate_bbox_roi']    = (x,y,w,h)
            it['plate_bbox_global'] = (gx1,gy1,gx2,gy2)
            saved += 1

    if debug:
        print(f"\n[plate_select] Toplam kaydedilen plaka: {saved}")
    return saved
