import cv2
import numpy as np
from pathlib import Path


def _counter_rotation_deg_from_rect(rect):
    # rect: ((cx,cy), (w,h), ang)  ang ∈ (-90, 0]
    (w, h) = rect[1]
    ang = float(rect[2])

    # Uzun kenarı yataya getir
    if w < h:
        ang += 90.0

    rot = ang  # KARŞI döndürme

    # En küçük eşdeğer döndürmeye indir (−90, 90] aralığına sıkıştır)
    if rot > 90.0:
        rot -= 180.0
    elif rot <= -90.0:
        rot += 180.0
    return rot










def normalize_src_orientation(src: np.ndarray) -> np.ndarray:
    """TL,TR üstte; BL,BR altta ve sol gerçekten solda kalsın."""
    src = np.asarray(src, dtype=np.float32)
    TL, TR, BR, BL = src
    # 180° ters mi?
    if (TL[1] + TR[1]) * 0.5 > (BL[1] + BR[1]) * 0.5:
        src = np.array([BL, BR, TR, TL], dtype=np.float32)
    # ayna mı?
    TL, TR, BR, BL = src
    if (TL[0] + BL[0]) * 0.5 > (TR[0] + BR[0]) * 0.5:
        src = np.array([TR, TL, BL, BR], dtype=np.float32)
    return src




def order_points(pts: np.ndarray) -> np.ndarray:
        pts = pts.astype(np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).ravel()
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        return np.array([tl, tr, br, bl], dtype=np.float32)
    
 
def _find_contours(img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE):
    
    res = cv2.findContours(img, mode, method)
    cnts = res[0] if len(res) == 2 else res[1]
    return cnts
 
    
    
def deskew(roi_bgr: np.ndarray, *,
           do_perspective: bool = True,
           min_contour_area: int = 120, 
           canny1: int = 60,
           canny2: int = 120,
           debug: bool = False
           ):
    
    
    H, W = roi_bgr.shape[:2]
    dbg = {}
    
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    ######***************
    gray_f = cv2.bilateralFilter(gray, 7, 50, 50)
    edges = cv2.Canny(gray_f, canny1, canny2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    #######*************
    
    if debug:
        dbg["edges"] = edges.copy()
    
    
    cnts = _find_contours(edges)
    cnt = max(cnts, key=cv2.contourArea) if cnts else None
    if cnt is not None and cv2.contourArea(cnt) < min_contour_area:
        cnt = None
        
    ####minarearect
    
    angle = 0.0
    if cnt is not None:
        rect = cv2.minAreaRect(cnt)             # ((cx,cy), (w,h), angle)  angle ∈ (-90,0]
        angle = _counter_rotation_deg_from_rect(rect)
# =============================================================================
#         (w, h) = rect[1]
#         ang = rect[2]
#         
#         if w < h:
#             ang += 90.0
#         angle = ang
#         
# =============================================================================
        
        
    #Deskew
    M = cv2.getRotationMatrix2D((W/2.0, H/2.0), angle, 1.0)           ##goruntunun merkezini secer , dondurmek icin
    rotated = cv2.warpAffine(roi_bgr, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)        #dondur
    
# =============================================================================
#     if not do_perspective:
#         return (rotated, dbg) if debug else rotated
# =============================================================================
    
    if debug:
        dbg["angle_deg"] = float(angle)  # burada artık 150-180 gibi değer görmeyeceksin
    
    #4 kose perspektif duzeltme
    gry = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    thr = cv2.threshold(gry, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=2)
    
    if debug:
        dbg["thr"] = thr.copy()
        
    cnts2= _find_contours(thr)
    if not cnts2:
        return (rotated , dbg) if debug else rotated
    
    cnt2 = max(cnts2, key = cv2.contourArea)
    peri = cv2.arcLength(cnt2, True)
    approx = cv2.approxPolyDP(cnt2, 0.02*peri, True)
    
    
    ####4 kose degilse
    if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(4, 2).astype(np.float32)   # <-- reshape
    else:
            box = cv2.boxPoints(cv2.minAreaRect(cnt2))
            quad = box.astype(np.float32)
    
    src = order_points(quad)
    src = normalize_src_orientation(src)
    
    
    #boyut
    
    widthA = np.linalg.norm(src[2] - src[3])
    widthB = np.linalg.norm(src[1] - src[0])
    heightA = np.linalg.norm(src[1] - src[2])
    heightB = np.linalg.norm(src[0] - src[3])
    
    
    maxW = int(max(widthA, widthB))
    maxH = int(max(heightA, heightB))
    maxW = max(maxW, 40)
    maxH = max(maxH, 20)
    
    
    aspect = maxW / max(1, maxH)
    if aspect > 2.0:
        tgtH = 64
        tgtW = int(tgtH * min(max(aspect, 3.0), 6.0))  # 3–6 aralığına sıkıştırıyo
    else:
        tgtH = 96
        tgtW = int(tgtH * min(max(aspect, 1.2), 2.2))

    dst = np.array([[0, 0],
                    [tgtW - 1, 0],
                    [tgtW - 1, tgtH - 1],
                    [0, tgtH - 1]], dtype=np.float32)

    M_p = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        rotated, M_p, (tgtW, tgtH),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    if warped.shape[1] < warped.shape[0]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    if debug:
        dbg.update({
            "angle_deg": angle,
            "quad_src": src.copy(),
            "rotated": rotated.copy(),
        })
        return warped, dbg
    else:
        return warped
    
    
    
    
    
# =============================================================================
# IN_DIR   = "data/ocr/ocr/"              
# OUT_DIR  = "data/ocr/ocr/prep_out"                 
# 
# WARP_DIR = Path(OUT_DIR, "warped")
# DBG_DIR  = Path(OUT_DIR, "debug")
# WARP_DIR.mkdir(parents=True, exist_ok=True)
# DBG_DIR.mkdir(parents=True, exist_ok=True)
#      
# roi = cv2.imread("plate_snaps/192.168.1.130_ch5_20250507163002_20250507170001_f000190_i00_x295y730x698y898.jpg")
# warped, dbg = deskew(roi, debug=True)
# # =============================================================================
# # cv2.imshow("edges", dbg["edges"]); cv2.imshow("thr", dbg["thr"])
# # =============================================================================
# cv2.imshow("warped", warped); cv2.waitKey(0); cv2.destroyAllWindows()
# =============================================================================
     #****************************************************************************************************************************   
    

import numpy as np


def list_images(in_dir, patterns=('*.jpg')):
    p = Path(in_dir)
    files = []
    for pat in patterns:
        files.extend(sorted(p.rglob(pat)))
    return [str(f) for f in files]


def process_folder_deskew(in_dir: str,
                          out_dir: str,
                          *,
                          save_debug: bool = True,
                          do_perspective: bool = True,
                          min_contour_area: int = 120,
                          canny1: int = 60,
                          canny2: int = 120):

    in_paths = list_images(in_dir)
    out_dir = Path(out_dir)
    warp_dir = out_dir / "warp"
    dbg_dir  = out_dir / "debug"
    warp_dir.mkdir(parents=True, exist_ok=True)
    if save_debug:
        dbg_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for ipath in in_paths:
        


        p = Path(ipath)
        if not (p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}):
            # print(f"skip non-image: {ipath}")  # istersen logla
            continue
        
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            # print(f"unreadable image: {ipath}")
            continue

        
        if img is None:
            continue

        if save_debug:
            warped, dbg = deskew(
                img,
                do_perspective=do_perspective,
                min_contour_area=min_contour_area,
                canny1=canny1,
                canny2=canny2,
                debug=True
            )
        else:
            warped = deskew(
                img,
                do_perspective=do_perspective,
                min_contour_area=min_contour_area,
                canny1=canny1,
                canny2=canny2,
                debug=False
            )
            dbg = None

        stem = Path(ipath).stem
        out_warp = warp_dir / f"{stem}_warped.png"
        cv2.imwrite(str(out_warp), warped)

        saved = {"input": ipath, "warped": str(out_warp)}

        # Debug görselleri (varsa) kaydet
        if save_debug and isinstance(dbg, dict):
            if "edges" in dbg:
                cv2.imwrite(str(dbg_dir / f"{stem}_1_edges.png"), dbg["edges"])
            if "thr" in dbg:
                cv2.imwrite(str(dbg_dir / f"{stem}_2_thr.png"), dbg["thr"])
            if "rotated" in dbg:
                rotated = dbg["rotated"].copy()
                # dörtgen bulunduysa çiz
                if "quad_src" in dbg:
                    pts = dbg["quad_src"].astype(np.int32)
                    cv2.polylines(rotated, [pts], True, (0, 255, 0), 2)
                # açı bilgisi varsa yaz
                ang = dbg.get("angle_deg", None)
                if ang is not None:
                    cv2.putText(rotated, f"angle={ang:.2f} deg",
                                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imwrite(str(dbg_dir / f"{stem}_3_rotated.png"), rotated)

            saved["debug_dir"] = str(dbg_dir)

        results.append(saved)

    return results

        


results = process_folder_deskew(
     in_dir="plate_snaps", 
     out_dir="plates4",
     save_debug=True,          
     do_perspective=True,
    canny1 = 60, canny2=120,      
)
print(f"{len(results)} görsel işlendi.")
