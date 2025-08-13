# lp_save.py
import os
import cv2
import numpy as np

def _ensure_u8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.bool_:
        return (img.astype(np.uint8)) * 255
    if img.dtype in (np.float32, np.float64):
        m, M = float(img.min()), float(img.max())
        if M <= 1.0:  # 0..1 float
            return (img * 255.0).astype(np.uint8)
        if 0.0 <= m <= 255.0 and 0.0 <= M <= 255.0:  # zaten 0..255 aralığı
            return img.astype(np.uint8)
        if M > m:
            img = (img - m) / (M - m) * 255.0
        else:
            img = np.zeros_like(img, dtype=np.float32)
        return img.astype(np.uint8)
    if img.dtype in (np.int16, np.uint16, np.int32):
        return np.clip(img, 0, 255).astype(np.uint8)
    return img.astype(np.uint8)

def save_steps(results, out_root='debug_out', frame_idx=0, step_order=None):
    """
    results: img_process çıktısı listesi
    out_root/frame_XXXXXX/det_YY/ içine yalnızca görüntü alanlarını .png kaydeder
    """
    base = os.path.join(out_root, f"frame_{frame_idx:06d}")
    for di, r in enumerate(results):
        det_dir = os.path.join(base, f"det_{di:02d}")
        os.makedirs(det_dir, exist_ok=True)

        # sadece ndarray olan anahtarları yaz
        if step_order:
            order = [k for k in step_order if (k in r and isinstance(r[k], np.ndarray))]
        else:
            order = [k for k, v in r.items() if isinstance(v, np.ndarray)]

        for step in order:
            img = r[step]
            cv2.imwrite(os.path.join(det_dir, f"{step}.png"), _ensure_u8(img))
