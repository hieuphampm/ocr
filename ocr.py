from __future__ import annotations

import argparse
import base64
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import sys

HERE = Path(__file__).resolve().parent  # everything that ships with the script lives next to it

try:
    from dotenv import load_dotenv

    load_dotenv(HERE / ".env")  # next to this script, whatever the current folder is
    load_dotenv()  # and/or a .env in the current folder; real environment variables always win
except ImportError:  # .env is optional
    pass

# Windows consoles default to a legacy code page: Vietnamese text in print() must not crash the run
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
W, H = 1240, 1754  # canonical page size (all coordinates below live in this space)

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")
MODEL_NAME = os.environ.get("MODEL_NAME")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

LOW_CONF = 0.6  # OCR confidence below this -> listed in needs_review
ALIGN_MIN_BOXES = 28  # of 43 printed squares must be located, otherwise the alignment is rejected


def validate_config() -> None:
    if not OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY is missing (set it in .env or the environment).")
    if not OPENAI_BASE_URL:
        raise SystemExit("OPENAI_BASE_URL is empty.")
    if not MODEL_NAME:
        raise SystemExit("MODEL_NAME is empty.")


CHECKBOXES = {
    "1_Nhan": [
        ("Nhăn trán", 140.2, 461.9),
        ("Cau mày", 349.0, 458.1),
        ("Nhăn mắt", 644.6, 455.7),
        ("Mũi thỏ", 902.1, 454.7),
        ("Nhăn má", 140.2, 506.3),
        ("Nhăn khóe miệng", 350.4, 503.3),
        ("Cơ cằn", 644.5, 501.7),
        ("Nhăn nọng", 903.5, 500.9),
        ("Nhăn cổ", 139.4, 551.3),
        ("Mồ hôi nách", 350.3, 549.2),
        ("Khác (ghi rõ)", 645.9, 547.0),
    ],
    "2_Day": [
        ("Trán baby", 140.1, 663.3),
        ("Mũi", 484.0, 660.7),
        ("Cằm", 819.6, 659.6),
        ("Thái dương", 140.1, 712.7),
        ("Ngấn má", 483.9, 711.8),
        ("Rãnh mũi", 819.5, 710.8),
        ("Giọt lệ", 140.8, 762.8),
        ("Ốp má trên", 483.8, 762.0),
        ("Ốp má dưới", 819.4, 760.5),
        ("Ngạch cằm", 141.5, 814.4),
        ("Line xương hàm", 485.1, 814.0),
        ("Môi", 819.3, 811.6),
        ("Dái tai", 141.5, 865.2),
        ("Bàn tay", 485.1, 864.0),
        ("Bàn chân", 819.2, 862.7),
        ("Bikini mu", 142.2, 916.0),
        ("Bikini môi lớn", 486.5, 915.0),
        ("Ốp tóc mai", 819.8, 913.8),
    ],
    "3_THD": [
        ("THD mũi", 143.6, 1024.9),
        ("THD môi", 397.0, 1024.5),
        ("THD cằm", 650.2, 1023.3),
        ("THD bàn tay", 903.7, 1022.2),
    ],
    "4_CangChi": [
        ("CC mũi", 145.0, 1172.0),
        ("CC mặt", 330.9, 1171.6),
        ("CC nọng", 518.9, 1170.5),
        ("CC Crews", 704.6, 1169.3),
        ("CC nâng cung mày", 890.1, 1168.1),
    ],
    "5_Ro": [
        ("Rẻ quạt ốp má", 146.4, 1316.9),
        ("Rẻ quạt vùng rỗ", 503.4, 1317.2),
        ("Tiêm từng vết rỗ kim châm", 812.3, 1315.6),
        ("Phá hủy rỗ bằng kim châm", 146.4, 1366.2),
        ("Chấm CA", 503.4, 1367.3),
    ],
}

SECTION_TITLES = {
    "1_Nhan": "Nhăn",
    "2_Day": "Đẩy",
    "3_THD": "THD",
    "4_CangChi": "Căng chỉ",
    "5_Ro": "Rỗ",
}

# ROIs are in the canonical page space that every photo is aligned to. Each one starts
# right after the printed label so the label itself is (mostly) excluded.
HEADER_ROIS = {
    "ndh": (145, 150, 440, 235),
    "bs": (478, 140, 770, 225),
    "phuTa": (848, 150, 1130, 252),  # tall enough for low-slung handwriting, stops above the "Ngày làm" label
    "stt": (138, 215, 225, 280),
    "mskh": (292, 205, 490, 275),
    "phongGiuong": (642, 205, 850, 275),
    "ngayLam": (962, 205, 1140, 285),
    # DV đăng ký has 3 lines; handwriting may be on any of them (below the printed label).
    "dvDangKy": (100, 300, 1130, 410),
}

# Items 6/7/8: the right edge stops at x=880 on purpose. The doctor's signature / "BS - <name>"
# sign-off sits at the far right of these lines (x ~ 900-1100) and was being read as real values.
SECTION_TEXT_ROIS = {
    "6_Meso": (190, 1395, 880, 1465),
    "7_ReQuatTHDHoacHoatChatTangSinh": (548, 1455, 880, 1535),
    "8_ThuocGiai": (248, 1535, 880, 1600),
}

# ---------------------------------------------------------------------------
# Geometry / alignment
# ---------------------------------------------------------------------------


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def find_document_corners(image: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image.shape[0] * image.shape[1]
    best: Optional[Tuple[float, np.ndarray]] = None
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.35 * image_area:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            if best is None or area > best[0]:
                best = (area, approx.reshape(4, 2))
    return order_points(best[1]) if best else None


def warp_by_corners(image: np.ndarray) -> Tuple[np.ndarray, bool]:
    corners = find_document_corners(image)
    if corners is None:
        return cv2.resize(image, (W, H), interpolation=cv2.INTER_CUBIC), False
    dst = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
    m = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(image, m, (W, H), flags=cv2.INTER_CUBIC), True


def _features(gray: np.ndarray):
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    if hasattr(cv2, "SIFT_create"):
        det, norm = cv2.SIFT_create(nfeatures=5000), cv2.NORM_L2
    else:
        det, norm = cv2.ORB_create(nfeatures=5000), cv2.NORM_HAMMING
    kp, des = det.detectAndCompute(g, None)
    return kp, des, norm


def match_homography(image: np.ndarray, ref_page: np.ndarray) -> Optional[Tuple[np.ndarray, int]]:
    """SIFT/ORB + RANSAC: homography mapping `image` pixels -> canonical page. (matrix, inliers) or None.

    The form is full of repeated elements (squares, dotted lines, similar labels), so the ratio
    test rejects many correct matches; thresholds are loose and the caller validates the result
    by counting how many printed squares end up where they should."""
    scale = 1.0
    long_side = max(image.shape[:2])
    small = image
    if long_side > 2400:
        scale = 2400.0 / long_side
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    kp1, d1, norm = _features(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    kp2, d2, _ = _features(cv2.cvtColor(ref_page, cv2.COLOR_BGR2GRAY))
    if d1 is None or d2 is None or len(kp1) < 30 or len(kp2) < 30:
        return None
    pairs = cv2.BFMatcher(norm).knnMatch(d1, d2, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
    if len(good) < 25:
        return None
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2) / scale
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    hm, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if hm is None:
        return None
    inliers = int(mask.sum())
    if inliers < 25 or inliers / len(good) < 0.25:
        return None
    return hm, inliers


def corner_homography(image: np.ndarray) -> Optional[np.ndarray]:
    corners = find_document_corners(image)
    if corners is None:
        return None
    dst = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
    return cv2.getPerspectiveTransform(corners, dst)


def warp_with(image: np.ndarray, hm: np.ndarray) -> np.ndarray:
    return cv2.warpPerspective(image, hm, (W, H), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


def refine_homography(image: np.ndarray, hm: np.ndarray) -> Tuple[np.ndarray, int]:
    """Second stage: the 43 printed squares are strong landmarks. Find each one on the
    warped page and fit a better homography to their nominal positions (2 passes)."""
    used = 0
    for search in (45, 28):
        gp = _pad_gray(warp_with(image, hm))
        src, dst = [], []
        for items in CHECKBOXES.values():
            for _, x, y in items:
                fx, fy, mf = locate_box(gp, x, y, search=search, thresh=0.5)
                if mf >= 0.5:
                    src.append((fx, fy))
                    dst.append((x, y))
        if len(src) < 12:
            break
        h2, mask = cv2.findHomography(np.float32(src), np.float32(dst), cv2.RANSAC, 5.0)
        if h2 is None or int(mask.sum()) < 12:
            break
        hm = h2 @ hm
        used = int(mask.sum())
    return hm, used


# ---------------------------------------------------------------------------
# Registration from the printed squares alone (no blank template, tolerant to big offsets)
# ---------------------------------------------------------------------------
TPL_PTS = np.float32([(x, y) for items in CHECKBOXES.values() for _, x, y in items])


def detect_square_candidates(gray: np.ndarray, lo: int = 14, hi: int = 46) -> np.ndarray:
    """Centres of empty, square-looking outlines in a page-sized grayscale image."""
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts: List[Tuple[float, float]] = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if not (lo <= w <= hi and lo <= h <= hi) or not (0.75 <= w / h <= 1.33):
            continue
        if cv2.contourArea(c) < 0.85 * w * h:  # round letters ("O", "D") fall below this
            continue
        inner = bw[y + h // 4 : y + h - h // 4, x + w // 4 : x + w - w // 4]
        if inner.size == 0 or inner.mean() > 255 * 0.25:  # must be hollow
            continue
        pts.append((x + w / 2.0, y + h / 2.0))
    return np.float32(pts).reshape(-1, 2)


def _match_pts(cand_t: np.ndarray, tol: float) -> Tuple[np.ndarray, np.ndarray]:
    """One-to-one nearest matches between transformed candidates and the 43 nominal centres."""
    if len(cand_t) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    d = np.linalg.norm(cand_t[:, None, :] - TPL_PTS[None, :, :], axis=2)
    j = d.argmin(axis=1)
    dist = d[np.arange(len(cand_t)), j]
    idx = np.nonzero(dist < tol)[0]
    keep, seen = [], set()
    for i in idx[np.argsort(dist[idx])]:
        if int(j[i]) in seen:
            continue
        seen.add(int(j[i]))
        keep.append(i)
    keep = np.array(keep, dtype=int)
    return keep, j[keep]


def register_by_boxes(image: np.ndarray) -> Optional[Tuple[np.ndarray, int]]:
    """image -> canonical homography estimated only from the printed checkbox lattice.

    1) coarse: vote on (scale_x, scale_y, shift_x, shift_y) so photos where the paper does not
       fill the frame, or is shifted by more than one row, still land on the right squares;
    2) fine: alternate nearest-neighbour matching and affine -> homography fitting."""
    h0, w0 = image.shape[:2]
    pre = np.diag([W / w0, H / h0, 1.0])
    small = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
    cand = detect_square_candidates(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    if len(cand) < 8:
        return None

    xb = np.arange(-520, 521, 12)
    yb = np.arange(-740, 741, 12)
    hyps = []
    for sx in np.arange(0.7, 1.361, 0.025):
        for sy in np.arange(0.7, 1.361, 0.025):
            c = cand * np.float32([sx, sy])
            t = (TPL_PTS[None, :, :] - c[:, None, :]).reshape(-1, 2)
            hist, _, _ = np.histogram2d(t[:, 0], t[:, 1], bins=[xb, yb])
            i, j = np.unravel_index(hist.argmax(), hist.shape)
            hyps.append((hist[i, j], sx, sy, xb[i] + 6.0, yb[j] + 6.0))
    hyps.sort(key=lambda t: -t[0])
    best: Optional[Tuple[int, float, float, float, float]] = None
    for _, sx, sy, tx, ty in hyps[:40]:
        n = len(_match_pts(cand * np.float32([sx, sy]) + np.float32([tx, ty]), 10)[0])
        if best is None or n > best[0]:
            best = (n, sx, sy, tx, ty)
    if best is None or best[0] < 8:
        return None
    _, sx, sy, tx, ty = best
    cur = np.array([[sx, 0, tx], [0, sy, ty], [0, 0, 1]], dtype=np.float64)

    for tol, kind in ((12, "affine"), (14, "affine"), (16, "homography"), (10, "homography")):
        ct = cv2.perspectiveTransform(cand.reshape(-1, 1, 2), cur).reshape(-1, 2)
        ci, tj = _match_pts(ct, tol)
        if len(ci) < 8:
            break
        src, dst = cand[ci].astype(np.float32), TPL_PTS[tj]
        if kind == "affine":
            m, _ = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=6.0)
            if m is None:
                break
            cur = np.vstack([m, [0, 0, 1]])
        else:
            if len(ci) < 12:
                break
            h2, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if h2 is None:
                break
            cur = h2
    ct = cv2.perspectiveTransform(cand.reshape(-1, 1, 2), cur).reshape(-1, 2)
    n_final = len(_match_pts(ct, 8)[0])
    return cur @ pre, n_final


def _align_one(image: np.ndarray, blank_page: Optional[np.ndarray], tag: str = "") -> Tuple[np.ndarray, Dict[str, Any]]:
    """Try every registration strategy, refine each on the printed squares, keep the one that
    lines up the most squares. A strategy that only "looks" plausible is not enough."""
    inits: List[Tuple[str, np.ndarray, Optional[int]]] = []
    if blank_page is not None:
        res = match_homography(image, blank_page)
        if res is not None:
            inits.append(("feature_homography", res[0], res[1]))
    reg = register_by_boxes(image)
    if reg is not None:
        inits.append(("box_lattice", reg[0], reg[1]))
    hm_c = corner_homography(image)
    if hm_c is not None:
        inits.append(("document_corners", hm_c, None))
    inits.append(("plain_resize", np.diag([W / image.shape[1], H / image.shape[0], 1.0]), None))

    best: Optional[Tuple[int, str, Optional[int], np.ndarray]] = None
    for method, hm0, inl in inits:
        hm1, n = refine_homography(image, hm0)
        print(f"[align] {tag}{method}: {n}/43 squares located")
        if best is None or n > best[0]:
            best = (n, method, inl, hm1)
    assert best is not None
    n, method, inl, hm = best
    return hm, {"method": method, "feature_inliers": inl, "boxes_used_to_refine": n, "ok": n >= ALIGN_MIN_BOXES}


def _rotation_matrix(code: Optional[int], w: int, h: int) -> np.ndarray:
    """3x3 matrix that maps pixel (x, y) of the original w x h image to the same pixel after cv2.rotate(code)."""
    if code is None:
        return np.eye(3)
    if code == cv2.ROTATE_90_CLOCKWISE:
        return np.array([[0, -1, h - 1], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    if code == cv2.ROTATE_180:
        return np.array([[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1]], dtype=np.float64)
    return np.array([[0, 1, 0], [-1, 0, w - 1], [0, 0, 1]], dtype=np.float64)  # 90 counter-clockwise


# (clockwise degrees the photo must be turned to stand upright, cv2.rotate code)
_ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def align_page(image: np.ndarray, blank_page: Optional[np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Align a photo to the canonical page, whatever way up it is.

    The 43-square lattice is not symmetric, so only the upright orientation lines up with it; a
    sideways or upside-down page matches a handful of squares by chance. We try the likely
    orientation first (a landscape photo is probably a page lying on its side), stop at the first
    one that is clearly aligned, and otherwise keep the orientation with the most squares.
    The rotation is folded into the returned homography, so warp_with(image, hm) is already upright.
    `info["rotation"]` is how many degrees clockwise the photo had to be turned."""
    h0, w0 = image.shape[:2]
    order = [0, 90, 270, 180] if h0 >= w0 else [90, 270, 0, 180]
    best: Optional[Tuple[int, int, np.ndarray, Dict[str, Any]]] = None
    for angle in order:
        code = _ROTATIONS[angle]
        img = image if code is None else cv2.rotate(image, code)
        hm, info = _align_one(img, blank_page, tag=f"rot{angle} ")
        hm = hm @ _rotation_matrix(code, w0, h0)
        info["rotation"] = angle
        if best is None or info["boxes_used_to_refine"] > best[0]:
            best = (info["boxes_used_to_refine"], angle, hm, info)
        if info["ok"]:
            break
    assert best is not None
    _, angle, hm, info = best
    if angle:
        print(f"[align] Ảnh bị xoay: đã xoay {angle}° theo chiều kim đồng hồ cho đúng hướng")
    return hm, info


# ---------------------------------------------------------------------------
# Checkbox detection (tri-state)
# ---------------------------------------------------------------------------
PAD = 48
R = 24  # half-size of the analysed patch
INNER = 8  # half-size of the interior region (printed border sits at about +-12)
CHECKED_RATIO = 0.08
UNCHECKED_RATIO = 0.02


def _pad_gray(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.copyMakeBorder(g, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=255)


def _patch(gp: np.ndarray, cx: float, cy: float, half: int) -> np.ndarray:
    x, y = int(round(cx)) + PAD, int(round(cy)) + PAD
    return gp[y - half : y + half, x - half : x + half]


def _ideal_box(side: int = 26, thick: int = 2, size: int = 40) -> np.ndarray:
    t = np.full((size, size), 255, np.uint8)
    o = (size - side) // 2
    cv2.rectangle(t, (o, o), (o + side, o + side), 40, thick)
    return t


_BOX_TPL = _ideal_box()


def locate_box(gp: np.ndarray, cx: float, cy: float, search: int = 34, thresh: float = 0.35) -> Tuple[float, float, float]:
    """Find the printed square near (cx, cy). Returns (x, y, match_score)."""
    win = _patch(gp, cx, cy, search)
    res = cv2.matchTemplate(win, _BOX_TPL, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(res)
    if mx < thresh:
        return cx, cy, mx
    half_t = _BOX_TPL.shape[0] // 2
    return cx + (loc[0] + half_t - search), cy + (loc[1] + half_t - search), mx


def _ink(gray: np.ndarray, delta: int = 40) -> np.ndarray:
    bg = float(np.median(gray))
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return (blur < bg - delta).astype(np.uint8)


def score_checkbox(fp: np.ndarray, nx: float, ny: float) -> Dict[str, Any]:
    fx, fy, mf = locate_box(fp, nx, ny, search=30, thresh=0.4)
    located = mf >= 0.4
    if not located:
        fx, fy = nx, ny
    ink = cv2.morphologyEx(_ink(_patch(fp, fx, fy, R)), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    yy, xx = np.mgrid[0 : 2 * R, 0 : 2 * R]
    inner = (np.abs(xx - R) <= INNER) & (np.abs(yy - R) <= INNER)
    inner_ratio = float(ink[inner].mean())

    if inner_ratio >= CHECKED_RATIO:
        status = "checked"
    elif located and inner_ratio < UNCHECKED_RATIO:
        status = "unchecked"
    else:
        status = "uncertain"  # never promoted to true
    return {
        "status": status,
        "inner_ratio": round(inner_ratio, 4),
        "match": round(float(mf), 3),
        "located": bool(located),
        "x": round(fx, 1),
        "y": round(fy, 1),
    }


def detect_checkboxes(page: np.ndarray) -> Dict[str, Dict[str, Any]]:
    fp = _pad_gray(page)
    out: Dict[str, Dict[str, Any]] = {}
    for key, items in CHECKBOXES.items():
        details = []
        for label, cx, cy in items:
            d = score_checkbox(fp, cx, cy)
            d["label"] = label
            details.append(d)
        out[key] = {
            "title": SECTION_TITLES[key],
            "checked": [d["label"] for d in details if d["status"] == "checked"],
            "uncertain": [d["label"] for d in details if d["status"] == "uncertain"],
            "details": details,
        }
    return out


def reject_boxes(boxes: Dict[str, Dict[str, Any]]) -> None:
    """Alignment failed: nothing may be reported as ticked; every box needs a human look."""
    for sec in boxes.values():
        for d in sec["details"]:
            d["status"] = "uncertain"
        sec["checked"] = []
        sec["uncertain"] = [d["label"] for d in sec["details"]]


# ---------------------------------------------------------------------------
# Adaptive regions: handwritten notes (dose / remarks) next to the checkboxes
# ---------------------------------------------------------------------------
# Nothing here is a fixed box. Print is neutral grey and the handwriting is blue, so the page is
# split into "ink" and "print"; ink strokes are grouped into words/lines and every group is
# attached to the nearest checkbox on its left in the same row. The OCR crop is the bounding box
# of that group with everything that is not handwriting painted white (no printed label, no dots).
NOTE_UP, NOTE_DOWN = 40, 36  # how far above / below the row of its box a note may sit
NOTE_MIN_PIXELS = 45  # smaller blobs are specks, not writing
NOTE_PAD = (12, 8)  # padding (x, y) around the group
# Free-text rules under sections 3 and 4 have no checkbox next to them: (y_min, y_max)
LINE_NOTE_ZONES = {"3_THD": (1048, 1100), "4_CangChi": (1193, 1245)}


# The form is a fixed printed template, so we know exactly where the printed text and dotted rules are:
# a photo of the blank form (aligned like every other photo) gives a "print template", and anything
# that coincides with it is print. What is left over in the writing areas is handwriting. This does
# not depend on the pen colour or on the lighting (a colour test breaks under warm/cool light).
PRINT_TEMPLATE_PATH = HERE / "template_print.png"
PRINT_TOLERANCE = 11  # px the (locally re-aligned) print is grown by before it is removed
NOTE_X_MAX = 1170  # right of this is the edge of the sheet, not writing
NOTE_MIN_HEIGHT = 12  # flatter than this is a strike-through / underline, not text


def stroke_map(page: np.ndarray) -> np.ndarray:
    """How much darker than its surroundings each pixel is, for dark features narrower than ~16 px
    (pen strokes, print). Broad shadows and lighting gradients are not strokes and drop out."""
    gray = cv2.GaussianBlur(cv2.cvtColor(page, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))).astype(np.float32)


def build_print_template(blank_raw: np.ndarray) -> np.ndarray:
    """Print template (0/1, canonical page size) from a photo or scan of the BLANK form."""
    hm, info = align_page(blank_raw, None)
    if not info["ok"]:
        raise ValueError("không căn chỉnh được ảnh form trống (cần thấy rõ các ô vuông)")
    t = (stroke_map(warp_with(blank_raw, hm)) > 14).astype(np.uint8)
    return cv2.morphologyEx(t, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def load_print_template(blank_raw: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """template_print.png next to this script if present; else built from the blank form (and saved)."""
    if PRINT_TEMPLATE_PATH.exists():
        t = cv2.imdecode(np.fromfile(str(PRINT_TEMPLATE_PATH), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if t is not None and t.shape == (H, W):
            return (t > 0).astype(np.uint8)
        print(f"[warn] {PRINT_TEMPLATE_PATH.name} không hợp lệ (cần {W}x{H}), bỏ qua.")
    if blank_raw is None:
        return None
    t = build_print_template(blank_raw)
    try:
        write_image(PRINT_TEMPLATE_PATH, t * 255)
        print(f"[info] Đã tạo {PRINT_TEMPLATE_PATH.name} từ form trống (lần sau dùng lại, không cần --blank-template để dò ghi chú).")
    except Exception as exc:  # read-only folder etc.: still usable for this run
        print(f"[warn] Không lưu được {PRINT_TEMPLATE_PATH.name}: {exc}")
    return t


def adapt_template(strokes: np.ndarray, tpl: np.ndarray, tile: Tuple[int, int] = (120, 100), search: int = 16,
                   min_px: int = 250, min_score: float = 0.30, sigma: float = 110) -> np.ndarray:
    """Bend the print template onto THIS photo. A sheet is never perfectly flat (paper curl, lens,
    alignment error): per tile, find the shift that best overlays the template's print on the page's
    strokes, smooth the shifts into a field, and warp the template with it. Handwriting does not
    matter much, the print dominates every tile that is used."""
    hh, ww = tpl.shape
    tw, th = tile
    page = (strokes > 18).astype(np.float32)
    t_f = tpl.astype(np.float32)
    sx, sy, sw = (np.zeros((hh, ww), np.float32) for _ in range(3))
    for y0 in range(0, hh - th + 1, th // 2):
        for x0 in range(0, ww - tw + 1, tw // 2):
            t = t_f[y0 : y0 + th, x0 : x0 + tw]
            if t.sum() < min_px:  # not enough print to lock on to
                continue
            ya, yb, xa, xb = max(0, y0 - search), min(hh, y0 + th + search), max(0, x0 - search), min(ww, x0 + tw + search)
            _, score, _, loc = cv2.minMaxLoc(cv2.matchTemplate(page[ya:yb, xa:xb], t, cv2.TM_CCOEFF_NORMED))
            if score < min_score:
                continue
            cy, cx = y0 + th // 2, x0 + tw // 2
            sx[cy, cx], sy[cy, cx], sw[cy, cx] = xa + loc[0] - x0, ya + loc[1] - y0, score
    num_x, num_y, den = (cv2.GaussianBlur(a, (0, 0), sigma) for a in (sx * sw, sy * sw, sw))
    dx, dy = num_x / (den + 1e-4), num_y / (den + 1e-4)
    dx[den < 1e-4] = dy[den < 1e-4] = 0  # nothing measured nearby: leave the template where it is
    gx, gy = np.meshgrid(np.arange(ww, dtype=np.float32), np.arange(hh, dtype=np.float32))
    return cv2.remap(tpl, gx - dx, gy - dy, cv2.INTER_NEAREST)


def handwriting_mask(strokes: np.ndarray, print_zone: np.ndarray, thresh: float = 22) -> np.ndarray:
    """1 where there is a dark stroke that is NOT printed text. The default is strict (sure ink);
    a lower threshold also catches faint strokes."""
    m = ((strokes > thresh) & (print_zone == 0)).astype(np.uint8)
    m[:, NOTE_X_MAX:] = 0
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def remove_ticks(mask: np.ndarray, items: List[Dict[str, Any]]) -> np.ndarray:
    """Tick marks are not notes. A stroke that touches a square is a tick (it may leave the square)."""
    _, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for it in items:
        x, y = int(round(it["x"])), int(round(it["y"]))
        ids = np.unique(lab[max(0, y - 12) : y + 13, max(0, x - 12) : x + 13])
        for i in ids[ids > 0]:
            if stats[i, cv2.CC_STAT_WIDTH] <= 100 and stats[i, cv2.CC_STAT_HEIGHT] <= 100:
                mask[lab == i] = 0
            else:  # tick fused with writing: only cut around the square
                mask[max(0, y - 50) : y + 18, max(0, x - 20) : x + 26] = 0
        if it["checked"]:  # the tick's tail runs up and to the right, out of the square
            mask[max(0, y - 52) : y + 20, max(0, x - 18) : x + 35] = 0
    return mask


def attach_note(items: List[Dict[str, Any]], x1: int, ys: np.ndarray) -> Optional[Tuple[str, Any]]:
    """Which checkbox (or free-text rule) does an ink group belong to? None = not a note."""
    gy = float(ys.mean())
    for key, (lo, hi) in LINE_NOTE_ZONES.items():
        if lo <= gy <= hi and ((ys >= lo - 6) & (ys <= hi + 6)).mean() >= 0.8:
            return "line", key
    cands = [it for it in items if it["y"] - NOTE_UP <= gy <= it["y"] + NOTE_DOWN and it["x"] + 8 <= x1]
    if not cands:
        return None
    row_y = min(cands, key=lambda it: abs(gy - it["y"]))["y"]
    it = max((c for c in cands if abs(c["y"] - row_y) <= 14), key=lambda c: c["x"])  # nearest box on the left
    if ((ys >= it["y"] - NOTE_UP - 6) & (ys <= it["y"] + NOTE_DOWN + 6)).mean() < 0.8:
        return None  # mostly outside the row (e.g. the tail of the registration lines above)
    return "box", it


def find_notes(page: np.ndarray, boxes: Dict[str, Dict[str, Any]], print_template: Optional[np.ndarray]):
    """-> (notes {(section, label): note}, line_notes {section: [note]}, mask for the OCR crops).
    Without a print template there is no reliable way to tell print from handwriting: nothing is returned."""
    if print_template is None:
        return {}, {}, np.zeros(page.shape[:2], np.uint8)
    items = [
        dict(key=k, label=d["label"], x=d["x"], y=d["y"], checked=d["status"] == "checked")
        for k, sec in boxes.items()
        for d in sec["details"]
    ]
    strokes = stroke_map(page)
    zone = cv2.dilate(adapt_template(strokes, print_template), np.ones((PRINT_TOLERANCE, PRINT_TOLERANCE), np.uint8))
    mask = remove_ticks(handwriting_mask(strokes, zone), items)
    link = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 9)))  # letters -> words -> a line
    n, lab, _, _ = cv2.connectedComponentsWithStats(link, connectivity=8)
    notes: Dict[Tuple[str, str], Dict[str, Any]] = {}
    lines: Dict[str, List[Dict[str, Any]]] = {}
    for i in range(1, n):
        ys, xs = np.nonzero((lab == i) & (mask > 0))
        if len(xs) < NOTE_MIN_PIXELS:
            continue
        x1, x2, y1, y2 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        if x2 - x1 < 8 or y2 - y1 < NOTE_MIN_HEIGHT:
            continue
        target = attach_note(items, x1, ys)
        if target is None:
            continue
        region = [max(0, x1 - NOTE_PAD[0]), max(0, y1 - NOTE_PAD[1]), min(W, x2 + NOTE_PAD[0] + 1), min(H, y2 + NOTE_PAD[1] + 1)]
        note: Dict[str, Any] = {"region": region, "ink_px": int(len(xs))}
        if target[0] == "line":
            lines.setdefault(target[1], []).append(note)
            continue
        it = target[1]
        old = notes.get((it["key"], it["label"]))
        if old is not None:  # several groups for one box -> one note covering all of them
            old["region"] = [min(old["region"][0], region[0]), min(old["region"][1], region[1]),
                             max(old["region"][2], region[2]), max(old["region"][3], region[3])]
            old["ink_px"] += note["ink_px"]
        else:
            note["checked"] = bool(it["checked"])
            notes[(it["key"], it["label"])] = note
    for lst in lines.values():
        lst.sort(key=lambda d: (d["region"][1], d["region"][0]))
    # what the OCR gets to see: sure ink plus the faint strokes right next to it (a strict mask alone
    # breaks thin letters), never the printed labels or dotted rules
    faint = handwriting_mask(strokes, zone, 12)
    near = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    keep = cv2.dilate(((faint > 0) & (near > 0)).astype(np.uint8), np.ones((3, 3), np.uint8))
    return notes, lines, keep


def note_crop(page: np.ndarray, keep_mask: np.ndarray, region: List[int]) -> np.ndarray:
    """The note's bounding box with only the handwriting kept; everything else is paper white."""
    x1, y1, x2, y2 = region
    keep = keep_mask[y1:y2, x1:x2] > 0
    crop = page[y1:y2, x1:x2].copy()
    crop[~keep] = 255
    return crop


# ---------------------------------------------------------------------------
# Multimodal OCR (header + items 6/7/8 + notes) - one request per field
# ---------------------------------------------------------------------------
NOTE_HINT = (
    "Ghi chú viết tay cạnh ô '{label}' (mục '{title}'): thường là liều lượng / số lượng hoặc ghi chú ngắn, "
    "ví dụ '10U', '20u', '1,2ml', '1ml T', 'Khóe miệng', 'Viền hàm (20u)'. Giữ nguyên số, đơn vị (U, u, ml, cc) và ký hiệu "
    "như T/P đúng như viết; không diễn giải, không mở rộng chữ viết tắt. Dấu phẩy thập phân giữ nguyên (1,2ml). "
    "Ảnh chỉ chứa nét viết tay."
)
NOTE_LINE_HINT = (
    "Ghi chú viết tay trên dòng kẻ của mục '{title}' (liều lượng / số lượng / vị trí). Giữ nguyên số, đơn vị và ký hiệu "
    "như viết; không diễn giải, không mở rộng chữ viết tắt. Ảnh chỉ chứa nét viết tay."
)
FIELD_HINTS = {
    "ndh": "Tên người ghi phiếu (NDH). Thường là tên riêng 1-3 từ, có thể kèm viết tắt như KT.",
    "bs": "Tên bác sĩ (BS). Một tên riêng ngắn.",
    "phuTa": "Tên phụ tá. Một tên riêng ngắn.",
    "stt": "Số thứ tự. Chỉ gồm chữ số.",
    "mskh": "Mã số khách hàng. Chuỗi chữ số.",
    "phongGiuong": "Phòng / Giường, dạng '<phòng>-<giường>' hoặc '<phòng>/<giường>' (ví dụ '202-6', '202/1'). "
    "Dấu gạch hoặc dấu xuyệt là dấu phân cách, KHÔNG phải chữ số 1.",
    "ngayLam": "Ngày làm, dạng ngày/tháng (ví dụ '2/10'). Dấu xuyệt không phải chữ số 1.",
    "dvDangKy": "Dịch vụ đăng ký, có thể viết trên 1-3 dòng. Bỏ qua dòng chữ in sẵn 'DV đăng ký (ghi rõ New, TK lần 1,2,3...)'. "
    "Từ chuyên môn thường gặp: meso, filler, THD, Radian, nhăn, mắt, mặt, mũi, má, cằm, rãnh mũi, viền hàm, "
    "khóe miệng, nâng cơ, căng chỉ, cấy, rạch, TK, New.",
    "6_Meso": "Mục 6 - Meso: tên sản phẩm/liều lượng meso, ví dụ 'Radian 1ml'.",
    "7_ReQuatTHDHoacHoatChatTangSinh": "Mục 7 - Rẻ quạt THD hoặc hoạt chất tăng sinh: tên chất và liều lượng.",
    "8_ThuocGiai": "Mục 8 - Thuốc giãi: tên thuốc và liều lượng.",
}

FIELD_PROMPT = """
You are a precise Vietnamese handwriting OCR engine for a medical/aesthetic checklist form.
The attached image is ONE field cropped from the form. Meaning of this field:
{hint}

Rules:
- Transcribe ONLY handwritten text. Ignore printed labels, dotted lines and underlines.
- Signatures, initials, stamps, scribbles and doctor sign-offs such as "BS - <name>" are NOT field
  values: return value "" and "is_signature": true.
- Do NOT invent or guess. If blank or unreadable: value "", confidence 0.0.
- Preserve Vietnamese diacritics when visible. Keep digits exactly as written.
- confidence is 0..1 and reflects your real certainty.

Return JSON ONLY, no Markdown fences:
{"value": "", "confidence": 0.0, "is_signature": false}
""".strip()


def make_openai_client():
    from openai import OpenAI

    return OpenAI(api_key=OPENAI_API_KEY or "local-dev", base_url=OPENAI_BASE_URL)


def crop_roi(image: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = roi
    x1 = max(0, min(W - 1, int(x1)))
    x2 = max(x1 + 1, min(W, int(x2)))
    y1 = max(0, min(H - 1, int(y1)))
    y2 = max(y1 + 1, min(H, int(y2)))
    return image[y1:y2, x1:x2]


def encode_crop(crop: np.ndarray, max_width: int = 1800, min_width: int = 0, pad: int = 0) -> str:
    # small crops (a dose like "10U") are scaled up so a vision model gets enough pixels
    scale = min(6.0, max(2.0, min_width / max(1, crop.shape[1])))
    img = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)  # keep colour
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    h, w = img.shape[:2]
    if w > max_width:
        img = cv2.resize(img, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("Could not JPEG-encode crop")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def extract_json_object(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if m:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"Model did not return valid JSON: {raw[:500]}")


def normalize_ocr_result(obj: Dict[str, Any], keys: List[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for key in keys:
        item = obj.get(key, {})
        if isinstance(item, dict):
            value, conf = item.get("value", ""), item.get("confidence")
        else:
            value, conf = (item if isinstance(item, str) else ""), None
        value = re.sub(r"\s+", " ", str(value or "")).strip(" .:")
        try:
            conf = max(0.0, min(1.0, float(conf))) if conf is not None else None
        except (TypeError, ValueError):
            conf = None
        out[key] = {"value": value, "confidence": round(conf, 3) if conf is not None else None}
    return out


def ocr_field(client, key: str, crop: np.ndarray, hint: Optional[str] = None, min_width: int = 0, pad: int = 0) -> Dict[str, Any]:
    prompt = FIELD_PROMPT.replace("{hint}", hint if hint is not None else FIELD_HINTS.get(key, ""))
    request = dict(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a precise Vietnamese handwriting OCR engine. Return only the requested JSON."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": encode_crop(crop, min_width=min_width, pad=pad), "detail": "high"}},
                ],
            },
        ],
        temperature=0,
        max_tokens=300,
    )
    try:
        resp = client.chat.completions.create(**request, response_format={"type": "json_object"})
    except Exception:  # some gateways reject response_format
        resp = client.chat.completions.create(**request)
    obj = extract_json_object(resp.choices[0].message.content or "")
    if obj.get("is_signature") is True:
        obj["value"], obj["confidence"] = "", 0.0
    return normalize_ocr_result({key: obj}, [key])[key]


def clean_text(key: str, v: Dict[str, Any]) -> Dict[str, Any]:
    """Post-filter for things the model should not have returned."""
    val = v["value"]
    # printed hint copied from the form
    val = re.sub(r"(?i)d\.?\s?v\.?\s*đăng\s*ký", "", val)
    val = re.sub(r"(?i)\(?\s*ghi\s*rõ\s*new\W*tk\s*lần\s*1\W*2\W*3\W*\)?", "", val)
    val = re.sub(r"(?i)^\s*new\W*tk\s*lần\s*1\W*2\W*3\W*", "", val)
    val = re.sub(r"\s+", " ", val).strip(" .:")
    if key in SECTION_TEXT_ROIS:
        # signatures/initials ("2", "Qy", "BS - Anh" ...) are not item content
        if len(re.sub(r"\W", "", val)) <= 2 or re.match(r"(?i)^bs\b", val):
            val = ""
    if not val:
        return {"value": "", "confidence": 0.0}
    return {"value": val, "confidence": v["confidence"]}


def ocr_all_fields(client, page: np.ndarray) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], List[str]]:
    rois = {**HEADER_ROIS, **SECTION_TEXT_ROIS}
    errors: List[str] = []

    def work(key: str):
        try:
            return key, ocr_field(client, key, crop_roi(page, rois[key]))
        except Exception as exc:  # keep going; the field is flagged for review
            print(f"[warn] OCR failed for {key}: {exc}")
            errors.append(key)
            return key, {"value": "", "confidence": None}

    with ThreadPoolExecutor(max_workers=4) as ex:
        res = dict(ex.map(work, rois))
    header = {k: clean_text(k, res[k]) for k in HEADER_ROIS}
    bottom = {k: clean_text(k, res[k]) for k in SECTION_TEXT_ROIS}
    return header, bottom, errors


def ocr_notes(client, page: np.ndarray, mask: np.ndarray, notes: Dict[Tuple[str, str], Dict[str, Any]],
              lines: Dict[str, List[Dict[str, Any]]]) -> None:
    """OCR every note crop; fills value/confidence in place (error=True when the request failed)."""
    jobs = [(n, NOTE_HINT.format(label=label, title=SECTION_TITLES[key])) for (key, label), n in notes.items()]
    jobs += [(n, NOTE_LINE_HINT.format(title=SECTION_TITLES[key])) for key, lst in lines.items() for n in lst]

    def work(job):
        n, hint = job
        try:
            r = clean_text("note", ocr_field(client, "note", note_crop(page, mask, n["region"]), hint=hint, min_width=420, pad=16))
            n["value"], n["confidence"] = r["value"], r["confidence"]
        except Exception as exc:  # keep going; flagged for review
            print(f"[warn] OCR failed for note {n['region']}: {exc}")
            n["value"], n["confidence"], n["error"] = "", None, True

    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(work, jobs))


# ---------------------------------------------------------------------------
# Debug image
# ---------------------------------------------------------------------------
def draw_debug(page: np.ndarray, boxes: Dict[str, Dict[str, Any]], align_ok: bool = True,
               notes: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
               lines: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> np.ndarray:
    out = page.copy()
    colors = {"checked": (0, 0, 255), "uncertain": (0, 165, 255), "unchecked": (0, 160, 0)}
    magenta = (255, 0, 255)
    where = {(k, d["label"]): (int(d["x"]), int(d["y"])) for k, sec in boxes.items() for d in sec["details"]}
    for (key, label), n in (notes or {}).items():  # adaptive note regions, linked to their checkbox
        x1, y1, x2, y2 = n["region"]
        bx, by = where[(key, label)]
        cv2.line(out, (bx + 14, by), (x1, (y1 + y2) // 2), magenta, 1, cv2.LINE_AA)
        cv2.rectangle(out, (x1, y1), (x2, y2), magenta, 2)
        cv2.putText(out, "NOTE", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, magenta, 1, cv2.LINE_AA)
    for lst in (lines or {}).values():
        for n in lst:
            x1, y1, x2, y2 = n["region"]
            cv2.rectangle(out, (x1, y1), (x2, y2), magenta, 2)
            cv2.putText(out, "LINE NOTE", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, magenta, 1, cv2.LINE_AA)
    for sec in boxes.values():
        for d in sec["details"]:
            x, y = int(d["x"]), int(d["y"])
            cv2.rectangle(out, (x - 13, y - 13), (x + 13, y + 13), colors[d["status"]], 2 if d["status"] != "unchecked" else 1)
            if d["status"] != "unchecked":
                cv2.putText(out, d["status"].upper(), (x - 13, y - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[d["status"]], 1, cv2.LINE_AA)
    for key, (x1, y1, x2, y2) in {**HEADER_ROIS, **SECTION_TEXT_ROIS}.items():
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv2.putText(out, key, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 0, 0), 1, cv2.LINE_AA)
    if not align_ok:
        cv2.putText(out, "ALIGNMENT FAILED - results rejected", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Image I/O + batch helpers
# ---------------------------------------------------------------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def read_image(path: Path) -> Optional[np.ndarray]:
    """cv2.imread that also works with non-ASCII paths (tên file/thư mục tiếng Việt, Windows)."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def write_image(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise RuntimeError(f"Could not encode {path.name}")
    buf.tofile(str(path))


def list_images(folder: Path, recursive: bool) -> List[Path]:
    it = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("."))


def output_names(paths: List[Path], root: Path) -> Dict[Path, str]:
    """Output base name per image. Sub-folders are folded into the name, and a.jpg / a.png
    in the same place become a_jpg / a_png so nothing overwrites anything."""
    base: Dict[Path, str] = {}
    for p in paths:
        try:
            rel = p.relative_to(root)
        except ValueError:
            rel = Path(p.name)
        base[p] = str(rel.with_suffix("")).replace(os.sep, "__")
    counts: Dict[str, int] = {}
    for b in base.values():
        counts[b] = counts.get(b, 0) + 1
    return {p: (b if counts[b] == 1 else f"{b}_{p.suffix.lower().lstrip('.')}") for p, b in base.items()}


# ---------------------------------------------------------------------------
# One image
# ---------------------------------------------------------------------------
def analyze_image(image: np.ndarray, source: str, blank_page: Optional[np.ndarray], client, args) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Whole pipeline on an image already in memory -> (result JSON, aligned page, debug image).
    Used by the folder batch (process_image) and by the web API; touches no files."""
    hm, align_info = align_page(image, blank_page)
    page = warp_with(image, hm)
    align_ok = bool(align_info["ok"])
    if not align_ok:
        print(f"[warn] Chỉ khớp được {align_info['boxes_used_to_refine']}/43 ô (cần >= {ALIGN_MIN_BOXES}); "
              "kết quả tick bị từ chối, nên chụp lại ảnh rõ/thẳng hơn.")

    boxes = detect_checkboxes(page)
    if not align_ok:
        reject_boxes(boxes)

    # adaptive note regions: only trusted when the page is aligned (positions are meaningless otherwise)
    notes: Dict[Tuple[str, str], Dict[str, Any]] = {}
    lines: Dict[str, List[Dict[str, Any]]] = {}
    ink = np.zeros(page.shape[:2], np.uint8)
    print_template = getattr(args, "print_template", None)
    if align_ok:
        notes, lines, ink = find_notes(page, boxes, print_template)

    ocr_errors: List[str] = []
    if client is None:
        header = {k: {"value": "", "confidence": None} for k in HEADER_ROIS}
        bottom = {k: {"value": "", "confidence": None} for k in SECTION_TEXT_ROIS}
    else:
        header, bottom, ocr_errors = ocr_all_fields(client, page)
        ocr_notes(client, page, ink, notes, lines)

    needs_review: List[str] = []
    if not align_ok:
        needs_review.append("ALIGNMENT_FAILED")
    for k, v in {**header, **bottom}.items():
        if k in ocr_errors or (v["value"] and (not align_ok or v["confidence"] is None or v["confidence"] < LOW_CONF)):
            needs_review.append(k)
    for key, sec in boxes.items():
        needs_review += [f"{key}:{lbl}" for lbl in sec["uncertain"]]

    def note_flag(n: Dict[str, Any]) -> bool:
        conf = n.get("confidence")
        if n.get("error") or (client is not None and not n.get("value")):  # request failed / ink but nothing readable
            return True
        return bool(n.get("value")) and (conf is None or conf < LOW_CONF)

    for (key, label), n in notes.items():
        if note_flag(n):
            needs_review.append(f"{key}:{label}:note")
        if not n["checked"] and "ghi rõ" not in label.lower():  # writing next to an empty square: a human decides
            needs_review.append(f"{key}:{label}:note_without_tick")
    for key, lst in lines.items():
        needs_review += [f"{key}:line_note{j + 1}" for j, n in enumerate(lst) if note_flag(n)]

    def public_note(n: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"value": n.get("value", ""), "confidence": n.get("confidence"), "region": n["region"]}
        if "checked" in n:
            out["checked"] = n["checked"]
        return out

    def section_json(k: str, s: Dict[str, Any]) -> Dict[str, Any]:
        sec = {
            "title": s["title"],
            "checked": s["checked"],
            "uncertain": s["uncertain"],
            "notes": {lbl: public_note(n) for (kk, lbl), n in notes.items() if kk == k},
        }
        if k in LINE_NOTE_ZONES:
            sec["line_notes"] = [public_note(n) for n in lines.get(k, [])]
        return sec

    result: Dict[str, Any] = {
        "form": "CHECKLIST HA",
        "source": source,
        "alignment": align_info,
        "notes_enabled": print_template is not None,
        "ocr": {"model": MODEL_NAME if client is not None else None, "base_url": OPENAI_BASE_URL if client is not None else None},
        "header": header,
        "sections": {
            **{k: section_json(k, s) for k, s in boxes.items()},
            **bottom,
        },
        "needs_review": needs_review,
    }
    if getattr(args, "verbose", False):
        result["_debug"] = {k: s["details"] for k, s in boxes.items()}
    return result, page, draw_debug(page, boxes, align_ok, notes, lines)


def process_image(path: Path, name: str, blank_page: Optional[np.ndarray], client, args, source: str) -> Dict[str, Any]:
    image = read_image(path)
    if image is None:
        raise ValueError("không đọc được ảnh (file hỏng hoặc không phải ảnh)")
    result, page, debug = analyze_image(image, source, blank_page, client, args)
    write_image(args.result_dir / "aligned" / f"{name}.jpg", page)
    write_image(args.result_dir / "debug" / f"{name}.png", debug)
    return result


def summarize(result: Dict[str, Any], source: str, json_name: str) -> Dict[str, Any]:
    """One line of _summary.json, built from a result dict (fresh or loaded from disk)."""
    entry: Dict[str, Any] = {"image": source, "json": json_name}
    if "error" in result:
        entry.update(status="error", error=result["error"])
        return entry
    entry["status"] = "ok" if result["alignment"]["ok"] else "alignment_failed"
    entry["squares_located"] = result["alignment"]["boxes_used_to_refine"]
    entry["checked"] = {k: v["checked"] for k, v in result["sections"].items() if v.get("checked")}
    entry["notes"] = sum(len(v.get("notes", {})) + len(v.get("line_notes", [])) for v in result["sections"].values())
    entry["needs_review"] = len(result["needs_review"])
    return entry


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CHECKLIST HA: cả thư mục ảnh -> JSON")
    p.add_argument("input", type=Path, nargs="?", default=Path("image"),
                   help="Thư mục chứa ảnh form đã điền (mặc định: image). Cũng nhận 1 file ảnh.")
    p.add_argument("--result-dir", type=Path, default=Path("result"),
                   help="Thư mục ảnh kết quả, chia 2 thư mục con: aligned/ (ảnh đã căn chỉnh) và debug/ (ảnh debug) (mặc định: result)")
    p.add_argument("--json-dir", type=Path, default=Path("result_json"),
                   help="Nơi ghi file JSON, mỗi ảnh một file + _summary.json (mặc định: result_json)")
    p.add_argument("--recursive", action="store_true", help="Quét cả thư mục con")
    p.add_argument("--skip-existing", action="store_true", help="Bỏ qua ảnh đã có file JSON (chạy tiếp khi bị ngắt, đỡ tốn API)")
    p.add_argument("--blank-template", type=Path, default=None, help="Ảnh form trống (tùy chọn, đã crop sát tờ giấy thì tốt hơn)")
    p.add_argument("--verbose", action="store_true", help="Ghi điểm từng checkbox vào JSON (_debug)")
    p.add_argument("--skip-ai", action="store_true", help="Bỏ qua OCR chữ viết tay")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not args.input.exists():
        raise SystemExit(f"Không tìm thấy: {args.input}")
    if args.input.is_dir():
        root, paths = args.input, list_images(args.input, args.recursive)
        if not paths:
            raise SystemExit(f"Không có ảnh nào trong {args.input} (hỗ trợ: {', '.join(sorted(IMAGE_EXTS))})")
    else:
        root, paths = args.input.parent, [args.input]
    names = output_names(paths, root)

    (args.result_dir / "aligned").mkdir(parents=True, exist_ok=True)
    (args.result_dir / "debug").mkdir(parents=True, exist_ok=True)
    args.json_dir.mkdir(parents=True, exist_ok=True)

    blank_page: Optional[np.ndarray] = None
    if args.blank_template is not None:
        blank_raw = read_image(args.blank_template)
        if blank_raw is None:
            raise SystemExit(f"Could not read blank template: {args.blank_template}")
        blank_page, blank_ok = warp_by_corners(blank_raw)
        if not blank_ok:
            print("[warn] Không tìm thấy 4 góc ở form trống; bỏ qua form trống (vẫn căn chỉnh được bằng lưới ô vuông).")
            blank_page = None

    args.print_template = load_print_template(read_image(args.blank_template) if args.blank_template is not None else None)
    if args.print_template is None:
        print(f"[warn] Không có {PRINT_TEMPLATE_PATH.name} (và không có --blank-template): BỎ QUA việc dò ghi chú cạnh ô. "
              "Chạy một lần với --blank-template <ảnh form trống> để tạo.")

    client = None
    if not args.skip_ai:
        validate_config()  # fail once, before the batch starts
        client = make_openai_client()

    summary: List[Dict[str, Any]] = []
    total = len(paths)
    for i, path in enumerate(paths, 1):
        name = names[path]
        source = str(path.relative_to(root)) if path != root and root in path.parents else path.name
        json_path = args.json_dir / f"{name}.json"
        if args.skip_existing and json_path.exists():
            # resume: keep finished images, but retry the ones that failed last time
            try:
                old = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                old = None
            if old is not None and "error" not in old:
                print(f"[{i}/{total}] {source}: đã có JSON, bỏ qua")
                summary.append(summarize(old, source, json_path.name))
                continue

        print(f"\n[{i}/{total}] {source}")
        try:
            result = process_image(path, name, blank_page, client, args, source)
        except Exception as exc:  # one bad file must not stop the batch
            print(f"[error] {source}: {type(exc).__name__}: {exc}")
            result = {"form": "CHECKLIST HA", "source": source, "error": f"{type(exc).__name__}: {exc}",
                      "needs_review": ["PROCESSING_FAILED"]}
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        summary.append(summarize(result, source, json_path.name))

    (args.json_dir / "_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    counts: Dict[str, int] = {}
    for e in summary:
        counts[e["status"]] = counts.get(e["status"], 0) + 1
    print("\n===== TỔNG KẾT =====")
    for e in summary:
        extra = f"{e['squares_located']}/43 ô, tick: {sum(len(v) for v in e['checked'].values())}, ghi chú: {e['notes']}, cần xem lại: {e['needs_review']}" \
            if "squares_located" in e else e.get("error", "")
        print(f"{e['status']:17s} {e['image']}  {extra}")
    print(f"\n{', '.join(f'{k}: {v}' for k, v in counts.items())}")
    print(f"JSON  -> {args.json_dir}/   (mỗi ảnh một file + _summary.json)")
    print(f"Ảnh   -> {args.result_dir}/aligned/ (ảnh căn chỉnh) và {args.result_dir}/debug/ (ảnh debug)")
    if counts.get("alignment_failed") or counts.get("error"):
        print("Có ảnh bị từ chối/lỗi: xem _summary.json, chụp lại các ảnh đó.")


if __name__ == "__main__":
    main()
