"""Synthetic regression test: draw a fake CHECKLIST HA page (ticks + handwritten notes), photograph
it at awkward angles/scales, then check alignment, tick detection and the adaptive note regions
(no real photos needed)."""
import sys
import tempfile
import types
from pathlib import Path

import cv2
import numpy as np

import ocr

rng = np.random.default_rng(0)
INK = (170, 40, 20)  # BGR: blue pen
TICKED = {"Nhăn mắt", "Rãnh mũi", "Ngạch cằm"}
# notes written next to a box: (section, label) -> (text, dx, dy) relative to the box centre
NOTES = {
    ("1_Nhan", "Nhăn mắt"): ("10U", 150, 12),
    ("2_Day", "Rãnh mũi"): ("1.2ml", 150, 12),
    ("1_Nhan", "Khác (ghi rõ)"): ("Vien Ham 20u", 215, 14),
    ("2_Day", "Cằm"): ("5u", 110, 12),  # next to an UNTICKED box -> must be flagged
}
LINE_NOTE = ("3_THD", "2 soi", 300, 1070)  # on the free-text rule under section 3


def make_page(ticks=TICKED, writing=True) -> np.ndarray:
    page = np.full((ocr.H, ocr.W, 3), 245, np.uint8)
    cv2.putText(page, "CHECKLIST HA", (390, 110), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (40, 40, 40), 4, cv2.LINE_AA)
    for y in (190, 250, 300, 340, 380, 1070, 1210, 1430, 1500, 1570):  # dotted lines
        for x in range(120, 1120, 8):
            cv2.circle(page, (x, y), 1, (120, 120, 120), -1)
    pos = {}
    for key, items in ocr.CHECKBOXES.items():
        for label, x, y in items:
            x, y = int(round(x)), int(round(y))
            pos[(key, label)] = (x, y)
            cv2.rectangle(page, (x - 13, y - 13), (x + 13, y + 13), (70, 70, 70), 2)
            cv2.putText(page, label, (x + 28, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 1, cv2.LINE_AA)
            if writing and label in ticks:
                pts = np.array([[x - 12, y], [x - 3, y + 11], [x + 24, y - 36]], np.int32)
                cv2.polylines(page, [pts], False, INK, 3, cv2.LINE_AA)
    if not writing:  # the blank form: print only
        return page
    # registration text (DV dang ky): scribbles plus one long line whose last stroke hangs down
    # into the first checkbox row, like the closing bracket of a real form
    for _ in range(30):
        x, y = int(rng.integers(150, 1000)), int(rng.integers(150, 360))
        cv2.line(page, (x, y), (x + int(rng.integers(10, 60)), y + int(rng.integers(-25, 25))), INK, 3, cv2.LINE_AA)
    zig = np.array([[x, 350 if i % 2 == 0 else 385] for i, x in enumerate(range(150, 1100, 40))] + [[1100, 435]], np.int32)
    cv2.polylines(page, [zig], False, INK, 3, cv2.LINE_AA)
    # handwritten notes next to the boxes + one on a free-text rule
    for (key, label), (text, dx, dy) in NOTES.items():
        x, y = pos[(key, label)]
        cv2.putText(page, text, (x + dx, y + dy), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.2, INK, 3, cv2.LINE_AA)
    cv2.putText(page, LINE_NOTE[1], (LINE_NOTE[2], LINE_NOTE[3] - 8), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.2, INK, 3, cv2.LINE_AA)
    return page


def photograph(page, scale=(0.70, 0.72), angle=1.5, shift=(10, -35), persp=2e-5, size=(853, 1245), bg=None):
    sx, sy = scale
    a = np.deg2rad(angle)
    m = np.array([[sx * np.cos(a), -sx * np.sin(a), shift[0]], [sy * np.sin(a), sy * np.cos(a), shift[1]], [persp, persp * 0.5, 1.0]])
    canvas = bg if bg is not None else np.full((size[1], size[0], 3), 200, np.uint8)
    cv2.warpPerspective(page, m, size, dst=canvas, borderMode=cv2.BORDER_TRANSPARENT)
    grad = np.linspace(0.85, 1.05, size[0])[None, :, None]
    img = np.clip(canvas.astype(np.float32) * grad + rng.normal(0, 4, canvas.shape), 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (3, 3), 1.0)


def warm_lamp(img):
    """Warm lamp as real cameras render it: the paper turns orange AND the dark print loses its colour."""
    img = np.clip(img.astype(np.float32) * np.array([0.72, 0.9, 1.15]), 0, 255)
    g = img.mean(axis=2, keepdims=True)
    a = np.clip((190 - g) / 120, 0, 1) * 0.6
    return np.clip(img * (1 - a) + g * a, 0, 255).astype(np.uint8)


# print template = photo of the BLANK form (what template_print.png is), built exactly like the real one
TPL = ocr.build_print_template(photograph(make_page(ticks=(), writing=False)))


def run(name, img, want_rot=None):
    hm, info = ocr.align_page(img, None)
    page = ocr.warp_with(img, hm)
    boxes = ocr.detect_checkboxes(page)
    if not info["ok"]:
        ocr.reject_boxes(boxes)
    got = {l for s in boxes.values() for l in s["checked"]}
    uncertain = sum(len(s["uncertain"]) for s in boxes.values())
    notes, lines, _ = ocr.find_notes(page, boxes, TPL) if info["ok"] else ({}, {}, None)
    got_notes = set(notes)
    line_ok = list(lines) == [LINE_NOTE[0]] and len(lines[LINE_NOTE[0]]) == 1
    ok = info["ok"] and got == TICKED and got_notes == set(NOTES) and line_ok
    if want_rot is not None:
        ok = ok and info["rotation"] == want_rot
    print(f"{name:34s} method={info['method']:18s} rot={info['rotation']:3d} squares={info['boxes_used_to_refine']:2d} ok={info['ok']} "
          f"checked={len(got)} uncertain={uncertain} notes={len(got_notes)}/{len(NOTES)} line_notes={'1' if line_ok else 'WRONG'} "
          f"-> {'PASS' if ok else 'FAIL'}")
    if got_notes != set(NOTES):
        print("    missing:", sorted(set(NOTES) - got_notes), "extra:", sorted(got_notes - set(NOTES)))
    return ok


page = make_page()
marble = rng.integers(120, 190, (1280, 960, 3)).astype(np.uint8)
marble = cv2.GaussianBlur(marble, (0, 0), 6)

cases = {
    "like failing photo (tight, tilted)": photograph(page),
    "rotated -3deg": photograph(page, angle=-3.0, shift=(30, 0)),
    "stronger perspective": photograph(page, persp=6e-5, angle=2.0),
    "paper small on textured background": photograph(page, scale=(0.60, 0.60), shift=(190, 120), size=(960, 1280), bg=marble.copy()),
    "high-res phone photo": photograph(page, scale=(2.2, 2.3), shift=(40, -100), size=(2700, 3600)),
}
cases["warm lamp (print turns neutral)"] = warm_lamp(cases["like failing photo (tight, tilted)"])
results = [run(n, im, want_rot=0) for n, im in cases.items()]

# sideways / upside-down photos: the pixels themselves are rotated (no EXIF), the page must come back upright
base_photo = cases["like failing photo (tight, tilted)"]
for turn, code in ((90, cv2.ROTATE_90_CLOCKWISE), (180, cv2.ROTATE_180), (270, cv2.ROTATE_90_COUNTERCLOCKWISE)):
    results.append(run(f"photo turned {turn} deg", cv2.rotate(base_photo, code), want_rot=(360 - turn) % 360))
results.append(run("high-res photo turned 90 deg", cv2.rotate(cases["high-res phone photo"], cv2.ROTATE_90_CLOCKWISE), want_rot=270))

# pages with NO handwriting must produce no notes at all - the printed labels are not notes.
# (This is the bug that was reported: unticked boxes got their own printed label as a "note".)
for tag, light in (("daylight", lambda im: im), ("warm lamp", warm_lamp)):
    b_img = light(photograph(make_page(ticks=(), writing=False), angle=-1.0, shift=(14, -30)))
    hm, info = ocr.align_page(b_img, None)
    bp = ocr.warp_with(b_img, hm)
    n_blank, l_blank, _ = ocr.find_notes(bp, ocr.detect_checkboxes(bp), TPL)
    blank_ok = info["ok"] and not n_blank and not l_blank
    print(f"print-only page, {tag} -> no notes: {blank_ok}" + ("" if blank_ok else f"  extra={sorted(n_blank)}"))
    results.append(blank_ok)

# without a print template notes are skipped (never guessed)
bp_notes, bp_lines, _ = ocr.find_notes(bp, ocr.detect_checkboxes(bp), None)
print("no template -> notes skipped:", not bp_notes and not bp_lines)
results.append(not bp_notes and not bp_lines)

# junk must be REJECTED, never reported as ticked
junk = np.full((1245, 853, 3), 210, np.uint8)
cv2.randn(junk, 210, 12)
hm, info = ocr.align_page(junk, None)
b = ocr.detect_checkboxes(ocr.warp_with(junk, hm))
if not info["ok"]:
    ocr.reject_boxes(b)
junk_ok = (not info["ok"]) and not any(s["checked"] for s in b.values())
print("junk image rejected:", junk_ok)
results.append(junk_ok)


# ---- whole pipeline with a stub model: JSON shape, confidence handling, needs_review ---------------
class _Resp:
    def __init__(self, text):
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=text))]


class _Stub:
    """Answers per prompt: 10U (sure), 1.2ml (unsure), unreadable for the line note, text otherwise."""

    def __init__(self):
        self.chat = types.SimpleNamespace(completions=self)

    def create(self, **kw):
        txt = [p for p in kw["messages"][1]["content"] if p["type"] == "text"][0]["text"]
        if "'Nhăn mắt'" in txt:
            return _Resp('{"value":"10U","confidence":0.95}')
        if "'Rãnh mũi'" in txt:
            return _Resp('{"value":"1,2ml","confidence":0.3}')
        if "'Khác (ghi rõ)'" in txt:
            return _Resp('{"value":"Viền Hàm 20u","confidence":0.9}')
        if "'Cằm'" in txt:
            return _Resp('{"value":"5u","confidence":0.9}')
        if "dòng kẻ" in txt:
            return _Resp('{"value":"","confidence":0.0}')
        return _Resp('{"value":"","confidence":0.0}')


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    (tmp / "aligned").mkdir()
    (tmp / "debug").mkdir()
    cv2.imwrite(str(tmp / "p.jpg"), cases["like failing photo (tight, tilted)"])
    args = types.SimpleNamespace(result_dir=tmp, json_dir=tmp, verbose=False, print_template=TPL)
    r = ocr.process_image(tmp / "p.jpg", "p", None, _Stub(), args, "p.jpg")
    s1, s2, s3 = r["sections"]["1_Nhan"], r["sections"]["2_Day"], r["sections"]["3_THD"]
    nr = set(r["needs_review"])
    checks = {
        "note value + checked flag": s1["notes"]["Nhăn mắt"]["value"] == "10U" and s1["notes"]["Nhăn mắt"]["checked"] is True,
        "note region is [x1,y1,x2,y2] ints": all(isinstance(v, int) for v in s1["notes"]["Nhăn mắt"]["region"]),
        "'ghi rõ' note kept, box NOT ticked": s1["notes"]["Khác (ghi rõ)"]["checked"] is False and "Khác (ghi rõ)" not in s1["checked"],
        "low-confidence note flagged": "2_Day:Rãnh mũi:note" in nr,
        "note beside empty box flagged": "2_Day:Cằm:note_without_tick" in nr,
        "'ghi rõ' note not flagged for missing tick": "1_Nhan:Khác (ghi rõ):note_without_tick" not in nr,
        "confident notes not flagged": "1_Nhan:Nhăn mắt:note" not in nr and "2_Day:Cằm:note" not in nr,
        "unreadable line note flagged": "3_THD:line_note1" in nr,
        "line_notes only on sections 3/4": "line_notes" in s3 and "line_notes" not in s1 and "line_notes" in r["sections"]["4_CangChi"],
        "ticks unchanged by notes": set(s1["checked"]) | set(s2["checked"]) == TICKED,
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        results.append(ok)
    # the same page with --skip-ai: regions found, no values, nothing flagged for OCR
    r2 = ocr.process_image(tmp / "p.jpg", "p2", None, None, args, "p.jpg")
    results.append(r["notes_enabled"] is True)
    skip_ok = r2["sections"]["1_Nhan"]["notes"]["Nhăn mắt"]["value"] == "" and not any(x.endswith(":note") for x in r2["needs_review"])
    print(f"  {'PASS' if skip_ok else 'FAIL'}  skip-ai keeps regions, no OCR flags")
    results.append(skip_ok)

sys.exit(0 if all(results) else 1)
