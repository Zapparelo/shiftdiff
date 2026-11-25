"""Shift-tolerant visual diff.

1. Both captures are scaled to a common width with area averaging (smooths anti-aliasing).
2. Every pixel row gets a coarse signature. Rows of the old and new capture are aligned like lines in a text diff
   (difflib), so content that only moved up or down (banner inserted, section grew) is matched to its new position.
3. Aligned row pairs are compared pixel by pixel with a colour tolerance; isolated changed pixels are dropped (noise).
4. Rows that exist only in the new capture count as changed where they differ from the page background,
   rows that disappeared count as removed. Masked regions are ignored in both captures.
"""
from __future__ import annotations

import difflib
import io
from collections import Counter
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

WORK_W = 640
TOL = 40          # max channel difference that still counts as equal (0-255)
SIG_BLOCK = 8     # horizontal pixels per signature cell
SIG_Q = 24        # quantisation step of the signature


def _load(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")


def _work(im: Image.Image, scale: float) -> np.ndarray:
    w = max(1, round(im.width * scale)); h = im.height  # only horizontal scaling: vertical shifts stay whole pixels
    return np.asarray(im.resize((w, h), Image.Resampling.BOX), dtype=np.int16)


def _fit_width(arr: np.ndarray, w: int) -> np.ndarray:
    if arr.shape[1] == w:
        return arr
    if arr.shape[1] > w:
        return arr[:, :w]
    pad = np.full((arr.shape[0], w - arr.shape[1], 3), 255, dtype=np.int16)
    return np.concatenate([arr, pad], axis=1)


def _signatures(arr: np.ndarray) -> list[bytes]:
    g = arr.mean(axis=2)
    w = (g.shape[1] // SIG_BLOCK) * SIG_BLOCK
    cells = g[:, :w].reshape(g.shape[0], -1, SIG_BLOCK).mean(axis=2)
    return [r.tobytes() for r in (cells // SIG_Q).astype(np.uint8)]


def _denoise(m: np.ndarray) -> np.ndarray:
    """Keep changed pixels that have at least 2 changed neighbours in their 3x3 block."""
    if not m.any():
        return m
    p = np.pad(m.astype(np.uint8), 1)
    n = sum(p[1 + dy:p.shape[0] - 1 + dy, 1 + dx:p.shape[1] - 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    return m & (n >= 3)


def _runs(sigs: list[bytes]) -> tuple[list[tuple[bytes, int]], list[int]]:
    """Collapse runs of identical rows (blank space, solid bars) into one token, so the alignment
    is anchored on real content instead of matching arbitrary blank rows."""
    toks: list[tuple[bytes, int]] = []; starts: list[int] = []
    for i, s in enumerate(sigs):
        if toks and toks[-1][0] == s:
            toks[-1] = (s, toks[-1][1] + 1)
        else:
            toks.append((s, 1)); starts.append(i)
    return toks, starts + [len(sigs)]


def _align(sa: list[bytes], sb: list[bytes]):
    """Returns row pairs from exactly matching content plus the blocks that still need an offset search."""
    ta, xa = _runs(sa); tb, xb = _runs(sb)
    sm = difflib.SequenceMatcher(None, [t[0] for t in ta], [t[0] for t in tb], autojunk=False)
    pa: list[int] = []; pb: list[int] = []; blocks: list[list[int]] = []; last_off = 0
    ops = sm.get_opcodes()
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            for k in range(i2 - i1):  # token by token: runs may differ in length (spacing grew or shrank)
                r1, r2, s1, s2 = xa[i1 + k], xa[i1 + k + 1], xb[j1 + k], xb[j1 + k + 1]
                n = min(r2 - r1, s2 - s1)
                pa += range(r1, r1 + n); pb += range(s1, s1 + n)
                if s2 - s1 > n:
                    blocks.append([r1 + n, r2, s1 + n, s2, s1 - r1, s1 - r1])
            last_off = xb[j2] - xa[i2]
        else:
            blocks.append([xa[i1], xa[i2], xb[j1], xb[j2], last_off, None])
    # fill in the offset of the next exact match for each block
    nxt = None
    for tag, i1, i2, j1, j2 in reversed(ops):
        if tag == "equal":
            nxt = xb[j1] - xa[i1]
    for b in blocks:
        if b[5] is None:
            later = [xb[j1] - xa[i1] for tag, i1, i2, j1, j2 in ops if tag == "equal" and xb[j1] >= b[3]]
            b[5] = later[0] if later else b[4]
    return pa, pb, blocks


def _best_pairs(a: np.ndarray, b: np.ndarray, blk: list[int]) -> tuple[list[int], list[int], list[int]]:
    """For a block of new rows, try the offsets of the surrounding matches (and top/bottom anchoring)
    and keep the one with the fewest differing pixels. Robust to JPEG noise and edits inside shifted content."""
    r1, r2, s1, s2, off_prev, off_next = blk
    if s2 <= s1:
        return [], [], []
    rows_b = np.arange(s1, s2)
    best = None
    for off in dict.fromkeys([off_prev, off_next, s1 - r1, s2 - r2]):
        ra = rows_b - off
        ok = (ra >= 0) & (ra < a.shape[0])
        if not ok.any():
            score = len(rows_b) * a.shape[1]
        else:
            score = int((np.abs(a[ra[ok]] - b[rows_b[ok]]).max(axis=2) > TOL).sum()) + int((~ok).sum()) * a.shape[1] // 4
        if best is None or score < best[0]:
            best = (score, off)
    ra = rows_b - best[1]
    ok = (ra >= 0) & (ra < a.shape[0])
    return ra[ok].tolist(), rows_b[ok].tolist(), rows_b[~ok].tolist()


def naive_pct(prev: bytes, cur: bytes, threshold: int = 24) -> float:
    """The previous position-by-position method (kept for comparison as pixelChangePct)."""
    a, b = _load(prev), _load(cur)
    w, h = max(a.width, b.width), max(a.height, b.height)
    ca = Image.new("RGB", (w, h), (255, 255, 255)); ca.paste(a, (0, 0))
    cb = Image.new("RGB", (w, h), (255, 255, 255)); cb.paste(b, (0, 0))
    s = max(1, w // 480)
    sz = (w // s, max(1, h // s))
    d = np.abs(np.asarray(ca.resize(sz).convert("L"), dtype=np.int16) - np.asarray(cb.resize(sz).convert("L"), dtype=np.int16))
    return round(100 * float((d > threshold).mean()), 2)


def visual_diff(prev: bytes, cur: bytes, regions: list[dict[str, float]] | None = None, make_image: bool = True) -> dict[str, Any]:
    """regions: rectangles {x, y, width, height} in pixels of the captured image, ignored in both captures."""
    A, B = _load(prev), _load(cur)
    scale = min(1.0, WORK_W / max(A.width, B.width))
    a, b = _work(A, scale), _work(B, scale)
    W = max(a.shape[1], b.shape[1])
    a, b = _fit_width(a, W), _fit_width(b, W)
    ignore_a = np.zeros(a.shape[:2], bool); ignore_b = np.zeros(b.shape[:2], bool)
    for r in regions or []:
        try:
            x0, y0 = int(float(r["x"]) * scale), int(float(r["y"]))
            x1, y1 = int((float(r["x"]) + float(r["width"])) * scale) + 1, int(float(r["y"]) + float(r["height"])) + 1
        except (KeyError, TypeError, ValueError):
            continue
        ignore_a[max(0, y0):y1, max(0, x0):x1] = True
        ignore_b[max(0, y0):y1, max(0, x0):x1] = True
    a[ignore_a] = 0; b[ignore_b] = 0
    # page background = most common colour of the new capture
    q = (b[::4, ::4] // 8).reshape(-1, 3)
    bg = np.array(Counter(map(tuple, q.tolist())).most_common(1)[0][0], dtype=np.int16) * 8 + 4

    pairs_a, pairs_b, blocks = _align(_signatures(a), _signatures(b))
    inserted: list[int] = []
    for blk in blocks:
        xa_, xb_, ins_ = _best_pairs(a, b, blk)
        pairs_a += xa_; pairs_b += xb_; inserted += ins_
    used = np.zeros(a.shape[0], bool)
    if pairs_a:
        used[np.array(pairs_a)] = True
    removed = np.where(~used)[0].tolist()
    removed_at: list[int] = []  # filled below for removed runs with real content
    mask = np.zeros(b.shape[:2], bool)
    if pairs_b:
        pa, pb = np.array(pairs_a), np.array(pairs_b)
        d = np.abs(a[pa] - b[pb]).max(axis=2) > TOL
        d &= ~ignore_b[pb]
        mask[pb] = d
        offsets = Counter((pb - pa).tolist())
    else:
        offsets = Counter()
    if inserted:
        ins = np.array(inserted)
        mask[ins] = (np.abs(b[ins] - bg).max(axis=2) > TOL) & ~ignore_b[ins]
    mask = _denoise(mask)
    removed_rows = len(removed)
    removed_px = 0
    if removed:
        rem = np.array(removed)  # content that is gone from the page; blank space that vanished does not count
        rmask = _denoise((np.abs(a[rem] - bg).max(axis=2) > TOL) & ~ignore_a[rem])
        removed_px = int(rmask.sum())
        has = rmask.any(axis=1)
        pa_arr, pb_arr = np.array(pairs_a or [0]), np.array(pairs_b or [0])
        order = np.argsort(pa_arr); pa_s, pb_s = pa_arr[order], pb_arr[order]
        k = 0
        while k < len(removed):  # mark runs of >= 8 removed rows that held content
            e = k
            while e + 1 < len(removed) and removed[e + 1] == removed[e] + 1:
                e += 1
            if e - k + 1 >= 8 and has[k:e + 1].any():
                q = np.searchsorted(pa_s, removed[k])
                removed_at.append(int(pb_s[q - 1]) + 1 if q > 0 else 0)
            k = e + 1
    union = W * (b.shape[0] + removed_rows)
    changed = int(mask.sum()) + removed_px
    pct = round(100 * changed / union, 3) if union else 0.0
    shift = max(((o, c) for o, c in offsets.items() if o != 0), key=lambda t: t[1], default=(0, 0))
    out: dict[str, Any] = {
        "changePct": pct,
        "contentShiftPx": shift[0] if shift[1] >= 0.05 * max(1, len(pairs_b)) else 0,
        "insertedPx": len(inserted),
        "removedPx": removed_rows,
        "ignoredPct": round(100 * float(ignore_b.mean()), 2),
        "regions": [],
        "image": None,
    }
    # changed bands (for the report and the diff image)
    rows = mask.any(axis=1)
    bands = []; start = None
    for y, v in enumerate(list(rows) + [False]):
        if v and start is None:
            start = y
        elif not v and start is not None:
            if bands and start - bands[-1][1] <= 16:
                bands[-1][1] = y
            else:
                bands.append([start, y])
            start = None
    for y0, y1 in bands:
        cols = np.where(mask[y0:y1].any(axis=0))[0]
        groups = np.split(cols, np.where(np.diff(cols) > 24)[0] + 1)  # separate boxes for separate columns
        for g in groups:
            rows_g = np.where(mask[y0:y1, g[0]:g[-1] + 1].any(axis=1))[0]
            out["regions"].append({"x": round(g[0] / scale), "y": y0 + int(rows_g[0]), "width": round((g[-1] + 1 - g[0]) / scale),
                                   "height": int(rows_g[-1] + 1 - rows_g[0])})
    if make_image and (pct > 0 or removed_at):
        big = Image.fromarray((mask * 255).astype(np.uint8)).resize(B.size, Image.Resampling.NEAREST)
        red = Image.new("RGB", B.size, (255, 0, 0))
        ov = Image.composite(Image.blend(B, red, 0.45), B, big)
        dr = ImageDraw.Draw(ov)
        for r in out["regions"][:50]:
            dr.rectangle([r["x"] - 3, r["y"] - 3, r["x"] + r["width"] + 3, r["y"] + r["height"] + 3], outline=(255, 0, 0), width=3)
        for y in removed_at:
            yy = y; dr.line([0, yy, B.width, yy], fill=(200, 0, 200), width=4)
        for r in regions or []:
            try:
                dr.rectangle([r["x"], r["y"], r["x"] + r["width"], r["y"] + r["height"]], outline=(120, 120, 120), width=2)
            except (KeyError, TypeError):
                pass
        buf = io.BytesIO(); ov.save(buf, "JPEG", quality=70)
        out["image"] = buf.getvalue()
    out["regions"] = out["regions"][:20]
    return out


def main(argv: list[str] | None = None) -> None:
    """Command line: shiftdiff old.png new.png [--out diff.jpg] [--ignore x,y,w,h ...]"""
    import argparse, json, pathlib
    ap = argparse.ArgumentParser(prog="shiftdiff", description="Shift-tolerant visual diff of two full-page screenshots.")
    ap.add_argument("old"); ap.add_argument("new")
    ap.add_argument("--out", default="diff.jpg", help="diff image (changes in red)")
    ap.add_argument("--ignore", nargs="*", default=[], help="rectangles x,y,width,height in image pixels")
    a = ap.parse_args(argv)
    regions = [dict(zip(("x", "y", "width", "height"), map(float, r.split(",")))) for r in a.ignore]
    old, new = pathlib.Path(a.old).read_bytes(), pathlib.Path(a.new).read_bytes()
    res = visual_diff(old, new, regions)
    if res["image"]:
        pathlib.Path(a.out).write_bytes(res["image"])
    print(json.dumps({"changePct": res["changePct"], "pixelByPixelPct": naive_pct(old, new), "contentShiftPx": res["contentShiftPx"],
                      "changedRegions": res["regions"], "diffImage": a.out if res["image"] else None}, indent=2))


if __name__ == "__main__":
    main()
