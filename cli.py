"""Compare two screenshots: python cli.py old.png new.png [--out diff.jpg] [--ignore x,y,w,h ...]"""
import argparse, json, pathlib
from shiftdiff import visual_diff, naive_pct

ap = argparse.ArgumentParser(description="Shift-tolerant visual diff of two full-page screenshots.")
ap.add_argument("old"); ap.add_argument("new")
ap.add_argument("--out", default="diff.jpg", help="diff image (changes in red)")
ap.add_argument("--ignore", nargs="*", default=[], help="rectangles x,y,width,height in image pixels")
a = ap.parse_args()
regions = [dict(zip(("x", "y", "width", "height"), map(float, r.split(",")))) for r in a.ignore]
old, new = pathlib.Path(a.old).read_bytes(), pathlib.Path(a.new).read_bytes()
res = visual_diff(old, new, regions)
if res["image"]:
    pathlib.Path(a.out).write_bytes(res["image"])
print(json.dumps({"changePct": res["changePct"], "pixelByPixelPct": naive_pct(old, new), "contentShiftPx": res["contentShiftPx"],
                  "changedRegions": res["regions"], "diffImage": a.out if res["image"] else None}, indent=2))
