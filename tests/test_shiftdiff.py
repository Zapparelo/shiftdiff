import io
import pathlib

import numpy as np
from PIL import Image

from shiftdiff import naive_pct, visual_diff

EX = pathlib.Path(__file__).resolve().parent.parent / "examples"


def _png(arr):
    b = io.BytesIO()
    Image.fromarray(arr).save(b, "PNG")
    return b.getvalue()


def test_identical_is_zero():
    img = (EX / "example-old.png").read_bytes()
    assert visual_diff(img, img, make_image=False)["changePct"] == 0


def test_example_shift_is_small():
    old, new = (EX / "example-old.png").read_bytes(), (EX / "example-new.png").read_bytes()
    res = visual_diff(old, new, make_image=False)
    assert res["changePct"] < naive_pct(old, new)
    assert res["changePct"] < 3


def test_inserted_banner_is_not_whole_page():
    rng = np.random.default_rng(0)
    page = np.full((1200, 400, 3), 255, np.uint8)
    for y in range(40, 1200, 60):
        page[y:y + 30, 20:380] = rng.integers(0, 255, 3)
    banner = np.zeros((50, 400, 3), np.uint8)
    banner[:] = (200, 30, 30)
    shifted = np.vstack([banner, page])[:1200]
    res = visual_diff(_png(page), _png(shifted), make_image=False)
    assert res["changePct"] < 10 < naive_pct(_png(page), _png(shifted))
