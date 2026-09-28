"""Visual debugger for the geometry controller: top-down map + planned vs actual path."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image, ImageDraw

from geo_play import STEP_JS, open_game


def render(v, actual, scale=6, zspan=160, ref=None):
    """v: debugView() dict; actual: list of real [x, y, z] after this step."""
    NX, NZ, DX, DZ, x0, z0 = v["NX"], v["NZ"], v["DX"], v["DZ"], v["x0"], v["z0"]
    L0 = np.array(v["L0"], np.float32).reshape(NZ, NX)
    L1 = np.array(v["L1"], np.float32).reshape(NZ, NX)
    bx, by, bz = v["ball"]
    rows = int(zspan / DZ)
    W, H = int(NX * DX * scale), int(rows * DZ * scale)
    img = np.zeros((H, W, 3), np.uint8)
    # expected surface height along the ball's slope: y ~ by - (z - bz); colour by height relative to it
    for j in range(rows):
        z = z0 + (j + 0.5) * DZ
        yexp = by - 0.7 - (z - bz)
        for layer, arr in ((1, L1), (0, L0)):
            row = arr[j]
            ok = row > -1e30
            if not ok.any():
                continue
            d = row - yexp
            col = np.zeros((NX, 3), np.uint8)
            col[:, 1] = np.clip(120 + d * 10, 40, 255).astype(np.uint8) * ok  # green = surface
            col[:, 2] = np.clip(-d * 10, 0, 255).astype(np.uint8) * ok        # blue tint = below
            if layer == 1:
                col //= 2
            y0 = H - int((j + 1) * DZ * scale)
            for i in np.nonzero(ok)[0]:
                img[max(0, y0):y0 + int(DZ * scale), int(i * DX * scale):int((i + 1) * DX * scale)] = col[i]
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    X = lambda x: (x - x0) * scale
    Z = lambda z: H - (z - z0) * scale
    for o in v["obs"]:
        d.polygon([(X(o[0]), Z(o[1])), (X(o[2]), Z(o[3])), (X(o[4]), Z(o[5]))], outline=(255, 40, 40))
    for j in range(rows):
        if not np.isnan(v["edgeL"][j]):
            z = z0 + (j + 0.5) * DZ
            d.point((X(v["edgeL"][j]), Z(z)), fill=(255, 255, 0)); d.point((X(v["edgeR"][j]), Z(z)), fill=(255, 255, 0))
    pts = [(X(p[0]), Z(p[1])) for p in v["path"]]
    if len(pts) > 1:
        d.line(pts, fill=(255, 255, 255), width=2)
    if v["path"] and v["path"][-1][3]:
        p = v["path"][-1]; d.ellipse([X(p[0]) - 5, Z(p[1]) - 5, X(p[0]) + 5, Z(p[1]) + 5], outline=(255, 0, 255), width=2)
    apts = [(X(p[0]), Z(p[2])) for p in actual]
    if len(apts) > 1:
        d.line(apts, fill=(0, 200, 255), width=2)
    if ref:
        rp = [(X(p[0]), Z(p[2])) for p in ref if bz - 5 < p[2] < bz + zspan]
        if len(rp) > 1:
            d.line(rp, fill=(255, 160, 0), width=2)
    d.ellipse([X(bx) - 4, Z(bx if False else bz) - 4, X(bx) + 4, Z(bz) + 4], fill=(255, 255, 255))
    d.text((4, 4), str(v["last"]), fill=(255, 255, 255))
    return im


def main():
    seed, steps = int(sys.argv[1]), [int(x) for x in sys.argv[2].split(",")]
    out = sys.argv[3]
    ref = None
    if len(sys.argv) > 4:   # overlay a reference path: positions of a planner run on the same seed
        import json
        ref = json.load(open(sys.argv[4]))
    e = open_game(seed); e.js("() => { __bot.norender = true; }")
    a, t, views, pos = 0, 0, {}, []
    while t <= max(steps) + 40:
        a, dead, score, last = e.js(STEP_JS, [a, 1, False])
        pos.append(e.js("() => __bot.ball()"))
        if t in steps:
            views[t] = e.js("() => __bot.ctl.debugView()")
        t += 1
        if dead:
            print("died at", t, "score", score)
            break
    ims = [render(v, pos[k:k + 60], ref=ref) for k, v in sorted(views.items())]
    W = sum(i.width for i in ims); H = max(i.height for i in ims)
    sheet = Image.new("RGB", (W + 10 * len(ims), H))
    x = 0
    for i in ims:
        sheet.paste(i, (x, 0)); x += i.width + 10
    sheet.save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
