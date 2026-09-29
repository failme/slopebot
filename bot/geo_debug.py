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


def frame_png(e):
    import base64
    w, h, px = e.js("""() => { const gl = __bot.gl, W = gl.drawingBufferWidth, H = gl.drawingBufferHeight;
      const p = new Uint8Array(W * H * 4); gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, p);
      let s = ''; for (let i = 0; i < p.length; i += 8192) s += String.fromCharCode.apply(null, p.subarray(i, i + 8192));
      return [W, H, btoa(s)]; }""")
    return Image.fromarray(np.frombuffer(base64.b64decode(px), np.uint8).reshape(h, w, 4)[::-1, :, :3].copy())


def crash_report(seed, out, before=(30, 15, 4)):
    """Play until the crash, then replay and show map + game view shortly before it."""
    e = open_game(seed, budget=1e9); e.js("() => { __bot.norender = true; }")
    a, t = 0, 0
    while True:
        a, dead, score, last = e.js(STEP_JS, [a, 1, False]); t += 1
        if dead or t > 20000:
            break
    e.close()
    death = t
    steps = [max(0, death - b) for b in before]
    e = open_game(seed, budget=1e9)
    a, t, views, frames, pos = 0, 0, {}, {}, []
    while t < death + 2:
        a, dead, score, last = e.js(STEP_JS, [a, 1, False])
        pos.append(e.js("() => __bot.ball()"))
        if t in steps:
            views[t] = e.js("() => __bot.ctl.debugView()")
            frames[t] = frame_png(e)
        t += 1
        if dead:
            break
    e.close()
    panels = []
    for k in sorted(views):
        m = render(views[k], pos[k:k + 60])
        f = frames[k].resize((m.width, int(frames[k].height * m.width / frames[k].width)))
        p = Image.new("RGB", (m.width, m.height + f.height)); p.paste(f, (0, 0)); p.paste(m, (0, f.height))
        panels.append(p)
    W = sum(p.width for p in panels) + 10 * len(panels); H = max(p.height for p in panels)
    sheet = Image.new("RGB", (W, H)); x = 0
    for p in panels:
        sheet.paste(p, (x, 0)); x += p.width + 10
    sheet.save(out)
    return death, score


def main():
    if sys.argv[1] == "crash":
        print(crash_report(int(sys.argv[2]), sys.argv[3]))
        return
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
