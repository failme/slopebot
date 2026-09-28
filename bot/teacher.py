"""Teacher = lookahead planner running in its own (never rendered) browser, kept in lockstep
with a second browser that renders what the player actually sees.

Both browsers load the same seed, so feeding them the same inputs keeps them identical;
the planner can freely save/restore its heap without disturbing the renderer's GL state.
"""
import base64
import os

import numpy as np

from env import SlopeEnv, ROOT

PLANNER_JS = os.path.join(ROOT, "bot", "planner.js")
ACTIONS = [None, "ArrowLeft", "ArrowRight"]
STEP_MS = 50  # one decision every 50ms
# During training each decision step is a single 50ms game frame: exactly what the planner's
# rollouts simulate, so the planner's predictions hold. (Watch mode renders 3 x 16.7ms frames.)
OBS_W, OBS_H = 128, 96

GRAB_JS = """([w, h]) => {
  const gl = __bot.gl, W = gl.drawingBufferWidth, H = gl.drawingBufferHeight;
  if (!__bot._px || __bot._px.length !== W * H * 4) __bot._px = new Uint8Array(W * H * 4);
  const px = __bot._px;
  gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, px);
  // Box-downsample to w x h, keep red and green channels, flip to top-down rows.
  const out = new Uint8Array(w * h * 2), sx = W / w, sy = H / h;
  for (let y = 0; y < h; y++) {
    const y0 = Math.floor(y * sy), y1 = Math.max(y0 + 1, Math.floor((y + 1) * sy));
    for (let x = 0; x < w; x++) {
      const x0 = Math.floor(x * sx), x1 = Math.max(x0 + 1, Math.floor((x + 1) * sx));
      let r = 0, g = 0, n = 0;
      for (let yy = y0; yy < y1; yy++) for (let xx = x0; xx < x1; xx++) {
        const o = ((H - 1 - yy) * W + xx) * 4; r += px[o]; g += px[o + 1]; n++;
      }
      out[(y * w + x) * 2] = r / n; out[(y * w + x) * 2 + 1] = g / n;
    }
  }
  let s = ''; for (let i = 0; i < out.length; i += 8192) s += String.fromCharCode.apply(null, out.subarray(i, i + 8192));
  return btoa(s);
}"""


def open_env(seed, width, height, headless=True, window=None, gpu=False):
    e = SlopeEnv(headless=headless, width=width, height=height, seed=seed, window=window, gpu=gpu).open()
    e.page.add_script_tag(path=PLANNER_JS)
    return e


class Renderer:
    """The browser the player 'sees'. Renders only the frame an observation is taken from."""

    def __init__(self, seed, width=256, height=192, headless=True, window=None, gpu=False):
        self.env = open_env(seed, width, height, headless, window, gpu)
        # The bot doesn't see the UI layer (score digits, "SPEED UP" banner): it would cover
        # the far end of the track.
        self.env.js("() => { __bot.hideUI = true; }")
        self.env.start_game()
        self.env.js("() => __bot.findBall()")

    def step(self, action, smooth=False):
        """Hold `action` for one decision step. Returns (ball position, dead).
        smooth: advance as three rendered 60fps frames instead of one 50ms frame."""
        return self.env.js("""([a, ms, smooth]) => {
          __bot.norender = false;
          if (smooth) __bot.run(3, ms / 3, a); else __bot.run(1, ms, a);
          return [__bot.ball(), __bot.stepped()];
        }""", [action, STEP_MS, smooth])

    def frames(self, n):
        self.env.frames(n)

    def score(self):
        return self.env.js("() => __bot.score()")

    def show_score(self, extra=""):
        """Watch mode: the game's own UI is hidden, so draw the score as an HTML overlay."""
        return self.env.js("""(extra) => {
          let d = document.getElementById('botscore');
          if (!d) {
            d = document.createElement('div'); d.id = 'botscore';
            d.style.cssText = 'position:fixed;top:10px;left:0;right:0;text-align:center;' +
              'font:bold 44px monospace;color:#3f3;text-shadow:0 0 8px #000;z-index:10;pointer-events:none;white-space:pre';
            document.body.appendChild(d);
          }
          const s = __bot.score();
          d.textContent = s + (extra ? '\n' + extra : '');
          return s;
        }""", extra)

    def obs(self):
        b = base64.b64decode(self.env.js(GRAB_JS, [OBS_W, OBS_H]))
        return np.frombuffer(b, np.uint8).reshape(OBS_H, OBS_W, 2)

    def frame(self):
        """Full-resolution RGB frame of the last rendered image."""
        w, h, b = self.env.js("""() => {
          const gl = __bot.gl, W = gl.drawingBufferWidth, H = gl.drawingBufferHeight;
          const px = new Uint8Array(W * H * 4);
          gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, px);
          let s = ''; for (let i = 0; i < px.length; i += 8192) s += String.fromCharCode.apply(null, px.subarray(i, i + 8192));
          return [W, H, btoa(s)];
        }""")
        a = np.frombuffer(base64.b64decode(b), np.uint8).reshape(h, w, 4)
        return a[::-1, :, :3].copy()

    def save_frame(self, path):
        from PIL import Image
        Image.fromarray(self.frame()).save(path)

    def close(self):
        self.env.close()


class Planner:
    def __init__(self, seed):
        self.env = open_env(seed, 64, 48)
        self.env.start_game()
        self.env.js("() => __bot.lite()")
        self.env.js("() => __bot.findBall()")

    def step(self, ai):
        """Take action index `ai` for one decision step. Returns (ball position, dead)."""
        return self.env.js("([ai, a, ms]) => { __bot.commit(ai); __bot.run(1, ms, a); return [__bot.ball(), __bot.stepped()]; }",
                           [ai, ACTIONS[ai], STEP_MS])

    def planned(self):
        return self.env.js("() => __bot.nextPlanned()")

    def eval(self, **cfg):
        return self.env.js("cfg => __bot.evalActions(cfg)", cfg)

    def close(self):
        self.env.close()
