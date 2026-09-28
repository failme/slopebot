"""Headless (or headed) Slope environment.

The Unity WebGL build is served locally and loaded in Chromium with bot/hooks.js
injected, which virtualises time so the game only advances when we step it.
"""
import functools
import http.server
import os
import socketserver
import threading

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAME_DIR = os.path.join(ROOT, "game")
HOOKS = os.path.join(ROOT, "bot", "hooks.js")
FRAME_MS = 1000.0 / 60.0


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def serve(port=0):
    handler = functools.partial(_Quiet, directory=GAME_DIR)
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", port), handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


_PW = None


def _playwright():
    # Only one sync Playwright instance may exist per thread; share it between envs.
    global _PW
    if _PW is None:
        _PW = sync_playwright().start()
    return _PW


class SlopeEnv:
    def __init__(self, headless=True, width=640, height=480, seed=1, window=None, gpu=False):
        """width/height: render resolution. window: (w, h) of the browser window (defaults to
        the render size). gpu: use the real GPU instead of SwiftShader (for watching)."""
        self.window = window or (width, height)
        self.gpu = gpu
        self.headless = headless
        self.seed = seed
        self.width, self.height = width, height

    def open(self):
        self.httpd = serve()
        port = self.httpd.server_address[1]
        self.browser = _playwright().chromium.launch(
            headless=self.headless,
            args=[] if self.gpu else ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
        )
        self.page = self.browser.new_page(viewport={"width": self.window[0], "height": self.window[1]})
        self.page.add_init_script(path=HOOKS)
        self.page.route("**/*", lambda r: r.continue_() if "127.0.0.1" in r.request.url else r.abort())
        # manual=1: time only advances when we step, so a run is reproducible for a given seed.
        self.page.goto(f"http://127.0.0.1:{port}/index.html?manual=1&seed={self.seed}")
        self.page.wait_for_function("window.__bot && __bot.pending() > 0 && gameInstance.Module && gameInstance.Module.HEAPU8", timeout=120000, polling=50)
        return self

    def set_res(self, w, h):
        """Fix the game's render resolution (independent of the window size)."""
        self.js("([w, h]) => __bot.setRes(w, h)", [w, h])

    def start_game(self):
        """Step through the intro to the menu, press Play and run until the ball is under control."""
        # The menu layout depends on the render resolution, so press Play at 640x480
        # (in a 640x480 window) and switch to the requested resolution afterwards.
        vp = self.page.viewport_size
        self.page.set_viewport_size({"width": 640, "height": 480})
        self.set_res(640, 480)
        self.frames(600)
        self.page.mouse.click(320, 282)
        self.frames(10)
        self.page.set_viewport_size(vp)
        self.set_res(self.width, self.height)
        self.frames(390)

    def close(self):
        self.browser.close()
        self.httpd.shutdown()

    def js(self, code, arg=None):
        return self.page.evaluate(code, arg)

    def frames(self, n, key=None):
        """Advance n frames, holding `key` ('ArrowLeft'/'ArrowRight'/None) for the duration."""
        return self.js("""([n, dt, key]) => { return __bot.run(n, dt, key); }""", [n, FRAME_MS, key])

    def set_manual(self, m=True):
        self.js("m => __bot.setManual(m)", m)

    def screenshot(self, path):
        self.page.screenshot(path=path)
