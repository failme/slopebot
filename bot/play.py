"""Run the bot on Slope.

    python bot/play.py --watch                    # trained CNN: browser window, real time
    python bot/play.py --episodes 5               # trained CNN: headless evaluation
    python bot/play.py --record run.webm          # trained CNN: headless, saved as video (.webm or .gif)
    python bot/play.py --replay runs/x.json --watch     # watch a recorded run (planner_run.py)
    python bot/play.py --replay runs/x.json --record x.webm
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from env import ROOT
from teacher import ACTIONS, STEP_MS, Renderer

DEFAULT_MODEL = os.path.join(ROOT, "models", "student.pt")


class VideoWriter:
    """Writes frames to .webm (via ffmpeg, e.g. the one bundled with Playwright) or .gif."""

    def __init__(self, path, fps=1000 // STEP_MS):
        self.path, self.fps, self.frames, self.proc = path, fps, [], None
        if path.endswith(".webm"):
            exe = shutil.which("ffmpeg") or next(iter(sorted(
                glob.glob(os.path.expanduser("~/.cache/ms-playwright/ffmpeg-*/ffmpeg-*")) +
                glob.glob("/opt/pw-browsers/ffmpeg-*/ffmpeg-*"))), None)
            if not exe:
                raise SystemExit("no ffmpeg found for .webm output; use a .gif path instead")
            self.proc = subprocess.Popen(
                [exe, "-y", "-loglevel", "error", "-f", "image2pipe", "-c:v", "mjpeg", "-framerate", str(fps),
                 "-i", "pipe:0", "-an", "-c:v", "vp8", "-b:v", "2M", path], stdin=subprocess.PIPE)

    def add(self, frame, text=None):
        from PIL import Image, ImageDraw
        im = Image.fromarray(frame)
        if text is not None:
            ImageDraw.Draw(im).text((im.width // 2 - 12, 6), str(text), fill=(60, 255, 60))
        if self.proc:
            im.save(self.proc.stdin, "JPEG", quality=90)
        else:
            self.frames.append(im)

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        elif self.frames:
            self.frames[0].save(self.path, save_all=True, append_images=self.frames[1:],
                                duration=1000 // self.fps, loop=0, optimize=True)


def pace(t_next):
    """Sleep so that one decision step takes STEP_MS of wall time."""
    t_next = max(t_next + STEP_MS / 1000.0, time.time() - 0.2)
    time.sleep(max(0.0, t_next - time.time()))
    return t_next


def run_episode(student, seed, watch=False, record=None, max_steps=100000, verbose=True, smooth=None,
                window=(960, 720)):
    """Let the trained CNN play one game. Returns (score, steps)."""
    smooth = watch if smooth is None else smooth
    R = Renderer(seed, headless=not watch, window=window if watch else None, gpu=watch)
    student.reset()
    video = VideoWriter(record) if record else None
    t_next = time.time()
    score, t = 0, 0
    try:
        for t in range(max_steps):
            a = student.act(R.obs())
            _, dead = R.step(ACTIONS[a], smooth=smooth)
            if video:
                video.add(R.frame(), R.score())
            if watch:
                if t % 4 == 0:
                    R.show_score()
                t_next = pace(t_next)
            if verbose and t % 200 == 0:
                print(f"  step {t} score {R.score()}", flush=True)
            if dead:
                break
        score = R.score()
        if watch:
            R.show_score("GAME OVER")
            time.sleep(3)
    finally:
        R.close()
        if video:
            video.close()
    return score, t + 1


def replay(path, watch=True, window=(960, 720), record=None, res=(640, 480)):
    """Replay a recorded input sequence (e.g. from planner_run.py). The game is deterministic,
    so the same seed + inputs reproduce the run exactly (at 20 frames/s, as it was played).
    The game's own HUD is shown."""
    with open(path) as f:
        run = json.load(f)
    R = Renderer(run["seed"], width=res[0], height=res[1], headless=not watch,
                 window=window if watch else None, gpu=watch)
    R.env.js("() => { __bot.hideUI = false; }")
    video = VideoWriter(record) if record else None
    t_next = time.time()
    try:
        for c in run["actions"]:
            _, dead = R.step(ACTIONS["NLR".index(c)])
            if watch:
                t_next = pace(t_next)
            if video:
                video.add(R.frame())
            if dead:
                break
        score = R.score()
        if video:  # hold the final frame for a second
            for _ in range(1000 // STEP_MS):
                video.add(R.frame())
        if watch:
            time.sleep(3)
    finally:
        R.close()
        if video:
            video.close()
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=None, help="level seed (default: random)")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--watch", action="store_true", help="show the game in a browser window, real time")
    ap.add_argument("--window", default="960x720", help="browser window size in watch mode")
    ap.add_argument("--record", default=None, help="save a video (.webm or .gif) of the (first) episode")
    ap.add_argument("--replay", default=None, help="replay a recorded run (from planner_run.py)")
    a = ap.parse_args()
    window = tuple(int(v) for v in a.window.split("x"))
    if a.replay:
        print("score", replay(a.replay, watch=a.watch, window=window, record=a.record))
        return
    if not os.path.exists(a.model):
        raise SystemExit(f"no model at {a.model}: train one with bot/train.py (see README), "
                         "or use --replay to watch a recorded planner game")
    import torch
    from model import Student
    torch.set_num_threads(2)
    student = Student(a.model)
    rng = np.random.default_rng()
    scores = []
    for ep in range(a.episodes):
        seed = a.seed + ep if a.seed is not None else int(rng.integers(1, 1 << 30))
        score, steps = run_episode(student, seed, watch=a.watch, record=a.record if ep == 0 else None,
                                   window=window)
        scores.append(score)
        print(f"episode {ep}: seed {seed} score {score} ({steps} steps)", flush=True)
    if len(scores) > 1:
        print(f"mean {np.mean(scores):.1f}  median {np.median(scores):.1f}  max {max(scores)}")


if __name__ == "__main__":
    main()
