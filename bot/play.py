"""Run the trained bot on Slope.

    python bot/play.py --watch            # opens a browser window and plays in real time
    python bot/play.py --episodes 5       # headless evaluation, prints scores
    python bot/play.py --record run.gif   # headless, saves an animated GIF of the run
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from model import Student
from teacher import ACTIONS, STEP_MS, Renderer
from env import ROOT

DEFAULT_MODEL = os.path.join(ROOT, "models", "student.pt")


def run_episode(student, seed, watch=False, record=None, max_steps=100000, verbose=True, smooth=None,
                window=(960, 720)):
    smooth = watch if smooth is None else smooth
    R = Renderer(seed, headless=not watch, window=window if watch else None, gpu=watch)
    student.reset()
    frames = []
    t_next = time.time()
    try:
        for t in range(max_steps):
            obs = R.obs()
            a = student.act(obs)
            _, dead = R.step(ACTIONS[a], smooth=smooth)
            if record and t % 2 == 0:
                frames.append((R.frame(), R.score()))
            if watch and t % 4 == 0:
                R.show_score()
            if watch:  # keep real-time pace
                t_next = max(t_next + STEP_MS / 1000.0, time.time() - 0.2)
                time.sleep(max(0.0, t_next - time.time()))
            if verbose and t % 200 == 0:
                print(f"  step {t} score {R.score()}", flush=True)
            if dead:
                break
        score = R.score()
        if watch:
            R.show_score("GAME OVER")
            try:
                R.frames(40)  # let the crash play out on screen
            except Exception:
                pass
            time.sleep(3)
    finally:
        R.close()
    if record and frames:
        save_gif(frames, record)
    return score, t + 1


def save_gif(frames, path):
    from PIL import Image, ImageDraw
    ims = []
    for f, score in frames:
        im = Image.fromarray(f)
        d = ImageDraw.Draw(im)
        d.text((im.width // 2 - 12, 6), str(score), fill=(60, 255, 60))
        ims.append(im)
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=100, loop=0, optimize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=None, help="level seed (default: random)")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--watch", action="store_true", help="show the game in a browser window, real-time")
    ap.add_argument("--record", default=None, help="save an animated GIF of the (first) episode")
    ap.add_argument("--window", default="960x720", help="browser window size in watch mode")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(2)
    student = Student(a.model)
    rng = np.random.default_rng()
    scores = []
    for ep in range(a.episodes):
        seed = a.seed + ep if a.seed is not None else int(rng.integers(1, 1 << 30))
        window = tuple(int(v) for v in a.window.split("x"))
        score, steps = run_episode(student, seed, watch=a.watch, record=a.record if ep == 0 else None, window=window)
        scores.append(score)
        print(f"episode {ep}: seed {seed} score {score} ({steps} steps)", flush=True)
    if len(scores) > 1:
        print(f"mean {np.mean(scores):.1f}  median {np.median(scores):.1f}  max {max(scores)}")


if __name__ == "__main__":
    main()
