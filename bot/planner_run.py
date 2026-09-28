"""Let the lookahead planner (the teacher) play a whole game on its own and save the inputs.

    python bot/planner_run.py --seed 7 --out runs/seed7.json

The planner cheats (it reads game memory and uses save states to look ahead), so this is
not the trained bot; it is how the training labels are made. Because the game is
deterministic, the saved inputs can be replayed and watched: play.py --replay runs/seed7.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collect import teacher_pick
from teacher import ACTIONS, Planner, Renderer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=0, help="stop once this score is reached (0 = play until death)")
    ap.add_argument("--max-steps", type=int, default=100000)
    ap.add_argument("--watch", action="store_true",
                    help="mirror the game in a visible browser while the planner plays (slow motion: "
                         "each decision takes the planner ~0.5s). Use play.py --replay for real time.")
    a = ap.parse_args()
    P = Planner(a.seed)
    view = None
    if a.watch:
        view = Renderer(a.seed, width=640, height=480, headless=False, window=(960, 720), gpu=True)
        view.env.js("() => { __bot.hideUI = false; }")
    acts, prev, t0 = [], 0, time.time()
    score = 0
    try:
        for t in range(a.max_steps):
            if t % 2 == 0:
                vals = P.eval()["vals"]
                ai = teacher_pick(vals, prev)
            else:
                ai = P.planned()
            (b, dead) = P.step(ai)
            if view:
                view.step(ACTIONS[ai])
            acts.append(ai)
            prev = ai
            if t % 100 == 0:
                score = P.env.js("() => __bot.score()")
                print(f"step {t} score {score} {time.time() - t0:.0f}s", flush=True)
                save(a.out, a.seed, acts, score, False)
            if dead:
                break
            if a.target and t % 20 == 0 and P.env.js("() => __bot.score()") >= a.target:
                break
        score = P.env.js("() => __bot.score()")
    finally:
        P.close()
        if view:
            view.close()
    save(a.out, a.seed, acts, score, dead)
    print(f"final score {score} after {len(acts)} steps ({'died' if dead else 'stopped'})")


def save(path, seed, acts, score, dead):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({"seed": seed, "score": score, "died": dead, "step_ms": 50,
                   "actions": "".join("NLR"[a] for a in acts)}, f)


if __name__ == "__main__":
    main()
