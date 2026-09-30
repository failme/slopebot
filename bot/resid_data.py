"""Collect training data for the learned correction of the ball model (train_resid.py).

The bot plays; at every step the controller simulates the step it just took with its own
model and logs the model's features (geo_controller.js residFeats) and how far the real ball's
move differed from the model's.

    python bot/resid_data.py 401-410 data/resid_401-410.jsonl
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geo_play import STEP_JS, open_game


def main():
    lo, _, hi = sys.argv[1].partition("-")
    out = sys.argv[2]
    with open(out, "a") as f:
        for seed in range(int(lo), int(hi or lo) + 1):
            e = open_game(seed)
            e.js("() => { __bot.norender = true; __bot.ctlParams.resid = 0; __bot.ctl.residLog = []; }")
            a, t = 0, 0
            try:
                while t < 8000:
                    a, dead, score, _ = e.js(STEP_JS, [a, 1, False])
                    t += 1
                    if dead:
                        break
                rows = e.js("() => __bot.ctl.residLog")
            finally:
                e.close()
            for k, r in enumerate(rows):
                f.write(json.dumps({"seed": seed, "t": k, "f": r[:-4], "err": r[-4:-1], "realAir": r[-1]}) + "\n")
            print(f"seed {seed}: score {score}, {len(rows)} steps", flush=True)


if __name__ == "__main__":
    main()
