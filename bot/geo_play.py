"""Run the real-time geometry controller (bot/geo_controller.js).

    python bot/geo_play.py --seeds 1-5          # headless evaluation, fast (virtual time)
    python bot/geo_play.py --seed 7 --watch     # visible browser, real time
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import ROOT, SlopeEnv

CTL_JS = os.path.join(ROOT, "bot", "geo_controller.js")
PLANNER_JS = os.path.join(ROOT, "bot", "planner.js")

# One decision step inside the page: capture this frame's geometry while advancing 50 ms
# with the chosen key, then decide the next key.
STEP_JS = """([a, n, smooth]) => {
  const A = [null, 'ArrowLeft', 'ArrowRight'];
  let dead = false;
  for (let i = 0; i < n && !dead; i++) {
    __bot.geo.on = true; __bot.geo.begin();
    if (smooth) __bot.run(3, 50 / 3, A[a]); else __bot.run(1, 50, A[a]);
    __bot.geo.on = false;
    dead = __bot.stepped();
    if (!dead) a = __bot.ctl.decide();
  }
  return [a, dead, __bot.score(), __bot.ctl.last];
}"""


def open_game(seed, watch=False, window=(960, 720), res=(320, 240)):
    e = SlopeEnv(headless=not watch, width=res[0], height=res[1], seed=seed, window=window if watch else None,
                 gpu=watch, geo=True).open()
    e.page.add_script_tag(path=PLANNER_JS)
    e.page.add_script_tag(path=CTL_JS)
    e.start_game()
    e.js("() => { __bot.findBall(); __bot.ctl.reset(); }")
    return e


def play(seed, watch=False, max_steps=20000, verbose=True, window=(960, 720)):
    e = open_game(seed, watch, window)
    if not watch:
        e.js("() => { __bot.norender = true; }")   # geometry is still captured
    a, t, t0, score, worst = 0, 0, time.time(), 0, 0.0
    try:
        chunk = 1 if watch else 20
        t_next = time.time()
        while t < max_steps:
            a, dead, score, last = e.js(STEP_JS, [a, chunk, watch])
            t += chunk
            if last:
                worst = max(worst, last["ms"])
            if watch:
                t_next = max(t_next + 0.05, time.time() - 0.2)
                time.sleep(max(0.0, t_next - time.time()))
            if verbose and t % 500 == 0:
                print(f"  seed {seed} step {t} score {score} (decision {last and round(last['ms'], 1)} ms, "
                      f"sees {last and round(last['zAhead'])} ahead)", flush=True)
            if dead:
                break
    finally:
        e.close()
    return score, t, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=None, help="e.g. 1-5")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--max-steps", type=int, default=20000)
    a = ap.parse_args()
    seeds = [a.seed] if a.seed is not None else []
    if a.seeds:
        lo, _, hi = a.seeds.partition("-")
        seeds += list(range(int(lo), int(hi or lo) + 1))
    if not seeds:
        import random
        seeds = [random.randint(1, 1 << 30)]
    scores = []
    for s in seeds:
        sc, steps, worst = play(s, a.watch, a.max_steps)
        scores.append(sc)
        print(f"seed {s}: score {sc} ({steps} steps, slowest decision {worst:.1f} ms)", flush=True)
    if len(scores) > 1:
        print("scores", scores, "mean", sum(scores) / len(scores))


if __name__ == "__main__":
    main()
