"""Run the real-time geometry controller (bot/geo_controller.js).

    python bot/geo_play.py --seeds 1-5          # headless evaluation, fast (virtual time)
    python bot/geo_play.py --seed 7 --watch     # visible browser, real time
    python bot/geo_play.py --seed 7 --out runs/geo7.json   # save the keys, then:
    python bot/play.py --replay runs/geo7.json --record geo7.webm
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import ROOT, SlopeEnv

CTL_JS = os.path.join(ROOT, "bot", "geo_controller.js")
PLANNER_JS = os.path.join(ROOT, "bot", "planner.js")
RESID_JSON = os.environ.get("GEO_RESID") or os.path.join(ROOT, "bot", "resid.json")   # learned model correction
SAFETY_JSON = os.environ.get("GEO_SAFETY") or os.path.join(ROOT, "bot", "safety.json")   # GEO_SAFETY: try other weights

# One decision step inside the page: capture this frame's geometry while advancing 50 ms
# with the chosen key, then decide the next key.
STEP_JS = """([a, n, smooth]) => {
  const A = [null, 'ArrowLeft', 'ArrowRight'];
  let dead = false;
  for (let i = 0; i < n && !dead; i++) {
    // smooth (for watching): three 16.7 ms frames per step; the geometry of the last one is used
    if (smooth) __bot.run(2, 50 / 3, A[a]);
    __bot.geo.on = true; __bot.geo.begin();
    __bot.run(1, smooth ? 50 / 3 : 50, A[a]);
    __bot.geo.on = false;
    dead = __bot.stepped();
    if (!dead) a = __bot.ctl.decide();
  }
  return [a, dead, __bot.score(), __bot.ctl.last];
}"""


def open_game(seed, watch=False, window=(960, 720), res=(320, 240), budget=None):
    """budget: planning time per decision in ms (default: the controller's real-time budget).
    Tools that replay a game and need the same decisions again pass a huge budget."""
    e = SlopeEnv(headless=not watch, width=res[0], height=res[1], seed=seed, window=window if watch else None,
                 gpu=watch, geo=True).open()
    e.page.add_script_tag(path=PLANNER_JS)
    e.page.add_script_tag(path=CTL_JS)
    e.start_game()
    e.js("() => { __bot.findBall(); __bot.ctl.reset(); }")
    if budget is not None:
        e.js("b => { __bot.ctlParams.budget = b; }", budget)
    if os.path.exists(SAFETY_JSON):   # the learned plan-safety estimate (train_safety.py)
        with open(SAFETY_JSON) as f:
            e.js("w => { __bot.ctlSafety = w; }", json.load(f))
    if os.path.exists(RESID_JSON):   # the learned correction of the ball model (train_resid.py)
        with open(RESID_JSON) as f:
            e.js("w => { __bot.ctlResid = w; }", json.load(f))
    if os.environ.get("GEO_PARAMS"):   # e.g. GEO_PARAMS='{"gamma": 1}' to try other settings
        e.js("p => { Object.assign(__bot.ctlParams, p); }", json.loads(os.environ["GEO_PARAMS"]))
    return e


def play(seed, watch=False, max_steps=20000, verbose=True, window=(960, 720), out=None):
    """out: save the keys pressed (planner_run.py format), to replay or record a video of the
    game with play.py --replay (the game is deterministic)."""
    budget = float(os.environ["GEO_BUDGET"]) if os.environ.get("GEO_BUDGET") else None   # ms per decision
    e = open_game(seed, watch, window, res=(640, 480) if watch else (320, 240), budget=budget)
    if not watch:
        e.js("() => { __bot.norender = true; }")   # geometry is still captured
    a, t, t0, score, worst, total_ms = 0, 0, time.time(), 0, 0.0, 0.0
    recent, keys = [], []
    try:
        chunk = 1
        t_next = time.time()
        while t < max_steps:
            keys.append(a)
            a, dead, score, last = e.js(STEP_JS, [a, chunk, watch])
            t += chunk
            if last:
                worst = max(worst, last["ms"]); total_ms += last["ms"]
                recent.append(last["surv"]); recent = recent[-40:]
            if watch:
                t_next = max(t_next + 0.05, time.time() - 0.2)
                time.sleep(max(0.0, t_next - time.time()))
            if verbose and t % 500 == 0:
                print(f"  seed {seed} step {t} score {score} (decision {last and round(last['ms'], 1)} ms, "
                      f"sees {last and round(last['zAhead'])} ahead)", flush=True)
            if dead:
                if verbose:
                    print(f"  seed {seed} died; planned survival over the last 40 steps: {recent}", flush=True)
                break
    finally:
        e.close()
        if out:
            with open(out, "w") as f:
                json.dump({"seed": seed, "score": score, "actions": "".join("NLR"[k] for k in keys)}, f)
    return score, t, worst, total_ms / max(1, t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=None, help="e.g. 1-5")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--out", default=None, help="save the game's keys (one seed), for play.py --replay")
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
        sc, steps, worst, mean_ms = play(s, a.watch, a.max_steps, out=a.out)
        scores.append(sc)
        print(f"seed {s}: score {sc} ({steps} steps, decisions {mean_ms:.1f} ms on average, slowest {worst:.1f} ms)", flush=True)
    if len(scores) > 1:
        print("scores", scores, "mean", sum(scores) / len(scores))


if __name__ == "__main__":
    main()
