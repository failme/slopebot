"""Labelled data for a learned plan-safety estimate.

The controller plays a game (browser A, deterministic, no time limit). A second browser
(B, no rendering) follows the same keys. Every `every` steps, all the plans A considered are
played in B from the same state (save / play / restore) to see whether each really survives,
and saved with the features of the model's prediction for that plan (see run() in
geo_controller.js). B has to be separate: restoring the heap corrupts A's geometry capture.

    python bot/geo_data.py 1-4 data/plans_1-4.jsonl [EVERY]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geo_play import STEP_JS, open_game
from teacher import open_env

H = 60
A = [None, "ArrowLeft", "ArrowRight"]

JUDGE_JS = """([plans, H]) => {
  const K = {0: null, '-1': 'ArrowLeft', '1': 'ArrowRight'};
  const s0 = __bot.save(), air0 = Object.assign({}, __bot.air);
  const out = plans.map(keys => {
    const us = keys.concat(Array(Math.max(0, H - keys.length)).fill(0)).slice(0, H);
    __bot.load(s0); __bot.air = Object.assign({}, air0);
    for (let k = 0; k < H; k++) { __bot.run(1, 50, K[us[k]]); __bot.trackAir(); if (__bot.isDead()) return k; }
    for (let k = 0; k < 40; k++) {
      if (__bot.air.n === 0) return 999;
      __bot.run(1, 50, null); __bot.trackAir();
      if (__bot.isDead()) return H + k;
    }
    return H + 40;
  });
  __bot.load(s0); __bot.air = air0;
  return out;
}"""


def collect(seed, out, every=50, warm=60, max_steps=6000):
    a_env = open_game(seed, budget=1e9)
    a_env.js("() => { __bot.norender = true; }")
    b_env = open_env(seed, 64, 48)
    b_env.start_game()
    b_env.js("() => { __bot.lite(); __bot.findBall(); }")
    n = 0
    try:
        act = 0
        for t in range(max_steps):
            now = t >= warm and t % every == 0
            if now:
                a_env.js("() => { __bot.ctl.collect = []; }")
            nxt, dead, score, last = a_env.js(STEP_JS, [act, 1, False])
            b_ball, b_dead = b_env.js("([a]) => { __bot.run(1, 50, a); const d = __bot.stepped(); return [__bot.ball(), d]; }", [A[act]])
            if dead or b_dead:
                break
            if now:
                plans = a_env.js("() => { const c = __bot.ctl.collect; __bot.ctl.collect = null; return c; }")
                a_ball = a_env.js("() => __bot.ball()")
                if max(abs(p - q) for p, q in zip(a_ball, b_ball)) > 1e-3:
                    print(f"seed {seed} step {t}: the two browsers disagree {a_ball} {b_ball}; stopping", flush=True)
                    break
                real = b_env.js(JUDGE_JS, [[p["keys"] for p in plans], H])
                for p, r in zip(plans, real):
                    out.write(json.dumps({"seed": seed, "t": t, "tag": p["tag"], "chosen": p["tag"] == last["best"],
                                          "surv": p["surv"], "sc": p["sc"], "feats": p["feats"], "real": r}) + "\n")
                    n += 1
                out.flush()
                ok = sum(r == 999 for r in real)
                print(f"seed {seed} step {t} score {score}: {len(plans)} plans, {ok} really survive, "
                      f"chosen {last['best']} {'survives' if real[[p['tag'] for p in plans].index(last['best'])] == 999 else 'dies'}",
                      flush=True)
            act = nxt
    finally:
        a_env.close()
        b_env.close()
    return n


def main():
    seeds, path = sys.argv[1], sys.argv[2]
    every = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    lo, _, hi = seeds.partition("-")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        for s in range(int(lo), int(hi or lo) + 1):
            print(f"seed {s}: {collect(s, f, every=every)} samples", flush=True)


if __name__ == "__main__":
    main()
