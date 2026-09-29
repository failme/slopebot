"""Were the bot's own candidate plans any good? At checkpoints before the end of a game, collect
every plan the controller considers, play each plan's keys in the real game (save/restore),
and compare with the model's verdicts and ranking.

    python bot/geo_plans.py 17 --before-death 30,60,90,130
    python bot/geo_plans.py 1-8 --params '{"robustK": 20}'   # another decision rule at the same states
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geo_eval import record
from geo_play import STEP_JS, open_game

H = 60

JUDGE_JS = """([H]) => {
  const K = {0: null, '-1': 'ArrowLeft', '1': 'ArrowRight'};
  const plans = __bot.ctl.collect; __bot.ctl.collect = null;
  const s0 = __bot.save(), air0 = Object.assign({}, __bot.air);
  const real = plans.map(p => {
    const us = p.keys.concat(Array(Math.max(0, H - p.keys.length)).fill(0)).slice(0, H);
    __bot.load(s0); __bot.air = Object.assign({}, air0);
    for (const u of us) { __bot.run(1, 50, K[u]); __bot.trackAir(); if (__bot.isDead()) return false; }
    for (let k = 0; k < 40; k++) {
      if (__bot.air.n === 0) return true;
      __bot.run(1, 50, null); __bot.trackAir();
      if (__bot.isDead()) return false;
    }
    return false;
  });
  __bot.load(s0); __bot.air = air0;
  return plans.map((p, i) => [p.tag, p.surv, +p.sc.toFixed(1), real[i]]);
}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seeds", help="e.g. 3 or 1-8")
    ap.add_argument("--before-death", default="30,60,90,130")
    ap.add_argument("--params", default=None, help="JSON: controller parameters for the decision at the checkpoint "
                    "(the game up to there is the bot's own, with its default parameters)")
    a = ap.parse_args()
    params = json.loads(a.params) if a.params else {}
    lo, _, hi = a.seeds.partition("-")
    tally = [0, 0, 0]   # checkpoints where some plan really survives / the chosen one does / all
    for seed in range(int(lo), int(hi or lo) + 1):
        tally = [x + y for x, y in zip(tally, run_seed(seed, a.before_death, params))]
    print(f"TOTAL: {tally[2]} checkpoints; some plan really survives at {tally[0]}; "
          f"the chosen plan really survives at {tally[1]}", flush=True)


def run_seed(seed, before_death, params):
    keys, score = record(seed)
    n = len(keys)
    print(f"seed {seed}: game ends at step {n} (score {score})", flush=True)
    tally = [0, 0, 0]
    for b in [int(x) for x in before_death.split(",")]:
        T = n - b
        if T < 5:
            continue
        e = open_game(seed, budget=1e9)
        e.js("() => { __bot.norender = true; }")
        try:
            # replay the bot's own decisions (deterministic) up to step T, collecting the plans there
            act = 0
            for t in range(T + 1):
                if t == T:
                    e.js("p => { __bot.ctl.collect = []; Object.assign(__bot.ctlParams, p); }", params)
                act, dead, sc, last = e.js(STEP_JS, [act, 1, False])
            chosen = last["best"]
            rows = e.js(JUDGE_JS, [H])
        finally:
            e.close()
        ok = [r for r in rows if r[3]]
        ranked = sorted(rows, key=lambda r: -r[2])
        rank = next((i for i, r in enumerate(ranked) if r[3]), None)
        ch = next((r for r in rows if r[0] == chosen), None)
        print(f"  {b:3d} steps before the end: {len(rows)} plans, {len(ok)} really survive; "
              f"chosen {chosen} (model {ch and ch[1]}, really {'survives' if ch and ch[3] else 'dies'}); "
              f"best-scored plan that really survives: {'rank ' + str(rank + 1) if rank is not None else 'none'}"
              + (f" ({ranked[rank][0]}, model survival {ranked[rank][1]})" if rank is not None else ""), flush=True)
        tally[2] += 1
        if ok:
            tally[0] += 1
            if ch and ch[3]:
                tally[1] += 1
    return tally


if __name__ == "__main__":
    main()
