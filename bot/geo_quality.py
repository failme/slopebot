"""How good is the controller's ball model for planning?

At checkpoints of a real game, sample many key sequences and ask both the model and the real
game (by saving and restoring the game state) whether the ball survives them. A useful model
agrees with the game, above all on the sequences it thinks are safe.

    python bot/geo_quality.py 1-4            # seeds 1..4, a checkpoint every 150 steps
    python bot/geo_quality.py 3 --every 100 --n 400
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geo_eval import record
from geo_play import open_game

H = 60  # steps per sequence

# Replay the recorded keys up to step T (keeping the controller's view of the game in sync),
# then judge the sequences with the model and with the real game.
JUDGE_JS = """([keys, T, seqs, H]) => {
  const A = [null, 'ArrowLeft', 'ArrowRight'], U = [0, -1, 1], K = {0: null, '-1': 'ArrowLeft', '1': 'ArrowRight'};
  for (let t = 0; t <= T; t++) {
    __bot.geo.on = true; __bot.geo.begin(); __bot.run(1, 50, A[keys[t]]); __bot.geo.on = false;
    __bot.stepped(); __bot.ctl.sense();
    if (t < T) __bot.ctl.commit(U[keys[t + 1]]);
  }
  const model = seqs.map(us => { const p = __bot.ctl.predict(us); return p.length === us.length && !p[p.length - 1][3]; });
  const s0 = __bot.save(), air0 = Object.assign({}, __bot.air);
  const real = seqs.map(us => {
    __bot.load(s0); __bot.air = Object.assign({}, air0);
    // alive through the sequence, and then back on the ground (not falling) within 40 steps
    for (const u of us) {
      __bot.run(1, 50, K[u]); __bot.trackAir();
      if (__bot.isDead()) return false;
    }
    for (let k = 0; k < 40; k++) {
      if (__bot.air.n === 0) return true;
      __bot.run(1, 50, null); __bot.trackAir();
      if (__bot.isDead()) return false;
    }
    return false;
  });
  __bot.load(s0); __bot.air = air0;
  return [model, real];
}"""


def sequences(rng, n, keys_after):
    out = [[[0, -1, 1][k] for k in keys_after[:H]]]            # what the controller really did
    out[0] += [0] * (H - len(out[0]))
    for a in (0, -1, 1):
        out.append([a] * H)
    while len(out) < n:
        s = []
        while len(s) < H:
            s += [rng.choice((0, -1, 1))] * rng.randint(2, 14)
        out.append(s[:H])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seeds")
    ap.add_argument("--every", type=int, default=150)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default=None, help="save per-checkpoint results (json)")
    ap.add_argument("--before-death", default=None,
                    help="checkpoints this many steps before the game's end instead, e.g. 60,90,130")
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-")
    rng = random.Random(1)
    tot = {"ms_rs": 0, "ms_rd": 0, "md_rs": 0, "md_rd": 0}
    results = []
    for seed in range(int(lo), int(hi or lo) + 1):
        keys, score = record(seed)
        points = range(a.every, len(keys) - 5, a.every)
        if a.before_death:
            points = [len(keys) - int(b) for b in a.before_death.split(",") if len(keys) > int(b)]
        for T in points:
            seqs = sequences(rng, a.n, keys[T + 1:])
            e = open_game(seed, budget=1e9)   # no time limit: the same decisions every time
            e.js("() => { __bot.norender = true; }")
            try:
                model, real = e.js(JUDGE_JS, [keys, T, seqs, H])
            finally:
                e.close()
            c = {"ms_rs": 0, "ms_rd": 0, "md_rs": 0, "md_rd": 0}
            for m, r in zip(model, real):
                c[("ms" if m else "md") + "_" + ("rs" if r else "rd")] += 1
            for k in c:
                tot[k] += c[k]
            results.append({"seed": seed, "T": T, **c, "did": [model[0], real[0]]})
            prec = c["ms_rs"] / max(1, c["ms_rs"] + c["ms_rd"])
            print(f"seed {seed} step {T}: model-safe {c['ms_rs'] + c['ms_rd']:3d} (really safe {prec:.0%}), "
                  f"really safe {c['ms_rs'] + c['md_rs']:3d} (model agrees {c['ms_rs'] / max(1, c['ms_rs'] + c['md_rs']):.0%}); "
                  f"what the bot did: model {'safe' if model[0] else 'dies'}, real {'safe' if real[0] else 'dies'}",
                  flush=True)
    prec = tot["ms_rs"] / max(1, tot["ms_rs"] + tot["ms_rd"])
    rec = tot["ms_rs"] / max(1, tot["ms_rs"] + tot["md_rs"])
    print(f"TOTAL: of sequences the model calls safe, {prec:.1%} really are; of really safe ones, the model finds {rec:.1%}"
          f"  ({tot})")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    main()
