"""Measure how well the controller's ball model predicts the real game.

Plays a game with the controller (recording its keys), then replays the same keys (the game
is deterministic) and, at every step, simulates the recorded future keys open-loop with
the controller's model (ctl.predict) and compares with where the ball really went.

    python bot/geo_eval.py 6 out.json        # seed 6, saves per-step predictions
    python bot/geo_eval.py 6 new.json out.json   # same keys as in out.json (after a model change)
    python bot/geo_eval.py death 3 9          # why did the controller die on seeds 3 and 9?
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geo_play import STEP_JS, open_game

K = 30  # prediction horizon (steps)

REPLAY_JS = """([keys, K]) => {
  const A = [null, 'ArrowLeft', 'ArrowRight'], U = [0, -1, 1];
  const out = [];
  for (let t = 0; t < keys.length; t++) {
    __bot.geo.on = true; __bot.geo.begin();
    __bot.run(1, 50, A[keys[t]]);
    __bot.geo.on = false;
    const dead = __bot.stepped(), b = __bot.ball();
    __bot.ctl.sense();
    const fut = keys.slice(t + 1, t + 1 + K).map(k => U[k]);
    const pred = fut.length ? __bot.ctl.predict(fut) : [];
    out.push({ b, pred, st: __bot.ctl.state() });
    if (t + 1 < keys.length) __bot.ctl.commit(U[keys[t + 1]]);
    if (dead) break;
  }
  return out;
}"""


def record(seed, max_steps=8000):
    e = open_game(seed, budget=1e9)   # no time limit: the same decisions every time
    e.js("() => { __bot.norender = true; }")
    keys, a, score = [0], 0, 0
    try:
        for _ in range(max_steps):
            a, dead, score, _ = e.js(STEP_JS, [a, 1, False])
            if dead:
                break
            keys.append(a)
    finally:
        e.close()
    return keys, score


def replay(seed, keys):
    e = open_game(seed, budget=1e9)   # no time limit: the same decisions every time
    e.js("() => { __bot.norender = true; }")
    try:
        out = []
        for c in range(0, len(keys), 200):   # chunks, so no single evaluate call is huge
            # each chunk needs the keys that follow it for its predictions
            out += e.js(REPLAY_JS.replace("keys.length; t++", f"Math.min(keys.length, {c + 200}); t++")
                        .replace("let t = 0", f"let t = {c}"), [keys, K])
            if len(out) < min(len(keys), c + 200):
                break
    finally:
        e.close()
    return out


def death_report(seed, look=60):
    """Why did the controller die on this seed? For the steps before the death, simulate the
    keys it really pressed: if the model foresaw the death, planning failed (or there was no
    way out); if not, the model was wrong, and the divergence shows where."""
    keys, score = record(seed)
    death = len(keys)
    e = open_game(seed, budget=1e9)   # no time limit: the same decisions every time
    e.js("() => { __bot.norender = true; }")
    try:
        rows = e.js(REPLAY_JS.replace("const out = [];", f"const out = []; const T0 = {max(0, death - look - 1)};")
                    .replace("const pred = fut.length ?", "const pred = (t >= T0 && fut.length) ?"), [keys, look + 2])
    finally:
        e.close()
    B = [r["b"] for r in rows]
    n = len(rows)
    print(f"seed {seed}: died at step {n} (score {score})")
    foresaw = None
    for t in range(max(0, n - look - 1), n - 1):
        p = rows[t]["pred"]
        if not p:
            continue
        end = t + len(p)
        why = p[-1][3]
        if why and end >= n - 3 and foresaw is None:
            foresaw = (t, why)
        if (n - 1 - t) % 5 == 0 or t >= n - 4:
            h = min(len(p), n - 1 - t)
            err = [round(p[h - 1][i] - B[t + h][i], 2) for i in range(3)] if h > 0 else None
            print(f"  t-{n - 1 - t:2d}: model with real keys: {'dies ' + ['', 'falling', 'obstacle', 'crash'][why] + f' at +{len(p)}' if why else 'survives'}"
                  f"; error at death {err}; real air {rows[t]['st'][0]}")
    if foresaw:
        print(f"  => the model foresaw it {n - 1 - foresaw[0]} steps ahead ({['', 'falling', 'obstacle', 'crash'][foresaw[1]]}): planning / no escape")
    else:
        print("  => the model did not foresee it: model error")


def main():
    if sys.argv[1] == "death":
        for s in sys.argv[2:]:
            death_report(int(s))
        return
    seed, path = int(sys.argv[1]), sys.argv[2]
    if len(sys.argv) > 3:   # reuse the keys of an earlier recording (to test model changes)
        with open(sys.argv[3]) as f:
            old = json.load(f)
        keys, score = old["keys"], old["score"]
    else:
        keys, score = record(seed)
    print(f"seed {seed}: score {score}, {len(keys)} steps", flush=True)
    rows = replay(seed, keys)
    with open(path, "w") as f:
        json.dump({"seed": seed, "keys": keys, "score": score, "rows": rows}, f)
    print("saved", path)


if __name__ == "__main__":
    main()
