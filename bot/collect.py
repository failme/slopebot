"""Collect teacher-labelled training data (DAgger style).

Each episode runs the planner and renderer in lockstep. At every decision step the
planner scores all three actions; the action actually taken comes from the teacher with
probability `beta`, otherwise from the current student, so the student learns to recover
from its own mistakes.
"""
import argparse
import os
import random
import time

import numpy as np

from teacher import ACTIONS, Planner, Renderer

def teacher_pick(vals, prev):
    m = max(vals)
    for i in [0, prev, 1, 2]:
        if vals[i] == m:
            return i


def episode(seed, student=None, beta=1.0, max_steps=4000, log_every=200, rng=None, snap_dir=None,
            flush=None, flush_every=500, label_every=2, intervene=False):
    """Play one episode; `flush(data_dict)` is called every `flush_every` labelled steps and at
    the end. Only every `label_every`-th step is labelled by the planner (and recorded)."""
    rng = rng or random.Random(seed)
    P = Planner(seed)
    R = Renderer(seed)
    if student:
        student.reset()
    obs_l, prev_l, val_l, act_l, pos_l, score_l = [], [], [], [], [], []
    prev = 0
    last_obs = None
    interventions = 0
    t0 = time.time()
    try:
        for t in range(max_steps):
            labelled = t % label_every == 0
            obs = R.obs()
            if labelled:
                vals = P.eval()["vals"]
                a_t = teacher_pick(vals, prev)
            else:
                a_t = P.planned()  # keep following the teacher's current plan
            a = a_t
            if student is not None:
                a_s = student.act(obs)
                if intervene:
                    # The student drives; the teacher only overrides a move it knows is fatal.
                    a = a_s
                    if labelled and vals[a_s] <= 0 < max(vals):
                        a = a_t
                        interventions += 1
                elif rng.random() >= beta:
                    a = a_s
            b, dead = P.step(a)
            R.step(ACTIONS[a])
            prev = a
            if labelled:
                obs_l.append(obs); prev_l.append(obs if last_obs is None else last_obs)
                val_l.append(vals); act_l.append(a); pos_l.append(b); score_l.append(R.score())
            last_obs = obs
            if log_every and t % log_every == 0:
                print(f"  seed {seed} step {t} score={R.score()} vals={[round(v, 2) for v in vals]} "
                      f"interventions={interventions} {time.time() - t0:.0f}s", flush=True)
                if snap_dir:
                    R.save_frame(os.path.join(snap_dir, f"s{seed}_{t:05d}_z{b[2]:.0f}.png"))
            if flush and len(act_l) >= flush_every:
                flush(pack(obs_l, prev_l, val_l, act_l, pos_l, score_l))
                obs_l, prev_l, val_l, act_l, pos_l, score_l = [], [], [], [], [], []
            if dead:
                break
    finally:
        P.close()
        R.close()
    d = pack(obs_l, prev_l, val_l, act_l, pos_l, score_l)
    if flush and len(act_l):
        flush(d)
    return d


def pack(obs_l, prev_l, val_l, act_l, pos_l, score_l):
    return dict(obs=np.array(obs_l, np.uint8), prev=np.array(prev_l, np.uint8), vals=np.array(val_l, np.float32),
                act=np.array(act_l, np.int8), pos=np.array(pos_l, np.float32), score=np.array(score_l, np.int32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True, help="e.g. 1-8 or 3,5,9")
    ap.add_argument("--student", default=None)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--out", default="data")
    ap.add_argument("--snap-dir", default=None)
    ap.add_argument("--intervene", action="store_true", help="student drives, teacher overrides fatal moves")
    a = ap.parse_args()
    seeds = []
    for part in a.seeds.split(","):
        if "-" in part:
            lo, hi = part.split("-"); seeds += list(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    student = None
    if a.student:
        from model import Student
        student = Student(a.student)
    os.makedirs(a.out, exist_ok=True)
    for s in seeds:
        t0 = time.time()
        tag = f"s{s}_b{a.beta:g}" + ("_stu" if a.student else "") + ("_int" if a.intervene else "")
        parts = []

        def flush(d, tag=tag, parts=parts):
            np.savez_compressed(os.path.join(a.out, f"{tag}_p{len(parts):02d}.npz"), **d)
            parts.append(len(d["act"]))
            print(f"  saved {tag} part {len(parts) - 1} (score {d['score'][-1]})", flush=True)

        d = episode(s, student, a.beta, a.max_steps, snap_dir=a.snap_dir, flush=flush, intervene=a.intervene)
        final = d["score"][-1] if len(d["score"]) else "?"
        print(f"seed {s}: {sum(parts)} steps, score {final}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
