# slopebot

A bot that plays [Slope](https://github.com/slopenexus/slopenexus.github.io/tree/test/games/19-04v2.1)
(the offline Unity WebGL build) on its own, in a real browser.

**Status:** the bot that reaches 300 is a lookahead planner (search with save states,
reading game memory), not a trained network (see [Results](#results)). The pipeline for
training a pixels-only network to imitate it is included, but the networks it produced
did not generalise, so no model is shipped — see [The learned (CNN) bot](#the-learned-cnn-bot).

## Quick start

```bash
pip install -r requirements.txt
python -m playwright install chromium     # if you don't already have a Playwright Chromium
./scripts/fetch_game.sh                   # downloads the game into ./game

# watch the recorded 301-point game in real time (about 5 minutes)
python bot/play.py --replay runs/planner_seed43.json --watch

# let the planner bot play (headless) until it reaches 300, recording its inputs
python bot/planner_run.py --seed 42 --target 300 --out runs/seed42.json

# ...and watch that game in real time in a browser window (or save it as a video)
python bot/play.py --replay runs/seed42.json --watch
python bot/play.py --replay runs/seed42.json --record seed42.webm

# or watch the planner live while it plays (slow motion: each move takes it ~0.5-1s)
python bot/planner_run.py --seed 42 --out runs/seed42.json --watch
```

`runs/` contains recorded games, so `play.py --replay` works straight after
`fetch_game.sh`.

## How it works

Everything runs in Chromium through Playwright. `bot/hooks.js` is injected into the page
before the game loads and takes control of it:

* **Virtual time.** `requestAnimationFrame`, `performance.now` and `Date.now` are replaced,
  so the game only advances when the bot steps it, by exactly the requested game time.
* **Determinism.** Emscripten's timer callbacks run on the virtual clock, network requests
  never complete and `Math.random` is seeded. The same seed and the same inputs always give
  the same game, even in different browsers. The seed changes the generated level.
* **Save states.** The Emscripten heap (plus virtual clock, timers and held keys) can be
  saved and restored in a few milliseconds.

`bot/planner.js` reads a few things straight from game memory (found by diffing memory
snapshots): the ball's position, the score, and two bytes the game sets when the ball is
lost (crash or fall). The planner then plays like model-predictive control:

* every 100 ms it restores a save state and simulates candidate input sequences 3.6 s ahead
  with rendering switched off (~0.5 ms per 20 ms physics tick);
* candidates are the plans that survived last time (shifted by one step), one-step
  deviations from them, and structured/random manoeuvres;
* it takes the first move of a plan that survives, preferring "no key" when that is as safe.

The long horizon matters: when the ball rolls off the side of the track the game only
declares it dead ~1.5 s later, and legit jumps keep the ball airborne for up to ~1.5 s, so
there is no quick "fell off" test.

Restoring the heap desynchronises Unity's GPU bookkeeping, so the planner runs in its own
browser with all WebGL calls stubbed out; what you watch is a second browser fed the same
inputs (`bot/teacher.py`), which stays identical because the game is deterministic.

## The learned (CNN) bot

The original plan was to distil the planner into a CNN that only sees the screen
(`bot/collect.py`, `bot/train.py`, `bot/model.py`, `python bot/play.py --watch`):

1. The planner plays; every other frame is labelled with how many of its candidate plans
   survive after each action (left / right / none).
2. A CNN (two 128x96 frames, UI hidden) learns to predict per-action safety; at play time
   it goes straight unless another action looks clearly safer.
3. DAgger-style: the student drives and the planner labels (and vetoes fatal moves).

With ~12k labelled frames the network did not generalise to unseen levels: on held-out
games it predicts "is going straight fatal?" at chance level, and plain behaviour cloning
of the planner's moves and predicting its path 1 s ahead do no better (while a directly
visible quantity, the ball's sideways speed, is learned fine). The planner's decisions
depend on precise dynamics 1-4 s ahead that aren't recoverable from a frame, so imitation
would need far more data (or a different target) than this machine produces
(~7k labels/hour). No trained model is shipped for that reason.

## Files

| file | what |
| --- | --- |
| `bot/hooks.js` | virtual time, determinism, save/restore, input injection, UI hiding, no-GL mode |
| `bot/planner.js` | ball/score/death readers, rollout planner |
| `bot/env.py` | serves `game/`, launches Chromium, starts a game |
| `bot/teacher.py` | planner + renderer browsers, screen capture |
| `bot/planner_run.py` | the planner bot: plays a game, records inputs, optional live view |
| `bot/play.py` | replay / record videos; run the CNN bot |
| `bot/collect.py`, `bot/train.py`, `bot/model.py` | CNN imitation-learning pipeline |

## Results

Planner bot, headless, one game per level seed (score = the game's own counter):

| seed | result | notes |
| --- | --- | --- |
| 43 | **301** (stopped at the 300 target) | 5861 moves, ~4.9 min of game time; `runs/planner_seed43.json` |
| 42 | **300** (stopped at the 300 target) | 5381 moves; `runs/planner_seed42.json` |
| 45 | 204, then fell off the track | before the "danger" search upgrade |
| 42, 43, 44 (first attempts) | stuck at 257 / 161 / 156 | fell into a void where the game never ends the run; fixed by the free-fall rule |

Replaying a recording in a fresh browser reproduces the game exactly, so it is an
independent check of the result: `python bot/play.py --replay runs/planner_seed43.json`
prints the final score. Both recordings were checked this way and end on 301 and 300.

It takes the planner roughly 1-2 s of CPU per move in the late game (the rollouts dominate),
so a 300-point game takes ~2 hours of wall time to compute on a 4-core machine, but only
~5 minutes to watch.
