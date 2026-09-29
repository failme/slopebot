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

## The real-time bot (geometry controller)

`bot/geo_controller.js` plays in real time on any seed: every 50 ms game step it decides in
~15-30 ms, using only what the game shows it at that moment. It never saves, restores or
fast-forwards the game.

```bash
python bot/geo_play.py --seed 7 --watch      # watch it play, real time
python bot/geo_play.py --seeds 1-8           # headless, faster than real time
./bot/geo_bench.sh /tmp/bench 1 24           # seeds 1..24, 4 games in parallel
```

### Race the bot

```bash
python bot/race.py              # a random level
python bot/race.py --seed 7
```

You play in the window (arrow keys; click it once if the keys don't respond) while the bot
plays the same level at the same time in a hidden copy of the game. Its ball is drawn in your
game as a blue ghost ball at its real position on the track, so you see it pull ahead or fall
behind; when it's out of sight the scoreboard (`PLAYER: X   BOT: X`, replacing the game's own
score) says how far ahead or behind it is. When you crash the race goes on until the bot is
out too; Enter starts a new race on a new level, Esc quits. (`bot/race.js` draws the ghost:
it re-draws the game's own ball mesh at the bot's position with a blue copy of its texture.
The same seed always builds the same track, however the two games are played.)

What it reads each step:

* the ball's position and the game's time scale, from game memory (`planner.js` finds the
  ball; the time scale grows from 2.75 to ~3.4 during a game and scales all the physics);
* the geometry the game draws that frame (`bot/geotap.js` mirrors WebGL buffers and decodes
  every draw call that uses the track or obstacle textures into world-space triangles).

What it does with it:

* **Map.** Track triangles are rasterised into a height map around the ball (up to three
  surface layers per cell, each stored as its exact plane); steep faces become walls; red
  obstacles are grouped into rigid objects and followed from frame to frame, and movers
  (pistons, sliders) are predicted as the constant-speed back-and-forth motion they follow.
* **Ball model.** A small physical model fitted on recorded games: steering response to
  the keys (with the game's 1-2 step input lag), rolling friction and banked-surface pull,
  gravity and drag scaled by the time scale, landings that lose speed to friction in
  proportion to the impact, bounces off walls, and the ways to die (hitting red, running
  head-on into a wall, falling off). `python bot/geo_eval.py SEED out.json` measures how
  well it predicts a real game.
* **Planning.** ~200 candidate steering policies (follow a lane across the track, switch
  lanes, steer to a fixed offset, keep the previous plan) are simulated 3 s ahead. They are
  scored on survival (discounted, so distant, less certain deaths count less), clearance
  from obstacles and edges, and time in the air; the best few are re-simulated from
  slightly perturbed starts and judged by their worst case. The first key of the winner is
  pressed. Planning stops after 35 ms.
* **Learned plan safety.** The physics model can't tell some fatal plans from safe ones, so
  a small network (`bot/safety.json`, trained by `bot/train_safety.py`) estimates from the
  features of each predicted path whether the plan really survives; `log(p)` is added to the
  plan score. Training data comes from `bot/geo_data.py`: while the bot plays, a second,
  render-free copy of the game follows the same keys and plays every candidate plan for real
  (~105k labelled plans from 24 games). It never runs at play time.

Results (no time limit, 4 games in parallel; seeds never used for training):

| version | seeds | mean score | median | best | games ≥ 100 |
| --- | --- | --- | --- | --- | --- |
| physics model only | 49-96 | 58.4 | 58 | 140 | 5 / 48 |
| + learned plan safety | 49-96 | 81.4 | 84 | 183 | 15 / 48 |
| + steering force fitted per game speed | 49-96 | **84.2** | 88 | 147 | 13 / 48 |

The last change makes the model's 10-30 step predictions of the ball's sideways position
~10-20% more accurate (median error); the score gain is within the noise of 48 games
(per-seed difference +2.8 ± 6.7). Tried and rejected: adding the plan's survival under
perturbed physics models as safety-network features (no better on 17k real-game plans).

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
| `bot/geotap.js` | captures the track / obstacle triangles of every frame from WebGL |
| `bot/geo_controller.js` | the real-time bot: map, ball model, planner |
| `bot/race.py`, `bot/race.js` | race the bot: play alongside it, its ball shown as a blue ghost |
| `bot/geo_play.py` | runs the real-time bot (headless, `--watch`, `--out` to save a game for `play.py --replay`) |
| `bot/geo_bench.sh` | benchmark over a range of seeds |
| `bot/geo_eval.py` | model accuracy on a replayed game; `death` mode explains why a game ended |
| `bot/geo_quality.py` | model vs real game: which sampled key sequences survive |
| `bot/geo_debug.py` | crash report image: game view + top-down map with planned and real paths |

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
