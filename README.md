# slopebot

A bot that learned to play [Slope](https://github.com/slopenexus/slopenexus.github.io/tree/test/games/19-04v2.1)
(the offline Unity WebGL build) from pixels.

The bot sees only the game screen (downscaled to 80x60, red + green channels, last two frames)
and presses left / right / nothing 20 times a second. It is a small CNN trained by imitation
learning from a lookahead planner that can see the future.

## Quick start

```bash
pip install -r requirements.txt
python -m playwright install chromium     # if you don't already have a Playwright Chromium
./scripts/fetch_game.sh                   # downloads the game into ./game

python bot/play.py --watch                # opens a browser window and plays in real time
python bot/play.py --episodes 5           # headless evaluation, prints scores
python bot/play.py --record run.gif       # headless run saved as an animated GIF
```

Options: `--seed N` picks the level (levels are generated from the seed), `--model` a
different checkpoint (default `models/student.pt`).

## How it works

Everything runs in Chromium through Playwright. `bot/hooks.js` is injected into the page
before the game loads and takes control of time:

* `requestAnimationFrame`, `performance.now` and `Date.now` are replaced, so the game only
  advances when the bot steps it, by exactly the requested amount of game time.
* Emscripten's timer callbacks run on that virtual clock, network requests never complete,
  and `Math.random` is seeded. With that the game is fully deterministic: the same seed and
  the same inputs always produce the same run, even in different browsers.
* The whole Emscripten heap (plus the virtual clock and timers) can be saved and restored,
  which gives save states.

`bot/planner.js` uses those save states as a teacher. For each of the three actions it
simulates candidate plans ~3 s ahead (rendering switched off, ~0.5 ms per physics tick),
counting how many avoid death. Death is read straight from game memory: two bytes the game
flips when the ball is lost (found by diffing memory across deaths). Surviving plans are
carried over between steps, like model-predictive control. Ball position and score are also
read from memory, for logging only.

Because restoring the heap desynchronises Unity's GPU-side bookkeeping, the planner runs in
its own never-rendering browser, kept in lockstep with a second browser that renders what a
player would see (`bot/teacher.py`).

Training (`bot/collect.py`, `bot/train.py`) is DAgger-style imitation learning:

1. The teacher plays; every frame is labelled with the teacher's action values.
2. A CNN (`bot/model.py`) is trained to pick the action the teacher rates safest.
3. The student then drives (with some teacher actions mixed in) while the teacher keeps
   labelling the states the student actually reaches, so it learns to recover from its own
   mistakes. Repeat.

At play time only the renderer browser and the CNN are used: no memory reading, no
lookahead.

## Files

| file | what |
| --- | --- |
| `bot/hooks.js` | virtual time, determinism, save/restore, input injection |
| `bot/planner.js` | ball/death/score readers, rollout planner (teacher) |
| `bot/env.py` | serves `game/`, launches Chromium, starts a run |
| `bot/teacher.py` | planner + renderer pair, observation capture |
| `bot/collect.py` | data collection (teacher and DAgger) |
| `bot/train.py` | trains the CNN |
| `bot/model.py` | the CNN policy |
| `bot/play.py` | runs / watches / records the trained bot |
