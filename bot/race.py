"""Race the bot: you play Slope in a window while the bot plays the same level at the same
time in a hidden copy of the game. The bot's ball is shown as a blue ghost ball in your game
(at its real position on the track, ahead of you or behind you), and the scoreboard shows
both scores.

    python bot/race.py              # a random level
    python bot/race.py --seed 7

Arrow keys steer. Enter starts a new race once you are out, Esc quits.
"""
import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import SlopeEnv
from geo_play import PLANNER_JS, STEP_JS, open_game

STEP = 0.05   # the bot decides every 50 ms of game time, like when it plays alone

# Called every step on the player's page: show the bot's ball and the scoreboard, and report
# the player's ball, score, whether they are out, and any key command (Enter / Esc).
UPDATE_JS = """([bpos, board, track]) => {
  const R = __bot.race;
  if (bpos) R.push(bpos); else R.hide();
  R.board(board);
  const dead = track ? __bot.stepped() : true;
  const cmd = R.cmd; R.cmd = null;
  return [__bot.ball(), __bot.score(), dead, cmd];
}"""


def gap_text(pz, bz):
    d = bz - pz
    if abs(d) < 2:
        return "NECK AND NECK"
    return f"BOT {abs(d):.0f} m {'AHEAD' if d > 0 else 'BEHIND'}"


def race(player, seed, first):
    """One race on this level. Returns False if the player quit."""
    if not first:
        player.load(seed)
    player.page.add_script_tag(path=PLANNER_JS)
    player.start_game()
    player.js("() => { __bot.findBall(); __bot.hideUI = true; }")   # hideUI: the game's own score
    board = {"player": 0, "bot": 0, "msg": "LOADING THE BOT…", "sub": f"level {seed}"}
    player.js(UPDATE_JS, [None, board, False])
    bot = open_game(seed)
    bot.js("() => { __bot.norender = true; }")   # it only needs the level's geometry
    try:
        for c in ("3", "2", "1"):
            board.update(msg=c, sub="arrow keys to steer")
            player.js(UPDATE_JS, [None, board, False])
            time.sleep(1)
        board.update(msg="GO!", sub="")
        # both games start now, from the same moment of the same level
        player.js("() => { __bot.realClock = true; __bot.setManual(false); }")
        t0 = time.time()
        steps, a = 0, 0
        bot_out = player_out = False
        bot_score = player_score = 0
        bpos, ppos = bot.js("() => __bot.ball()"), None
        while True:
            # the bot: catch up with real time (a few steps at most per round)
            if not bot_out:
                behind = int((time.time() - t0) / STEP) - steps
                for _ in range(min(3, max(0, behind))):
                    a, bot_out, bot_score, _ = bot.js(STEP_JS, [a, 1, False])
                    steps += 1
                    if bot_out:
                        break
                bpos = bot.js("() => __bot.ball()")
            if time.time() - t0 > 1.5 and board["msg"] == "GO!":
                board["msg"] = ""
            board.update(player=player_score, bot=bot_score, playerOut=player_out, botOut=bot_out)
            if not player_out and not bot_out and ppos:
                board["gap"] = gap_text(ppos[2], bpos[2])
            elif bot_out and not player_out:
                board["gap"] = f"THE BOT IS OUT — {'YOU LEAD!' if player_score > bot_score else f'BEAT {bot_score}!'}"
            elif player_out:
                board["gap"] = ""
            if player_out:
                if bot_out:
                    board["gap"] = ""
                    board["msg"] = ("YOU WIN!" if player_score > bot_score else
                                    "THE BOT WINS" if bot_score > player_score else "IT'S A TIE")
                    board["sub"] = "Enter: new race    Esc: quit"
                else:
                    board["msg"] = "YOU'RE OUT"
                    board["sub"] = "the bot is still going…    Enter: new race    Esc: quit"
            ppos, score, dead, cmd = player.js(UPDATE_JS, [None if bot_out else bpos, board, not player_out])
            if not player_out:
                player_score = score
                if dead:
                    player_out = True
            if cmd == "quit":
                return False
            if cmd == "again" and player_out:
                return True
            time.sleep(max(0.0, t0 + (steps + 1) * STEP - time.time()) if not bot_out else STEP)
    finally:
        bot.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None, help="level (default: random)")
    ap.add_argument("--window", default="960x720", help="initial window size (it can be resized)")
    a = ap.parse_args()
    w, h = (int(v) for v in a.window.split("x"))
    seed = a.seed if a.seed is not None else random.randint(1, 1 << 30)
    player = SlopeEnv(headless=False, width=640, height=480, seed=seed, window=(w, h), gpu=True, race=True).open()
    try:
        first = True
        while race(player, seed, first):
            first = False
            seed = random.randint(1, 1 << 30)
    finally:
        player.close()


if __name__ == "__main__":
    main()
