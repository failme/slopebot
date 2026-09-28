// In-page lookahead planner ("teacher"). Uses heap save/restore to try action
// sequences with rendering disabled, and picks one that keeps the ball alive.
(() => {
  const B = window.__bot;
  const DT = 1000 / 60;
  const ACTS = [null, 'ArrowLeft', 'ArrowRight'];
  const F = () => new Float32Array(window.gameInstance.Module.HEAPU8.buffer);

  // Locate the ball in memory: its physics AABB is stored as min/max vec3s exactly one
  // diameter apart. Pick the AABB that moves with steering and falls down the slope.
  B.findBall = function () {
    const M = window.gameInstance.Module;
    const s0 = this.save();
    const grab = (k) => { this.load(s0); this.run(20, DT, k); return new Float32Array(M.HEAPU8.slice(0, s0.heap.length).buffer); };
    const A = grab('ArrowLeft'), R = grab('ArrowRight'), C = grab(null);
    const base = new Float32Array(s0.heap.buffer, 0, s0.heap.length >> 2);
    const cands = [];
    for (let i = 0; i < C.length - 6; i++) {
      const e0 = C[i + 3] - C[i];
      if (!(e0 > 0.9 && e0 < 1.1)) continue;
      if (Math.abs(C[i + 4] - C[i + 1] - e0) > 1e-3 || Math.abs(C[i + 5] - C[i + 2] - e0) > 1e-3) continue;
      const dy = C[i + 1] - base[i + 1], dz = C[i + 2] - base[i + 2];
      if (!(dy < -2 && dz > 2)) continue;
      if (!(A[i] < C[i] - 0.3 && C[i] + 0.3 < R[i])) continue;
      cands.push([i, R[i] - A[i], e0]);
    }
    this.load(s0);
    cands.sort((a, b) => Math.abs(a[2] - 1) - Math.abs(b[2] - 1));
    this.ballIdx = cands.length ? cands[0][0] : -1;
    return cands.slice(0, 5).map(c => [c[0] * 4, c[1], c[2]]);
  };
  // Ball centre = middle of its AABB.
  B.ballAt = function (f) { const i = this.ballIdx; return [(f[i] + f[i + 3]) / 2, (f[i + 1] + f[i + 4]) / 2, (f[i + 2] + f[i + 5]) / 2]; };
  B.ball = function () { return this.ballAt(F()); };

  const RDT = 50;  // rollout timestep (ms); physics runs at a fixed 20ms step so this is exact enough

  // Bytes that the game sets to 1 the moment the ball is lost (crash or fall), found by
  // diffing memory across deaths on several levels.
  B.deadAddrs = [2271113, 2269980];
  // The score shown on the HUD (two copies of the counter live at these addresses).
  B.scoreAddrs = [33012020, 30493240];
  B.score = function () {
    const H = window.gameInstance.Module.HEAP32, v = this.scoreAddrs.map(a => H[a >> 2]);
    return v[0] === v[1] ? v[0] : Math.max(...v.filter(x => x >= 0 && x < 1e6));
  };
  B.isDead = function () { const H = window.gameInstance.Module.HEAPU8; return this.deadAddrs.some(a => H[a] !== 0); };

  // Roll `seq` ([[action, steps], ...], steps of RDT ms) forward without rendering.
  // Returns the number of steps survived and whether the ball died.
  B.rollout = function (seq) {
    let t = 0, dead = false;
    const wasNR = this.norender; this.norender = true;
    outer: for (const [a, n] of seq) {
      for (let k = 0; k < n; k++) {
        this.run(1, RDT, a); t++;
        if (this.isDead()) { dead = true; break outer; }
      }
    }
    this.norender = wasNR;
    return { t, dead };
  };

  // Deterministic PRNG for random plan proposals.
  let rs = 12345;
  const rnd = () => { rs ^= rs << 13; rs >>>= 0; rs ^= rs >>> 17; rs ^= rs << 5; rs >>>= 0; return rs / 4294967296; };
  const L = 'ArrowLeft', R = 'ArrowRight';

  // Continuations tried after the first action, ordered so the usually-safe ones come first.
  // Every plan ends by going straight (no key) until the horizon.
  const conts = (H, nRandom) => {
    const out = [[[null, H]]];
    for (const d of [2, 4, 7, 11]) out.push([[L, d], [null, H - d]], [[R, d], [null, H - d]]);
    for (const d of [2, 5, 9]) for (const w of [4, 10, 18]) {
      out.push([[null, d], [L, w], [null, H - d - w]], [[null, d], [R, w], [null, H - d - w]]);
    }
    for (const d of [3, 6, 10]) for (const e of [3, 6, 10]) {
      out.push([[L, d], [R, e], [null, H - d - e]], [[R, d], [L, e], [null, H - d - e]]);
    }
    for (let k = 0; k < nRandom; k++) {
      const seq = []; let used = 0;
      while (used < H * 0.6) {
        const n = 2 + Math.floor(rnd() * 12), a = ACTS[Math.floor(rnd() * 3)];
        seq.push([a, n]); used += n;
      }
      seq.push([null, Math.max(1, H - used)]);
      out.push(seq);
    }
    return out;
  };
  const shift = (seq, n) => {
    const out = [];
    for (const [a, k] of seq) { const m = k - n; n = Math.max(0, n - k); if (m > 0) out.push([a, m]); }
    return out;
  };
  const len = (seq) => seq.reduce((s, [, k]) => s + k, 0);

  // Score each immediate action (held for one decision step): the number of candidate plans
  // (capped at cfg.cap) that survive the whole horizon, or, if none of cfg.tries candidates
  // does, -(fraction of the horizon lost by the best one). The horizon must be long because
  // after rolling off the track the game only flags the ball as dead up to ~1.7s later.
  // Surviving plans are carried over to the next step (see commit), so usually the first
  // candidate tried already survives.
  B.evalActions = function (cfg) {
    cfg = Object.assign({ horizon: 72, cap: 2, tries: 12, random: 30 }, cfg || {});
    const s0 = this.save(this._ps); this._ps = s0;
    const H = cfg.horizon;
    rs = (Math.floor(this.now()) * 2654435761) >>> 0 || 1;
    const base = conts(H - 1, cfg.random);
    // Interleave structured and random proposals after the straight-ahead one.
    const structured = base.slice(1, base.length - cfg.random), random = base.slice(base.length - cfg.random);
    const mixed = [base[0]];
    for (let i = 0; i < Math.max(structured.length, random.length); i++) {
      if (i < random.length) mixed.push(random[i]);
      if (i < structured.length) mixed.push(structured[i]);
    }
    const vals = [], survivors = [];
    let rolls = 0;
    for (const a of ACTS) {
      let ok = 0, best = 0, tries = 0, bestSeq = null;
      const surv = [];
      // Candidates: plans carried over from the last step that start with `a`; then every
      // carried-over plan with its first step replaced by `a` (a one-step deviation usually
      // survives too, which keeps the labels of the non-planned actions honest); then the
      // generic proposals.
      const kept = (this._plans && this._plans[ACTS.indexOf(a)]) || [];
      const dev = [];
      if (this._plans) for (const ps of this._plans) for (const p of ps) {
        const d = [[a, 1], ...shift(p, 1)];
        if (p[0][0] !== a) dev.push(d);
      }
      const C = kept.concat(dev, mixed);
      for (let i = 0; i < C.length; i++) {
        const isKept = i < kept.length + dev.length;
        if (!isKept && tries >= cfg.tries) break;
        const seq = isKept ? C[i] : [[a, 1], ...C[i]];
        const n = len(seq);
        if (n < H) seq.push([null, H - n]);
        this.load(s0);
        const r = this.rollout(seq); rolls++;
        if (!isKept) tries++;
        if (!r.dead) { ok++; surv.push(seq); if (ok >= cfg.cap) break; }
        if (r.t > best) { best = r.t; bestSeq = seq; }
      }
      vals.push(ok > 0 ? ok : best / H - 1);
      // With no survivor, carry over the plan that lived longest: following it buys time
      // for the moving horizon to reveal a way out.
      survivors.push(ok > 0 ? surv : (bestSeq ? [bestSeq] : []));
    }
    this.load(s0);
    this._cand = survivors;
    this._fresh = true;
    return { vals, rolls };
  };

  // The next action of the plan the planner is currently following (index into ACTS).
  B.nextPlanned = function () { return this._next || 0; };

  // Tell the planner which action was actually taken: the surviving plans that started with
  // it, minus the step just taken, become the first candidates for the next step.
  // Without a fresh evaluation since the last step, the carried-over plans themselves are
  // advanced instead.
  B.commit = function (ai) {
    const src = this._fresh ? this._cand : this._plans;
    this._fresh = false;
    const next = [[], [], []];
    this._next = 0;
    let first = true;
    for (const plan of (src && src[ai]) || []) {
      const rest = shift(plan, 1);
      if (rest.length) {
        rest.push([null, 1]);
        const k = ACTS.indexOf(rest[0][0]);
        next[k].push(rest);
        if (first) { this._next = k; first = false; }
      }
    }
    this._plans = next;
  };
})();
