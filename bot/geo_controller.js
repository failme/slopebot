// Real-time Slope controller.
//
// Every decision (50 ms) it:
//  1. rasterises the track and obstacle triangles captured by geotap.js this frame into a
//     top-down height map around the ball;
//  2. simulates the ball on that map for a few hundred candidate steering plans, using a
//     small fitted model of the ball (sideways response to the keys, rolling on the surface,
//     ballistic flight off edges);
//  3. returns the first key of the best plan.
// It needs only the ball's position (read from memory) and the rendered geometry; it never
// saves/restores the game or looks at the future, so it runs in real time.
(() => {
  const B = window.__bot, G = B.geo;

  // ---- model constants (units per 50 ms step), see fit_model.py -----------------------------
  const P = B.ctlParams = {
    // fitted on ~5800 grounded steps of the controller's own games (R^2 0.89)
    k1: 0.0366, k1v: 0.0011,  // sideways velocity change from the key pressed one step ago (+ per unit of forward speed)
    k2: 0.0404, k2v: 0.0016,  // ... and from the key pressed two steps ago
    drag: 0.0393,             // sideways velocity decay per step
    bank: 0.075,              // sideways pull per unit of sideways surface slope (banked track)
    g: 0.18,                  // gravity (per step^2)
    off: 0.7,                 // ball centre height above the surface
    snap: 1.0,                // how far below free fall the surface may be for the ball to stay on it (+0.3 per unit of speed)
    rad: 0.5,                 // ball radius (death test); safety margins are handled in plan scoring
    airDead: 40,              // steps airborne that count as falling off
    voidDead: 12,             // steps with no surface anywhere below that count as falling off
    horizon: 60,              // simulated steps per plan
    edgeW: 1.2,               // lateral distance at which a missing floor / obstacle counts as "close"
    decal: -1e9,              // red geometry less than this above the floor is a marking, not an obstacle
    up: 0.4, upV: 0.35,       // how far above the ball's continued path a surface may be and still be rolled onto
    clearSteps: 30,           // steps over which clearance to obstacles / edges is measured
    clearCap: 1.5,            // clearance beyond this doesn't count
    clearW: 25,               // score per unit of clearance (1 step of survival = 10)
    airW: 3,                  // penalty per airborne step (leaving the surface is where predictions are worst)
  };

  // ---- height map --------------------------------------------------------------------------
  const DX = 0.5, NX = 128, DZ = 1.0, NZ = 640;
  // Up to three surface heights per cell (highest first), so a tunnel roof or a bridge above
  // doesn't hide the floor under it.
  const L0 = new Float32Array(NX * NZ), L1 = new Float32Array(NX * NZ), L2 = new Float32Array(NX * NZ);
  const floor = L0;
  let x0 = 0, z0 = 0, zMax = 0;

  function clear(bx, bz) {
    x0 = bx - NX * DX / 2; z0 = bz - 8;
    L0.fill(-Infinity); L1.fill(-Infinity); L2.fill(-Infinity);
    zMax = -Infinity;
  }
  function addSurface(k, y) {
    const a = L0[k], b = L1[k];
    if (Math.abs(y - a) < 0.3 || Math.abs(y - b) < 0.3 || Math.abs(y - L2[k]) < 0.3) {
      if (Math.abs(y - a) < 0.3 && y > a) L0[k] = y;   // same surface: keep the higher sample
      return;
    }
    if (y > a) { L2[k] = b; L1[k] = a; L0[k] = y; }
    else if (y > b) { L2[k] = b; L1[k] = y; }
    else if (y > L2[k]) L2[k] = y;
  }
  // Rasterise one track triangle (world coords) into the surface layers.
  function tri(ax, ay, az, bx, by, bz, cx, cy, cz) {
    const ux = bx - ax, uy = by - ay, uz = bz - az, vx = cx - ax, vy = cy - ay, vz = cz - az;
    const ny = uz * vx - ux * vz, nx = uy * vz - uz * vy, nz = ux * vy - uy * vx;
    const nl = Math.hypot(nx, ny, nz) || 1;
    if (Math.abs(ny) / nl < 0.3) return;             // walls/sides: not something to roll on
    const minx = Math.min(ax, bx, cx), maxx = Math.max(ax, bx, cx), minz = Math.min(az, bz, cz), maxz = Math.max(az, bz, cz);
    let i0 = Math.floor((minx - x0) / DX), i1 = Math.floor((maxx - x0) / DX), j0 = Math.floor((minz - z0) / DZ), j1 = Math.floor((maxz - z0) / DZ);
    if (i1 < 0 || j1 < 0 || i0 >= NX || j0 >= NZ) return;
    if (maxz > zMax) zMax = maxz;
    i0 = Math.max(i0, 0); j0 = Math.max(j0, 0); i1 = Math.min(i1, NX - 1); j1 = Math.min(j1, NZ - 1);
    const d = ux * vz - uz * vx;
    if (Math.abs(d) < 1e-9) return;
    for (let j = j0; j <= j1; j++) {
      const pz = z0 + (j + 0.5) * DZ;
      for (let i = i0; i <= i1; i++) {
        const px = x0 + (i + 0.5) * DX, wx = px - ax, wz = pz - az;
        const s = (wx * vz - wz * vx) / d, t = (ux * wz - uz * wx) / d;
        if (s < -0.02 || t < -0.02 || s + t > 1.02) continue;
        addSurface(j * NX + i, ay + s * uy + t * vy);
      }
    }
  }
  function buildMap(ball) {
    clear(ball[0], ball[2]);
    const T = G.track, O = G.obst;
    for (let i = 0; i + 8 < T.length; i += 9) tri(T[i], T[i + 1], T[i + 2], T[i + 3], T[i + 4], T[i + 5], T[i + 6], T[i + 7], T[i + 8]);
    bucketObstacles();
  }
  // Highest floor at (x, z) that is not far above y (so an overhead structure isn't taken as floor).
  const cell = (x, z) => {
    const i = Math.floor((x - x0) / DX), j = Math.floor((z - z0) / DZ);
    return (i < 0 || j < 0 || i >= NX || j >= NZ) ? -1 : j * NX + i;
  };
  // Highest surface at (x, z) that is not above yMax (default: the top surface).
  const floorAt = (x, z, yMax = Infinity) => {
    const k = cell(x, z);
    if (k < 0) return -Infinity;
    if (L0[k] <= yMax) return L0[k];
    if (L1[k] <= yMax) return L1[k];
    return L2[k] <= yMax ? L2[k] : -Infinity;
  };
  // Obstacles: exact triangles, bucketed by z, tested against the ball sphere.
  const BZ = 2;
  let obBuckets = new Map();
  // Obstacle motion: match every red triangle to the same-shaped triangle nearest to it in the
  // previous frame; the offset is its velocity per step (some red blocks slide back and forth).
  let prevShapes = new Map(), vel = new Float32Array(0);
  const shapeKey = (O, i) => [O[i + 3] - O[i], O[i + 4] - O[i + 1], O[i + 5] - O[i + 2], O[i + 6] - O[i], O[i + 7] - O[i + 1], O[i + 8] - O[i + 2]]
    .map(v => Math.round(v * 20)).join(',');
  function trackObstacles() {
    const O = G.obst, n = O.length / 9, shapes = new Map();
    vel = new Float32Array(n * 3);
    for (let t = 0; t < n; t++) {
      const i = t * 9, key = shapeKey(O, i);
      const cx = (O[i] + O[i + 3] + O[i + 6]) / 3, cy = (O[i + 1] + O[i + 4] + O[i + 7]) / 3, cz = (O[i + 2] + O[i + 5] + O[i + 8]) / 3;
      if (!shapes.has(key)) shapes.set(key, []);
      shapes.get(key).push(cx, cy, cz);
      const prev = prevShapes.get(key);
      if (!prev) continue;
      let best = 0.8 * 0.8, bx = 0, by = 0, bz = 0;
      for (let q = 0; q < prev.length; q += 3) {
        const dx = cx - prev[q], dy = cy - prev[q + 1], dz = cz - prev[q + 2], d = dx * dx + dy * dy + dz * dz;
        if (d < best) { best = d; bx = dx; by = dy; bz = dz; }
      }
      vel[t * 3] = bx; vel[t * 3 + 1] = by; vel[t * 3 + 2] = bz;
    }
    prevShapes = shapes;
  }
  function bucketObstacles() {
    obBuckets = new Map();
    const O = G.obst, H = P.horizon;
    for (let i = 0; i + 8 < O.length; i += 9) {
      const zlo = Math.min(O[i + 2], O[i + 5], O[i + 8]), zhi = Math.max(O[i + 2], O[i + 5], O[i + 8]);
      // skip red markings lying flat on the track
      const cxm = (O[i] + O[i + 3] + O[i + 6]) / 3, czm = (zlo + zhi) / 2, ym = Math.max(O[i + 1], O[i + 4], O[i + 7]);
      const f = floorAt(cxm, czm, ym + 1);
      if (f > -Infinity && ym <= f + P.decal) continue;
      const vz = vel[i / 3 + 2] * H;
      for (let b = Math.floor((zlo + Math.min(0, vz)) / BZ); b <= Math.floor((zhi + Math.max(0, vz)) / BZ); b++) {
        if (!obBuckets.has(b)) obBuckets.set(b, []);
        obBuckets.get(b).push(i);
      }
    }
  }
  // squared distance from point p to triangle (a, b, c) (Ericson, Real-Time Collision Detection)
  function dist2(px, py, pz, O, i) {
    const ax = O[i], ay = O[i + 1], az = O[i + 2], bx = O[i + 3], by = O[i + 4], bz = O[i + 5], cx = O[i + 6], cy = O[i + 7], cz = O[i + 8];
    const abx = bx - ax, aby = by - ay, abz = bz - az, acx = cx - ax, acy = cy - ay, acz = cz - az;
    const apx = px - ax, apy = py - ay, apz = pz - az;
    const d1 = abx * apx + aby * apy + abz * apz, d2 = acx * apx + acy * apy + acz * apz;
    let qx, qy, qz;
    if (d1 <= 0 && d2 <= 0) { qx = ax; qy = ay; qz = az; }
    else {
      const bpx = px - bx, bpy = py - by, bpz = pz - bz;
      const d3 = abx * bpx + aby * bpy + abz * bpz, d4 = acx * bpx + acy * bpy + acz * bpz;
      if (d3 >= 0 && d4 <= d3) { qx = bx; qy = by; qz = bz; }
      else {
        const vc = d1 * d4 - d3 * d2;
        if (vc <= 0 && d1 >= 0 && d3 <= 0) { const v = d1 / (d1 - d3); qx = ax + v * abx; qy = ay + v * aby; qz = az + v * abz; }
        else {
          const cpx = px - cx, cpy = py - cy, cpz = pz - cz;
          const d5 = abx * cpx + aby * cpy + abz * cpz, d6 = acx * cpx + acy * cpy + acz * cpz;
          if (d6 >= 0 && d5 <= d6) { qx = cx; qy = cy; qz = cz; }
          else {
            const vb = d5 * d2 - d1 * d6;
            if (vb <= 0 && d2 >= 0 && d6 <= 0) { const w = d2 / (d2 - d6); qx = ax + w * acx; qy = ay + w * acy; qz = az + w * acz; }
            else {
              const va = d3 * d6 - d5 * d4;
              if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) { const w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); qx = bx + w * (cx - bx); qy = by + w * (cy - by); qz = bz + w * (cz - bz); }
              else { const den = 1 / (va + vb + vc), v = vb * den, w = vc * den; qx = ax + abx * v + acx * w; qy = ay + aby * v + acy * w; qz = az + abz * v + acz * w; }
            }
          }
        }
      }
    }
    return (px - qx) ** 2 + (py - qy) ** 2 + (pz - qz) ** 2;
  }
  // Does a ball of radius r at (x, y, z), t steps from now, touch an obstacle (moved on by t steps)?
  function hitsObstacle(x, y, z, r, t = 0) {
    const O = G.obst, r2 = r * r;
    for (let b = Math.floor((z - r) / BZ); b <= Math.floor((z + r) / BZ); b++) {
      const L = obBuckets.get(b);
      if (L) for (const i of L) {
        const v = i / 3;
        if (dist2(x - vel[v] * t, y - vel[v + 1] * t, z - vel[v + 2] * t, O, i) < r2) return true;
      }
    }
    return false;
  }

  // Distance from the ball centre (at time t) to the nearest obstacle within `lim` (else lim).
  function obstacleDist(x, y, z, t, lim) {
    const O = G.obst;
    let best = lim * lim;
    for (let b = Math.floor((z - lim) / BZ); b <= Math.floor((z + lim) / BZ); b++) {
      const L = obBuckets.get(b);
      if (L) for (const i of L) {
        const v = i / 3;
        const d = dist2(x - vel[v] * t, y - vel[v + 1] * t, z - vel[v + 2] * t, O, i);
        if (d < best) best = d;
      }
    }
    return Math.sqrt(best);
  }

  // Sideways slope (dy/dx) of the surface near height yRef at (x, z).
  function slopeX(x, yRef, z) {
    const ok = (h) => h !== -Infinity && Math.abs(h - yRef) <= 1.5;
    const a = floorAt(x - 0.5, z, yRef + 0.8), c = floorAt(x, z, yRef + 0.8), b = floorAt(x + 0.5, z, yRef + 0.8);
    if (ok(a) && ok(b)) return b - a;
    if (ok(c) && ok(b)) return (b - c) * 2;     // near the left edge: one-sided
    if (ok(a) && ok(c)) return (c - a) * 2;     // near the right edge
    return 0;
  }

  // ---- ball simulation ---------------------------------------------------------------------
  // s = {x, y, z, vx, vy, vz, u1, u2, air}; returns false when the ball dies.
  function simStep(s, u, t = 0) {
    const p = P;
    s.vx += (p.k1 + p.k1v * s.vz) * s.u1 + (p.k2 + p.k2v * s.vz) * s.u2 - p.drag * s.vx
          - (s.air === 0 ? p.bank * slopeX(s.x, s.y - p.off, s.z) : 0);
    s.u2 = s.u1; s.u1 = u;
    s.x += s.vx; s.z += s.vz; s.vz += s.az || 0;
    const yFree = s.y + s.vy - p.g;
    // The ball can only move onto surfaces near where its current slope would take it (a
    // ramp tilts up gradually); a raised block's top is out of reach, its sides are walls.
    const h = floorAt(s.x, s.z, s.y + s.vy - p.off + p.up + p.upV * Math.max(0, s.vz));
    if (h > -Infinity && h + p.off >= yFree - p.snap - 0.3 * s.vz) {
      // rolling on the surface (or landing): vertical speed follows the surface's slope
      const hb = floorAt(s.x, s.z - 1, h + 2);
      const slope = hb === -Infinity ? -1 : Math.max(-3, Math.min(3, h - hb));
      s.y = h + p.off; s.vy = slope * s.vz; s.air = 0; s.void = 0;
    } else {
      s.vy -= p.g; s.y = yFree; s.air++;
      // nothing at all below: falling off the side (a jump over a short gap lands in time)
      s.void = floorAt(s.x, s.z, s.y) === -Infinity ? (s.void || 0) + 1 : 0;
      if (s.air > p.airDead || s.void > p.voidDead) { s.why = 1; return false; }   // fell off
    }
    if (hitsObstacle(s.x, s.y, s.z, p.rad, t + 1)) { s.why = 2; return false; }
    return true;
  }

  // Sideways danger near (x, z): no floor, or an obstacle, within edgeW.
  function nearDanger(s, t) {
    const w = P.edgeW;
    for (const dx of [-w, w]) {
      if (s.air === 0 && floorAt(s.x + dx, s.z, s.y + 1) < s.y - P.off - 3) return 1;
      if (hitsObstacle(s.x + dx, s.y, s.z, 0.5, t + 1)) return 1;
    }
    return 0;
  }

  // Controller that steers toward lateral target xt (accounting for the key delay).
  // `rate`: fraction of the remaining gap to close per step.
  function toward(s, xt, rate = 0.12) {
    const p = P;
    const pend = (p.k1 + p.k1v * s.vz) * s.u1 + (p.k2 + p.k2v * s.vz) * s.u2 + (p.k2 + p.k2v * s.vz) * s.u1;
    const want = Math.max(-0.9, Math.min(0.9, rate * (xt - s.x)));
    const err = want - (s.vx + pend);
    const dead = 0.02 + 0.004 * s.vz;
    return err > dead ? 1 : err < -dead ? -1 : 0;
  }

  // Simulate a plan. plan(s, t) -> u.
  // Returns [stepsSurvived, danger, keyPresses, firstU, alive, clearance]; clearance is the
  // smallest gap (beyond the ball's radius) to an obstacle or a floor edge over the near future.
  function run(start, plan, H) {
    const s = Object.assign({}, start);
    let danger = 0, keys = 0, first = null, clear = P.clearCap, air = 0;
    for (let t = 0; t < H; t++) {
      if (s.z > zMax - 2) return [H, danger, keys, first ?? 0, true, clear, air];   // past what we can see
      const u = plan(s, t);
      if (first === null) first = u;
      if (u) keys++;
      if (!simStep(s, u, t)) return [t, danger, keys, first, false, 0, air];
      danger += nearDanger(s, t) * (1 - t / H);
      if (s.air > 0 && t < P.clearSteps) air++;
      if (t < P.clearSteps) {
        clear = Math.min(clear, obstacleDist(s.x, s.y, s.z, t + 1, P.clearCap + P.rad) - P.rad);
        if (s.air === 0) {
          const y = s.y - P.off;
          for (const dx of [0.6, 1.0, 1.5]) {
            if (dx - 0.5 >= clear) break;
            if (floorAt(s.x - dx, s.z, y + 0.8) < y - 1.5 || floorAt(s.x + dx, s.z, y + 0.8) < y - 1.5) { clear = Math.min(clear, dx - 0.5); break; }
          }
        }
      }
    }
    return [H, danger, keys, first, true, clear, air];
  }

  // ---- the track ahead --------------------------------------------------------------------
  // For every map row ahead, the left/right edge of the stretch of track the ball is on,
  // traced forward from the ball (so a track that shifts sideways is followed).
  const edgeL = new Float32Array(NZ), edgeR = new Float32Array(NZ);
  function traceTrack(bx, bz) {
    let cx = bx, have = false;
    const j0 = Math.max(0, Math.floor((bz - z0) / DZ));
    for (let j = 0; j < NZ; j++) { edgeL[j] = NaN; edgeR[j] = NaN; }
    for (let j = j0; j < NZ; j++) {
      // runs of cells with any surface in this row; pick the one containing / nearest to cx
      let best = null, bestD = Infinity, i = 0;
      while (i < NX) {
        while (i < NX && L0[j * NX + i] === -Infinity) i++;
        if (i >= NX) break;
        const a = i;
        while (i < NX && L0[j * NX + i] > -Infinity) i++;
        const l = x0 + a * DX, r = x0 + i * DX;
        const d = cx < l ? l - cx : cx > r ? cx - r : 0;
        if (d < bestD) { bestD = d; best = [l, r]; }
      }
      if (!best || bestD > 4) { if (have) continue; else continue; }
      edgeL[j] = best[0]; edgeR[j] = best[1]; have = true;
      cx = (best[0] + best[1]) / 2;
    }
  }
  // lateral target at depth z for a lane at fraction f across the track (0 = left edge)
  function laneX(z, f, fallback) {
    let j = Math.floor((z - z0) / DZ);
    if (j < 0) j = 0; if (j >= NZ) j = NZ - 1;
    for (let k = 0; k < 6 && isNaN(edgeL[j]); k++) j = Math.max(0, j - 1);
    if (isNaN(edgeL[j])) return fallback;
    const l = edgeL[j] + 0.6, r = edgeR[j] - 0.6;
    return l < r ? l + f * (r - l) : (edgeL[j] + edgeR[j]) / 2;
  }

  // ---- decision ----------------------------------------------------------------------------
  const hist = [];   // recent ball positions for velocity
  // Recent forward acceleration (per step^2), from the last few positions.
  const accel = () => {
    const n = hist.length;
    if (n < 5) return 0;
    const a = ((hist[n - 1][2] - hist[n - 2][2]) - (hist[n - 4][2] - hist[n - 5][2])) / 3;
    return Math.max(0, Math.min(0.05, a));
  };
  let u1 = 0, u2 = 0;
  B.ctl = {
    reset() { hist.length = 0; u1 = 0; u2 = 0; },
    // Call once per decision step, after the frame was rendered with geo.on = true.
    decide() {
      const t0 = performance.now ? B.realNow() : 0;
      const b = B.ball();
      hist.push(b); if (hist.length > 6) hist.shift();
      trackObstacles();
      const n = hist.length;
      const v = n >= 2 ? [0, 1, 2].map(i => hist[n - 1][i] - hist[n - 2][i]) : [0, -1, 1];
      buildMap(b);
      const start = { x: b[0], y: b[1], z: b[2], vx: v[0], vy: v[1], vz: Math.max(v[2], 0.5), az: accel(), u1, u2, air: 0 };
      if (floorAt(b[0], b[2], b[1]) + P.off < b[1] - 1) start.air = 5;
      const H = P.horizon;
      let best = null, bestScore = -Infinity, nPlans = 0;
      const consider = (plan, tag) => {
        const r = run(start, plan, H); nPlans++;
        const sc = r[0] * 10 - r[1] * 2 - r[2] * 0.02 + P.clearW * Math.max(0, r[5]) - P.airW * r[6];
        if (sc > bestScore) { bestScore = sc; best = { u: r[3], r, tag, plan }; }
      };
      // hold still (no key), steer toward a lateral target, and two-stage target changes
      consider(() => 0, 'none');
      const offs = [];
      for (let d = -7; d <= 7.01; d += 0.5) offs.push(d);
      for (const d of offs) consider((s) => toward(s, b[0] + d), 'x' + d);
      // lanes that follow the track: a fraction f across its width, possibly changing lane
      traceTrack(b[0], b[2]);
      const F = [0, 0.15, 0.3, 0.5, 0.7, 0.85, 1];
      // pure pursuit: aim at the lane L steps ahead and close the gap over L steps
      const pursue = (s, f, L) => toward(s, laneX(s.z + L * s.vz, f, s.x), 1 / L);
      for (const f of F) for (const L of [3, 5, 8]) consider((s) => pursue(s, f, L), `lane${f}/${L}`);
      for (const f1 of F) for (const T1 of [6, 14, 24]) for (const f2 of F) {
        if (f1 === f2) continue;
        consider((s, t) => pursue(s, t < T1 ? f1 : f2, 5), `lane${f1}>${f2}@${T1}`);
      }
      const ui = best ? best.u : 0;
      this.lastPlan = best && best.plan; this.lastStart = start;
      if (this.log) this.log.push([b[0], b[1], b[2], ui, slopeX(b[0], b[1] - P.off, b[2]), start.air > 0 ? 1 : 0]);
      u2 = u1; u1 = ui;
      this.last = { plans: nPlans, best: best && best.tag, surv: best && best.r[0], zAhead: zMax - b[2], ms: B.realNow() - t0 };
      return ui === -1 ? 1 : ui === 1 ? 2 : 0;   // index into [none, left, right]
    },
    // Simulate the given key sequence (-1/0/1 per step) from the current state, using the
    // geometry captured this frame. For validating the model against the real game.
    predict(us) {
      const b = B.ball();
      const n = hist.length;
      const v = n >= 2 ? [0, 1, 2].map(i => hist[n - 1][i] - hist[n - 2][i]) : [0, -1, 1];
      buildMap(b);
      const s = { x: b[0], y: b[1], z: b[2], vx: v[0], vy: v[1], vz: v[2], az: accel(), u1, u2, air: 0 };
      if (floorAt(b[0], b[2], b[1]) + P.off < b[1] - 1) s.air = 5;
      const out = [];
      for (let t = 0; t < us.length; t++) {
        const u = us[t], alive = simStep(s, u, t);
        out.push([s.x, s.y, s.z, alive ? 0 : s.why, s.air, floorAt(s.x, s.z, s.y + 1)]);
        if (!alive) break;
      }
      return out;
    },
    // Keep the velocity/key history in sync when something else is driving.
    observe(ui) { const b = B.ball(); hist.push(b); if (hist.length > 6) hist.shift(); trackObstacles(); u2 = u1; u1 = ui; },
    // For the visual debugger: map layers, traced edges, obstacles and the best plan's path.
    debugView() {
      const b = B.ball();
      const best = this.lastPlan;
      const path = [];
      if (best) {
        const s = Object.assign({}, this.lastStart);
        for (let t = 0; t < P.horizon; t++) { const u = best(s, t); const ok = simStep(s, u, t); path.push([s.x, s.z, s.y, ok ? 0 : s.why]); if (!ok) break; }
      }
      const obs = [];
      const O = G.obst;
      for (let i = 0; i + 8 < O.length; i += 9) obs.push([O[i], O[i + 2], O[i + 3], O[i + 5], O[i + 6], O[i + 8], vel[i / 3], vel[i / 3 + 2]]);
      return { x0, z0, NX, NZ, DX, DZ, L0: Array.from(L0), L1: Array.from(L1), ball: b,
               edgeL: Array.from(edgeL), edgeR: Array.from(edgeR), path, obs, last: this.last };
    },
    // Sideways slope of the surface under the ball (dy/dx), from this frame's geometry.
    slopeHere() { const b = B.ball(); buildMap(b); return slopeX(b[0], b[1] - P.off, b[2]); },
    debugMap() { return { x0, z0, NX, NZ, DX, DZ, floor: Array.from(L0) }; },
  };
})();
