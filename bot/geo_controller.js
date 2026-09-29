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

  // ---- model constants (units per 50 ms step), fitted on recorded games (see geo_eval.py) ---
  const P = B.ctlParams = {
    // Per-step quantities scale with the game's time scale ts (Time.timeScale, read from
    // memory; it grows from ~2.75 to ~3.4 during a game): accelerations with ts^2, drag with ts.
    // Fitted on ~5000 steps of real games (R^2 0.99 for the sideways response).
    k1: 0.00408, k2: 0.00456, // sideways velocity change from the key held last step / the step before (x ts^2)
    dg: 0.00865,              // sideways rolling friction on the ground (x ts)
    bank: 0.0069,             // sideways pull per unit of sideways surface slope (x ts^2)
    a0: -0.00439, a1: 0.00655, a2: 0.0005,  // along-track acceleration on the ground (see simStep)
    g: 0.01834,               // gravity (x ts^2)
    da: 0.00086, dax: 0.00127, fz: 0.00152,  // air drag, sideways air drag (x ts), forward push in the air (x ts^2)
    rad: 0.495,               // ball radius
    tsAddrs: [26580708, 22398436, 31244388],  // copies of Time.timeScale (float) in the Emscripten heap
    reach: 0.3,               // a surface whose plane passes this far above the ball's last contact is a wall/block top, not a ramp
    airDead: 40,              // steps airborne that count as falling off
    voidDead: 3,              // airborne steps under the track with no track below and ahead that count as falling off
    aheadRows: 60, underW: 8, // ... looking this far ahead (rows), and this far sideways for track above
    horizon: 60,              // simulated steps per plan
    budget: 35,               // ms of planning per decision (a decision step is 50 ms)
    minPlans: 24,             // ... but at least this many plans are always tried
    robustK: 6, robustW: 0.7, // the best robustK plans are re-run from perturbed starts; weight of their worst case
    perturb: [[0.04, 0], [-0.04, 0], [0, 0.03], [0, -0.03]],  // (sideways speed + dvx, forward speed * (1 + f))
    wallE: 0.3,               // restitution when bouncing off a wall
    crashV: 0.8,              // running into a wall facing back at more than this (per step) is fatal
    margin0: 0.05, margin1: 0.01, marginMax: 0.35,  // planning keeps this far from red: margin0 + margin1 * steps ahead
    edgeW: 1.2,               // lateral distance at which a missing floor / obstacle counts as "close"
    decal: -1e9,              // red geometry less than this above the floor is a marking, not an obstacle
    clearSteps: 30,           // steps over which clearance to obstacles / edges is measured
    clearCap: 1.5,            // clearance beyond this doesn't count
    survW: 200, gamma: 0.95,  // survival score: survW * (1 - gamma^steps), so distant (less certain) deaths weigh less
    clearW: 25,               // score per unit of clearance
    stickW: 10,               // bonus for continuing the previous plan (avoids dithering between equal options)
    airW: 3,                  // penalty per airborne step (leaving the surface is where predictions are worst)
    wallW: 150,               // penalty for bouncing off a wall (predictions after that are poor)
  };

  // ---- height map --------------------------------------------------------------------------
  const DX = 0.5, NX = 128, DZ = 1.0, NZ = 640;
  // Up to three surfaces per cell (highest first), so a tunnel roof or a bridge above doesn't
  // hide the floor under it. Each is the plane of its triangle: height at the cell centre and
  // slopes dy/dx, dy/dz, so heights are exact anywhere in the cell.
  const NL = 3, LH = new Float32Array(NX * NZ * NL), LGX = new Float32Array(NX * NZ * NL), LGZ = new Float32Array(NX * NZ * NL);
  const L0 = new Float32Array(NX * NZ);   // top surface height per cell (for tracing / debugging)
  let x0 = 0, z0 = 0, zMax = 0;

  function clear(bx, bz) {
    x0 = bx - NX * DX / 2; z0 = bz - 8;
    LH.fill(-Infinity); L0.fill(-Infinity);
    zMax = -Infinity;
    TRI = []; TRN = []; triGrid = new Array(GNX * GNZ);
  }
  function addSurface(k, y, gx, gz) {
    const b = k * NL;
    let l = 0;
    for (; l < NL; l++) {
      const h = LH[b + l];
      if (Math.abs(y - h) < 0.3) { if (y > h) { LH[b + l] = y; LGX[b + l] = gx; LGZ[b + l] = gz; } return; }  // same surface
      if (y > h) break;
    }
    if (l >= NL) return;
    for (let m = NL - 1; m > l; m--) { LH[b + m] = LH[b + m - 1]; LGX[b + m] = LGX[b + m - 1]; LGZ[b + m] = LGZ[b + m - 1]; }
    LH[b + l] = y; LGX[b + l] = gx; LGZ[b + l] = gz;
    if (l === 0) L0[k] = y;
  }
  // Rasterise one track triangle (world coords) into the surface layers.
  function tri(ax, ay, az, bx, by, bz, cx, cy, cz) {
    const ux = bx - ax, uy = by - ay, uz = bz - az, vx = cx - ax, vy = cy - ay, vz = cz - az;
    const ny = uz * vx - ux * vz, nx = uy * vz - uz * vy, nz = ux * vy - uy * vx;
    const nl = Math.hypot(nx, ny, nz) || 1;
    if (Math.abs(ny) / nl < 0.3) {                   // walls/sides: not something to roll on
      // the ball bounces off these (see contacts()); floors go into the height map
      const w = TRI.length;
      TRI.push(ax, ay, az, bx, by, bz, cx, cy, cz);
      TRN.push(nx / nl, ny / nl, nz / nl);
      gridAdd(triGrid, w, Math.min(ax, bx, cx), Math.max(ax, bx, cx), Math.min(az, bz, cz), Math.max(az, bz, cz), 0.6);
      return;
    }
    const gx = -nx / ny, gz = -nz / ny;              // plane slopes dy/dx, dy/dz
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
        addSurface(j * NX + i, ay + s * uy + t * vy, gx, gz);
      }
    }
  }
  function buildMap(ball) {
    clear(ball[0], ball[2]);
    const T = G.track;
    for (let i = 0; i + 8 < T.length; i += 9) tri(T[i], T[i + 1], T[i + 2], T[i + 3], T[i + 4], T[i + 5], T[i + 6], T[i + 7], T[i + 8]);
    buildLowAhead();
    bucketObstacles();
  }
  const cell = (x, z) => {
    const i = Math.floor((x - x0) / DX), j = Math.floor((z - z0) / DZ);
    return (i < 0 || j < 0 || i >= NX || j >= NZ) ? -1 : j * NX + i;
  };
  // Height at (x, z) of the highest surface there that is not above yMax (-Infinity if none).
  // Its slopes are left in hit.gx / hit.gz.
  const hit = { gx: 0, gz: 0 };
  const floorAt = (x, z, yMax = Infinity) => {
    const k = cell(x, z);
    if (k < 0) return -Infinity;
    const cx = x0 + ((k % NX) + 0.5) * DX, cz = z0 + (Math.floor(k / NX) + 0.5) * DZ;
    for (let l = k * NL, e = l + NL; l < e; l++) {
      if (LH[l] === -Infinity) return -Infinity;
      const h = LH[l] + LGX[l] * (x - cx) + LGZ[l] * (z - cz);
      if (h <= yMax) { hit.gx = LGX[l]; hit.gz = LGZ[l]; return h; }
    }
    return -Infinity;
  };
  // The surface at (x, z) that the ball, last touching height yLast at (xl, zl), can roll onto:
  // the highest one whose plane, extended back to (xl, zl), isn't above yLast (a ramp continues
  // the surface; a raised block's top doesn't, its side is a wall).
  const reachable = (x, z, xl, zl, yLast) => {
    const k = cell(x, z);
    if (k < 0) return -Infinity;
    const cx = x0 + ((k % NX) + 0.5) * DX, cz = z0 + (Math.floor(k / NX) + 0.5) * DZ;
    for (let l = k * NL, e = l + NL; l < e; l++) {
      if (LH[l] === -Infinity) return -Infinity;
      const gx = LGX[l], gz = LGZ[l], h = LH[l] + gx * (x - cx) + gz * (z - cz);
      if (h - gx * (x - xl) - gz * (z - zl) <= yLast + P.reach) { hit.gx = gx; hit.gz = gz; return h; }
    }
    return -Infinity;
  };
  // Obstacles (red triangles) and walls (track faces too steep to roll on) are kept as exact
  // triangles in a coarse x/z grid over the map: each cell lists the triangles within GM of it
  // (over the whole horizon, for moving obstacles), so a query only looks at one cell.
  const GX = 1, GZ = 2, GNX = NX * DX / GX, GNZ = NZ * DZ / GZ, GM = 2.1;
  let obGrid = [], triGrid = [], TRI = [], TRN = [];
  const gcell = (x, z) => {
    const i = Math.floor((x - x0) / GX), j = Math.floor((z - z0) / GZ);
    return (i < 0 || j < 0 || i >= GNX || j >= GNZ) ? -1 : j * GNX + i;
  };
  function gridAdd(grid, idx, xlo, xhi, zlo, zhi, m = GM) {
    const i0 = Math.max(0, Math.floor((xlo - m - x0) / GX)), i1 = Math.min(GNX - 1, Math.floor((xhi + m - x0) / GX));
    const j0 = Math.max(0, Math.floor((zlo - m - z0) / GZ)), j1 = Math.min(GNZ - 1, Math.floor((zhi + m - z0) / GZ));
    for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) {
      const k = j * GNX + i;
      (grid[k] || (grid[k] = [])).push(idx);
    }
  }
  // Obstacle motion. Red geometry comes as rigid objects (triangles sharing vertices), which
  // are followed from frame to frame. Movers (pistons, sliders) go back and forth at constant
  // speed between two turning points; the turning points are learned while an object is in
  // view, and its motion is predicted as that ping-pong.
  const OH = 64;                       // offset tables cover this many steps (>= horizon + 1)
  let triObj = new Int32Array(0);      // red triangle -> object this frame
  let objOff = [];                     // object -> null (static) or Float32Array(OH * 3) of offsets
  let objExt = [];                     // object -> [min dx, max dx, min dz, max dz] of its offsets
  let tracks = [];                     // followed objects
  const ranges = [];                   // turning-point distances seen so far (default for new movers)
  function trackObstacles() {
    const O = G.obst, n = O.length / 9;
    const par = new Int32Array(n);
    for (let i = 0; i < n; i++) par[i] = i;
    const find = (i) => { while (par[i] !== i) { par[i] = par[par[i]]; i = par[i]; } return i; };
    const vk = new Map();
    for (let t = 0; t < n; t++) for (let v = 0; v < 3; v++) {
      const i = t * 9 + v * 3, key = Math.round(O[i] * 1000) + ',' + Math.round(O[i + 1] * 1000) + ',' + Math.round(O[i + 2] * 1000);
      const o = vk.get(key);
      if (o === undefined) vk.set(key, t); else { const a = find(t), b = find(o); if (a !== b) par[a] = b; }
    }
    const idx = new Map(), objs = [];
    triObj = new Int32Array(n);
    for (let t = 0; t < n; t++) {
      const r = find(t);
      let k = idx.get(r);
      if (k === undefined) { k = objs.length; idx.set(r, k); objs.push({ c: [0, 0, 0], lo: [1e9, 1e9, 1e9], hi: [-1e9, -1e9, -1e9], n: 0 }); }
      triObj[t] = k;
      const ob = objs[k];
      for (let v = 0; v < 3; v++) for (let a = 0; a < 3; a++) {
        const x = O[t * 9 + v * 3 + a];
        ob.c[a] += x; if (x < ob.lo[a]) ob.lo[a] = x; if (x > ob.hi[a]) ob.hi[a] = x;
      }
      ob.n++;
    }
    for (const ob of objs) {
      for (let a = 0; a < 3; a++) ob.c[a] /= ob.n * 3;
      ob.sig = ob.n + ':' + ob.lo.map((l, a) => Math.round((ob.hi[a] - l) * 10)).join(',');
    }
    // match to the followed objects: same shape, nearest to where it was expected to be
    for (const tr of tracks) tr.used = false;
    for (const ob of objs) {
      let best = null, bd = 1.5 * 1.5;
      for (const tr of tracks) {
        if (tr.used || tr.sig !== ob.sig) continue;
        let d = 0;
        for (let a = 0; a < 3; a++) d += (ob.c[a] - tr.pos[a] - tr.vel[a]) ** 2;
        if (d < bd) { bd = d; best = tr; }
      }
      if (!best) {
        best = { sig: ob.sig, pos: ob.c.slice(), vel: [0, 0, 0], pmin: ob.c.slice(), pmax: ob.c.slice(),
                 turnLo: [false, false, false], turnHi: [false, false, false], speed: [0, 0, 0], dir: [0, 0, 0], age: 0 };
        tracks.push(best);
      } else {
        for (let a = 0; a < 3; a++) {
          const v = ob.c[a] - best.pos[a], sp = Math.max(best.speed[a] * 0.9, Math.abs(v));
          // a turn: clearly moving the other way from the last clear motion
          if (Math.abs(v) > 0.5 * sp && sp > 0.02) {
            const dir = Math.sign(v), last = best.dir[a];
            if (last > 0 && dir < 0 && !best.turnHi[a]) { best.turnHi[a] = true; if (best.turnLo[a]) ranges.push(best.pmax[a] - best.pmin[a]); }
            if (last < 0 && dir > 0 && !best.turnLo[a]) { best.turnLo[a] = true; if (best.turnHi[a]) ranges.push(best.pmax[a] - best.pmin[a]); }
            best.dir[a] = dir;
          }
          best.vel[a] = v; best.speed[a] = sp;
          best.pmin[a] = Math.min(best.pmin[a], ob.c[a]); best.pmax[a] = Math.max(best.pmax[a], ob.c[a]);
        }
        best.pos = ob.c.slice();
      }
      best.used = true; best.age = 0; ob.track = best;
    }
    tracks = tracks.filter(tr => tr.used || ++tr.age < 5);
    if (ranges.length > 50) ranges.splice(0, ranges.length - 50);
    // predicted offsets over the horizon
    const R = ranges.length ? ranges.slice().sort((x, y) => x - y)[ranges.length >> 1] : 20;
    objOff = objs.map(ob => {
      const tr = ob.track;
      if (!tr.speed.some(v => v > 0.02)) return null;
      const off = new Float32Array(OH * 3);
      for (let a = 0; a < 3; a++) {
        // constant speed, in the direction it was last clearly moving (at a turning point the
        // measured displacement is partial or zero)
        if (tr.speed[a] <= 0.02 || !tr.dir[a]) continue;
        let v = tr.dir[a] * tr.speed[a];
        let lo = tr.turnLo[a] ? tr.pmin[a] : -Infinity, hi = tr.turnHi[a] ? tr.pmax[a] : Infinity;
        if (lo === -Infinity && hi < Infinity) lo = hi - R;
        if (hi === Infinity && lo > -Infinity) hi = lo + R;
        let p = tr.pos[a];
        for (let k = 1; k < OH; k++) {
          p += v;
          if (p > hi) { p = 2 * hi - p; v = -v; }
          if (p < lo) { p = 2 * lo - p; v = -v; }
          off[k * 3 + a] = p - tr.pos[a];
        }
      }
      return off;
    });
    objExt = objOff.map(off => {
      if (!off) return null;
      let a = 0, b = 0, c = 0, d = 0;
      for (let k = 0; k < OH; k++) { const x = off[k * 3], z = off[k * 3 + 2]; if (x < a) a = x; if (x > b) b = x; if (z < c) c = z; if (z > d) d = z; }
      return [a, b, c, d];
    });
  }
  // Offset of red triangle i (flat index) t steps from now; sets ox, oy, oz.
  let ox = 0, oy = 0, oz = 0;
  function offsetAt(i, t) {
    const off = objOff[triObj[i / 9]];
    if (!off) { ox = oy = oz = 0; return; }
    if (t < 0) t = 0; if (t > OH - 1.001) t = OH - 1.001;
    const k = Math.floor(t), f = t - k, a = k * 3, b = a + 3;
    ox = off[a] + f * (off[b] - off[a]); oy = off[a + 1] + f * (off[b + 1] - off[a + 1]); oz = off[a + 2] + f * (off[b + 2] - off[a + 2]);
  }
  function bucketObstacles() {
    obGrid = new Array(GNX * GNZ);
    const O = G.obst, H = P.horizon;
    for (let i = 0; i + 8 < O.length; i += 9) {
      const zlo = Math.min(O[i + 2], O[i + 5], O[i + 8]), zhi = Math.max(O[i + 2], O[i + 5], O[i + 8]);
      const xlo = Math.min(O[i], O[i + 3], O[i + 6]), xhi = Math.max(O[i], O[i + 3], O[i + 6]);
      // skip red markings lying flat on the track
      const cxm = (xlo + xhi) / 2, czm = (zlo + zhi) / 2, ym = Math.max(O[i + 1], O[i + 4], O[i + 7]);
      const f = floorAt(cxm, czm, ym + 1);
      if (f > -Infinity && ym <= f + P.decal) continue;
      const ext = objExt[triObj[i / 9]] || [0, 0, 0, 0];
      gridAdd(obGrid, i, xlo + ext[0], xhi + ext[1], zlo + ext[2], zhi + ext[3]);
    }
  }
  // squared distance from point p to triangle (a, b, c) (Ericson, Real-Time Collision Detection);
  // the closest point is left in cq
  const cq = { x: 0, y: 0, z: 0 };
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
    cq.x = qx; cq.y = qy; cq.z = qz;
    return (px - qx) ** 2 + (py - qy) ** 2 + (pz - qz) ** 2;
  }
  // Does a ball of radius r (<= GM) at (x, y, z), t steps from now, touch an obstacle (moved on by t steps)?
  function hitsObstacle(x, y, z, r, t = 0) {
    const k = gcell(x, z), L = k < 0 ? null : obGrid[k];
    if (!L) return false;
    const O = G.obst, r2 = r * r;
    for (const i of L) {
      offsetAt(i, t);
      if (dist2(x - ox, y - oy, z - oz, O, i) < r2) return true;
    }
    return false;
  }

  // Distance from the ball centre (at time t) to the nearest obstacle within lim (<= GM) (else lim).
  function obstacleDist(x, y, z, t, lim) {
    const k = gcell(x, z), L = k < 0 ? null : obGrid[k];
    if (!L) return lim;
    const O = G.obst;
    let best = lim * lim;
    for (const i of L) {
      offsetAt(i, t);
      const d = dist2(x - ox, y - oy, z - oz, O, i);
      if (d < best) best = d;
    }
    return Math.sqrt(best);
  }

  // An impact on a surface with unit normal n, while moving into it (vn = v.n < 0): the ball
  // bounces back with restitution e and loses speed along the surface to friction, about
  // landLoss(|vn|) * |vn| (fitted on ~120 real landings).
  const landLoss = (a) => a <= 0.4 ? 0.913 : a <= 1.15 ? 0.913 - (a - 0.4) * 0.137 : a <= 1.85 ? 0.81 - (a - 1.15) * 0.057 : Math.max(0.6, 0.77 - (a - 1.85) * 0.09);
  function impact(s, nx, ny, nz, vn, e) {
    const tx = s.vx - vn * nx, ty = s.vy - vn * ny, tz = s.vz - vn * nz, vt = Math.hypot(tx, ty, tz);
    const r = vt > 1e-6 ? Math.max(0, (vt + landLoss(-vn) * vn) / vt) : 0;
    s.vx = tx * r - e * vn * nx; s.vy = ty * r - e * vn * ny; s.vz = tz * r - e * vn * nz;
  }
  // The ball against the walls (track faces too steep to roll on: sides of platforms, pipe
  // walls; their edges are the rims of platforms). It is pushed out and, if it was moving into
  // the wall, bounces off it (see impact()). Returns 1 if it hit one, 2 if it crashed into one
  // head-on (the game ends the run when the ball is stopped like that).
  function contacts(s) {
    const k = gcell(s.x, s.z), L = k < 0 ? null : triGrid[k];
    if (!L) return 0;
    const r = P.rad, r2 = r * r;
    let res = 0;
    for (const i of L) {
      const d2 = dist2(s.x, s.y, s.z, TRI, i);
      if (d2 >= r2) continue;
      const d = Math.sqrt(d2) || 1e-6, nx = (s.x - cq.x) / d, ny = (s.y - cq.y) / d, nz = (s.z - cq.z) / d;
      s.x = cq.x + nx * r; s.y = cq.y + ny * r; s.z = cq.z + nz * r;
      const vn = s.vx * nx + s.vy * ny + s.vz * nz;
      if (vn < 0) {
        if (nz < -0.7 && -vn > P.crashV) return 2;
        impact(s, nx, ny, nz, vn, P.wallE); res |= 1;
      }
    }
    return res;
  }

  // For each cell, the lowest surface in its column (x) over the next P.aheadRows rows (z):
  // an airborne ball with nothing below it there has gone over the side for good, while one
  // jumping a gap or dropping down a chute still has track somewhere below and ahead.
  const lowAhead = new Float32Array(NX * NZ), lowCell = new Float32Array(NZ);
  function buildLowAhead() {
    const W = P.aheadRows, dq = new Int32Array(NZ);
    for (let i = 0; i < NX; i++) {
      for (let j = 0; j < NZ; j++) {
        const b = (j * NX + i) * NL;
        let m = Infinity;
        for (let l = 0; l < NL; l++) if (LH[b + l] > -Infinity) m = LH[b + l];   // layers are highest first
        lowCell[j] = m;
      }
      // sliding-window minimum over rows j .. j + W (monotonic deque), from the far end back
      let h = 0, t = 0;
      for (let j = NZ - 1; j >= 0; j--) {
        while (t > h && lowCell[dq[t - 1]] >= lowCell[j]) t--;
        dq[t++] = j;
        while (dq[h] > j + W) h++;
        lowAhead[j * NX + i] = lowCell[dq[h]];
      }
    }
  }
  // Nothing below height y ahead of (x, z), in the ball's lane (or anywhere within P.underW
  // sideways, if wide)?
  const LANE = [-0.5, 0, 0.5], WIDE = [-8, -6, -4, -2, -1, 0, 1, 2, 4, 6, 8];
  function nothingAhead(x, y, z, wide) {
    for (const dx of wide ? WIDE : LANE) {
      const k = cell(x + dx, z);
      if (k >= 0 && lowAhead[k] < y) return false;
    }
    return true;
  }
  // Is there track surface within P.underW sideways of (x, z) that is more than 1 unit above y?
  function underTrack(x, y, z) {
    for (let dx = -P.underW; dx <= P.underW + 1e-6; dx += 1) {
      const k = cell(x + dx, z);
      if (k >= 0 && L0[k] > y + 1) return true;
    }
    return false;
  }

  // Extra distance to keep from red obstacles when planning, growing with how far ahead the
  // prediction is (its error grows): a near miss in the model is a hit in reality too often.
  let planning = false;
  const margin = (t) => planning ? Math.min(P.marginMax, P.margin0 + P.margin1 * t) : 0;

  // ---- ball simulation ---------------------------------------------------------------------
  // s = {x, y, z, vx, vy, vz (displacement per step), u1, u2 (keys held the last two steps),
  //      air (steps airborne), ts (time scale), gx, gz (slopes of the surface it rolls on)}.
  // Returns false when the ball dies.
  function simStep(s, u, t = 0) {
    const p = P, ts = s.ts, s2 = ts * ts;
    // Velocity changes over the step (the fitted per-step model; velocities are displacements per step).
    s.vx += s2 * (p.k1 * s.u1 + p.k2 * s.u2);
    s.u2 = s.u1; s.u1 = u;
    if (s.air === 0) {
      // rolling: sideways friction and the pull down a banked surface; along the track the
      // slope's pull (horizontal part, for a rolling ball) against a constant resistance
      s.vx -= p.dg * ts * s.vx + p.bank * s2 * s.gx;
      s.vz += s2 * (p.a0 - p.a1 * 2 * s.gz / (1 + s.gz * s.gz)) - p.a2 * ts * s.vz;
    } else {
      s.vx -= p.dax * ts * s.vx;
      s.vz += s2 * p.fz - p.da * ts * s.vz;
    }
    s.vy -= p.g * s2 + p.da * ts * s.vy;
    // The move, in sub-steps of at most ~1 unit, so short ramps and thin obstacles aren't skipped.
    const K = Math.min(6, Math.max(1, Math.ceil(Math.hypot(s.vx, s.vy, s.vz))));
    const f = 1 / K;
    let air = true;
    for (let k = 0; k < K; k++) {
      const xp = s.x, yp = s.y, zp = s.z;
      s.x += f * s.vx; s.z += f * s.vz;
      const yFree = s.y + f * s.vy;
      const h = reachable(s.x, s.z, xp, zp, yp);
      const N = Math.sqrt(1 + hit.gx * hit.gx + hit.gz * hit.gz), yc = h + p.rad * N;
      if (h > -Infinity && yFree <= yc + 0.02) {
        // on the surface. Landing, or running onto a differently sloped face (a ramp), is an
        // impact: the velocity into the surface is lost, and with it some speed along it
        // (friction; fitted on real landings). Then the ball rolls along the surface.
        if (s.air > 0 || Math.abs(hit.gx - s.gx) + Math.abs(hit.gz - s.gz) > 0.05) {
          const nx = -hit.gx / N, ny = 1 / N, nz = -hit.gz / N;
          const vn = s.vx * nx + s.vy * ny + s.vz * nz;
          if (vn < 0) impact(s, nx, ny, nz, vn, 0);
        }
        s.vy = (yc - yp) / f; s.y = yc; s.air = 0; s.gx = hit.gx; s.gz = hit.gz; s.hs = h; air = false;
      } else {
        s.y = yFree; air = true;
        if (s.air === 0) s.air = 1;
      }
      // sides of platforms, walls of pipes, rims
      const c = contacts(s);
      if (c === 2) { s.why = 3; return false; }
      if (c) s.wall = (s.wall || 0) + 1;
      // obstacles, along the whole move
      const n = Math.max(1, Math.ceil(Math.hypot(s.x - xp, s.y - yp, s.z - zp) / 0.5));
      for (let q = 1; q <= n; q++) {
        const w = q / n;
        if (hitsObstacle(xp + w * (s.x - xp), yp + w * (s.y - yp), zp + w * (s.z - zp), p.rad + margin(t), t + (k + w) * f)) { s.why = 2; return false; }
      }
    }
    if (air) {
      s.air++;
      // fallen off: no track below anywhere ahead in its lane (see buildLowAhead), and either
      // under the track that is beside it or nowhere near any track; failing that, airDead
      if (s.air > 2 && nothingAhead(s.x, s.y, s.z, false) &&
          (underTrack(s.x, s.y, s.z) || nothingAhead(s.x, s.y, s.z, true))) s.void = (s.void || 0) + 1;
      else s.void = 0;
      if (s.air > p.airDead || s.void > p.voidDead) { s.why = 1; return false; }   // fell off
    } else { s.void = 0; s.gT = t; }
    return true;
  }

  // Sideways danger near (x, z): no floor, or an obstacle, within edgeW.
  function nearDanger(s, t) {
    const w = P.edgeW;
    for (const dx of [-w, w]) {
      if (s.air === 0 && floorAt(s.x + dx, s.z, s.hs + 1) < s.hs - 3) return 1;
      if (hitsObstacle(s.x + dx, s.y, s.z, 0.5, t + 1)) return 1;
    }
    return 0;
  }

  // Controller that steers toward lateral target xt (accounting for the key delay).
  // `rate`: fraction of the remaining gap to close per step.
  function toward(s, xt, rate = 0.12) {
    const p = P;
    const s2 = s.ts * s.ts, pend = s2 * (p.k1 * s.u1 + p.k2 * s.u2 + p.k2 * s.u1);
    const want = Math.max(-0.9, Math.min(0.9, rate * (xt - s.x)));
    const err = want - (s.vx + pend);
    const dead = 0.02 + 0.004 * s.vz;
    return err > dead ? 1 : err < -dead ? -1 : 0;
  }

  // Simulate a plan. plan(s, t) -> u.
  // Returns [stepsSurvived, danger, keyPresses, firstU, alive, clearance, airSteps, wallHits];
  // clearance is the smallest gap (beyond the ball's radius) to an obstacle or a floor edge
  // over the near future.
  function run(start, plan, H) {
    const s = Object.assign({}, start);
    let danger = 0, keys = 0, first = null, clear = P.clearCap, air = 0;
    for (let t = 0; t < H; t++) {
      if (s.z > zMax - 2) return [H, danger, keys, first ?? 0, true, clear, air, s.wall || 0];   // past what we can see
      const u = plan(s, t);
      if (first === null) first = u;
      if (u) keys++;
      // Dying later than the near future doesn't erase how close the near future passes to
      // things. A fall counts from when the ball left the ground (it is only noticed later).
      if (!simStep(s, u, t)) {
        const td = s.why === 1 ? Math.min(t, (s.gT ?? -1) + 3) : t;
        return [td, danger, keys, first, false, td < P.clearSteps ? 0 : clear, air, s.wall || 0];
      }
      danger += nearDanger(s, t) * (1 - t / H);
      if (s.air > 0 && t < P.clearSteps) air++;
      if (t < P.clearSteps) {
        clear = Math.min(clear, obstacleDist(s.x, s.y, s.z, t + 1, P.clearCap + P.rad) - P.rad);
        if (s.air === 0) {
          const y = s.hs;
          for (const dx of [0.6, 1.0, 1.5]) {
            if (dx - 0.5 >= clear) break;
            if (floorAt(s.x - dx, s.z, y + 0.8) < y - 1.5 || floorAt(s.x + dx, s.z, y + 0.8) < y - 1.5) { clear = Math.min(clear, dx - 0.5); break; }
          }
        }
      }
    }
    return [H, danger, keys, first, true, clear, air, s.wall || 0];
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
  // The time scale lives in several places; not every copy is kept up to date in every game.
  // Take a value that two copies agree on.
  let lastTs = 2.76;
  const readTs = () => {
    const F = new Float32Array(window.gameInstance.Module.HEAPU8.buffer);
    const v = P.tsAddrs.map(a => F[a >> 2]).filter(x => x > 1 && x < 20);
    for (let i = 0; i < v.length; i++) for (let j = i + 1; j < v.length; j++)
      if (Math.abs(v[i] - v[j]) < 0.01) return (lastTs = v[i]);
    return lastTs;
  };
  let u1 = 0, u2 = 0;
  // Model state of the ball now (after sense()), from its last two positions and this frame's geometry.
  function current() {
    const b = hist[hist.length - 1], n = hist.length;
    const v = n >= 2 ? [0, 1, 2].map(i => hist[n - 1][i] - hist[n - 2][i]) : [0, -1, 1];
    buildMap(b);
    const s = { x: b[0], y: b[1], z: b[2], vx: v[0], vy: v[1], vz: Math.max(v[2], 0.3), u1, u2, air: 1, ts: readTs(),
                gx: 0, gz: -1, hs: b[1] - 0.7, void: 0 };
    const h = floorAt(b[0], b[2], b[1]);
    if (h > -Infinity) {
      const N = Math.sqrt(1 + hit.gx * hit.gx + hit.gz * hit.gz);
      if (b[1] - (h + P.rad * N) < 0.15) { s.air = 0; s.gx = hit.gx; s.gz = hit.gz; s.hs = h; }
    }
    return s;
  }
  // Take in this frame: ball position and obstacle motion.
  function sense() { hist.push(B.ball()); if (hist.length > 6) hist.shift(); trackObstacles(); }
  function commit(ui) { u2 = u1; u1 = ui; }
  B.ctl = {
    reset() { hist.length = 0; u1 = 0; u2 = 0; this.seq = []; this.last = null; tracks = []; ranges.length = 0; },
    movers(all) { return tracks.filter(t => all || t.vel.some(v => Math.abs(v) > 0.02)).map(t => ({ pos: t.pos, vel: t.vel, pmin: t.pmin, pmax: t.pmax, lo: t.turnLo, hi: t.turnHi, sig: t.sig })); },
    ranges() { return ranges.slice(); },
    // Call once per decision step, after the frame was rendered with geo.on = true.
    decide() {
      const t0 = B.realNow();
      sense();
      const start = current(), b = [start.x, start.y, start.z];
      planning = true;
      const H = P.horizon;
      let best = null, bestScore = -Infinity, nPlans = 0;
      const prevTag = this.last && this.last.best, prevSeq = this.seq || [];
      const survScore = (n) => P.survW * (1 - Math.pow(P.gamma, n));
      const top = [];   // the best few plans, re-checked for robustness below
      const consider = (plan, tag) => {
        // real time: the plans are tried in order of importance; stop when time is up (but
        // always try the basic ones)
        if (nPlans >= P.minPlans && B.realNow() - t0 > P.budget) return;
        const r = run(start, plan, H); nPlans++;
        const sc = survScore(r[0]) - r[1] * 2 - r[2] * 0.02 + P.clearW * Math.max(0, r[5])
          - P.airW * r[6] - P.wallW * Math.min(1, r[7]) + (tag === prevTag || tag === 'prev' ? P.stickW : 0);
        if (sc > bestScore) { bestScore = sc; best = { u: r[3], r, tag, plan, sc }; }
        if (top.length < P.robustK || sc > top[top.length - 1].sc) {
          top.push({ u: r[3], r, tag, plan, sc });
          top.sort((p, q) => q.sc - p.sc);
          if (top.length > P.robustK) top.pop();
        }
      };
      // the previous best plan, as the key sequence it produced, one step on
      if (prevSeq.length > 1) consider((s, t) => prevSeq[t + 1] ?? 0, 'prev');
      // hold still (no key)
      consider(() => 0, 'none');
      // lanes that follow the track: a fraction f across its width, possibly changing lane
      traceTrack(b[0], b[2]);
      const F = [0, 0.15, 0.3, 0.5, 0.7, 0.85, 1];
      // pure pursuit: aim at the lane L steps ahead and close the gap over L steps
      const pursue = (s, f, L) => toward(s, laneX(s.z + L * s.vz, f, s.x), 1 / L);
      for (const f of F) for (const L of [3, 5, 8]) consider((s) => pursue(s, f, L), `lane${f}/${L}`);
      // steer toward a fixed sideways target
      for (let d = -7; d <= 7.01; d += 0.5) consider((s) => toward(s, b[0] + d), 'x' + d);
      // two-stage lane changes
      for (const f1 of F) for (const T1 of [6, 14, 24]) for (const f2 of F) {
        if (f1 === f2) continue;
        consider((s, t) => pursue(s, t < T1 ? f1 : f2, 5), `lane${f1}>${f2}@${T1}`);
      }
      // Robustness: the model is never exact, so replay the best few plans from slightly
      // different starts (sideways speed, forward speed) and judge each by its worst case.
      if (top.length > 1 && B.realNow() - t0 < P.budget) {
        let bestR = -Infinity;
        for (const c of top) {
          let worst = c.r[0];
          for (const [dvx, fz] of P.perturb) {
            const r = run(Object.assign({}, start, { vx: start.vx + dvx, vz: start.vz * (1 + fz) }), c.plan, H);
            if (r[0] < worst) worst = r[0];
          }
          const sc = c.sc - P.robustW * (survScore(c.r[0]) - survScore(worst));
          if (sc > bestR) { bestR = sc; best = c; }
        }
      }
      const ui = best ? best.u : 0;
      this.lastPlan = best && best.plan; this.lastStart = start;
      // remember the chosen plan's keys, to be continued next time
      this.seq = [];
      if (best) {
        const s = Object.assign({}, start);
        for (let t = 0; t < H; t++) { const u = best.plan(s, t); this.seq.push(u); if (!simStep(s, u, t)) break; }
      }
      planning = false;
      commit(ui);
      this.last = { plans: nPlans, best: best && best.tag, surv: best && best.r[0], clear: best && +best.r[5].toFixed(2),
                    zAhead: zMax - b[2], ms: B.realNow() - t0 };
      return ui === -1 ? 1 : ui === 1 ? 2 : 0;   // index into [none, left, right]
    },
    // When something else drives: sense() after each frame, then commit(key) with the key it holds next.
    sense, commit,
    observe(ui) { sense(); commit(ui); },
    // Simulate the key sequence us (-1/0/1 per step) from the current state (call after sense(),
    // before commit()), using the geometry captured this frame. For validating the model.
    predict(us) {
      const s = current(), out = [];
      for (let t = 0; t < us.length; t++) {
        const alive = simStep(s, us[t], t);
        out.push([s.x, s.y, s.z, alive ? 0 : s.why, s.air, s.gx, s.gz, s.wall || 0, s.vx, s.vy, s.vz]);
        if (!alive) break;
      }
      return out;
    },
    state() { const s = current(); return [s.air, s.gx, s.gz, s.ts, s.hs]; },
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
      for (let i = 0; i + 8 < O.length; i += 9) {
        offsetAt(i, 1);
        obs.push([O[i], O[i + 2], O[i + 3], O[i + 5], O[i + 6], O[i + 8], ox, oz, oy, Math.min(O[i + 1], O[i + 4], O[i + 7]), Math.max(O[i + 1], O[i + 4], O[i + 7])]);
      }
      const L1 = new Float32Array(NX * NZ);
      for (let k = 0; k < NX * NZ; k++) L1[k] = LH[k * NL + 1];
      return { x0, z0, NX, NZ, DX, DZ, L0: Array.from(L0), L1: Array.from(L1), ball: b,
               edgeL: Array.from(edgeL), edgeR: Array.from(edgeR), path, obs, last: this.last };
    },
  };
})();
