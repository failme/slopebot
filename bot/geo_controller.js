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
    // Per-step quantities scale with the game's time scale ts (Time.timeScale, read from
    // memory; it grows from ~2.75 to ~3.4 during a game): accelerations with ts^2, drag with ts.
    // Fitted on ~5000 steps of real games (R^2 0.99 for the sideways response).
    k1: 0.00408, k2: 0.00456, // sideways velocity change from the key held last step / the step before (x ts^2)
    dg: 0.00865,              // sideways rolling friction on the ground (x ts)
    bank: 0.0069,             // sideways pull per unit of sideways surface slope (x ts^2)
    a0: -0.00439, a1: 0.00655, a2: 0.0005,  // along-track acceleration on the ground (see simStep)
    g: 0.01834,               // gravity (x ts^2)
    landT: 0.92, landN: 0.51, // impact: speed along the surface after = landT * before + landN * (normal speed, < 0)
    wallE: 0.3,               // restitution when bouncing off a wall
    da: 0.00086, dax: 0.00127, fz: 0.00152,  // air drag, sideways air drag (x ts), forward push in the air (x ts^2)
    rad: 0.495,               // ball radius
    tsAddr: 26580708,         // Time.timeScale (float) in the Emscripten heap
    reach: 0.3,               // a surface whose plane passes this far above the ball's last contact is a wall/block top, not a ramp
    airDead: 40,              // steps airborne that count as falling off
    voidDead: 12,             // steps with no surface anywhere below that count as falling off
    horizon: 60,              // simulated steps per plan
    edgeW: 1.2,               // lateral distance at which a missing floor / obstacle counts as "close"
    decal: -1e9,              // red geometry less than this above the floor is a marking, not an obstacle
    clearSteps: 30,           // steps over which clearance to obstacles / edges is measured
    clearCap: 1.5,            // clearance beyond this doesn't count
    clearW: 25,               // score per unit of clearance (1 step of survival = 10)
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
    WALLS = []; wallGrid = new Array(GNX * GNZ);
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
      const w = WALLS.length;
      WALLS.push(ax, ay, az, bx, by, bz, cx, cy, cz);
      gridAdd(wallGrid, w, Math.min(ax, bx, cx), Math.max(ax, bx, cx), Math.min(az, bz, cz), Math.max(az, bz, cz));
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
  let obGrid = [], wallGrid = [], WALLS = [];
  const gcell = (x, z) => {
    const i = Math.floor((x - x0) / GX), j = Math.floor((z - z0) / GZ);
    return (i < 0 || j < 0 || i >= GNX || j >= GNZ) ? -1 : j * GNX + i;
  };
  function gridAdd(grid, idx, xlo, xhi, zlo, zhi) {
    const i0 = Math.max(0, Math.floor((xlo - GM - x0) / GX)), i1 = Math.min(GNX - 1, Math.floor((xhi + GM - x0) / GX));
    const j0 = Math.max(0, Math.floor((zlo - GM - z0) / GZ)), j1 = Math.min(GNZ - 1, Math.floor((zhi + GM - z0) / GZ));
    for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) {
      const k = j * GNX + i;
      (grid[k] || (grid[k] = [])).push(idx);
    }
  }
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
    obGrid = new Array(GNX * GNZ);
    const O = G.obst, H = P.horizon;
    for (let i = 0; i + 8 < O.length; i += 9) {
      const zlo = Math.min(O[i + 2], O[i + 5], O[i + 8]), zhi = Math.max(O[i + 2], O[i + 5], O[i + 8]);
      const xlo = Math.min(O[i], O[i + 3], O[i + 6]), xhi = Math.max(O[i], O[i + 3], O[i + 6]);
      // skip red markings lying flat on the track
      const cxm = (xlo + xhi) / 2, czm = (zlo + zhi) / 2, ym = Math.max(O[i + 1], O[i + 4], O[i + 7]);
      const f = floorAt(cxm, czm, ym + 1);
      if (f > -Infinity && ym <= f + P.decal) continue;
      const vx = vel[i / 3] * H, vz = vel[i / 3 + 2] * H;
      gridAdd(obGrid, i, xlo + Math.min(0, vx), xhi + Math.max(0, vx), zlo + Math.min(0, vz), zhi + Math.max(0, vz));
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
      const v = i / 3;
      if (dist2(x - vel[v] * t, y - vel[v + 1] * t, z - vel[v + 2] * t, O, i) < r2) return true;
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
      const v = i / 3;
      const d = dist2(x - vel[v] * t, y - vel[v + 1] * t, z - vel[v + 2] * t, O, i);
      if (d < best) best = d;
    }
    return Math.sqrt(best);
  }

  // An impact on a surface with unit normal n, while moving into it (vn = v.n < 0): the ball
  // bounces back with restitution e and loses speed along the surface to friction (fitted on
  // real landings: after = landT * before + landN * vn, for impacts of 0.5+ units/step).
  function impact(s, nx, ny, nz, vn, e) {
    const tx = s.vx - vn * nx, ty = s.vy - vn * ny, tz = s.vz - vn * nz, vt = Math.hypot(tx, ty, tz);
    const a = 1 - (1 - P.landT) * Math.min(1, -vn / 0.5);
    const r = vt > 1e-6 ? Math.max(0, (a * vt + P.landN * vn) / vt) : 0;
    s.vx = tx * r - e * vn * nx; s.vy = ty * r - e * vn * ny; s.vz = tz * r - e * vn * nz;
  }
  // Push the ball out of any wall it overlaps; an impact if it was moving into it.
  function walls(s) {
    const k = gcell(s.x, s.z), L = k < 0 ? null : wallGrid[k];
    if (!L) return false;
    const r = P.rad;
    let hitAny = false;
    for (const i of L) {
      const d2 = dist2(s.x, s.y, s.z, WALLS, i);
      if (d2 >= r * r) continue;
      const d = Math.sqrt(d2) || 1e-6, nx = (s.x - cq.x) / d, ny = (s.y - cq.y) / d, nz = (s.z - cq.z) / d;
      s.x = cq.x + nx * r; s.y = cq.y + ny * r; s.z = cq.z + nz * r;
      const vn = s.vx * nx + s.vy * ny + s.vz * nz;
      if (vn < 0) { impact(s, nx, ny, nz, vn, P.wallE); hitAny = true; }
    }
    return hitAny;
  }

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
      // sides of platforms, walls of pipes: the ball bounces off them
      if (walls(s)) s.wall = (s.wall || 0) + 1;
      // obstacles, along the whole move
      const n = Math.max(1, Math.ceil(Math.hypot(s.x - xp, s.y - yp, s.z - zp) / 0.5));
      for (let q = 1; q <= n; q++) {
        const w = q / n;
        if (hitsObstacle(xp + w * (s.x - xp), yp + w * (s.y - yp), zp + w * (s.z - zp), p.rad, t + (k + w) * f)) { s.why = 2; return false; }
      }
    }
    if (air) {
      s.air++;
      // nothing at all below: falling off the side (a jump over a short gap lands in time)
      s.void = floorAt(s.x, s.z, s.y) === -Infinity ? (s.void || 0) + 1 : 0;
      if (s.air > p.airDead || s.void > p.voidDead) { s.why = 1; return false; }   // fell off
    } else s.void = 0;
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
      if (!simStep(s, u, t)) return [t, danger, keys, first, false, 0, air, s.wall || 0];
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
  const readTs = () => {
    const v = new Float32Array(window.gameInstance.Module.HEAPU8.buffer, P.tsAddr, 1)[0];
    return v > 0.5 && v < 20 ? v : 3;
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
    reset() { hist.length = 0; u1 = 0; u2 = 0; },
    // Call once per decision step, after the frame was rendered with geo.on = true.
    decide() {
      const t0 = B.realNow();
      sense();
      const start = current(), b = [start.x, start.y, start.z];
      const H = P.horizon;
      let best = null, bestScore = -Infinity, nPlans = 0;
      const consider = (plan, tag) => {
        const r = run(start, plan, H); nPlans++;
        const sc = r[0] * 10 - r[1] * 2 - r[2] * 0.02 + P.clearW * Math.max(0, r[5]) - P.airW * r[6] - P.wallW * Math.min(1, r[7]);
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
      commit(ui);
      this.last = { plans: nPlans, best: best && best.tag, surv: best && best.r[0], zAhead: zMax - b[2], ms: B.realNow() - t0 };
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
        out.push([s.x, s.y, s.z, alive ? 0 : s.why, s.air]);
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
      for (let i = 0; i + 8 < O.length; i += 9) obs.push([O[i], O[i + 2], O[i + 3], O[i + 5], O[i + 6], O[i + 8], vel[i / 3], vel[i / 3 + 2]]);
      const L1 = new Float32Array(NX * NZ);
      for (let k = 0; k < NX * NZ; k++) L1[k] = LH[k * NL + 1];
      return { x0, z0, NX, NZ, DX, DZ, L0: Array.from(L0), L1: Array.from(L1), ball: b,
               edgeL: Array.from(edgeL), edgeR: Array.from(edgeR), path, obs, last: this.last };
    },
  };
})();
