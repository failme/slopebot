// Race mode (bot/race.py): the bot's ball, playing the same level in another (hidden) copy of
// the game, is drawn as a blue ghost ball in the player's game, and a "PLAYER / BOT"
// scoreboard replaces the game's own score. Same seed = same track, so the bot's ball can be
// drawn at its real world position, however far ahead of or behind the player it is.
// Installed (with ?race=1) by hooks.js right after the WebGL context is created.
window.__installRace = (gl) => {
  const B = window.__bot;
  const R = B.race = {
    samples: [],        // [realTime ms, [x, y, z]] of the bot's ball, newest last
    delay: 60,          // show the bot's ball this many ms in the past, interpolating between samples
    ghostOn: false,
    push(pos) { this.samples.push([B.realNow(), pos]); if (this.samples.length > 8) this.samples.shift(); this.ghostOn = true; },
    hide() { this.ghostOn = false; this.samples = []; },
    // where the bot's ball is (interpolated), or null
    ghost() {
      const S = this.samples;
      if (!this.ghostOn || !S.length) return null;
      const t = B.realNow() - this.delay;
      if (t <= S[0][0]) return S[0][1];
      for (let i = S.length - 1; i > 0; i--) {
        if (S[i - 1][0] <= t) {
          const [t0, a] = S[i - 1], [t1, b] = S[i];
          const f = Math.min(1, (t - t0) / Math.max(1, t1 - t0));
          return a.map((v, k) => v + f * (b[k] - v));
        }
      }
      return S[S.length - 1][1];
    },
    cmd: null,          // set by the keyboard (Enter / Escape) for race.py
  };

  // --- the ghost ball: the game's own ball mesh drawn a second time, moved and recoloured ---
  const uniLoc = new Map();          // uniform location -> [program, name]
  const o2w = new Map();             // program -> [location, current unity_ObjectToWorld]
  let prog = null, blueTex = null;
  const W = (name, fn) => { const o = gl[name].bind(gl); gl[name] = (...a) => { const r = o(...a); fn(a, r); return r; }; return o; };
  W('getUniformLocation', (a, r) => { if (r) uniLoc.set(r, [a[0], a[1]]); });
  W('useProgram', a => { prog = a[0]; });
  const setMat = W('uniformMatrix4fv', a => {
    const u = uniLoc.get(a[0]);
    if (u && u[1] === 'unity_ObjectToWorld') o2w.set(u[0], [a[0], Float32Array.from(a[2].subarray ? a[2].subarray(0, 16) : a[2])]);
  });
  // Keep a copy of every small texture's pixels (the ball's is a 32x32 RGB grid) so that a blue
  // copy of the ball's own texture can be made.
  const texData = new Map();
  const texBound = () => gl.getParameter(gl.TEXTURE_BINDING_2D);
  W('texImage2D', a => {
    const d = a[8];
    if (a[1] === 0 && a.length >= 9 && d && d.buffer && a[3] * a[4] <= 4096 && a[7] === gl.UNSIGNED_BYTE)
      texData.set(texBound(), { w: a[3], h: a[4], fmt: a[6], data: new Uint8Array(d.buffer, d.byteOffset, d.byteLength).slice() });
  });
  // Made once, from the ball's texture (bound on unit 0 while the ball is drawn): its green grid
  // lines turn light blue and its black body dark blue.
  const makeTex = () => {
    const src = texData.get(texBound());
    const t = gl.createTexture();
    const params = [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER, gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]
      .map(k => [k, gl.getTexParameter(gl.TEXTURE_2D, k)]);
    const prev = texBound();
    gl.bindTexture(gl.TEXTURE_2D, t);
    const align = gl.getParameter(gl.UNPACK_ALIGNMENT), flip = gl.getParameter(gl.UNPACK_FLIP_Y_WEBGL);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1); gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    if (src && (src.fmt === gl.RGB || src.fmt === gl.RGBA)) {
      const n = src.fmt === gl.RGB ? 3 : 4, d = src.data.slice();
      for (let i = 0; i + n <= d.length; i += n) {
        const v = Math.max(d[i], d[i + 1], d[i + 2]) / 255;   // line intensity
        d[i] = 8 + v * (90 - 8); d[i + 1] = 30 + v * (185 - 30); d[i + 2] = 110 + v * (255 - 110);
      }
      gl.texImage2D(gl.TEXTURE_2D, 0, src.fmt, src.w, src.h, 0, src.fmt, gl.UNSIGNED_BYTE, d);
    } else {   // (unknown texture: plain blue)
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, 1, 1, 0, gl.RGB, gl.UNSIGNED_BYTE, new Uint8Array([40, 110, 230]));
    }
    const pot = !src || ((src.w & (src.w - 1)) === 0 && (src.h & (src.h - 1)) === 0);
    for (const [k, v] of params) {
      if (k === gl.TEXTURE_MIN_FILTER && v !== gl.LINEAR && v !== gl.NEAREST && !pot) gl.texParameteri(gl.TEXTURE_2D, k, gl.LINEAR);
      else gl.texParameteri(gl.TEXTURE_2D, k, v);
    }
    if (pot) gl.generateMipmap(gl.TEXTURE_2D);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, align); gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, flip);
    gl.bindTexture(gl.TEXTURE_2D, prev);
    return t;
  };
  const draw = gl.drawElements;   // (already wrapped by hooks.js for norender / hideUI)
  gl.drawElements = (...a) => {
    draw(...a);
    if (a[1] !== 2304 || B.norender) return;   // 2304 indices: the ball mesh
    const g = R.ghost(), m = o2w.get(prog);
    if (!g || !m || B.ballIdx === undefined || B.ballIdx < 0) return;
    const p = B.ball(), M = m[1].slice();
    M[12] += g[0] - p[0]; M[13] += g[1] - p[1]; M[14] += g[2] - p[2];
    const prevUnit = gl.getParameter(gl.ACTIVE_TEXTURE);
    gl.activeTexture(gl.TEXTURE0);
    if (!blueTex) blueTex = makeTex();
    const prevTex = gl.getParameter(gl.TEXTURE_BINDING_2D);
    gl.bindTexture(gl.TEXTURE_2D, blueTex);
    setMat(m[0], false, M);
    draw(...a);
    setMat(m[0], false, m[1]);
    gl.bindTexture(gl.TEXTURE_2D, prevTex);
    gl.activeTexture(prevUnit);
  };

  // --- scoreboard ---
  let box = null, shade = null;
  const el = (css) => { const d = document.createElement('div'); d.style.cssText = css; return d; };
  R.board = (s) => {
    if (!box) {
      box = el('position:fixed;left:0;right:0;top:12px;z-index:10;text-align:center;pointer-events:none;' +
               'font:bold 30px monospace;color:#fff;text-shadow:0 0 6px #000,0 0 2px #000');
      box.innerHTML = '<div><span id="rp" style="color:#7dff5a"></span>&nbsp;&nbsp;&nbsp;<span id="rb" style="color:#59b8ff"></span></div>' +
                      '<div id="rg" style="font-size:18px;margin-top:4px"></div>' +
                      '<div id="rm" style="font-size:40px;margin-top:18vh"></div><div id="rs" style="font-size:20px;margin-top:10px"></div>';
      document.body.appendChild(box);
      // covers the game (and its game-over screen, which would restart it if clicked) once the player is out
      shade = el('position:fixed;inset:0;z-index:2147483646;background:rgba(0,0,0,0.95);display:none');
      document.body.appendChild(shade);
      box.style.zIndex = '2147483647';
      window.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') R.cmd = 'again';
        else if (e.key === 'Escape') R.cmd = 'quit';
      });
    }
    const q = (id) => box.querySelector('#' + id);
    q('rp').textContent = `PLAYER: ${s.player}${s.playerOut ? ' ✖' : ''}`;
    q('rb').textContent = `BOT: ${s.bot}${s.botOut ? ' ✖' : ''}`;
    q('rg').textContent = s.gap || '';
    q('rm').textContent = s.msg || '';
    q('rs').textContent = s.sub || '';
    shade.style.display = s.playerOut ? 'block' : 'none';
  };
};
