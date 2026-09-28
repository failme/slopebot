// Geometry tap. Mirrors WebGL buffer contents so that every draw call can be decoded into
// world-space triangles. Slope draws the green track with one texture and the deadly red
// obstacles with another, so each frame yields the exact track and obstacle geometry.
// Installed (with ?geo=1) by hooks.js right after the WebGL context is created.
window.__installGeoTap = (gl) => {
  const data = new Map();            // WebGLBuffer -> Uint8Array copy of its contents
  let arr = null, el = null, tex = null;
  const ptr = {};                    // attribute index -> pointer info
  const uniLoc = new Map();          // uniform location -> [program, name]
  const o2w = new Map();             // program -> its current unity_ObjectToWorld
  let prog = null;
  const texId = new Map(); let texN = 0;
  const W = (name, fn) => { const o = gl[name].bind(gl); gl[name] = (...a) => { const r = o(...a); fn(a, r); return r; }; };
  W('getUniformLocation', (a, r) => { if (r) uniLoc.set(r, [a[0], a[1]]); });
  W('useProgram', a => { prog = a[0]; });
  W('bindBuffer', a => { if (a[0] === gl.ARRAY_BUFFER) arr = a[1]; else el = a[1]; });
  W('bufferData', a => {
    const b = a[0] === gl.ARRAY_BUFFER ? arr : el;
    if (typeof a[1] === 'number') data.set(b, new Uint8Array(a[1]));
    else { const s = a[1]; data.set(b, new Uint8Array(s.buffer.slice(s.byteOffset, s.byteOffset + s.byteLength))); }
  });
  W('bufferSubData', a => {
    const b = a[0] === gl.ARRAY_BUFFER ? arr : el, d = data.get(b);
    if (!d) return;
    const s = a[2];
    d.set(new Uint8Array(s.buffer, s.byteOffset, Math.min(s.byteLength, d.length - a[1])), a[1]);
  });
  W('vertexAttribPointer', a => { ptr[a[0]] = { stride: a[4], off: a[5], buf: arr }; });
  W('uniformMatrix4fv', a => {
    const u = uniLoc.get(a[0]);
    if (u && u[1] === 'unity_ObjectToWorld') o2w.set(u[0], Float32Array.from(a[2].subarray ? a[2].subarray(0, 16) : a[2]));
  });
  W('bindTexture', a => { if (a[1]) { if (!texId.has(a[1])) texId.set(a[1], texN++); tex = texId.get(a[1]); } });

  const G = window.__bot.geo = {
    TRACK: 34, OBSTACLE: 35,   // texture ids (load order is deterministic)
    TRACK2: 45,                 // a few special track tiles use this one
    on: false,
    track: [], obst: [],        // flat [x,y,z, x,y,z, x,y,z, ...] world-space triangles, last frame
    begin() { this.track = []; this.obst = []; this.other = {}; },
    other: {},                  // debugging (all = true): triangles drawn with any other texture
    all: false,
  };
  W('drawElements', a => {
    if (!G.on) return;
    const isTrack = tex === G.TRACK || tex === G.TRACK2;
    if (!isTrack && tex !== G.OBSTACLE && !G.all) return;
    const cnt = a[1], type = a[2], off = a[3];
    if (cnt === 2304) return;  // the ball mesh (same texture as the track)
    const ib = data.get(el), p = ptr[0];
    if (!ib || !p) return;
    const vb = data.get(p.buf);
    if (!vb) return;
    const idx = type === gl.UNSIGNED_SHORT ? new Uint16Array(ib.buffer, off, cnt) : new Uint32Array(ib.buffer, off, cnt);
    const f = new Float32Array(vb.buffer, 0, vb.byteLength >> 2), m = o2w.get(prog);
    const out = isTrack ? G.track : tex === G.OBSTACLE ? G.obst : (G.other[tex] = G.other[tex] || []);
    for (let k = 0; k < cnt; k++) {
      const v = (idx[k] * p.stride + p.off) >> 2;
      const x = f[v], y = f[v + 1], z = f[v + 2];
      if (m) out.push(m[0] * x + m[4] * y + m[8] * z + m[12], m[1] * x + m[5] * y + m[9] * z + m[13], m[2] * x + m[6] * y + m[10] * z + m[14]);
      else out.push(x, y, z);
    }
  });
};
