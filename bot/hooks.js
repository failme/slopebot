// Injected before the game loads: makes time virtual and frames manually steppable,
// and keeps the WebGL drawing buffer readable.
(() => {
  const params = new URLSearchParams(location.search);
  const seedParam = (+params.get('seed') >>> 0) || 1;
  // Virtual clock. Its start depends on the seed, which is what varies the generated level.
  let vt = (seedParam % 100000) * 1000.37;
  const realNow = performance.now.bind(performance);
  const dateBase = 1600000000000 + seedParam * 7919000;
  performance.now = () => vt;
  const RealDate = Date;
  Date.now = () => dateBase + vt;
  // Deterministic Math.random (seed can be set via localStorage before load).
  let seed = seedParam;
  Math.random = () => { seed ^= seed << 13; seed >>>= 0; seed ^= seed >>> 17; seed ^= seed << 5; seed >>>= 0; return seed / 4294967296; };
  // Network / timer activity is a source of non-determinism (callbacks arrive on wall-clock
  // time and call back into the game). Log it so it can be inspected via __bot.netlog.
  const netlog = [];
  const isLocal = (u) => { try { return new URL(u, location.href).origin === location.origin; } catch (e) { return true; } };
  const xOpen = XMLHttpRequest.prototype.open, xSend = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.open = function (m, u, ...r) { this.__url = String(u); return xOpen.call(this, m, u, ...r); };
  XMLHttpRequest.prototype.send = function (...a) {
    if (!isLocal(this.__url)) { netlog.push(['xhr', this.__url, vt]); return; }  // never sent, never completes
    return xSend.apply(this, a);
  };
  const rFetch = window.fetch && window.fetch.bind(window);
  if (rFetch) window.fetch = (u, ...r) => {
    const url = typeof u === 'string' ? u : (u && u.url) || '';
    if (!isLocal(url)) { netlog.push(['fetch', url, vt]); return new Promise(() => {}); }
    return rFetch(u, ...r);
  };
  // Emscripten's Browser.safeSetTimeout callbacks call into the game; run them on virtual
  // time (inside step) instead of wall-clock time. They are part of save/restore state.
  const rST = window.setTimeout.bind(window);
  const rCT = window.clearTimeout.bind(window);
  const VID0 = 1 << 30;
  let vtimers = [], vid = VID0;
  window.setTimeout = (f, ms, ...r) => {
    if (typeof f === 'function' && /allowAsyncCallbacks/.test(String(f))) {
      const id = ++vid;
      vtimers.push({ due: vt + (+ms || 0), f, id });
      return id;
    }
    return rST(f, ms, ...r);
  };
  window.clearTimeout = (id) => {
    if (id > VID0) { vtimers = vtimers.filter(t => t.id !== id); return; }
    return rCT(id);
  };
  const fireTimers = () => {
    if (!vtimers.length) return;
    const due = vtimers.filter(t => t.due <= vt);
    if (!due.length) return;
    vtimers = vtimers.filter(t => t.due > vt);
    due.sort((a, b) => a.due - b.due);
    for (const t of due) t.f();
  };
  // Optional fixed render resolution: the canvas drawing buffer stays at fixedRes while CSS
  // stretches it over the window (index.html would otherwise resize it to the window).
  let fixedRes = null;
  const rSI = window.setInterval.bind(window);
  window.setInterval = (f, ms, ...r) => {
    if (typeof f === 'function' && /innerWidth/.test(String(f))) {
      const orig = f;
      f = () => {
        if (!fixedRes) return orig();
        const c = document.getElementById('#canvas');
        if (!c) return;
        if (c.width !== fixedRes[0]) c.width = fixedRes[0];
        if (c.height !== fixedRes[1]) c.height = fixedRes[1];
        c.style.width = '100vw'; c.style.height = '100vh';
      };
    }
    return rSI(f, ms, ...r);
  };
  // No gamepads, ever (also saves a per-frame navigator query).
  try { navigator.getGamepads = () => []; } catch (e) {}
  let rafQ = [];
  let manual = !!params.get('manual');
  const realRaf = window.requestAnimationFrame.bind(window);
  window.requestAnimationFrame = (cb) => {
    rafQ.push(cb);
    if (!manual) realRaf(() => { if (!manual) window.__bot.step(1000 / 60); });
    return rafQ.length;
  };
  window.cancelAnimationFrame = () => {};
  const origGetContext = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = function (type, attrs) {
    // Force WebGL1: WebGL2 fence objects live in JS tables that heap save/restore can't track.
    if (type === 'webgl2') return null;
    if (type && type.startsWith('webgl')) {
      attrs = Object.assign({}, attrs || {}, { preserveDrawingBuffer: true });
      const ctx = origGetContext.call(this, type, attrs);
      if (ctx && !ctx.__wrapped) {
        ctx.__wrapped = true;
        window.__bot.gl = ctx;
        // Draw calls can be switched off for frames nobody looks at (lookahead / skipped frames).
        for (const fn of ['drawElements', 'drawArrays', 'clear']) {
          const orig = ctx[fn].bind(ctx);
          ctx[fn] = (...a) => { if (!window.__bot.norender) orig(...a); };
        }
        // Track which shader program is bound so UI draws (Unity's UI shader) can be hidden.
        const progSrc = new Map(), shaderSrc = new Map();
        const oSS = ctx.shaderSource.bind(ctx), oAS = ctx.attachShader.bind(ctx), oUP = ctx.useProgram.bind(ctx);
        ctx.shaderSource = (sh, src) => { shaderSrc.set(sh, src); return oSS(sh, src); };
        ctx.attachShader = (pr, sh) => { progSrc.set(pr, (progSrc.get(pr) || '') + (shaderSrc.get(sh) || '')); return oAS(pr, sh); };
        let curIsUI = false;
        const isUI = (pr) => {
          // Unity UI shaders, and the text shader (colour from vertices, alpha from the font).
          if (pr && pr.__isUI === undefined) {
            const src = progSrc.get(pr) || '';
            pr.__isUI = /_TextureSampleAdd|_ClipRect/.test(src) ||
              /col_1\.xyz = xlv_COLOR\.xyz;\s*col_1\.w = \(xlv_COLOR\.w \* texture2D \(_MainTex, xlv_TEXCOORD0\)\.w\)/.test(src);
          }
          return !!(pr && pr.__isUI);
        };
        ctx.useProgram = (pr) => { curIsUI = isUI(pr); return oUP(pr); };
        for (const fn of ['drawElements', 'drawArrays']) {
          const orig = ctx[fn].bind(ctx);
          ctx[fn] = (...a) => { if (!(curIsUI && window.__bot.hideUI)) orig(...a); };
        }
        // In a browser that never renders (the planner), GPU uploads are pure waste, and
        // objects created during rolled-back rollouts would never be freed.
        for (const fn of ['bufferData', 'bufferSubData', 'texImage2D', 'texSubImage2D',
                          'compressedTexImage2D', 'compressedTexSubImage2D']) {
          const orig = ctx[fn].bind(ctx);
          ctx[fn] = (...a) => { if (!window.__bot.nogl) orig(...a); };
        }
        // Installed last so it sees every draw call, even when drawing is switched off.
        if (params.get('geo') && window.__installGeoTap) window.__installGeoTap(ctx);
      }
      return ctx;
    }
    return origGetContext.call(this, type, attrs);
  };
  window.__bot = {
    gl: null,
    norender: false,
    nogl: false,
    hideUI: false,
    buf: null,
    setManual(m) { manual = m; if (!m) this.step(1000 / 60); },
    step(dt) {
      vt += dt;
      fireTimers();
      const q = rafQ; rafQ = [];
      for (const cb of q) cb(vt);
      return q.length;
    },
    now: () => vt,
    pending: () => rafQ.length,
    // For a browser that never renders (the planner): turn WebGL state/draw/upload calls into
    // no-ops and stop the per-frame canvas size queries from touching layout.
    lite() {
      const gl = this.gl, noop = () => {};
      for (const fn of ['vertexAttribPointer', 'bindBuffer', 'bindTexture', 'uniform1f', 'uniform1fv',
        'uniform1i', 'uniform1iv', 'uniform2f', 'uniform2fv', 'uniform3f', 'uniform3fv', 'uniform4f',
        'uniform4fv', 'uniformMatrix3fv', 'uniformMatrix4fv', 'enable', 'disable', 'blendFunc',
        'blendFuncSeparate', 'blendEquation', 'blendEquationSeparate', 'depthFunc', 'depthMask',
        'colorMask', 'cullFace', 'frontFace', 'viewport', 'scissor', 'clearColor', 'clearDepth',
        'clearStencil', 'activeTexture', 'enableVertexAttribArray', 'disableVertexAttribArray',
        'useProgram', 'stencilFunc', 'stencilFuncSeparate', 'stencilOp', 'stencilOpSeparate',
        'stencilMask', 'polygonOffset', 'texParameteri', 'pixelStorei', 'bindFramebuffer',
        'bindRenderbuffer', 'drawElements', 'drawArrays', 'clear', 'bufferData', 'bufferSubData',
        'texImage2D', 'texSubImage2D', 'compressedTexImage2D', 'compressedTexSubImage2D', 'generateMipmap']) {
        if (gl[fn]) gl[fn] = noop;
      }
      const c = document.getElementById('#canvas');
      if (c) {
        const w = c.clientWidth, h = c.clientHeight;
        Object.defineProperty(c, 'clientWidth', { get: () => w });
        Object.defineProperty(c, 'clientHeight', { get: () => h });
      }
      this.norender = true; this.nogl = true;
    },
    setRes(w, h) {
      fixedRes = w ? [w, h] : null;
      const c = document.getElementById('#canvas');
      if (c && fixedRes) { c.width = w; c.height = h; c.style.width = '100vw'; c.style.height = '100vh'; }
    },
    netlog,
    // Save/restore the whole emscripten heap together with virtual time.
    // Only the in-use part of the heap (below the sbrk break) needs saving.
    save(into) {
      const M = window.gameInstance.Module;
      const top = M._sbrk(0);
      let heap = into && into.heap;
      if (!heap || heap.buffer.byteLength < top) heap = new Uint8Array(new ArrayBuffer(top + (16 << 20)), 0, top);
      else heap = new Uint8Array(heap.buffer, 0, top);
      heap.set(M.HEAPU8.subarray(0, top));
      return { heap, vt, held: this.held, timers: vtimers.slice(), vid };
    },
    load(st) {
      const M = window.gameInstance.Module;
      const cur = M._sbrk(0);
      M.HEAPU8.set(st.heap);
      // Memory the heap grew into since the save must look fresh (zeroed) again.
      if (cur > st.heap.length) M.HEAPU8.fill(0, st.heap.length, cur);
      vt = st.vt; this.held = st.held; vtimers = st.timers.slice(); vid = st.vid;
    },
    held: null,
    // Hold `key` (or nothing) and advance n frames.
    run(n, dt, key) {
      if (key !== this.held) {
        if (this.held) this.key(this.held, false);
        if (key) this.key(key, true);
        this.held = key;
      }
      for (let i = 0; i < n; i++) this.step(dt);
      return vt;
    },
    key(code, down) {
      const map = { ArrowLeft: 37, ArrowRight: 39, Space: 32 };
      const ev = new KeyboardEvent(down ? 'keydown' : 'keyup', { key: code, code, keyCode: map[code], which: map[code], bubbles: true });
      Object.defineProperty(ev, 'keyCode', { get: () => map[code] });
      Object.defineProperty(ev, 'which', { get: () => map[code] });
      window.dispatchEvent(ev);
    },
    realNow,
  };
})();
