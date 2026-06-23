// ── all tunable parameters in one place ──────────────────────────────────
const CONFIG = {

  // server
  SERVER:    'http://localhost:5001',
  CHUNK_MS:  2000,        // how often mic sends a chunk to server (ms)

  // warp effect
  WARP: {
    scale:    4.0,        // zoom into noise space — higher = tighter patterns
    strength: 0.35,       // how far pixels displace — higher = more distortion
    sigma:    0.22,       // gaussian width — higher = longer transition tail
    speed:    0.008,      // transition speed per frame — lower = slower
  },

};
