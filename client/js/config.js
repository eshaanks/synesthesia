const CONFIG = {
  SERVER:   '',     // same-origin: Flask serves both client and API on one port
  CHUNK_MS: 2000,  // 2s chunks — better model context, EMA on server smooths variance
};
