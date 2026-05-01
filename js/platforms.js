// Platform surfaces from pixel-level grass-top detection on bg.png (1920×1080).
// yFrac, xFrac, wFrac all expressed as fractions of canvas dimensions.

const DETECTED = [
  { yFrac: 0.164, xFrac: 0.190, wFrac: 0.572 },  // second platform (confirmed)
  { yFrac: 0.401, xFrac: 0.230, wFrac: 0.578 },  // mid platform (confirmed)
  { yFrac: 0.639, xFrac: 0.082, wFrac: 0.813 },  // lower wide platform (confirmed)
  { yFrac: 0.934, xFrac: 0.177, wFrac: 0.741 },  // ground level
];

export const SPAWN_PLATFORMS = DETECTED;

export function getPlatforms(canvasW, canvasH) {
  return DETECTED.map(p => ({
    x: p.xFrac * canvasW,
    y: p.yFrac * canvasH,
    w: p.wFrac * canvasW,
  }));
}

// No-op — detection already done offline, kept for import compatibility
export function detectPlatforms() {}
