// Sprite sheet loader for assets/mushrooms.png (262×308, original user art)
//
// Layout: 4 cols × 4 rows, each cell 65×77px
//
//  Row 0  (y=  0): small idle frames  — col 0, col 1
//  Row 1  (y= 77): main walk frames   — col 0, col 1, col 2, col 3
//  Row 2  (y=154): hit / alt frames   — col 0, col 1, col 2
//  Row 3  (y=231): death frames       — col 0 (body), col 1 (cap), col 2 (ball)

const CELL_W = 65;
const CELL_H = 77;
const SHEET_W = 262;
const SHEET_H = 308;

// [col, row] into the grid
export const FRAMES = {
  idle:  [0, 0],
  walk0: [0, 1],
  walk1: [1, 1],
  walk2: [2, 1],
  hit:   [0, 2],
  dead:  [0, 3],
};

let _clean  = null;
let _loaded = false;
let _failed = false;

export function isLoaded()  { return _loaded; }
export function hasFailed() { return _failed; }

export function loadSheet() {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      _clean  = _buildCleanCanvas(img);
      _loaded = true;
      resolve();
    };
    img.onerror = () => { _failed = true; reject(new Error('mushrooms.png failed')); };
    img.src = 'assets/mushrooms.png';
  });
}

// Chroma-key: make magenta (#FF00FF ±threshold) fully transparent
function _buildCleanCanvas(img) {
  const oc  = document.createElement('canvas');
  oc.width  = SHEET_W;
  oc.height = SHEET_H;
  const oc2 = oc.getContext('2d');
  oc2.drawImage(img, 0, 0);
  const id = oc2.getImageData(0, 0, SHEET_W, SHEET_H);
  const d  = id.data;
  for (let i = 0; i < d.length; i += 4) {
    const r = d[i], g = d[i+1], b = d[i+2];
    if (r > 180 && g < 80 && b > 180) d[i+3] = 0;
  }
  oc2.putImageData(id, 0, 0);
  return oc;
}

// Draw a sprite frame centered horizontally at (cx, cy), feet touching cy.
// frameName : key of FRAMES
// facing    : 1 = right, -1 = left (horizontal mirror)
// scale     : uniform scale
// alpha     : 0–1
export function drawFrame(ctx, cx, cy, frameName, facing, scale = 1, alpha = 1) {
  if (!_loaded || !_clean) return false;

  const [col, row] = FRAMES[frameName] || FRAMES.walk0;
  const sx = col * CELL_W;
  const sy = row * CELL_H;
  const dw = CELL_W * scale;
  const dh = CELL_H * scale;

  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.translate(Math.round(cx), Math.round(cy));
  ctx.scale(facing, 1);          // horizontal flip for facing direction

  // Feet sit at ~88% of cell height, so shift up to align feet with cy
  const footFrac = 0.72;
  ctx.drawImage(
    _clean,
    sx, sy, CELL_W, CELL_H,
    -dw / 2,                     // centered horizontally
    -dh * footFrac,              // feet (88% down the cell) land exactly at cy
    dw, dh
  );

  ctx.restore();
  return true;
}
