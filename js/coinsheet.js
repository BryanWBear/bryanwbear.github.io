// Coin sprite sheet — assets/coins.png (188x258, gray background #e0e0e0)
// Bounding boxes determined by pixel analysis.

// Per-tier sprite bounding boxes: each entry is [sx, sy, sw, sh]
const COIN_SPRITES = {
  bronze: [
    [14, 10, 46, 48],
    [64, 10, 50, 48],
    [118, 10, 46, 48],
  ],
  gold: [
    [14, 62, 46, 48],
    [64, 62, 50, 48],
    [118, 62, 46, 48],
  ],
  bill: [
    [14, 114, 62, 60],
    [80, 114, 64, 60],
  ],
  meso: [
    [14, 180, 64, 62],
    [82, 180, 64, 62],
  ],
};

export const COIN_TIERS = {
  bronze: { frames: 3, label: 'Bronze Coin' },
  gold:   { frames: 3, label: 'Gold Coin' },
  bill:   { frames: 2, label: 'Meso Bills' },
  meso:   { frames: 2, label: 'Meso Bag' },
};

export const COIN_AMOUNTS = {
  bronze: [10, 50],
  gold:   [100, 500],
  bill:   [800, 3000],
  meso:   [2000, 8000],
};

let _sheet = null;
let _loaded = false;

export function loadCoinSheet() {
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => { _sheet = _removeGray(img); _loaded = true; resolve(); };
    img.onerror = () => resolve();
    img.src = 'assets/coins.png';
  });
}

function _removeGray(img) {
  const c = document.createElement('canvas');
  c.width = img.naturalWidth;
  c.height = img.naturalHeight;
  const cx = c.getContext('2d');
  cx.drawImage(img, 0, 0);
  const id = cx.getImageData(0, 0, c.width, c.height);
  const d = id.data;
  for (let i = 0; i < d.length; i += 4) {
    const r = d[i], g = d[i+1], b = d[i+2];
    // Gray background: ~(224,224,224)
    if (r > 200 && g > 200 && b > 200 && Math.abs(r-g) < 20 && Math.abs(g-b) < 20) {
      d[i+3] = 0;
    }
    // Also remove magenta separators if any
    if (r > 180 && g < 80 && b > 180) d[i+3] = 0;
  }
  cx.putImageData(id, 0, 0);
  return c;
}

export function isCoinSheetLoaded() { return _loaded; }

export function drawCoinSprite(ctx, cx, cy, tier, frame, size = 26) {
  if (!_loaded || !_sheet) return false;
  const sprites = COIN_SPRITES[tier];
  if (!sprites) return false;
  const [sx, sy, sw, sh] = sprites[frame % sprites.length];
  ctx.save();
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(_sheet, sx, sy, sw, sh, cx - size/2, cy - size/2, size, size);
  ctx.restore();
  return true;
}
