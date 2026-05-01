// Item sprite sheet — assets/items.png (1582x1364, magenta background)
// Bounding boxes from pixel analysis: 17 cols × 14 rows

const ROW_Y = [10, 108, 205, 304, 402, 501, 600, 698, 797, 883, 994, 1094, 1191, 1288];
const ROW_H = [53,  54,  56,  57,  56,  55,  55,  56,  57,  82,  55,   54,   57,   59];
const COL_X = [30, 123, 211, 305, 398, 490, 581, 673, 765, 848,  938, 1031, 1124, 1214, 1306, 1403, 1496];
const COL_W = [58,  57,  62,  57,  57,  55,  57,  57,  57,  72,   76,   71,   70,   75,   74,   62,   59];

let _sheet = null;
let _loaded = false;

export function loadItemSheet() {
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => { _sheet = _removeMagenta(img); _loaded = true; resolve(); };
    img.onerror = () => resolve();
    img.src = 'assets/items.png';
  });
}

function _removeMagenta(img) {
  const c = document.createElement('canvas');
  c.width = img.naturalWidth;
  c.height = img.naturalHeight;
  const cx = c.getContext('2d');
  cx.drawImage(img, 0, 0);
  const id = cx.getImageData(0, 0, c.width, c.height);
  const d = id.data;
  for (let i = 0; i < d.length; i += 4) {
    const r = d[i], g = d[i+1], b = d[i+2];
    if (r > 180 && g < 80 && b > 180) d[i+3] = 0;
  }
  cx.putImageData(id, 0, 0);
  return c;
}

export function isItemSheetLoaded() { return _loaded; }

// Draw item at grid [col, row], centered on canvas point (cx, cy).
export function drawItemSprite(ctx, cx, cy, col, row, size = 28) {
  if (!_loaded || !_sheet) return false;
  if (row >= ROW_Y.length || col >= COL_X.length) return false;
  const sx = COL_X[col], sy = ROW_Y[row], sw = COL_W[col], sh = ROW_H[row];
  ctx.save();
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(_sheet, sx, sy, sw, sh, cx - size/2, cy - size/2, size, size);
  ctx.restore();
  return true;
}
