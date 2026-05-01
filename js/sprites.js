// Sprites and background rendering

export function getGroundY(h) {
  return Math.floor(h * 0.710); // matches PLATFORM_DEFS ground yFrac
}

// ── BACKGROUND IMAGE ──────────────────────────────────────────────────────
let _bgImg = null;
let _bgLoaded = false;

export function loadBackground() {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => { _bgImg = img; _bgLoaded = true; resolve(img); };
    img.onerror = () => resolve(null);
    img.src = 'assets/bg.png';
  });
}

export function drawBackground(ctx, w, h) {
  if (_bgLoaded && _bgImg) {
    ctx.drawImage(_bgImg, 0, 0, w, h);
    return;
  }
  // ── CANVAS FALLBACK ───────────────────────────────────────────────────
  const groundY = getGroundY(h);

  // Sky — pale blue gradient like Henesys
  const sky = ctx.createLinearGradient(0, 0, 0, groundY);
  sky.addColorStop(0,   '#b0d8f8');
  sky.addColorStop(0.6, '#d8eefa');
  sky.addColorStop(1,   '#e8f8e8');
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, w, groundY);

  // Distant rolling hills (3 layers, pastel greens)
  _drawHills(ctx, w, groundY, 0.65, '#8ec46a', 220, 70);
  _drawHills(ctx, w, groundY, 0.78, '#a8d47a', 160, 55);
  _drawHills(ctx, w, groundY, 0.90, '#c0e490', 110, 40);

  // Cherry blossom trees (Henesys signature)
  const treeXs = [0.06, 0.19, 0.38, 0.54, 0.70, 0.87, 0.96];
  treeXs.forEach((fx, i) => {
    const tx = w * fx;
    const ty = groundY - 10 + ((i % 3) - 1) * 12;
    _drawCherryTree(ctx, tx, ty, 18 + (i % 3) * 5);
  });

  // Ground fill
  const groundGrad = ctx.createLinearGradient(0, groundY, 0, h);
  groundGrad.addColorStop(0,   '#5a8c2a');
  groundGrad.addColorStop(0.08,'#4a7020');
  groundGrad.addColorStop(1,   '#3a5818');
  ctx.fillStyle = groundGrad;
  ctx.fillRect(0, groundY, w, h - groundY);

  // Grass top row (bright)
  ctx.fillStyle = '#78b830';
  ctx.fillRect(0, groundY, w, 5);
  ctx.fillStyle = '#90d040';
  ctx.fillRect(0, groundY, w, 2);

  // Grass tufts
  ctx.fillStyle = '#a0e050';
  for (let x = 4; x < w; x += 18) {
    const jitter = (x * 13 % 9) - 4;
    _drawTuft(ctx, x + jitter, groundY);
  }

  // Wooden platform planks (MapleStory brown platforms)
  _drawPlatform(ctx, 0, groundY + 2, w);

  // Clouds (MapleStory fluffy white clouds)
  ctx.globalAlpha = 0.92;
  _drawCloud(ctx, w * 0.09, h * 0.09, 55);
  _drawCloud(ctx, w * 0.32, h * 0.06, 72);
  _drawCloud(ctx, w * 0.60, h * 0.10, 48);
  _drawCloud(ctx, w * 0.80, h * 0.07, 63);
  ctx.globalAlpha = 1;
}

function _drawHills(ctx, w, groundY, yFrac, color, rx, ry) {
  ctx.fillStyle = color;
  const baseY = groundY * yFrac;
  // Draw overlapping ellipses as rolling hills
  for (let x = -rx; x < w + rx; x += rx * 1.3) {
    ctx.beginPath();
    ctx.ellipse(x, baseY, rx, ry, 0, Math.PI, 0);
    ctx.fill();
  }
  ctx.fillRect(0, baseY, w, groundY - baseY);
}

function _drawCherryTree(ctx, cx, by, r) {
  // Trunk
  ctx.fillStyle = '#7a4a1a';
  ctx.fillRect(cx - 3, by - r * 1.8, 6, r * 1.8);
  ctx.fillStyle = '#a06828';
  ctx.fillRect(cx - 1, by - r * 1.8, 2, r * 1.8);

  // Foliage — pink blobs (3 overlapping ellipses)
  const blobs = [
    [cx, by - r * 2.2, r * 1.1, r * 0.75],
    [cx - r * 0.7, by - r * 1.7, r * 0.85, r * 0.65],
    [cx + r * 0.7, by - r * 1.7, r * 0.85, r * 0.65],
  ];
  blobs.forEach(([ex, ey, rx2, ry2]) => {
    ctx.fillStyle = '#f080a0';
    ctx.beginPath();
    ctx.ellipse(ex, ey, rx2, ry2, 0, 0, Math.PI * 2);
    ctx.fill();
    // Highlight
    ctx.fillStyle = '#f8b0c8';
    ctx.beginPath();
    ctx.ellipse(ex - rx2 * 0.25, ey - ry2 * 0.25, rx2 * 0.45, ry2 * 0.4, -0.3, 0, Math.PI * 2);
    ctx.fill();
  });

  // Petals scattered
  ctx.fillStyle = '#ffffff';
  for (let i = 0; i < 5; i++) {
    const angle = (i / 5) * Math.PI * 2;
    const pr = r * 0.9;
    ctx.fillRect(
      cx + Math.cos(angle) * pr - 1,
      (by - r * 2.0) + Math.sin(angle) * r * 0.6 - 1,
      2, 2
    );
  }
}

function _drawPlatform(ctx, x, y, w) {
  // MapleStory-style brown wood plank
  ctx.fillStyle = '#8b5a1a';
  ctx.fillRect(x, y, w, 14);
  // Top highlight
  ctx.fillStyle = '#c8841e';
  ctx.fillRect(x, y, w, 3);
  // Bottom shadow
  ctx.fillStyle = '#5a3008';
  ctx.fillRect(x, y + 11, w, 3);
  // Plank lines
  ctx.fillStyle = '#7a4a14';
  for (let px = 0; px < w; px += 48) {
    ctx.fillRect(px, y + 3, 2, 8);
  }
}

function _drawTuft(ctx, x, y) {
  ctx.fillRect(x,     y - 4, 2, 5);
  ctx.fillRect(x + 3, y - 6, 2, 7);
  ctx.fillRect(x + 6, y - 4, 2, 5);
}

function _drawCloud(ctx, cx, cy, r) {
  ctx.fillStyle = '#ffffff';
  [[0, 0, r, r * 0.55],
   [-r * 0.5, r * 0.12, r * 0.6, r * 0.42],
   [r * 0.52, r * 0.12, r * 0.58, r * 0.42],
   [-r * 0.25, r * 0.25, r * 0.4, r * 0.32],
   [r * 0.28, r * 0.25, r * 0.38, r * 0.32],
  ].forEach(([dx, dy, ex, ey]) => {
    ctx.beginPath();
    ctx.ellipse(cx + dx, cy + dy, ex, ey, 0, 0, Math.PI * 2);
    ctx.fill();
  });
  // Bottom shadow
  ctx.fillStyle = 'rgba(160,180,210,0.3)';
  ctx.beginPath();
  ctx.ellipse(cx, cy + r * 0.35, r * 0.85, r * 0.18, 0, 0, Math.PI * 2);
  ctx.fill();
}

// ── MUSHROOM SPRITES ──────────────────────────────────────────────────────
// Faithfully recreated from knowledge of MapleStory's Orange/Green/Blue Mushrooms
// frame: 0|1  facing: 1=right, -1=left
export function drawMushroom(ctx, x, y, type, frame, facing, alpha = 1, scaleX = 1, scaleY = 1) {
  const PALETTE = {
    orange: {
      cap:    '#e84800',  // deep orange-red
      capMid: '#ff6a00',  // mid orange
      capHi:  '#ff9040',  // highlight
      capSh:  '#8a2800',  // shadow underside
      dot:    '#ffffff',  // white polka dots
      dotSh:  '#dddddd',
      stalk:  '#f5e0b0',  // cream/beige
      stalkSh:'#c8b070',
      eye:    '#1a0800',
      mouth:  '#1a0800',
    },
    green: {
      cap:    '#1a6b10',
      capMid: '#28a018',
      capHi:  '#50d030',
      capSh:  '#0a3808',
      dot:    '#ffffff',
      dotSh:  '#ccddcc',
      stalk:  '#d8f0b8',
      stalkSh:'#a8c888',
      eye:    '#081a04',
      mouth:  '#081a04',
    },
    blue: {
      cap:    '#1a2888',
      capMid: '#2840cc',
      capHi:  '#5070ee',
      capSh:  '#0a1248',
      dot:    '#ffffff',
      dotSh:  '#ccccee',
      stalk:  '#c8d8f8',
      stalkSh:'#8898c8',
      eye:    '#04081a',
      mouth:  '#04081a',
    },
    dark: {
      cap:    '#3a3028',
      capMid: '#5a4a38',
      capHi:  '#7a6a58',
      capSh:  '#1a1410',
      dot:    '#ffffff',
      dotSh:  '#aaaaaa',
      stalk:  '#e8e4e0',  // very pale/white round body
      stalkSh:'#b8b0a8',
      eye:    '#100c08',
      mouth:  '#100c08',
    },
  };
  const p = PALETTE[type] || PALETTE.orange;

  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.translate(Math.round(x), Math.round(y));
  ctx.scale(facing * scaleX, scaleY);

  if (type === 'dark') {
    _drawDarkMushroom(ctx, p, frame, alpha);
  } else {
    _drawStandardMushroom(ctx, p, frame, alpha);
  }

  ctx.restore();
}

function _drawDarkMushroom(ctx, p, frame, alpha) {
  // Same oval body silhouette as standard but puffier (slightly larger)
  // Feet
  ctx.fillStyle = p.stalkSh;
  if (frame === 0) {
    ctx.beginPath(); ctx.ellipse(-6, 8, 5, 3, 0, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.ellipse( 6, 6, 5, 4, 0, 0, Math.PI * 2); ctx.fill();
  } else {
    ctx.beginPath(); ctx.ellipse(-6, 6, 5, 4, 0, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.ellipse( 6, 8, 5, 3, 0, 0, Math.PI * 2); ctx.fill();
  }
  ctx.fillStyle = p.stalk;
  if (frame === 0) {
    ctx.beginPath(); ctx.ellipse(-6, 7, 4, 2.5, 0, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.ellipse( 6, 5, 4, 3,   0, 0, Math.PI * 2); ctx.fill();
  } else {
    ctx.beginPath(); ctx.ellipse(-6, 5, 4, 3,   0, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.ellipse( 6, 7, 4, 2.5, 0, 0, Math.PI * 2); ctx.fill();
  }

  // Body — extra round, ghost-white
  ctx.fillStyle = p.stalkSh;
  ctx.beginPath(); ctx.ellipse(0, -2, 15, 13, 0, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = p.stalk;
  ctx.beginPath(); ctx.ellipse(0, -3, 14, 12, 0, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = '#ffffff';
  ctx.globalAlpha = alpha * 0.5;
  ctx.beginPath(); ctx.ellipse(4, -9, 4, 3, 0.4, 0, Math.PI * 2); ctx.fill();
  ctx.globalAlpha = alpha;

  // Wide droopy cap with downturned brim (flatter than orange)
  // Underside
  ctx.fillStyle = p.capSh;
  ctx.beginPath(); ctx.ellipse(0, -13, 19, 4.5, 0, 0, Math.PI * 2); ctx.fill();
  // Brim (wider than dome)
  ctx.fillStyle = p.cap;
  ctx.beginPath(); ctx.ellipse(0, -14, 18, 4, 0, 0, Math.PI * 2); ctx.fill();
  // Dome (low and flat)
  ctx.fillStyle = p.capMid;
  ctx.beginPath();
  ctx.ellipse(0, -18, 14, 7, 0, 0, Math.PI);
  ctx.fill();
  ctx.fillRect(-14, -18, 28, 6);
  // Dome highlight
  ctx.fillStyle = p.capHi;
  ctx.beginPath(); ctx.ellipse(-4, -21, 5, 3, -0.3, 0, Math.PI * 2); ctx.fill();
  // Dome edge shadows
  ctx.fillStyle = p.capSh;
  ctx.beginPath(); ctx.ellipse(-11, -17, 3, 5, -0.2, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.ellipse( 11, -17, 3, 5,  0.2, 0, Math.PI * 2); ctx.fill();

  // Lightning bolt price tag on cap right-brim
  ctx.fillStyle = '#ffdd00';
  ctx.fillRect(8, -19, 9, 12);
  ctx.fillStyle = '#ffaa00';
  ctx.fillRect(8, -19, 9, 2);
  ctx.fillRect(8, -8, 9, 1);
  // Bolt
  ctx.fillStyle = '#884400';
  ctx.fillRect(11, -17, 3, 4);
  ctx.fillRect(10, -13, 4, 2);
  ctx.fillRect(12, -11, 3, 3);

  // Face
  ctx.fillStyle = p.eye;
  ctx.beginPath(); ctx.ellipse(-5, -5, 2, 2, 0, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.ellipse( 4, -5, 2, 2, 0, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(-6, -6, 1, 1);
  ctx.fillRect( 3, -6, 1, 1);
  // Droopy eyes (sad expression)
  ctx.fillStyle = p.mouth;
  ctx.fillRect(-4, -1, 8, 2);
  ctx.fillRect(-5, -2, 2, 2);
  ctx.fillRect( 3, -2, 2, 2);
}

function _drawStandardMushroom(ctx, p, frame, alpha) {
  // ── FEET — small round stubs ──
  ctx.fillStyle = p.stalkSh;
  if (frame === 0) {
    // right foot forward
    ctx.beginPath(); ctx.ellipse(-6, 7, 5, 3, 0, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.ellipse( 6, 5, 5, 4, 0, 0, Math.PI * 2); ctx.fill();
  } else {
    ctx.beginPath(); ctx.ellipse(-6, 5, 5, 4, 0, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.ellipse( 6, 7, 5, 3, 0, 0, Math.PI * 2); ctx.fill();
  }
  ctx.fillStyle = p.stalk;
  if (frame === 0) {
    ctx.beginPath(); ctx.ellipse(-6, 6, 4, 2.5, 0, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.ellipse( 6, 4, 4, 3,   0, 0, Math.PI * 2); ctx.fill();
  } else {
    ctx.beginPath(); ctx.ellipse(-6, 4, 4, 3,   0, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.ellipse( 6, 6, 4, 2.5, 0, 0, Math.PI * 2); ctx.fill();
  }

  // ── BODY — wide squat oval ──
  ctx.fillStyle = p.stalkSh;
  ctx.beginPath();
  ctx.ellipse(0, -2, 14, 12, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = p.stalk;
  ctx.beginPath();
  ctx.ellipse(0, -3, 13, 11, 0, 0, Math.PI * 2);
  ctx.fill();
  // Body highlight top-right
  ctx.fillStyle = '#ffffff';
  ctx.globalAlpha = alpha * 0.45;
  ctx.beginPath();
  ctx.ellipse(4, -8, 4, 3, 0.4, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalAlpha = alpha;

  // ── CAP — wide dome sitting on top of oval body ──
  // Underside shadow strip
  ctx.fillStyle = p.capSh;
  ctx.beginPath();
  ctx.ellipse(0, -13, 17, 4, 0, 0, Math.PI * 2);
  ctx.fill();
  // Brim droop (flat ellipse wider than dome)
  ctx.fillStyle = p.cap;
  ctx.beginPath();
  ctx.ellipse(0, -14, 16, 3.5, 0, 0, Math.PI * 2);
  ctx.fill();
  // Main dome
  ctx.fillStyle = p.capMid;
  ctx.beginPath();
  ctx.ellipse(0, -19, 13, 9, 0, 0, Math.PI);
  ctx.fill();
  ctx.fillRect(-13, -19, 26, 7);
  // Dome highlight
  ctx.fillStyle = p.capHi;
  ctx.beginPath();
  ctx.ellipse(-4, -22, 5, 3, -0.3, 0, Math.PI * 2);
  ctx.fill();
  // Dome shadow edges
  ctx.fillStyle = p.capSh;
  ctx.beginPath();
  ctx.ellipse(-10, -18, 3, 5, -0.2, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.ellipse(10, -18, 3, 5, 0.2, 0, Math.PI * 2);
  ctx.fill();

  // ── POLKA DOTS ──
  const dots = [[-5, -24], [2, -27], [7, -20]];
  dots.forEach(([dx, dy]) => {
    ctx.fillStyle = p.dot;
    ctx.beginPath(); ctx.ellipse(dx, dy, 2.5, 2.5, 0, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = p.dotSh;
    ctx.beginPath(); ctx.ellipse(dx + 0.5, dy + 1, 1.5, 1, 0, 0, Math.PI * 2); ctx.fill();
  });

  // ── FACE ──
  ctx.fillStyle = p.eye;
  ctx.beginPath(); ctx.ellipse(-5, -6, 2, 2, 0, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.ellipse( 4, -6, 2, 2, 0, 0, Math.PI * 2); ctx.fill();
  // Eye shine
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(-6, -7, 1, 1);
  ctx.fillRect( 3, -7, 1, 1);
  // Frown
  ctx.fillStyle = p.mouth;
  ctx.fillRect(-4, -2, 8, 2);
  ctx.fillRect(-5, -3, 2, 2);
  ctx.fillRect( 3, -3, 2, 2);
}

// ── SWORD CURSOR ──────────────────────────────────────────────────────────
// MapleStory starter sword style (Wooden Sword / Iron Sword aesthetic)
export function drawSword(ctx, x, y, swingAngle = 0) {
  ctx.save();
  ctx.translate(Math.round(x), Math.round(y));
  ctx.rotate(swingAngle - Math.PI / 4);
  ctx.imageSmoothingEnabled = false;

  // Blade (light steel color)
  ctx.fillStyle = '#d8e8f8';
  ctx.fillRect(-2, -24, 4, 22);
  // Blade left edge (darker)
  ctx.fillStyle = '#8ab0d0';
  ctx.fillRect(-2, -24, 1, 22);
  // Blade center shine
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(-1, -22, 1, 18);
  // Blade tip triangle
  ctx.fillStyle = '#d8e8f8';
  ctx.beginPath();
  ctx.moveTo(-2, -24);
  ctx.lineTo(2,  -24);
  ctx.lineTo(0,  -31);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(-1, -29, 1, 3);

  // Cross-guard (gold, MapleStory swords have a prominent guard)
  ctx.fillStyle = '#d4900a';
  ctx.fillRect(-9, -2, 18, 5);
  ctx.fillStyle = '#ffcc30';
  ctx.fillRect(-8, -2, 16, 3);
  ctx.fillStyle = '#a06000';
  ctx.fillRect(-9,  2, 18, 1);
  // Guard end gems
  ctx.fillStyle = '#ff4040';
  ctx.fillRect(-11, -1, 3, 4);
  ctx.fillRect(  8, -1, 3, 4);

  // Handle (wrapped leather look)
  ctx.fillStyle = '#6a3010';
  ctx.fillRect(-3, 3, 6, 13);
  // Wrap lines
  ctx.fillStyle = '#4a1808';
  for (let i = 0; i < 4; i++) {
    ctx.fillRect(-3, 4 + i * 3, 6, 1);
  }
  ctx.fillStyle = '#a05030';
  ctx.fillRect(-2, 3, 1, 13);

  // Pommel (round, gold)
  ctx.fillStyle = '#d4900a';
  ctx.fillRect(-5, 16, 10, 6);
  ctx.fillStyle = '#ffcc30';
  ctx.fillRect(-4, 16, 8, 3);
  ctx.fillStyle = '#a06000';
  ctx.fillRect(-5, 21, 10, 1);

  ctx.restore();
}

// ── ITEM ICONS ────────────────────────────────────────────────────────────
export function drawItemIcon(ctx, x, y, iconType, size = 24) {
  const fns = {
    scroll:     _drawScroll,
    potion_red:  (c, cx, cy, s) => _drawPotion(c, cx, cy, s, '#dd1a1a', '#ff6060', '#ff9090'),
    potion_blue: (c, cx, cy, s) => _drawPotion(c, cx, cy, s, '#1a40dd', '#4488ff', '#88bbff'),
    sword:      _drawSwordIcon,
    map:        _drawMapIcon,
  };
  const fn = fns[iconType];
  if (fn) fn(ctx, x, y, size);
}

function _drawScroll(ctx, x, y, s) {
  ctx.save();
  ctx.translate(x, y);
  const u = s / 24;
  // Parchment
  ctx.fillStyle = '#f0d880';
  ctx.fillRect(-8 * u, -10 * u, 16 * u, 20 * u);
  ctx.fillStyle = '#c8a030';
  ctx.fillRect(-10 * u, -13 * u, 20 * u, 5 * u);
  ctx.fillRect(-10 * u,   8 * u, 20 * u, 5 * u);
  // Roll ends highlight
  ctx.fillStyle = '#ffd050';
  ctx.fillRect(-10 * u, -13 * u, 20 * u, 2 * u);
  ctx.fillRect(-10 * u,   8 * u, 20 * u, 2 * u);
  // Text lines
  ctx.fillStyle = '#8a6010';
  for (let i = 0; i < 4; i++) ctx.fillRect(-5 * u, (-5 + i * 4) * u, 10 * u, 1.5 * u);
  // Blue glow tint (scroll of knowledge)
  ctx.fillStyle = 'rgba(80,180,255,0.18)';
  ctx.fillRect(-10 * u, -13 * u, 20 * u, 26 * u);
  ctx.restore();
}

function _drawPotion(ctx, x, y, s, col, hi, rim) {
  ctx.save();
  ctx.translate(x, y);
  const u = s / 24;
  // Bottle body
  ctx.fillStyle = col;
  ctx.beginPath();
  ctx.ellipse(0, 5 * u, 8 * u, 9 * u, 0, 0, Math.PI * 2);
  ctx.fill();
  // Highlight
  ctx.fillStyle = hi;
  ctx.beginPath();
  ctx.ellipse(-2.5 * u, 1 * u, 3 * u, 5 * u, -0.3, 0, Math.PI * 2);
  ctx.fill();
  // Neck
  ctx.fillStyle = col;
  ctx.fillRect(-3 * u, -9 * u, 6 * u, 12 * u);
  ctx.fillStyle = rim;
  ctx.fillRect(-3 * u, -11 * u, 6 * u, 4 * u);
  // Cork
  ctx.fillStyle = '#d4a050';
  ctx.fillRect(-2 * u, -14 * u, 4 * u, 5 * u);
  ctx.fillStyle = '#f0c070';
  ctx.fillRect(-2 * u, -14 * u, 4 * u, 2 * u);
  // Bubble
  ctx.fillStyle = 'rgba(255,255,255,0.5)';
  ctx.beginPath();
  ctx.ellipse(2 * u, 7 * u, 2 * u, 2 * u, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function _drawSwordIcon(ctx, x, y, s) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(-Math.PI / 4);
  const u = s / 24;
  // Blade
  ctx.fillStyle = '#c8d8e8';
  ctx.fillRect(-2 * u, -14 * u, 4 * u, 16 * u);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(-1 * u, -12 * u, 1 * u, 12 * u);
  // Tip
  ctx.beginPath();
  ctx.moveTo(-2 * u, -14 * u);
  ctx.lineTo( 2 * u, -14 * u);
  ctx.lineTo( 0,     -20 * u);
  ctx.closePath();
  ctx.fillStyle = '#c8d8e8';
  ctx.fill();
  // Guard
  ctx.fillStyle = '#cc8800';
  ctx.fillRect(-7 * u, 2 * u, 14 * u, 3 * u);
  ctx.fillStyle = '#ffcc30';
  ctx.fillRect(-6 * u, 2 * u, 12 * u, 2 * u);
  // Handle
  ctx.fillStyle = '#6a3010';
  ctx.fillRect(-2 * u, 5 * u, 4 * u, 9 * u);
  // Pommel
  ctx.fillStyle = '#cc8800';
  ctx.fillRect(-3 * u, 14 * u, 6 * u, 4 * u);
  ctx.restore();
}

function _drawMapIcon(ctx, x, y, s) {
  ctx.save();
  ctx.translate(x, y);
  const u = s / 24;
  // Parchment base
  ctx.fillStyle = '#e8d060';
  ctx.fillRect(-10 * u, -11 * u, 20 * u, 22 * u);
  // Left fold shadow
  ctx.fillStyle = '#b89830';
  ctx.fillRect(-10 * u, -11 * u, 4 * u, 22 * u);
  // Right fold
  ctx.fillStyle = '#d4b040';
  ctx.fillRect( 6 * u,  -11 * u, 4 * u, 22 * u);
  // Map lines
  ctx.fillStyle = '#6a5010';
  ctx.fillRect(-4 * u,  -8 * u, 10 * u, 1.5 * u);
  ctx.fillRect(-6 * u,  -3 * u, 12 * u, 1.5 * u);
  ctx.fillRect(-2 * u,   2 * u,  8 * u, 1.5 * u);
  ctx.fillRect(-5 * u,   7 * u,  9 * u, 1.5 * u);
  // Red X marker
  ctx.fillStyle = '#cc2200';
  ctx.fillRect( 0,  -1 * u, 4 * u, 4 * u);
  ctx.fillStyle = '#ff4422';
  ctx.fillRect( 0,  -1 * u, 2 * u, 2 * u);
  ctx.restore();
}

// ── HP BAR (canvas helper) ────────────────────────────────────────────────
export function drawHpBar(ctx, x, y, w, h, ratio, color = '#cc2020') {
  // Inset background
  ctx.fillStyle = '#100808';
  ctx.fillRect(x, y, w, h);
  // Fill
  ctx.fillStyle = color;
  ctx.fillRect(x + 1, y + 1, Math.max(0, (w - 2) * ratio), h - 2);
  // Highlight on fill
  ctx.fillStyle = 'rgba(255,255,255,0.25)';
  ctx.fillRect(x + 1, y + 1, Math.max(0, (w - 2) * ratio), 1);
  // Border
  ctx.strokeStyle = '#2a1004';
  ctx.lineWidth = 1;
  ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
}
