import { drawItemIcon } from './sprites.js';
import { getArticle, COIN_AMOUNTS } from './articles.js';
import { drawItemSprite, isItemSheetLoaded } from './itemsheet.js';
import { drawCoinSprite, isCoinSheetLoaded, COIN_TIERS } from './coinsheet.js';

const GRAVITY = 0.55;

const GLOW_COLORS = {
  scroll:      '#44aaff',
  potion_red:  '#ff4444',
  potion_blue: '#4488ff',
  sword:       '#ffcc44',
  map:         '#44cc88',
  bronze:      '#cc8822',
  gold:        '#ffcc00',
  bill:        '#88cc44',
  meso:        '#ffaa00',
};

let _nextItemId = 1;

// ── Shared physics base ───────────────────────────────────────────────────────
class DroppedBase {
  constructor(x, groundY) {
    this.id        = _nextItemId++;
    this.x         = x + (Math.random() - 0.5) * 30;
    this.y         = groundY - 20;
    this.vx        = (Math.random() - 0.5) * 3;
    this.vy        = -(4 + Math.random() * 3);
    this.groundY   = groundY;
    this.landed    = false;
    this.pickedUp  = false;
    this.bobTimer  = 0;
    this.glowTimer = 0;
    this.age       = 0;
  }

  update(dt) {
    if (this.pickedUp) return;
    this.age       += dt;
    this.glowTimer += dt * 0.003;
    if (!this.landed) {
      this.vy += GRAVITY;
      this.y  += this.vy;
      this.x  += this.vx;
      this.vx *= 0.92;
      if (this.y >= this.groundY - 12) {
        this.y      = this.groundY - 12;
        this.landed = true;
        this.vy     = 0;
        this.vx     = 0;
      }
    } else {
      this.bobTimer += dt * 0.002;
    }
  }

  _drawGlow(ctx, color) {
    const bobY  = this.landed ? Math.sin(this.bobTimer * Math.PI * 2) * 2.5 : 0;
    const drawY = this.y + bobY;
    const r     = 14 + Math.sin(this.glowTimer * Math.PI * 2) * 3;
    const grd   = ctx.createRadialGradient(this.x, drawY, 2, this.x, drawY, r);
    grd.addColorStop(0, color + '88');
    grd.addColorStop(1, color + '00');
    ctx.fillStyle = grd;
    ctx.beginPath();
    ctx.ellipse(this.x, drawY, r, r * 0.5, 0, 0, Math.PI * 2);
    ctx.fill();
    return drawY;
  }

  _drawSparkles(ctx, drawY, color) {
    if (!this.landed) return;
    for (let i = 0; i < 3; i++) {
      const angle = this.glowTimer * Math.PI * 2 + (i / 3) * Math.PI * 2;
      const sr    = 10 + Math.sin(this.glowTimer * Math.PI * 4 + i) * 4;
      const sx    = this.x + Math.cos(angle) * sr;
      const sy    = drawY  + Math.sin(angle) * sr * 0.4;
      ctx.fillStyle  = color;
      ctx.globalAlpha = 0.6 + Math.sin(this.glowTimer * Math.PI * 6 + i) * 0.4;
      ctx.fillRect(sx - 1, sy - 1, 2, 2);
      ctx.globalAlpha = 1;
    }
  }

  isNear(px, py) {
    const dx = px - this.x;
    const dy = py - (this.y - 4);
    return Math.sqrt(dx * dx + dy * dy) < 28;
  }
}

// ── Article (blog) drop ───────────────────────────────────────────────────────
export class DroppedItem extends DroppedBase {
  constructor(x, groundY, articleId) {
    super(x, groundY);
    this.articleId = articleId;
    this.article   = getArticle(articleId);
    this.icon      = this.article ? this.article.icon : 'scroll';
    this.sprite    = this.article ? this.article.sprite : null;
  }

  draw(ctx) {
    if (this.pickedUp) return;
    const glowColor = GLOW_COLORS[this.icon] || '#ffffff';
    const drawY = this._drawGlow(ctx, glowColor);

    // Prefer sprite sheet; fall back to canvas-drawn icon
    if (this.sprite && isItemSheetLoaded()) {
      drawItemSprite(ctx, this.x, drawY - 4, this.sprite.col, this.sprite.row, 28);
    } else {
      drawItemIcon(ctx, this.x, drawY - 4, this.icon, 22);
    }

    this._drawSparkles(ctx, drawY, glowColor);
  }
}

// ── Coin drop ─────────────────────────────────────────────────────────────────
export class DroppedCoin extends DroppedBase {
  constructor(x, groundY, tier) {
    super(x, groundY);
    this.tier      = tier;
    const range    = COIN_AMOUNTS[tier] || [10, 50];
    this.amount    = Math.floor(range[0] + Math.random() * (range[1] - range[0]));
    this.animFrame = 0;
    this.animTimer = 0;
    this.frames    = (COIN_TIERS[tier] || {}).frames || 4;
  }

  update(dt) {
    super.update(dt);
    this.animTimer += dt;
    if (this.animTimer > 120) { this.animFrame = (this.animFrame + 1) % this.frames; this.animTimer = 0; }
  }

  draw(ctx) {
    if (this.pickedUp) return;
    const glowColor = GLOW_COLORS[this.tier] || '#ffcc00';
    const drawY = this._drawGlow(ctx, glowColor);

    if (isCoinSheetLoaded()) {
      drawCoinSprite(ctx, this.x, drawY - 2, this.tier, this.animFrame, 26);
    } else {
      // Fallback: simple pixel coin
      ctx.fillStyle = glowColor;
      ctx.beginPath();
      ctx.ellipse(this.x, drawY - 2, 10, 7, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#ffffff';
      ctx.globalAlpha = 0.4;
      ctx.beginPath();
      ctx.ellipse(this.x - 3, drawY - 5, 3, 2, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    this._drawSparkles(ctx, drawY, glowColor);
  }
}

// ── Spawn helpers ─────────────────────────────────────────────────────────────

export function spawnDrops(x, groundY, { articles = [], coins = [] }) {
  const drops = [];
  for (const id of articles)  drops.push(new DroppedItem(x, groundY, id));
  for (const tier of coins)   drops.push(new DroppedCoin(x, groundY, tier));
  return drops;
}

export function showPickupFlash(x, y, name) {
  const el = document.createElement('div');
  el.className  = 'pickup-flash';
  el.textContent = `+ ${name}`;
  el.style.left = `${x - 30}px`;
  el.style.top  = `${y - 20}px`;
  document.body.appendChild(el);
  el.addEventListener('animationend', () => el.remove());
}
