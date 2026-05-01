import { drawMushroom, drawHpBar } from './sprites.js';
import { drawFrame } from './spritesheet.js';

const TYPES = ['orange', 'green', 'blue', 'dark'];
const HP    = { orange: 300, green: 400, blue: 500, dark: 600 };
const SPEED = { orange: 1.2, green: 0.9, blue: 1.5, dark: 0.7 };

const STATE = { WALK: 'walk', HIT: 'hit', DYING: 'dying', DEAD: 'dead' };
const GRAVITY = 0.55;
let _nextId = 1;

// Pick a platform weighted by width so wider platforms get proportionally more mushrooms
function _pickPlatform(platforms) {
  const totalW = platforms.reduce((s, p) => s + p.w, 0);
  let r = Math.random() * totalW;
  for (const p of platforms) {
    r -= p.w;
    if (r <= 0) return p;
  }
  return platforms[platforms.length - 1];
}

export class Mushroom {
  constructor(canvasW, canvasH, platforms) {
    this.id    = _nextId++;
    this.type  = TYPES[Math.floor(Math.random() * TYPES.length)];
    this.maxHp = HP[this.type];
    this.hp    = this.maxHp;
    this.speed = SPEED[this.type] * (0.8 + Math.random() * 0.5);

    // Pick platform weighted by width
    const plat = _pickPlatform(platforms);
    this.platform = plat;
    // Spawn somewhere along the platform, with a little inset so they don't start at the edge
    this.x  = plat.x + 20 + Math.random() * Math.max(0, plat.w - 40);
    this.y  = plat.y;
    this.vy = 0;
    this.vx = (Math.random() < 0.5 ? 1 : -1) * this.speed;
    this.facing = this.vx > 0 ? 1 : -1;

    this.frame      = 0;
    this.frameTimer = 0;
    this.state      = STATE.WALK;
    this.stateTimer = 0;
    this.hitFlash   = 0;
    this.deathScale = 1;
    this.deathAngle = 0;
    this.showHp     = false;
    this.hpTimer    = 0;
  }

  takeDamage(amount) {
    if (this.state === STATE.DYING || this.state === STATE.DEAD) return false;
    this.hp = Math.max(0, this.hp - amount);
    this.state      = STATE.HIT;
    this.stateTimer = 120;
    this.hitFlash   = 1;
    this.showHp     = true;
    this.hpTimer    = 2000;
    if (this.hp <= 0) {
      this.state      = STATE.DYING;
      this.stateTimer = 600;
      this.vx         = 0;
    }
    return true;
  }

  isDead()  { return this.state === STATE.DEAD; }
  isDying() { return this.state === STATE.DYING; }

  update(dt, canvasW, _platforms) {
    if (this.state === STATE.DYING) {
      this.stateTimer -= dt;
      this.deathAngle += 0.15;
      this.deathScale  = Math.max(0, this.stateTimer / 600);
      if (this.stateTimer <= 0) this.state = STATE.DEAD;
      return;
    }

    if (this.state === STATE.HIT) {
      this.stateTimer -= dt;
      this.hitFlash    = this.stateTimer / 120;
      if (this.stateTimer <= 0) { this.state = STATE.WALK; this.hitFlash = 0; }
    }

    if (this.hpTimer > 0) {
      this.hpTimer -= dt;
      if (this.hpTimer <= 0) this.showHp = false;
    }

    const p = this.platform;

    // Horizontal movement — bounce at assigned platform edges
    this.x += this.vx;
    const left  = p.x + 10;
    const right = p.x + p.w - 10;
    if (this.x < left)  { this.x = left;  this.vx =  Math.abs(this.vx); this.facing =  1; }
    if (this.x > right) { this.x = right; this.vx = -Math.abs(this.vx); this.facing = -1; }
    if (this.vx > 0) this.facing = 1;
    else if (this.vx < 0) this.facing = -1;

    // Gravity — snap back to assigned platform surface when falling
    if (this.y < p.y) {
      this.vy += GRAVITY;
      this.y  += this.vy;
      if (this.y >= p.y) {
        this.y  = p.y;
        this.vy = 0;
      }
    } else {
      // Already on platform surface
      this.y  = p.y;
      this.vy = 0;
    }

    // Walk animation (3 frames)
    this.frameTimer += dt;
    if (this.frameTimer > 180) {
      this.frame = (this.frame + 1) % 3;
      this.frameTimer = 0;
    }
  }

  draw(ctx) {
    if (this.state === STATE.DEAD) return;

    const dying = this.state === STATE.DYING;
    const alpha = dying ? this.deathScale : 1;
    const scale = dying ? this.deathScale : 1;

    let frameName;
    if (dying)                         frameName = 'dead';
    else if (this.state === STATE.HIT) frameName = 'hit';
    else                               frameName = ['walk0','walk1','walk2'][this.frame];

    ctx.save();
    if (dying) {
      ctx.translate(this.x, this.y);
      ctx.rotate(this.deathAngle);
      ctx.translate(-this.x, -this.y);
    }

    const drawn = drawFrame(ctx, this.x, this.y, frameName, this.facing, scale, alpha);
    if (!drawn) drawMushroom(ctx, this.x, this.y, this.type, this.frame % 2, this.facing, alpha, scale, scale);

    if (this.hitFlash > 0) {
      ctx.save();
      ctx.globalAlpha = this.hitFlash * 0.5;
      ctx.fillStyle   = '#ff2200';
      ctx.fillRect(this.x - 30 * scale, this.y - 55 * scale, 60 * scale, 60 * scale);
      ctx.restore();
    }

    ctx.restore();

    if (this.showHp && !dying) {
      const bw = 48;
      drawHpBar(ctx, this.x - bw / 2, this.y - 62, bw, 6, this.hp / this.maxHp);
    }
  }
}

export function spawnMushroom(canvasW, canvasH, platforms) {
  return new Mushroom(canvasW, canvasH, platforms);
}
