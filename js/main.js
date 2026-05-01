import { drawBackground, loadBackground, getGroundY } from './sprites.js';
import { loadSheet } from './spritesheet.js';
import { Cursor, loadSword } from './cursor.js';
import { attackAt, spawnDamageNumber } from './combat.js';
import { spawnDrops, showPickupFlash, DroppedItem, DroppedCoin } from './items.js';
import { Inventory, setupModal } from './inventory.js';
import { Hud } from './hud.js';
import { spawnMushroom } from './mushroom.js';
import { getPlatforms, detectPlatforms, SPAWN_PLATFORMS } from './platforms.js';
import { rollDrops } from './articles.js';
import { loadItemSheet } from './itemsheet.js';
import { loadCoinSheet } from './coinsheet.js';

const canvas = document.getElementById('gameCanvas');
const ctx    = canvas.getContext('2d');
let platforms = [];

function resize() {
  canvas.width  = window.innerWidth;
  canvas.height = window.innerHeight;
  platforms = getPlatforms(canvas.width, canvas.height);
}
window.addEventListener('resize', () => {
  resize();
  mushrooms.forEach(m => {
    if (!m.isDead()) {
      platforms = getPlatforms(canvas.width, canvas.height);
      const nearest = platforms.reduce((best, p) => {
        const d = Math.abs(m.y - p.y);
        return d < Math.abs(m.y - best.y) ? p : best;
      });
      m.platform = nearest;
      m.y = nearest.y;
    }
  });
});
resize();

// ── SYSTEMS ───────────────────────────────────────────────────────────────
const cursor    = new Cursor(canvas);
const inventory = new Inventory();
const hud       = new Hud();
setupModal();

// ── GAME STATE ────────────────────────────────────────────────────────────
const mushrooms    = [];
const droppedItems = [];
const MAX_MUSHROOMS = 20;
const RESPAWN_DELAY = 3500;
const respawnQueue  = [];

function getSpawnPlatforms() {
  return SPAWN_PLATFORMS.map(p => ({
    x: p.xFrac * canvas.width,
    y: p.yFrac * canvas.height,
    w: p.wFrac * canvas.width,
  }));
}

function spawnInitial() {
  const sp = getSpawnPlatforms();
  for (let i = 0; i < MAX_MUSHROOMS; i++) {
    mushrooms.push(spawnMushroom(canvas.width, canvas.height, sp));
  }
}

function manageMushrooms(now) {
  for (let i = mushrooms.length - 1; i >= 0; i--) {
    if (mushrooms[i].isDead()) {
      mushrooms.splice(i, 1);
      respawnQueue.push(now + RESPAWN_DELAY);
    }
  }
  while (respawnQueue.length > 0 && respawnQueue[0] <= now) {
    respawnQueue.shift();
    if (mushrooms.length < MAX_MUSHROOMS)
      mushrooms.push(spawnMushroom(canvas.width, canvas.height, getSpawnPlatforms()));
  }
  while (mushrooms.length + respawnQueue.length < MAX_MUSHROOMS)
    respawnQueue.push(now + RESPAWN_DELAY);
}

// ── INPUT ─────────────────────────────────────────────────────────────────
canvas.addEventListener('mousedown', e => {
  if (e.button !== 0) return;
  cursor.swing();

  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

  // Attack mushrooms
  const hits = attackAt(mx, my, mushrooms);
  hits.forEach(hit => {
    spawnDamageNumber(hit.damage, hit.x, hit.y, hit.isCrit);
    if (hit.mushroom.isDying()) {
      const drops   = rollDrops(hit.mushroom.type);
      const groundY = hit.mushroom.y;
      droppedItems.push(...spawnDrops(hit.mushroom.x, groundY, drops));
    }
  });

  // Pick up items
  for (let i = droppedItems.length - 1; i >= 0; i--) {
    const item = droppedItems[i];
    if (item.pickedUp || !item.isNear(mx, my)) continue;
    item.pickedUp = true;

    if (item instanceof DroppedCoin) {
      inventory.addMesos(item.amount);
      showPickupFlash(e.clientX, e.clientY, `${item.amount.toLocaleString()} Mesos`);
    } else if (item instanceof DroppedItem) {
      const added = inventory.addItem(item.articleId);
      if (added && item.article) showPickupFlash(e.clientX, e.clientY, item.article.title);
    }

    droppedItems.splice(i, 1);
  }
});

// ── GAME LOOP ─────────────────────────────────────────────────────────────
let lastTime = 0;

function loop(timestamp) {
  const dt = Math.min(timestamp - lastTime, 50);
  lastTime  = timestamp;

  const W = canvas.width;
  const H = canvas.height;

  manageMushrooms(timestamp);
  cursor.update(dt);
  mushrooms.forEach(m => m.update(dt, W, platforms));
  droppedItems.forEach(item => item.update(dt));
  hud.update(dt);

  ctx.clearRect(0, 0, W, H);
  drawBackground(ctx, W, H);
  droppedItems.forEach(item => item.draw(ctx));
  mushrooms.forEach(m => m.draw(ctx));

  cursor.draw(ctx);


  requestAnimationFrame(loop);
}

// ── BOOT ──────────────────────────────────────────────────────────────────
Promise.all([
  loadBackground().catch(() => null),
  loadSheet().catch(() => {}),
  loadSword().catch(() => {}),
  loadItemSheet().catch(() => {}),
  loadCoinSheet().catch(() => {}),
]).then(([bgImg]) => {
  if (bgImg) detectPlatforms(bgImg, canvas.width, canvas.height);
  platforms = getPlatforms(canvas.width, canvas.height);
  spawnInitial();
  requestAnimationFrame(ts => { lastTime = ts; loop(ts); });
}).catch(err => {
  console.error('BOOT ERROR:', err);
  document.body.insertAdjacentHTML('beforeend',
    `<div style="position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:#200;color:#f88;font-family:monospace;padding:20px;z-index:9999;border:2px solid #f44">BOOT ERROR: ${err.message}</div>`);
});
