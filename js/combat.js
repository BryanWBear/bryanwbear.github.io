// Attack detection and floating damage numbers

const ATTACK_RADIUS = 48;

export function attackAt(x, y, mushrooms) {
  const hits = [];
  for (const m of mushrooms) {
    if (m.isDead() || m.isDying()) continue;
    const dx = m.x - x;
    const dy = (m.y - 18) - y; // center of mushroom body
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist <= ATTACK_RADIUS) {
      const isCrit = Math.random() < 0.15;
      const base = 100 + Math.floor(Math.random() * 101);
      const dmg = isCrit ? Math.floor(base * 1.8) : base;
      const hit = m.takeDamage(dmg);
      if (hit) hits.push({ mushroom: m, damage: dmg, isCrit, x: m.x, y: m.y - 30 });
    }
  }
  return hits;
}

// ── FLOATING DAMAGE NUMBERS (DOM-based) ──────────────────────────────────
export function spawnDamageNumber(dmg, x, y, isCrit) {
  const el = document.createElement('div');
  el.className = 'dmg-num' + (isCrit ? ' crit' : '');
  el.textContent = isCrit ? `${dmg}!!` : String(dmg);
  el.style.left = `${x - 15}px`;
  el.style.top = `${y}px`;
  document.body.appendChild(el);
  el.addEventListener('animationend', () => el.remove());
}
