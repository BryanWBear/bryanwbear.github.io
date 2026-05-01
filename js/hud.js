// HUD: HP/MP bars (DOM) + minimap (Canvas)

export class Hud {
  constructor() {
    this.hp = 999;
    this.maxHp = 999;
    this.mp = 500;
    this.maxMp = 500;
    this._drainTimer = 0;
    this._regenTimer = 0;
  }

  update(dt) {
    // Cosmetic HP drain over time, regens slowly
    this._drainTimer += dt;
    if (this._drainTimer > 8000) {
      this._drainTimer = 0;
      this.hp = Math.max(100, this.hp - Math.floor(Math.random() * 40));
    }
    this._regenTimer += dt;
    if (this._regenTimer > 3000) {
      this._regenTimer = 0;
      this.hp = Math.min(this.maxHp, this.hp + 8);
      this.mp = Math.min(this.maxMp, this.mp + 5);
    }
    this._updateDOM();
  }

  _updateDOM() {
    const hpFill = document.getElementById('hp-fill');
    const mpFill = document.getElementById('mp-fill');
    const hpVal = document.getElementById('hp-val');
    const mpVal = document.getElementById('mp-val');
    if (hpFill) hpFill.style.width = `${(this.hp / this.maxHp) * 100}%`;
    if (mpFill) mpFill.style.width = `${(this.mp / this.maxMp) * 100}%`;
    if (hpVal) hpVal.textContent = this.hp;
    if (mpVal) mpVal.textContent = this.mp;
  }

  drawMinimap(ctx, mushrooms, canvasW, canvasH) {
    const mmW = 90, mmH = 56;
    const mmX = canvasW - mmW - 8;
    const mmY = 8;

    // Background
    ctx.fillStyle = '#0e0804';
    ctx.fillRect(mmX, mmY, mmW, mmH);
    ctx.strokeStyle = '#7a5910';
    ctx.lineWidth = 2;
    ctx.strokeRect(mmX + 1, mmY + 1, mmW - 2, mmH - 2);

    // Label
    ctx.fillStyle = '#c8981e';
    ctx.font = '5px "Press Start 2P", monospace';
    ctx.fillText('MAP', mmX + 4, mmY + 9);

    // Ground line
    ctx.fillStyle = '#336622';
    ctx.fillRect(mmX + 2, mmY + mmH - 12, mmW - 4, 4);

    // Mushroom dots
    for (const m of mushrooms) {
      if (m.isDead()) continue;
      const mx = mmX + 2 + (m.x / canvasW) * (mmW - 4);
      const my = mmY + mmH - 14;
      const dotColor = m.type === 'orange' ? '#ff6600' : m.type === 'green' ? '#44cc44' : '#4466ff';
      ctx.fillStyle = dotColor;
      ctx.fillRect(Math.round(mx) - 1, Math.round(my) - 1, 3, 3);
    }

    // Player dot (center-ish)
    ctx.fillStyle = '#ffff44';
    const px = mmX + mmW / 2;
    const py = mmY + mmH - 14;
    ctx.fillRect(Math.round(px) - 2, Math.round(py) - 2, 4, 4);
  }
}
