// Custom sword cursor using assets/sword.png

let _swordImg = null;
let _swordLoaded = false;

export function loadSword() {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => { _swordImg = img; _swordLoaded = true; resolve(); };
    img.onerror = () => { resolve(); }; // fallback to canvas sword silently
    img.src = 'assets/sword.png';
  });
}

export class Cursor {
  constructor(canvas) {
    this.x = -200;
    this.y = -200;
    this.swingAngle = 0;
    this.swinging = false;
    this.swingTimer = 0;
    this.swingDuration = 220;

    canvas.addEventListener('mousemove', e => {
      const r = canvas.getBoundingClientRect();
      this.x = e.clientX - r.left;
      this.y = e.clientY - r.top;
    });

    canvas.addEventListener('mouseleave', () => {
      this.x = -200;
      this.y = -200;
    });
  }

  swing() {
    this.swinging = true;
    this.swingTimer = 0;
  }

  update(dt) {
    if (this.swinging) {
      this.swingTimer += dt;
      const t = Math.min(1, this.swingTimer / this.swingDuration);
      // Swing from -45° rest down to 0° (horizontal), then back
      this.swingAngle = Math.sin(t * Math.PI) * (Math.PI / 4);
      if (t >= 1) { this.swinging = false; this.swingAngle = 0; }
    }
  }

  draw(ctx) {
    const SIZE = 52; // display size in pixels

    ctx.save();
    ctx.translate(Math.round(this.x), Math.round(this.y));
    // Rotate so sword tip points upper-right at rest; swing arc on click
    ctx.rotate(this.swingAngle);

    if (_swordLoaded && _swordImg) {
      // Draw image centered on the handle (lower-left of image = grip point)
      ctx.drawImage(_swordImg, -SIZE * 0.25, -SIZE * 0.75, SIZE, SIZE);
    } else {
      // Fallback: simple canvas sword
      _drawFallbackSword(ctx);
    }

    ctx.restore();
  }
}

function _drawFallbackSword(ctx) {
  ctx.fillStyle = '#d8e8f8';
  ctx.fillRect(-2, -26, 4, 22);
  ctx.fillStyle = '#ffffff';
  ctx.beginPath(); ctx.moveTo(-2,-26); ctx.lineTo(2,-26); ctx.lineTo(0,-33); ctx.fill();
  ctx.fillStyle = '#cc8800';
  ctx.fillRect(-8, -4, 16, 4);
  ctx.fillStyle = '#6a3010';
  ctx.fillRect(-3, 0, 6, 12);
  ctx.fillStyle = '#cc8800';
  ctx.fillRect(-4, 12, 8, 5);
}
