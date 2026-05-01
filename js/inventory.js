import { drawItemIcon } from './sprites.js';
import { getArticle } from './articles.js';
import { drawItemSprite, isItemSheetLoaded } from './itemsheet.js';

const MAX_SLOTS = 32;

export class Inventory {
  constructor() {
    this.items  = [{ articleId: 'about-me', slot: 0 }];
    this.mesos  = 0;
    this.activeTab = 'all';
    this._buildGrid();
    this._bindTabs();
    this._render();
  }

  _buildGrid() {
    const grid = document.getElementById('inv-grid');
    grid.innerHTML = '';
    for (let i = 0; i < MAX_SLOTS; i++) {
      const slot = document.createElement('div');
      slot.className = 'inv-slot';
      slot.dataset.slot = i;
      grid.appendChild(slot);
    }
  }

  _bindTabs() {
    document.querySelectorAll('.inv-tab').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.inv-tab').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.activeTab = btn.dataset.cat;
        this._render();
      });
    });
  }

  addItem(articleId) {
    if (this.items.length >= MAX_SLOTS) return false;
    if (this.items.some(i => i.articleId === articleId)) return false;
    const usedSlots = new Set(this.items.map(i => i.slot));
    let slot = 0;
    while (usedSlots.has(slot)) slot++;
    this.items.push({ articleId, slot });
    this._render();
    this._updateFooter();
    return true;
  }

  addMesos(amount) {
    this.mesos += amount;
    this._updateFooter();
  }

  _render() {
    const grid = document.getElementById('inv-grid');
    grid.querySelectorAll('.inv-slot').forEach(el => {
      el.className = 'inv-slot';
      el.innerHTML = '';
      el.onclick = null;
    });

    const filtered = this.activeTab === 'all'
      ? this.items
      : this.items.filter(item => {
          const art = getArticle(item.articleId);
          return art && art.category === this.activeTab;
        });

    filtered.forEach((item, displayIdx) => {
      const art = getArticle(item.articleId);
      if (!art) return;
      const slotEl = grid.children[displayIdx];
      if (!slotEl) return;

      slotEl.classList.add('occupied');

      const iconCanvas = document.createElement('canvas');
      iconCanvas.width  = 32;
      iconCanvas.height = 32;
      const ictx = iconCanvas.getContext('2d');

      // Use sprite sheet if loaded, otherwise canvas fallback
      if (art.sprite && isItemSheetLoaded()) {
        drawItemSprite(ictx, 16, 16, art.sprite.col, art.sprite.row, 28);
      } else {
        drawItemIcon(ictx, 16, 16, art.icon, 28);
      }

      slotEl.appendChild(iconCanvas);

      const tip = document.createElement('div');
      tip.className   = 'slot-tooltip';
      tip.textContent = art.title;
      slotEl.appendChild(tip);

      slotEl.onclick = () => openModal(art);
    });
  }

  _updateFooter() {
    const mesos = document.getElementById('inv-mesos');
    if (mesos) mesos.textContent = this.mesos.toLocaleString();
  }
}

// ── MODAL ─────────────────────────────────────────────────────────────────
const CAT_LABELS = { tech: 'TECH', personal: 'LIFE', projects: 'PROJECTS', about: 'ABOUT' };

export function openModal(article) {
  const overlay   = document.getElementById('modal-overlay');
  const nameEl    = document.getElementById('modal-item-name');
  const catEl     = document.getElementById('modal-cat-badge');
  const contentEl = document.getElementById('modal-content');
  const iconCanvas = document.getElementById('modal-icon-canvas');

  nameEl.textContent = article.title;
  catEl.textContent  = `CATEGORY: ${CAT_LABELS[article.category] || article.category.toUpperCase()}`;

  const ictx = iconCanvas.getContext('2d');
  ictx.clearRect(0, 0, 32, 32);
  if (article.sprite && isItemSheetLoaded()) {
    drawItemSprite(ictx, 16, 16, article.sprite.col, article.sprite.row, 28);
  } else {
    drawItemIcon(ictx, 16, 16, article.icon, 28);
  }

  if (window.marked) {
    // Protect math blocks from marked's HTML escaping
    const mathBlocks = [];
    let src = article.content.trim();
    // Display math $$...$$
    src = src.replace(/\$\$([\s\S]+?)\$\$/g, (_, tex) => {
      mathBlocks.push({ display: true, tex });
      return `MATHPLACEHOLDER${mathBlocks.length - 1}ENDMATH`;
    });
    // Inline math $...$
    src = src.replace(/\$([^\n$]+?)\$/g, (_, tex) => {
      mathBlocks.push({ display: false, tex });
      return `MATHPLACEHOLDER${mathBlocks.length - 1}ENDMATH`;
    });
    let html = window.marked.parse(src);
    // Restore math blocks
    html = html.replace(/MATHPLACEHOLDER(\d+)ENDMATH/g, (_, i) => {
      const { display, tex } = mathBlocks[+i];
      return display ? `$$${tex}$$` : `$${tex}$`;
    });
    contentEl.innerHTML = html;
  } else {
    contentEl.textContent = article.content;
  }

  if (window.renderMathInElement) {
    renderMathInElement(contentEl, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '$',  right: '$',  display: false },
      ],
      throwOnError: false,
    });
  }

  overlay.classList.remove('hidden');
}

export function setupModal() {
  const overlay  = document.getElementById('modal-overlay');
  const closeBtn = document.getElementById('modal-close-btn');
  const invClose = document.getElementById('inv-close');

  closeBtn.addEventListener('click', () => overlay.classList.add('hidden'));
  overlay.addEventListener('click', e => {
    if (e.target === overlay) overlay.classList.add('hidden');
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') overlay.classList.add('hidden');
  });
  if (invClose) invClose.addEventListener('click', () => {});
}
