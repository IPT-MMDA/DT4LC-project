// Layout audit for the responsive group (flows/responsive.e2e.mjs): horizontal overflow
// and overlapping text, measured in the page.
//
// Overflow: the document is no wider than the viewport; no visible element reaches past
// either edge unless it sits in a named clipper (`clippers`, e.g. the map's canvas) or is
// a truncated line (text-overflow: ellipsis); every horizontal scroller is named
// (`scrollers`).
//
// Overlap: every visible element that carries text (headings, paragraphs, links, buttons,
// labels, table cells, list items, and any element with its own text) is measured by the
// line boxes of its text, clipped by its overflow ancestors; a button or a block link by
// its whole box, which hides what it covers. No two of them, neither the
// other's ancestor, may intersect by more than 1 px in both directions, except inside the
// named overlays (`overlays`, e.g. a label drawn on the map). Each vertical scroller is
// stepped through its height, so text below its fold is checked too.

/** Runs in the page. Returns { scrollWidth, innerWidth, wide[], scrollers[], overlaps[] }. */
function audit({ scrollers: allowedScrollers, clippers, overlays, scope }) {
  const vw = window.innerWidth;
  const describe = (el) => {
    const id = el.getAttribute('data-testid');
    const label = el.getAttribute('aria-label');
    const text = (el.innerText || el.textContent || '').trim().replace(/\s+/g, ' ').slice(0, 40);
    return `${el.tagName.toLowerCase()}${id ? `[data-testid=${id}]` : ''}${label ? `[aria-label="${label}"]` : ''}${text ? ` "${text}"` : ''}`;
  };
  const inAny = (el, selectors) => selectors.some((s) => el.closest(s));
  const clips = (el) => {
    const cs = getComputedStyle(el);
    return cs.overflowX !== 'visible' || cs.overflowY !== 'visible';
  };
  const all = [...document.body.querySelectorAll('*')];

  const scrollers = all
    .filter((el) => {
      const ox = getComputedStyle(el).overflowX;
      return (ox === 'auto' || ox === 'scroll') && el.scrollWidth > el.clientWidth + 1;
    })
    .map((el) => ({ el: describe(el), named: allowedScrollers.some((s) => el.matches(s)) }));

  const wide = [];
  for (const el of all) {
    const r = el.getBoundingClientRect();
    if (r.width <= 1 || r.height <= 1) continue;
    if (!el.checkVisibility({ visibilityProperty: true })) continue;
    if (r.left >= -1 && r.right <= vw + 1) continue;
    const parent = el.parentElement;
    if (parent && inAny(parent, [...allowedScrollers, ...clippers])) continue;
    // A truncated line clips its own text; the full text stays in the DOM (and its title).
    let truncated = false;
    for (let a = parent; a && a !== document.body; a = a.parentElement) {
      if (getComputedStyle(a).textOverflow === 'ellipsis' && a.getBoundingClientRect().right <= vw + 1) truncated = true;
    }
    if (truncated) continue;
    wide.push(`${describe(el)} [${Math.round(r.left)}, ${Math.round(r.right)}]`);
  }

  // Text units and their clipped line boxes.
  const TAGS = new Set(['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'P', 'A', 'BUTTON', 'LABEL', 'TD', 'TH', 'LI', 'DT', 'DD', 'SUMMARY', 'LEGEND', 'FIGCAPTION']);
  const ownText = (el) => [...el.childNodes].some((n) => n.nodeType === Node.TEXT_NODE && n.textContent.trim());
  const units = all.filter((el) => (TAGS.has(el.tagName) || ownText(el)) && !inAny(el, overlays) && (!scope || el.closest(scope)));

  const clipRect = (el) => {
    let box = { left: -Infinity, top: -Infinity, right: Infinity, bottom: Infinity };
    for (let a = el; a && a !== document.documentElement; a = a.parentElement) {
      if (!clips(a)) continue;
      const r = a.getBoundingClientRect();
      box = { left: Math.max(box.left, r.left), top: Math.max(box.top, r.top), right: Math.min(box.right, r.right), bottom: Math.min(box.bottom, r.bottom) };
    }
    // What is on screen.
    return { left: Math.max(box.left, 0), top: Math.max(box.top, 0), right: Math.min(box.right, vw), bottom: Math.min(box.bottom, window.innerHeight) };
  };

  const measure = () => {
    const measured = [];
    for (const el of units) {
      if (!el.checkVisibility({ opacityProperty: true, visibilityProperty: true })) continue;
      const rects = [];
      // A control's box covers the text under it.
      if ((el.tagName === 'BUTTON' || el.tagName === 'A') && getComputedStyle(el).display !== 'inline') {
        const r = el.getBoundingClientRect();
        const clip = clipRect(el.parentElement ?? el);
        const c = { left: Math.max(r.left, clip.left), top: Math.max(r.top, clip.top), right: Math.min(r.right, clip.right), bottom: Math.min(r.bottom, clip.bottom) };
        if (c.right - c.left > 1 && c.bottom - c.top > 1 && (el.innerText || '').trim()) measured.push({ el, rects: [c] });
        continue;
      }
      const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
      for (let n = walker.nextNode(); n; n = walker.nextNode()) {
        if (!n.textContent.trim() || !n.parentElement) continue;
        const parent = n.parentElement;
        if (!parent.checkVisibility({ opacityProperty: true, visibilityProperty: true })) continue;
        const clip = clipRect(parent);
        const range = document.createRange();
        range.selectNodeContents(n);
        for (const r of range.getClientRects()) {
          const c = { left: Math.max(r.left, clip.left), top: Math.max(r.top, clip.top), right: Math.min(r.right, clip.right), bottom: Math.min(r.bottom, clip.bottom) };
          if (c.right - c.left > 1 && c.bottom - c.top > 1) rects.push(c);
        }
      }
      if (rects.length) measured.push({ el, rects });
    }
    return measured;
  };

  const found = new Map();
  const compare = () => {
    const m = measure();
    for (let i = 0; i < m.length; i++) {
      for (let j = i + 1; j < m.length; j++) {
        const a = m[i];
        const b = m[j];
        if (a.el.contains(b.el) || b.el.contains(a.el)) continue;
        for (const ra of a.rects) {
          const hit = b.rects.find((rb) => Math.min(ra.right, rb.right) - Math.max(ra.left, rb.left) > 1 && Math.min(ra.bottom, rb.bottom) - Math.max(ra.top, rb.top) > 1);
          if (hit) {
            const key = `${describe(a.el)} × ${describe(b.el)}`;
            if (!found.has(key)) found.set(key, `${key} at (${Math.round(hit.left)}, ${Math.round(hit.top)})`);
            break;
          }
        }
      }
    }
  };

  // Every vertical scroller through its height (the others at the top), then the document.
  const vertical = all.filter((el) => {
    const oy = getComputedStyle(el).overflowY;
    return (oy === 'auto' || oy === 'scroll') && el.scrollHeight > el.clientHeight + 1;
  });
  const saved = vertical.map((el) => el.scrollTop);
  const savedY = window.scrollY;
  vertical.forEach((el) => (el.scrollTop = 0));
  window.scrollTo(0, 0);
  compare();
  for (const el of vertical) {
    for (let y = el.clientHeight * 0.8; y < el.scrollHeight; y += el.clientHeight * 0.8) {
      el.scrollTop = y;
      compare();
    }
    el.scrollTop = 0;
  }
  const docHeight = document.documentElement.scrollHeight;
  for (let y = window.innerHeight * 0.8; y < docHeight; y += window.innerHeight * 0.8) {
    window.scrollTo(0, y);
    compare();
  }
  vertical.forEach((el, i) => (el.scrollTop = saved[i]));
  window.scrollTo(0, savedY);

  return { scrollWidth: document.documentElement.scrollWidth, innerWidth: vw, wide, scrollers, overlaps: [...found.values()] };
}

/**
 * Audit the page's layout. `scope`: with a menu open over the page, only the menu's text
 * is checked for overlaps (the page under it was checked with the menu closed).
 * @param {import('playwright-core').Page} page
 * @param {{scrollers?: string[], clippers?: string[], overlays?: string[], scope?: string}} allow  CSS selectors, each named in the flow
 */
export async function auditLayout(page, { scrollers = [], clippers = [], overlays = [], scope = null } = {}) {
  return page.evaluate(audit, { scrollers, clippers, overlays, scope });
}

/** The audit's failures as lines; empty when the layout passes. */
export function layoutProblems(result) {
  const problems = [];
  if (result.scrollWidth > result.innerWidth) problems.push(`document scrollWidth ${result.scrollWidth} > innerWidth ${result.innerWidth}`);
  for (const w of result.wide) problems.push(`past the viewport: ${w}`);
  for (const s of result.scrollers) if (!s.named) problems.push(`unnamed horizontal scroller: ${s.el}`);
  for (const o of result.overlaps) problems.push(`overlapping text: ${o}`);
  return problems;
}
