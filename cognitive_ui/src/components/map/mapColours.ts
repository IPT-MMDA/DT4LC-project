import { useEffect, useState } from 'react';

// Map paint colours come from the design tokens in src/index.css, never from
// hard-coded hex. MapLibre's colour parser does not understand oklch(), so token
// values are converted to rgb() here, and re-read whenever data-theme changes.

/** Which token paints what on the map. */
const ROLES = {
  // Light amber, as the design draws regions; the selected one is bolder (MapContainer).
  regionSelected: 'region',
  region: 'region',
  drawing: 'canc',
  vertexFill: 'surface',
  boundaries: 'canc',
  footprint: 'canc',  // a raster's outline (the sample's footprint)
} as const;

export type MapColours = Record<keyof typeof ROLES, string>;

// Only when a token is unset (jsdom, or a stylesheet that failed to load).
const UNSET = 'rgb(128, 128, 128)';

const OKLCH = /^oklch\(\s*([\d.]+)(%?)\s+([\d.]+)\s+([\d.]+)(?:deg)?\s*(?:\/\s*([\d.]+)(%?)\s*)?\)$/i;

function srgbChannel(linear: number): number {
  const v = linear <= 0.0031308 ? 12.92 * linear : 1.055 * Math.pow(linear, 1 / 2.4) - 0.055;
  return Math.round(Math.min(1, Math.max(0, v)) * 255);
}

/**
 * A CSS colour MapLibre can parse: `oklch(L C H [/ A])` becomes `rgb()` / `rgba()`
 * (OKLab to linear sRGB, then gamma, clamped to the sRGB gamut); anything else
 * (hex, rgb(), named colours) is returned unchanged.
 */
export function toMapColour(value: string): string {
  const css = value.trim();
  const m = OKLCH.exec(css);
  if (!m) return css || UNSET;

  const L = m[2] ? Number(m[1]) / 100 : Number(m[1]);
  const C = Number(m[3]);
  const h = (Number(m[4]) * Math.PI) / 180;
  const a = C * Math.cos(h);
  const b = C * Math.sin(h);

  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const mm = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;

  const r = srgbChannel(4.0767416621 * l - 3.3077115913 * mm + 0.2309699292 * s);
  const g = srgbChannel(-1.2684380046 * l + 2.6097574011 * mm - 0.3413193965 * s);
  const bl = srgbChannel(-0.0041960863 * l - 0.7034186147 * mm + 1.707614701 * s);

  if (m[5] === undefined) return `rgb(${r}, ${g}, ${bl})`;
  const alpha = m[6] ? Number(m[5]) / 100 : Number(m[5]);
  return `rgba(${r}, ${g}, ${bl}, ${alpha})`;
}

/** The current theme's map colours, read from the token custom properties on `root`. */
export function readMapColours(root: HTMLElement = document.documentElement): MapColours {
  const style = getComputedStyle(root);
  const entries = Object.entries(ROLES).map(([role, token]) => [
    role,
    toMapColour(style.getPropertyValue(`--${token}`)),
  ]);
  return Object.fromEntries(entries) as MapColours;
}

/** Map colours that follow the theme: re-read when `data-theme` changes on <html>. */
export function useMapColours(): MapColours {
  const [colours, setColours] = useState(() => readMapColours());

  useEffect(() => {
    const root = document.documentElement;
    const update = () => setColours(readMapColours(root));
    update();
    const observer = new MutationObserver(update);
    observer.observe(root, { attributes: true, attributeFilter: ['data-theme'] });
    return () => observer.disconnect();
  }, []);

  return colours;
}
