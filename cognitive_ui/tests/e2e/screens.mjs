// The app's screens reached through the UI, for the groups that look at every screen in
// both themes (flows/a11y.e2e.mjs, flows/dark.e2e.mjs). Guest + E2E_LLM=none: the landing
// as a first-time guest, the sample and an NDWI change in the Workspace, the result On map
// with a stored drawn region, Capabilities, the Library and the result's detail page.
export const NDWI = 'Compare water before and after the dam breach';
export const NDMI = 'Map moisture change with NDMI between the two dates';
export const ANALYSIS_TIMEOUT = 60_000;
const STORAGE = 'dt4lc-app-storage';

// About 410 km² of the Lower Dnipro floodplain, as the map store persists a drawn region.
export const REGION = {
  id: 'region-e2e',
  name: 'Dnipro floodplain · Kherson',
  type: 'polygon',
  geometry: {
    type: 'Feature',
    properties: {},
    geometry: { type: 'Polygon', coordinates: [[[32.4, 46.47], [32.66, 46.47], [32.66, 46.655], [32.4, 46.655], [32.4, 46.47]]] },
  },
  bbox: [32.4, 46.47, 32.66, 46.655],
  createdAt: '2026-10-23T00:00:00.000Z',
};

/**
 * A 1440×900 context with reduced motion (still shots, no colour transitions mid-check),
 * the stored drawn region, and `theme` as the stored theme.
 */
export async function newContext(browser, { theme = 'light', region = true } = {}) {
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, reducedMotion: 'reduce' });
  await context.addInitScript(
    ({ key, r, t }) => {
      if (!localStorage.getItem(key)) {
        localStorage.setItem(key, JSON.stringify({ state: { ...(r ? { drawnRegions: [r], selectedRegion: r } : {}), theme: t }, version: 0 }));
      }
    },
    { key: STORAGE, r: region ? REGION : null, t: theme },
  );
  return context;
}

export async function settle(page) {
  await page.evaluate(() => document.fonts.ready);
  await page.waitForLoadState('networkidle', { timeout: 10_000 }).catch(() => {});
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
}

/**
 * Switch to `theme`: the rail's toggle in the app; outside it (the landing) the stored
 * theme and a reload. Resolves once <html data-theme> says so.
 */
export async function setTheme(page, theme) {
  if ((await page.evaluate(() => document.documentElement.dataset.theme)) === theme) return;
  const toggle = page.getByRole('navigation', { name: 'Main' }).getByRole('button', { name: /^Switch to (dark|light) theme$/ });
  if (await toggle.count()) {
    await toggle.click();
  } else {
    await page.evaluate(
      ({ key, t }) => {
        const stored = JSON.parse(localStorage.getItem(key) || '{"state":{},"version":0}');
        stored.state.theme = t;
        localStorage.setItem(key, JSON.stringify(stored));
      },
      { key: STORAGE, t: theme },
    );
    await page.reload();
  }
  await page.waitForFunction((t) => document.documentElement.dataset.theme === t, theme, { timeout: 5_000 });
  await settle(page);
}

/** Send `text` from the composer. */
export async function send(page, text) {
  const input = page.getByLabel('Message', { exact: true });
  await input.fill(text);
  await input.press('Enter');
}

/** The analysis message for job N (0-based) once its phase is `phase`. */
export async function analysis(page, phase, n = 0) {
  const message = page.getByTestId('analysis-message').nth(n);
  await message.and(page.locator(`[data-phase="${phase}"]`)).waitFor({ timeout: ANALYSIS_TIMEOUT });
  return message;
}

/** The main navigation (the rail at 1440). */
export async function navigate(page, name, url) {
  await page.getByRole('navigation', { name: 'Main' }).getByRole('link', { name, exact: true }).click();
  await page.waitForURL(url);
}

/** The landing as a first-time guest. */
export async function landing(page, baseURL) {
  await page.goto(`${baseURL}/`);
  await page.getByTestId('landing').waitFor();
  await settle(page);
}

/** The Workspace with the sample and an NDWI change done; returns its analysis message. */
export async function workspaceDone(page, baseURL) {
  await page.goto(`${baseURL}/app?sample=1`);
  await page.getByTestId('attachment-chip').waitFor();
  await send(page, NDWI);
  const message = await analysis(page, 'done');
  await message.getByTestId('legend').waitFor();
  await settle(page);
  return message;
}

/** Map mode with the NDWI result (On map from `message`), the sample and the stored region. */
export async function mapWithLayers(page, message) {
  await message.getByRole('group', { name: 'Result actions' }).getByRole('button', { name: 'On map' }).click();
  await navigate(page, 'Map', /\/app\/map$/);
  await page.waitForFunction(() => window.__dt4lcMap);
  await page.getByTestId('map-legend').waitFor();
  await page.getByRole('complementary', { name: 'Imagery' }).getByRole('region', { name: 'Map layers' }).getByTestId('map-layer-row').first().waitFor();
  await page.evaluate(() => new Promise((resolve) => (window.__dt4lcMap.loaded() ? resolve() : window.__dt4lcMap.once('idle', resolve))));
  await settle(page);
}

export async function capabilities(page) {
  await navigate(page, 'Can do', /\/app\/capabilities$/);
  await page.getByTestId('builtin-item').first().waitFor();
  await settle(page);
}

export async function library(page) {
  await navigate(page, 'Library', /\/app\/library$/);
  await page.getByTestId('library-row').first().waitFor();
  await settle(page);
}

/** From the Library: the first row's detail page, its images loaded. */
export async function resultDetail(page) {
  await page.getByTestId('library-row').first().getByTestId('library-name').click();
  await page.waitForURL(/\/app\/results\//);
  await page.getByTestId('detail-downloads').waitFor();
  await page.waitForFunction(() => [...document.querySelectorAll('img[data-testid="detail-image"]')].every((i) => i.complete && i.naturalWidth > 0));
  await settle(page);
}

/** Operator Settings (E2E_DEV_ROLE=operator, E2E_SETTINGS=1). */
export async function settings(page, baseURL) {
  await page.goto(`${baseURL}/app/settings`);
  await page.getByTestId('settings-section-assistant').waitFor();
  await page.getByTestId('assistant-connected').waitFor();
  await settle(page);
}
