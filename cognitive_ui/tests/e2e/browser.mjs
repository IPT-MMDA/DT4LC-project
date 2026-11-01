// Locate and launch chrome-headless-shell without downloading anything.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const here = path.dirname(fileURLToPath(import.meta.url));
export const LIBROOT_DIR = path.join(here, '.libroot', 'usr', 'lib', 'x86_64-linux-gnu');

/**
 * PLAYWRIGHT_CHROMIUM_EXECUTABLE, or else the newest
 * ~/.cache/ms-playwright/chromium_headless_shell-<rev>/chrome-headless-shell-linux64/chrome-headless-shell.
 */
export function resolveChromiumExecutable() {
  const explicit = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE;
  if (explicit) {
    if (!fs.existsSync(explicit)) throw new Error(`PLAYWRIGHT_CHROMIUM_EXECUTABLE=${explicit} does not exist`);
    return explicit;
  }
  const cache = path.join(os.homedir(), '.cache', 'ms-playwright');
  const candidates = (fs.existsSync(cache) ? fs.readdirSync(cache) : [])
    .map((name) => /^chromium_headless_shell-(\d+)$/.exec(name))
    .filter(Boolean)
    .sort((a, b) => Number(b[1]) - Number(a[1]))
    .map((m) => path.join(cache, m[0], 'chrome-headless-shell-linux64', 'chrome-headless-shell'))
    .filter((p) => fs.existsSync(p));
  if (candidates.length === 0) {
    throw new Error(
      `No chrome-headless-shell under ${cache}. Run \`npx playwright-core install chromium-headless-shell\` ` +
        'or set PLAYWRIGHT_CHROMIUM_EXECUTABLE.',
    );
  }
  return candidates[0];
}

/** Launch a browser server that flows connect to over its wsEndpoint. */
export async function launchBrowserServer() {
  const executablePath = resolveChromiumExecutable();
  const env = { ...process.env };
  if (fs.existsSync(LIBROOT_DIR)) {
    env.LD_LIBRARY_PATH = [LIBROOT_DIR, env.LD_LIBRARY_PATH].filter(Boolean).join(path.delimiter);
  }
  try {
    // run.mjs owns signal handling: Playwright's own handlers exit before its teardown finishes.
    return await chromium.launchServer({
      executablePath,
      headless: true,
      env,
      handleSIGINT: false,
      handleSIGTERM: false,
      handleSIGHUP: false,
    });
  } catch (err) {
    const missing = /error while loading shared libraries: (\S+?):/.exec(String(err?.message ?? err));
    if (missing) {
      err.message =
        `${executablePath} cannot start: ${missing[1]} is missing.\n` +
        `Run tests/e2e/setup-libs.sh (no root needed) to extract it into ${LIBROOT_DIR}, then retry.\n\n` +
        err.message;
    }
    throw err;
  }
}
