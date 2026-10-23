// Shared helper for flows/*.e2e.mjs. The runner (run.mjs) passes everything
// through E2E_* env vars; flows never start servers or browsers themselves.
import path from 'node:path';
import { chromium } from 'playwright-core';

/** Provider keys of the E2E_SETTINGS=1 API env: present for the server, never in the page. */
export const FAKE_PROVIDER_KEYS = { GEMINI_API_KEY: 'e2e-secret-key-123', GROQ_API_KEY: 'e2e-secret-key-456' };

/** Pick who signs in next at the fake GitHub (E2E_SIGNIN=fake): a key of FAKE_USERS in fake-github.mjs. */
export async function signInAs(user) {
  const res = await fetch(`${required('E2E_FAKE_GITHUB_URL')}/__e2e/user`, {
    method: 'PUT',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ user }),
  });
  if (!res.ok) throw new Error(`signInAs: ${res.status} ${await res.text()}`);
}

function required(name) {
  const value = process.env[name];
  if (!value) throw new Error(`${name} is not set: run flows through \`npm run e2e\`, not directly`);
  return value;
}

/**
 * Connect to the runner's browser.
 * @returns {Promise<{baseURL: string, apiURL: string, browser: import('playwright-core').Browser, outDir: string, llm: string}>}
 */
export async function setup() {
  const browser = await chromium.connect(required('E2E_WS_ENDPOINT'));
  return {
    baseURL: required('E2E_BASE_URL'),
    apiURL: required('E2E_API_URL'),
    browser,
    outDir: required('E2E_OUT'),
    llm: required('E2E_LLM'),
  };
}

/** Absolute path for an artefact (screenshot, trace) in the output directory. */
export function outPath(name) {
  return path.join(required('E2E_OUT'), name);
}

/** Collect uncaught page errors; assert the returned array is empty at the end. */
export function trackPageErrors(page) {
  const errors = [];
  page.on('pageerror', (err) => errors.push(err.message));
  return errors;
}

function fakeLLMURL() {
  const url = process.env.E2E_FAKE_LLM_URL;
  if (!url) throw new Error('Fake LLM rules need E2E_LLM=fake');
  return url;
}

/**
 * Replace the fake Ollama's rules until resetLLMRules(). The first rule whose
 * `match` matches the last user or system message wins; otherwise `default`.
 * `delay_ms` holds that rule's reply back, e.g. to catch a step mid-run.
 * @param {{default?: string, rules: Array<{match: RegExp | string, reply: string, delay_ms?: number}>}} rules
 */
export async function setLLMRules({ default: fallback = '', rules }) {
  const body = {
    default: fallback,
    rules: rules.map(({ match, reply, delay_ms }) => ({
      ...(match instanceof RegExp ? { match: match.source, flags: match.flags } : { match }),
      reply,
      ...(delay_ms ? { delay_ms } : {}),
    })),
  };
  const res = await fetch(`${fakeLLMURL()}/__e2e/rules`, {
    method: 'PUT',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`setLLMRules: ${res.status} ${await res.text()}`);
}

/** Restore the rules the runner loaded from E2E_LLM_RULES. */
export async function resetLLMRules() {
  const res = await fetch(`${fakeLLMURL()}/__e2e/reset`, { method: 'POST' });
  if (!res.ok) throw new Error(`resetLLMRules: ${res.status} ${await res.text()}`);
}
