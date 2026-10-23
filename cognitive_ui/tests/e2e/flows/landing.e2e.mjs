// The guest landing at 1440x900: design/screens/landing-hero.png and landing-below-fold.png.
// A first-time guest sees it at /; a guest with a stored conversation goes to /app.
// The hero buttons open the Workspace with and without the sample. Reduced motion stops
// every animation and the demo; the step buttons scrub the demo either way.
// Guest group (`npm run e2e:landing:guest`); the member redirect is landing-member.e2e.mjs.
import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { outPath, setup, trackPageErrors } from '../harness.mjs';

const guest = { skip: (process.env.E2E_DEV_ROLE || 'guest') !== 'guest' };

let h;
before(async () => {
  if (guest.skip) return;
  h = await setup();
});
after(async () => {
  await h?.browser.close();
});

// Records whether the landing ever rendered, so a redirect is checked for a flash too.
function watchLanding() {
  window.__landingSeen = false;
  new MutationObserver(() => {
    if (document.querySelector('[data-testid="landing"]')) window.__landingSeen = true;
  }).observe(document, { childList: true, subtree: true });
}

async function open(url, { reducedMotion, init } = {}) {
  const context = await h.browser.newContext({ viewport: { width: 1440, height: 900 } });
  if (init) await context.addInitScript(init.fn, init.arg);
  await context.addInitScript(watchLanding);
  const page = await context.newPage();
  const errors = trackPageErrors(page);
  if (reducedMotion) await page.emulateMedia({ reducedMotion });
  await page.goto(`${h.baseURL}${url}`);
  return { context, page, errors };
}

async function landingReady(page) {
  await page.getByTestId('landing').waitFor();
  await page.evaluate(() => document.fonts.ready);
  await page.waitForFunction(() => [...document.images].every((img) => img.complete));
}

// The demo's own animations: they finish once the 8-s timeline has played.
const demoAnimationsFinished = (page) =>
  page.evaluate(() =>
    Promise.all(
      document
        .getAnimations()
        .filter((a) => a.effect?.target?.closest?.('[data-demo]'))
        .map((a) => a.finished),
    ),
  );

const opacity = (page, anim) =>
  page.evaluate((name) => getComputedStyle(document.querySelector(`[data-demo] [data-anim="${name}"]`)).opacity, anim);

async function scrollToHow(page) {
  // As in landing-below-fold.png: the end of the tier cards, then "How it works".
  await page.evaluate(() => window.scrollTo(0, document.getElementById('how').offsetTop - 153));
}

test('a first-time guest sees hero B at /', guest, async () => {
  const { context, page, errors } = await open('/');
  try {
    await landingReady(page);
    assert.equal(new URL(page.url()).pathname, '/');
    await page.getByRole('heading', { level: 1, name: 'What changed on the ground? Ask in plain language.' }).waitFor();
    await page.getByText('Digital Twin for Land Cover · open-source research software').waitFor();
    assert.match(
      await page.getByTestId('landing-caption').innerText(),
      /^Example · Lower Dnipro, 1 → 9 June 2023 · \+157 km² of water after the Kakhovka dam breach · NASA HLS L30/,
    );
    const hero = page.getByRole('img', { name: 'Water change layer over the lower Dnipro' });
    assert.match(await hero.evaluate((img) => img.currentSrc), /kakhovka_water_change_06-01_to_06-09_1920\.jpg$/);
    // The headline has settled; the image keeps drifting.
    await page.evaluate(() => Promise.all(document.getAnimations().filter((a) => a.effect?.target?.tagName === 'H1').map((a) => a.finished)));
    assert.deepEqual(
      await page.evaluate(() => document.getAnimations().filter((a) => a.playState === 'running').map((a) => a.animationName)),
      ['drift'],
      'before the demo is in view only the drift runs',
    );
    await page.screenshot({ path: outPath('landing-hero.png') });
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});

test('below the fold: tier cards, the self-running demo, the footer', guest, async () => {
  const { context, page, errors } = await open('/');
  try {
    await landingReady(page);
    assert.equal(await page.getByTestId('landing-tier').count(), 3);
    const demo = page.getByTestId('landing-demo');
    assert.equal(await demo.getAttribute('data-mode'), 'idle');

    await scrollToHow(page);
    await page.waitForFunction(() => document.querySelector('[data-demo]').dataset.mode === 'play');
    await demoAnimationsFinished(page);
    await page.waitForFunction(() => document.querySelector('[data-demo]').dataset.frame === '3');
    assert.equal(await page.getByTestId('landing-step-3').getAttribute('aria-current'), 'step');
    assert.equal(await opacity(page, 'follow'), '1');
    await page.screenshot({ path: outPath('landing-below-fold.png') });

    const footer = page.getByRole('contentinfo');
    await footer.scrollIntoViewIfNeeded();
    assert.deepEqual(await footer.getByTestId('logo-text-mark').allInnerTexts(), ['UMD', 'SNSF']);
    assert.equal(await footer.getByRole('img', { name: 'University of Geneva' }).evaluate((img) => img.naturalWidth > 0), true);
    await footer.getByText('Grant 2023.01/0040', { exact: false }).waitFor();
    await footer.getByText('Imagery: NASA HLS, Copernicus Sentinel-2').waitFor();
    await page.screenshot({ path: outPath('landing-footer.png') });
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});

test('"Try with the Kakhovka sample" opens the Workspace with the sample chip', guest, async () => {
  const { context, page, errors } = await open('/');
  try {
    await landingReady(page);
    const sample = page.waitForRequest((req) => req.method() === 'POST' && new URL(req.url()).pathname === '/v1/sample');
    const button = page.getByRole('link', { name: 'Try with the Kakhovka sample' });
    assert.equal(await button.getAttribute('href'), '/app?sample=1');
    await button.click();
    await sample;
    const chip = page.getByTestId('attachment-chip');
    await chip.waitFor();
    assert.match(await chip.innerText(), /kakhovka_hls_l30/);
    await page.waitForURL((url) => url.pathname === '/app' && !url.searchParams.has('sample'));
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});

test('"Describe your own analysis" opens the empty Workspace (?sample=0)', guest, async () => {
  const { context, page, errors } = await open('/');
  try {
    await landingReady(page);
    await page.getByRole('link', { name: 'Describe your own analysis' }).click();
    await page.waitForURL((url) => url.pathname === '/app' && url.searchParams.get('sample') === '0');
    await page.getByRole('heading', { name: 'What do you want to know?' }).waitFor();
    assert.equal(await page.getByTestId('attachment-chip').count(), 0);
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});

test('a guest with a stored conversation goes to /app, with no flash of the landing', guest, async () => {
  const now = new Date().toISOString();
  const stored = {
    state: {
      conversations: [
        {
          id: 'conv_e2e',
          title: 'Earlier chat',
          createdAt: now,
          updatedAt: now,
          messages: [{ id: 'msg_e2e', kind: 'user', text: 'Earlier chat', createdAt: now }],
        },
      ],
      activeConversationId: 'conv_e2e',
      keptHours: 24,
    },
    version: 4,
  };
  const seed = (value) => localStorage.setItem('dt4lc-chat-storage', value);
  const { context, page, errors } = await open('/', { init: { fn: seed, arg: JSON.stringify(stored) } });
  try {
    await page.waitForURL((url) => url.pathname === '/app');
    await page.getByRole('complementary', { name: 'Conversations' }).getByText('Earlier chat').first().waitFor();
    assert.equal(await page.evaluate(() => window.__landingSeen), false);
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});

test('prefers-reduced-motion: no animation at all, a still demo, the steps still scrub', guest, async () => {
  const { context, page, errors } = await open('/', { reducedMotion: 'reduce' });
  try {
    await landingReady(page);
    assert.equal(await page.evaluate(() => document.getAnimations().length), 0);

    await scrollToHow(page);
    const demo = page.getByTestId('landing-demo');
    await page.waitForTimeout(500);
    assert.equal(await demo.getAttribute('data-mode'), 'poster');
    assert.equal(await page.getByRole('button', { name: 'Play the demo from the start' }).count(), 0);
    assert.equal(await page.evaluate(() => document.getAnimations().length), 0);
    const first = await demo.screenshot();
    await page.waitForTimeout(3000);
    const later = await demo.screenshot();
    assert.ok(first.equals(later), 'the demo does not advance');
    assert.equal(await page.evaluate(() => document.getAnimations().length), 0);
    await page.screenshot({ path: outPath('landing-below-fold-reduced-motion.png') });

    await page.getByTestId('landing-step-1').click();
    await page.waitForFunction(() => document.querySelector('[data-demo]').dataset.mode === 'scrub');
    assert.equal(await demo.getAttribute('data-frame'), '1');
    assert.equal(await opacity(page, 'tl'), '0', 'step 1 is before the timeline');
    assert.ok(!(await demo.screenshot()).equals(later), 'step 1 shows another frame');
    // A scrubbed frame holds still.
    assert.equal(await page.evaluate(() => document.getAnimations().filter((a) => a.playState === 'running').length), 0);
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});

test('the step buttons scrub the demo to their frame', guest, async () => {
  const { context, page, errors } = await open('/');
  try {
    await landingReady(page);
    await scrollToHow(page);
    const demo = page.getByTestId('landing-demo');
    await page.waitForFunction(() => document.querySelector('[data-demo]').dataset.mode === 'play');

    const scrub = async (step) => {
      await page.getByTestId(`landing-step-${step}`).click();
      await page.waitForFunction((s) => {
        const d = document.querySelector('[data-demo]');
        return d.dataset.mode === 'scrub' && d.dataset.frame === String(s);
      }, step);
      assert.equal(await page.getByTestId(`landing-step-${step}`).getAttribute('aria-current'), 'step');
      return demo.screenshot();
    };

    // Step 2, mid-run: the first two steps done, the third still running, no result yet.
    const two = await scrub(2);
    assert.deepEqual(
      [await opacity(page, 'tl'), await opacity(page, 'check2'), await opacity(page, 'check3'), await opacity(page, 'card')],
      ['1', '1', '0', '0'],
    );
    // Step 3: the result card with the difference map and the interpretation.
    const three = await scrub(3);
    assert.deepEqual([await opacity(page, 'card'), await opacity(page, 'diff'), await opacity(page, 'interp')], ['1', '1', '1']);
    // Step 1: the question and its files, before the plan.
    const one = await scrub(1);
    assert.deepEqual([await opacity(page, 'chip2'), await opacity(page, 'tl')], ['1', '0']);
    assert.ok(!one.equals(two) && !two.equals(three) && !one.equals(three), 'each step is its own frame');
    assert.deepEqual(errors, []);
  } finally {
    await context.close();
  }
});
