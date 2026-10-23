import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { HowItWorks } from '../components/landing/HowItWorks';
import '../components/landing/landing.css';
import type { SessionInfo } from '../types';

// The guest landing (design/DESIGN.md section 7, design/prototype/Landing v2.dc.html):
// hero B, "What it can do", "How it works" and the footer. The hero sits on imagery
// in both themes, so its ink is literal white.

const GITHUB = 'https://github.com/IPT-MMDA/DT4LC-project';
// The JOSS paper is in review and has no DOI yet: link its source in the repository.
const PAPER = `${GITHUB}/blob/master/paper.md`;
const HERO = '/landing/kakhovka_water_change_06-01_to_06-09';
// The caption's figure is the bundled crop's; the hero shows the full scene (DESIGN §8).
const FULL_SCENE =
  'Computed by the app on the bundled 768 × 768 px crop. The full scene in this image shows more: +428 km² of water gained and 43 km² lost (ΔNDWI > 0.3).';

const TIERS = [
  {
    label: '1 · Ask',
    title: 'Questions, explanations, follow-ups',
    text: 'Methods, results and what to do next, answered in plain language by the assistant.',
    prompt: 'Why is NDWI better than NDVI for water?',
  },
  {
    label: '2 · Composed analyses',
    title: 'New pipelines from the built-in steps',
    text: 'Any index from bands, masked classification, multi-date comparison, parameter changes. Needs the assistant.',
    prompt: 'Map moisture change with NDMI between the two dates',
  },
  {
    label: '3 · Built-in analyses',
    title: 'Eight analyses that run without the assistant',
    text: 'Spectral indices, change detection, land-cover and snow classification, statistics, field boundaries, Prithvi reconstruction.',
    prompt: 'Compare water before and after the dam breach',
  },
];

function External({ href, children, className }: { href: string; children: ReactNode; className?: string }) {
  return (
    <a href={href} target="_blank" rel="noopener noreferrer" className={className}>
      {children}
    </a>
  );
}

function Header() {
  return (
    <header className="absolute inset-x-0 top-0 z-[5] h-16 flex items-center px-[80px] gap-[28px] text-[#fff]">
      <Link to="/" aria-label="DT4LC home" className="flex items-center gap-[10px]">
        <span aria-hidden className="w-7 h-7 rounded-[7px] bg-[#fff] text-[#1a1d23] flex items-center justify-center text-[11px] font-bold">
          DT
        </span>
        <span className="font-semibold tracking-[-.01em]">DT4LC</span>
      </Link>
      <div className="flex-1" />
      <div className="flex items-center gap-[6px] py-1 pr-1 pl-[14px] rounded-pill bg-[rgba(10,12,16,.55)] border border-[rgba(255,255,255,.25)]">
        <nav aria-label="Landing" className="flex gap-[18px] text-[13.5px] pr-[10px]">
          <Link to="/app/capabilities">Capabilities</Link>
          <a href="#how">How it works</a>
          <a href="#about">About</a>
          <External href={GITHUB}>GitHub ↗</External>
        </nav>
        <Link to="/app?sample=0" className="text-[13px] px-3 py-[7px] rounded-pill bg-[#fff] text-[#1a1d23] font-medium">
          Open workspace
        </Link>
      </div>
    </header>
  );
}

function Hero() {
  return (
    <section aria-labelledby="landing-title" className="relative h-screen overflow-hidden bg-[#0a0c10] text-[#fff]">
      <img
        src={`${HERO}_1920.jpg`}
        srcSet={`${HERO}_960.jpg 960w, ${HERO}_1920.jpg 1920w`}
        sizes="100vw"
        alt="Water change layer over the lower Dnipro"
        className="landing-drift absolute inset-0 w-full h-full object-cover"
      />
      <div className="absolute inset-0 bg-[linear-gradient(90deg,rgba(0,0,0,.55)_0%,rgba(0,0,0,0)_60%)]" />
      <div className="absolute inset-0 bg-[linear-gradient(0deg,rgba(0,0,0,.3)_0%,rgba(0,0,0,0)_40%)]" />
      <div className="absolute left-[80px] right-[80px] top-[84px] bottom-[72px] flex flex-col items-start justify-center pt-[2%]">
        <div className="text-[11px] tracking-[.08em] uppercase text-[rgba(255,255,255,.8)] font-semibold mb-4">
          Digital Twin for Land Cover · open-source research software
        </div>
        <h1 id="landing-title" className="landing-settle mb-5 text-hero max-w-[900px]">
          What changed on the ground?
          <br />
          Ask in plain language.
        </h1>
        <p className="mb-8 text-hero-sub text-[rgba(255,255,255,.88)] max-w-[560px]">
          DT4LC plans the analysis, runs it on Sentinel-2, Landsat or your own imagery, and answers with maps, numbers and an explanation.
        </p>
        <div className="flex gap-3 flex-wrap">
          <Link to="/app?sample=1" className="inline-flex items-center h-12 px-[22px] rounded-[8px] bg-[#fff] text-[#1a1d23] font-semibold text-[15px]">
            Try with the Kakhovka sample
          </Link>
          <Link to="/app?sample=0" className="inline-flex items-center h-12 px-[22px] rounded-[8px] border border-[rgba(255,255,255,.7)] font-medium text-[15px]">
            Describe your own analysis
          </Link>
        </div>
      </div>
      {/* The bundled crop's NDWI change as the app computes it (+157 km², DESIGN §8); the full scene in the image differs. */}
      <div data-testid="landing-caption" className="absolute left-[80px] bottom-6 flex items-center gap-3 font-mono text-[12.5px] text-[rgba(255,255,255,.85)]">
        <span>
          Example · Lower Dnipro, 1 → 9 June 2023 · +157 km² of water after the Kakhovka dam breach{' '}
          <span data-testid="landing-caption-crop" title={FULL_SCENE} className="underline decoration-dotted underline-offset-2 cursor-help">
            (bundled crop)
          </span>{' '}
          · NASA HLS L30
          <span className="sr-only">. {FULL_SCENE}</span>
        </span>
        <span className="flex items-center gap-[5px]">
          <span className="w-[10px] h-[10px] rounded-[2px] bg-legend-gained" />
          gained
        </span>
        <span className="flex items-center gap-[5px]">
          <span className="w-[10px] h-[10px] rounded-[2px] bg-legend-lost" />
          lost
        </span>
      </div>
    </section>
  );
}

function Tiers() {
  return (
    <section id="capabilities" aria-labelledby="capabilities-title" className="px-10 py-14 border-t border-border bg-surface">
      <div className="max-w-[1360px] mx-auto flex flex-col gap-[22px]">
        <div className="flex items-baseline justify-between gap-5">
          <h2 id="capabilities-title" className="text-[22px] font-semibold tracking-[-.015em]">
            What it can do
          </h2>
          <Link to="/app/capabilities" className="text-[13px] text-accent">
            All capabilities →
          </Link>
        </div>
        <div className="grid grid-cols-3 gap-3">
          {TIERS.map((tier) => (
            <Link
              key={tier.label}
              to="/app/capabilities"
              data-testid="landing-tier"
              className="border border-border rounded-card bg-bg py-[18px] px-5 flex flex-col gap-2"
            >
              <div className="text-label uppercase text-muted">{tier.label}</div>
              <div className="font-semibold text-[15px]">{tier.title}</div>
              <div className="text-text2 leading-[1.5] text-[13px]">{tier.text}</div>
              <div className="mt-auto py-2 px-[10px] rounded-[7px] bg-surface2 text-[12.5px] leading-[1.45]">"{tier.prompt}"</div>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}

/** A partner logo; UMD and SNSF have no file yet (public/brand/README.md), so their slot holds a text mark. */
function Logo({ src, name, mark, height }: { src?: string; name: string; mark: string; height: 56 | 72 }) {
  const h = height === 72 ? 'h-[72px]' : 'h-[56px]';
  if (src) return <img src={src} alt={name} className={`${h} w-auto`} />;
  return (
    <span role="img" aria-label={name} title={name} data-testid="logo-text-mark" className={`${h} flex items-center text-[15px] font-semibold tracking-[-.01em] text-text2`}>
      {mark}
    </span>
  );
}

function Footer() {
  return (
    <footer id="about" className="border-t border-border bg-surface pt-8 pb-9 px-10">
      <div className="max-w-[1360px] mx-auto flex flex-col gap-[22px]">
        <div className="grid grid-cols-3 gap-3 items-center">
          <div className="flex gap-7 items-center">
            <Logo src="/brand/logo_mmda.png" name='MMDA, Igor Sikorsky Kyiv Polytechnic Institute' mark="MMDA" height={72} />
            <Logo src="/brand/logo_unige.png" name="University of Geneva" mark="UNIGE" height={72} />
            <Logo name="University of Maryland" mark="UMD" height={72} />
            <Logo name="Swiss National Science Foundation" mark="SNSF" height={56} />
          </div>
          <p className="col-span-2 min-w-0 text-[12.5px] text-muted leading-[1.6]">
            Developed at the National Technical University of Ukraine "Igor Sikorsky Kyiv Polytechnic Institute" with the University of Geneva
            and the University of Maryland. Supported by the project "DT4LC" (Grant 2023.01/0040) of the Ukrainian-Swiss Joint Research
            Programme, funded by the Swiss National Science Foundation.
          </p>
        </div>
        <div className="flex gap-5 text-[12.5px] text-text2 border-t border-border2 pt-4 flex-wrap">
          <span>
            Open source · AGPL-3.0 ·{' '}
            <External href={GITHUB} className="text-accent">
              GitHub ↗
            </External>
          </span>
          <External href={PAPER} className="text-accent">
            Paper in the Journal of Open Source Software ↗
          </External>
          <span className="text-muted">Imagery: NASA HLS, Copernicus Sentinel-2</span>
        </div>
      </div>
    </footer>
  );
}

/** `/` for a first-time guest; routes.tsx decides who sees it. */
export function Landing({ session }: { session?: SessionInfo }) {
  const hours = session?.kept_hours ?? 24;
  const limit = session?.analyses.limit ?? 5;
  return (
    <div data-testid="landing" className="relative bg-bg text-text text-[14px] font-sans">
      <Header />
      <main>
        <Hero />
        <Tiers />
        <HowItWorks note={`No account. Conversations stay in this browser for ${hours} h. Up to ${limit} analyses per hour.`} />
      </main>
      <Footer />
    </div>
  );
}
