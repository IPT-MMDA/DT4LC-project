import { useEffect, useRef, useState, type ReactNode } from 'react';
import { Menu, X } from 'lucide-react';
import { Link } from 'react-router-dom';
import { roleLabel } from '../api/hooks/useSession';
import { HowItWorks } from '../components/landing/HowItWorks';
import '../components/landing/landing.css';
import { PHONE_QUERY, useMediaQuery } from '../hooks/useMediaQuery';
import type { SessionInfo } from '../types';

// The landing (design/DESIGN.md section 7, design/prototype/Landing v2.dc.html):
// hero B, "What it can do", "How it works" and the footer. The hero sits on imagery
// in both themes, so its ink (and its focus ring) is literal white.

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

function NavLinks({ className, onPick }: { className: string; onPick?: () => void }) {
  return (
    <nav aria-label="Landing" id="landing-nav" className={className}>
      <Link to="/app/capabilities" onClick={onPick}>
        Capabilities
      </Link>
      <a href="#how" onClick={onPick}>
        How it works
      </a>
      <a href="#about" onClick={onPick}>
        About
      </a>
      <External href={GITHUB}>GitHub ↗</External>
    </nav>
  );
}

/** The header; on a phone its links collapse behind a Menu button beside Open workspace. */
function Header() {
  const phone = useMediaQuery(PHONE_QUERY);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLElement>(null);
  const menuOpen = phone && open;

  useEffect(() => {
    if (!menuOpen) return;
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false);
    const onPointer = (e: PointerEvent) => ref.current && !ref.current.contains(e.target as Node) && setOpen(false);
    document.addEventListener('keydown', onKey);
    document.addEventListener('pointerdown', onPointer);
    return () => {
      document.removeEventListener('keydown', onKey);
      document.removeEventListener('pointerdown', onPointer);
    };
  }, [menuOpen]);

  return (
    <header ref={ref} className="absolute inset-x-0 top-0 z-[5] h-16 flex items-center px-[80px] max-md:px-4 md:max-lg:px-8 gap-[28px] text-[#fff] [&_:focus-visible]:outline-[#fff]">
      {/* The name holds the visible "DT" and "DT4LC" (WCAG 2.5.3, label in name). */}
      <Link to="/" aria-label="DT, DT4LC home" className="flex items-center gap-[10px]">
        <span className="w-7 h-7 rounded-[7px] bg-[#fff] text-[#1a1d23] flex items-center justify-center text-[11px] font-bold">
          DT
        </span>{' '}
        <span className="font-semibold tracking-[-.01em]">DT4LC</span>
      </Link>
      <div className="flex-1" />
      <div className="flex items-center gap-[6px] py-1 pr-1 pl-[14px] max-md:pl-1 rounded-pill bg-[rgba(10,12,16,.55)] border border-[rgba(255,255,255,.25)]">
        {!phone && <NavLinks className="flex gap-[18px] text-[13.5px] pr-[10px]" />}
        <Link to="/app" className="text-[13px] px-3 py-[7px] rounded-pill bg-[#fff] text-[#1a1d23] font-medium">
          Open workspace
        </Link>
        {phone && (
          <button
            type="button"
            aria-label="Menu"
            aria-expanded={open}
            aria-controls={open ? 'landing-nav' : undefined}
            onClick={() => setOpen((o) => !o)}
            className="w-8 h-8 rounded-pill flex items-center justify-center"
          >
            {open ? <X aria-hidden className="w-[18px] h-[18px]" /> : <Menu aria-hidden className="w-[18px] h-[18px]" />}
          </button>
        )}
      </div>
      {menuOpen && (
        <NavLinks
          onPick={() => setOpen(false)}
          className="absolute top-16 inset-x-4 p-2 flex flex-col rounded-card bg-[#0a0c10] border border-[rgba(255,255,255,.25)] text-[15px] [&>a]:px-3 [&>a]:py-[10px]"
        />
      )}
    </header>
  );
}

/** The sample's figure; under the buttons on a phone, at the bottom of the hero otherwise. */
function Caption({ className }: { className: string }) {
  return (
    // The bundled crop's NDWI change as the app computes it (+157 km², DESIGN §8); the full scene in the image differs.
    <div data-testid="landing-caption" className={`${className} font-mono text-[12.5px] text-[rgba(255,255,255,.85)]`}>
      <span className="group relative">
        Example · Lower Dnipro, 1 → 9 June 2023 · +157 km² of water after the Kakhovka dam breach{' '}
        {/* Focusable, so a keyboard shows the note the pointer gets from the title. */}
        <span
          data-testid="landing-caption-crop"
          tabIndex={0}
          title={FULL_SCENE}
          className="underline decoration-dotted underline-offset-2 cursor-help rounded-[2px]"
        >
          (bundled crop)
        </span>{' '}
        · NASA HLS L30
        <span className="sr-only">. {FULL_SCENE}</span>
        <span
          aria-hidden
          data-testid="landing-caption-note"
          className="hidden group-focus-within:block absolute left-0 bottom-full mb-2 w-[min(420px,calc(100vw-32px))] px-3 py-2 rounded-[8px] bg-[rgba(10,12,16,.92)] border border-[rgba(255,255,255,.25)] font-sans text-[12.5px] leading-[1.45] text-[#fff]"
        >
          {FULL_SCENE}
        </span>
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
  );
}

function Hero() {
  const phone = useMediaQuery(PHONE_QUERY);
  return (
    <section
      aria-labelledby="landing-title"
      data-testid="landing-hero"
      className="relative h-screen max-md:h-auto max-md:min-h-[100dvh] overflow-hidden bg-[#0a0c10] text-[#fff] [&_:focus-visible]:outline-[#fff]"
    >
      <img
        src={`${HERO}_1920.jpg`}
        srcSet={`${HERO}_960.jpg 960w, ${HERO}_1920.jpg 1920w`}
        sizes="100vw"
        alt="Water change layer over the lower Dnipro"
        className="landing-drift absolute inset-0 w-full h-full object-cover"
      />
      <div className="absolute inset-0 bg-[linear-gradient(90deg,rgba(0,0,0,.55)_0%,rgba(0,0,0,0)_60%)]" />
      <div className="absolute inset-0 bg-[linear-gradient(0deg,rgba(0,0,0,.3)_0%,rgba(0,0,0,0)_40%)]" />
      <div className="absolute left-[80px] right-[80px] top-[84px] bottom-[72px] md:max-lg:inset-x-8 max-md:relative max-md:inset-auto max-md:min-h-[100dvh] max-md:px-4 max-md:pt-[84px] max-md:pb-8 flex flex-col items-start justify-center pt-[2%]">
        <div className="text-[11px] tracking-[.08em] uppercase text-[rgba(255,255,255,.8)] font-semibold mb-4">
          Digital Twin for Land Cover · open-source research software
        </div>
        <h1 id="landing-title" className="landing-settle mb-5 text-hero max-md:text-hero-phone max-w-[900px]">
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
        {phone && <Caption className="mt-6 flex flex-wrap items-center gap-x-3 gap-y-1" />}
      </div>
      {!phone && <Caption className="absolute left-[80px] md:max-lg:left-8 md:max-lg:right-8 md:max-lg:flex-wrap bottom-6 flex items-center gap-3" />}
    </section>
  );
}

function Tiers() {
  return (
    <section id="capabilities" aria-labelledby="capabilities-title" className="px-10 max-md:px-4 md:max-lg:px-6 py-14 max-md:py-10 border-t border-border bg-surface">
      <div className="max-w-[1360px] mx-auto flex flex-col gap-[22px]">
        <div className="flex items-baseline justify-between gap-5">
          <h2 id="capabilities-title" className="text-[22px] font-semibold tracking-[-.015em]">
            What it can do
          </h2>
          <Link to="/app/capabilities" className="text-[13px] text-accent">
            All capabilities →
          </Link>
        </div>
        <div className="grid grid-cols-3 max-md:grid-cols-1 gap-3">
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
    <footer id="about" className="border-t border-border bg-surface pt-8 pb-9 px-10 max-md:px-4 md:max-lg:px-6">
      <div className="max-w-[1360px] mx-auto flex flex-col gap-[22px]">
        <div className="grid grid-cols-3 max-lg:grid-cols-1 gap-3 max-lg:gap-5 items-center">
          <div className="flex gap-7 max-md:gap-5 items-center max-lg:flex-wrap">
            <Logo src="/brand/logo_mmda.png" name='MMDA, Igor Sikorsky Kyiv Polytechnic Institute' mark="MMDA" height={72} />
            <Logo src="/brand/logo_unige.png" name="University of Geneva" mark="UNIGE" height={72} />
            <Logo name="University of Maryland" mark="UMD" height={72} />
            <Logo name="Swiss National Science Foundation" mark="SNSF" height={56} />
          </div>
          <p className="col-span-2 max-lg:col-span-1 min-w-0 text-[12.5px] text-muted leading-[1.6]">
            Developed at the National Technical University of Ukraine "Igor Sikorsky Kyiv Polytechnic Institute" with the University of Geneva
            and the University of Maryland. Supported by the project "DT4LC" (Grant 2023.01/0040) of the Ukrainian-Swiss Joint Research
            Programme, funded by the Swiss National Science Foundation.
          </p>
        </div>
        <div className="flex gap-5 text-[12.5px] text-text2 border-t border-border2 pt-4 flex-wrap">
          <span>
            Open source · AGPL-3.0 ·{' '}
            <External href={GITHUB} className="text-accent underline underline-offset-2">
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

/**
 * The line under "How it works" (DESIGN §6): a guest's retention and quota, also while
 * the session loads or when it failed; a member's or operator's has no quota.
 */
function roleNote(session?: SessionInfo): string {
  if (session && !session.guest) {
    return `${roleLabel(session.role)}. Conversations stay in this browser and results on this server until you delete them. No analysis limit.`;
  }
  const hours = session?.kept_hours ?? 24;
  const limit = session?.analyses.limit ?? 5;
  return `No account. Conversations stay in this browser for ${hours} h. Up to ${limit} analyses per hour.`;
}

/** `/` for every visitor; routes.tsx sends only Workspace deep links on to `/app`. */
export function Landing({ session }: { session?: SessionInfo }) {
  return (
    <div data-testid="landing" className="relative bg-bg text-text text-[14px] font-sans">
      <Header />
      <main>
        <Hero />
        <Tiers />
        <HowItWorks note={roleNote(session)} />
      </main>
      <Footer />
    </div>
  );
}
