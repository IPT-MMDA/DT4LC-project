import type { CSSProperties, ReactNode } from 'react';
import { cn } from '../../utils/cn';
import { SCRUB_AT, type DemoStep } from './demoMachine';
import { useDemo } from './useDemo';

const STEPS: { step: Exclude<DemoStep, 0>; title: string; text: string }[] = [
  {
    step: 1,
    title: 'Ask',
    text: 'Describe what you want to know. Attach GeoTIFFs, or pick a region and dates on the map and the imagery is fetched from Sentinel-2, Landsat or MODIS.',
  },
  {
    step: 2,
    title: 'It plans and runs',
    text: 'The assistant turns the question into steps, shows them as they run, and lets you cancel. Nothing is hidden in a job queue.',
  },
  {
    step: 3,
    title: 'You get maps, numbers and an explanation',
    text: 'Results arrive in the conversation and on the map, with downloads as GeoTIFF, PNG or GeoPackage. Ask a follow-up to go further.',
  },
];

// The bundled crop's own NDWI change result (src/test/fixtures/jobs/ndwi-change.json,
// stats.water), in the result card's km² format: not the full scene's +428 km² (DESIGN §8).
const WATER = { before: '13.9 km²', after: '170 km²', gained: '+157 km²' };

const Monogram = ({ size }: { size: 20 | 22 }) => (
  <div
    aria-hidden
    className={cn(
      'flex-none flex items-center justify-center bg-ink text-ink-bg text-[8px] font-bold',
      size === 20 ? 'w-5 h-5 rounded-[5px]' : 'w-[22px] h-[22px] rounded-control mt-[2px]',
    )}
  >
    DT
  </div>
);

function FileChip({ anim, name, src }: { anim: string; name: string; src: string }) {
  return (
    <div data-anim={anim} className="flex gap-[7px] items-center py-1 pl-1 pr-[9px] border border-border rounded-[7px] bg-surface">
      <img src={src} alt="" className="w-[26px] h-[26px] rounded-chip object-cover" />
      <div>
        <div className="text-[11.5px] font-mono">{name}</div>
        <div className="text-[10px] text-muted">HLS L30 · 5 bands · 30 m</div>
      </div>
    </div>
  );
}

function TimelineRow({ n, label, time }: { n: 1 | 2 | 3; label: string; time: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="relative w-[14px] h-[14px] flex-none">
        <span data-anim={`ring${n}`} className="absolute inset-0 rounded-full border-[1.5px] border-dashed border-faint" />
        <span data-anim={`check${n}`} className="absolute inset-0 rounded-full bg-done text-[#fff] text-[9px] flex items-center justify-center">
          ✓
        </span>
      </span>
      <span className="flex-1 text-text2">{label}</span>
      <span data-anim={`check${n}`} className="font-mono text-[11px] text-muted">
        {time}
      </span>
    </div>
  );
}

function Stat({ anim, label, value, className }: { anim: string; label: string; value: string; className?: string }) {
  return (
    <div>
      <div className="text-[10px] text-muted uppercase tracking-[.06em]">{label}</div>
      <div data-anim={anim} className={cn('font-mono text-stat mt-[2px]', className)}>
        {value}
      </div>
    </div>
  );
}

const Swatch = ({ className }: { className: string }) => <span className={cn('w-[9px] h-[9px] rounded-[2px]', className)} />;

const Label = ({ anim, children }: { anim: string; children: ReactNode }) => (
  <span data-anim={anim} className="absolute left-2 top-2 font-mono font-medium text-[10px] bg-[rgba(255,255,255,.92)] text-[#3a4150] px-[6px] py-[2px] rounded-[3px]">
    {children}
  </span>
);

/** The compact workspace that plays the three steps: one sample analysis, start to finish. */
function Demo({ demo }: { demo: ReturnType<typeof useDemo> }) {
  const { ref, state, play } = demo;
  const style = state.mode === 'scrub' && state.step ? ({ '--scrub': SCRUB_AT[state.step] } as CSSProperties) : undefined;
  return (
    <div
      ref={ref}
      data-demo
      data-mode={state.mode}
      data-frame={state.step}
      data-testid="landing-demo"
      style={style}
      className="relative border border-border rounded-[14px] bg-surface overflow-hidden max-w-[720px]"
    >
      <div className="h-10 flex items-center px-[14px] gap-[10px] border-b border-border2 text-[12px] text-muted">
        <Monogram size={20} />
        <span className="font-semibold text-text">Lower Dnipro — flooding after the dam breach</span>
        <span className="font-mono">2 files</span>
        <div className="flex-1" />
        {!state.reduced && (
          <button
            type="button"
            onClick={play}
            title="Play from the start"
            aria-label="Play the demo from the start"
            className="h-[26px] px-[9px] rounded-control border border-border bg-surface text-muted flex items-center gap-[5px] text-[12px]"
          >
            ▶ Play
          </button>
        )}
      </div>
      <div data-anim="scene" className="pt-4 px-4 pb-[18px] flex flex-col gap-[14px]">
        <div className="self-end max-w-[520px] flex flex-col items-end gap-[6px]">
          <div className="bg-bubble rounded-[12px_12px_3px_12px] py-[9px] px-[13px] leading-[1.5] text-[13.5px] relative">
            <span data-anim="type" className="inline-block whitespace-nowrap max-md:whitespace-normal">
              Compare water before and after the dam breach using NDWI.
            </span>
            <span data-anim="caret" className="absolute right-2 bottom-[11px] w-[2px] h-[14px] bg-text" />
          </div>
          <div className="flex gap-[6px]">
            <FileChip anim="chip1" name="hlsl_20230601.tif" src="/landing/demo_before.jpg" />
            <FileChip anim="chip2" name="hlsl_20230609.tif" src="/landing/demo_after.jpg" />
          </div>
        </div>
        <div className="flex gap-[10px]">
          <Monogram size={22} />
          <div className="min-w-0 flex-1 flex flex-col gap-[10px]">
            <div data-anim="intent" className="leading-[1.5] text-[13.5px]">
              I'll compute NDWI on both dates, then difference them to map where water changed.
            </div>
            <div data-anim="tl" className="flex flex-col gap-[6px] text-[12px]">
              <TimelineRow n={1} label="NDWI · 2023-06-01" time="9 s" />
              <TimelineRow n={2} label="NDWI · 2023-06-09" time="8 s" />
              <TimelineRow n={3} label="Change detection · after − before" time="31 s" />
            </div>
            <div data-anim="card" className="border border-border rounded-card bg-surface overflow-hidden">
              <div className="flex items-center justify-between py-2 px-3 border-b border-border2 gap-2">
                <div className="text-[12.5px] font-semibold">Water change · NDWI</div>
                <span data-anim="pill" className="inline-flex items-center gap-[5px] py-[2px] pr-2 pl-[6px] rounded-pill bg-done-tint text-done-ink font-semibold text-[11px]">
                  <span className="w-[11px] h-[11px] rounded-full bg-done text-[#fff] text-[7px] flex items-center justify-center">✓</span>
                  Done · 48 s
                </span>
              </div>
              <div className="relative h-[210px] overflow-hidden bg-[#e9e3d6]">
                <img src="/landing/demo_before.jpg" alt="True colour, 1 June 2023" className="absolute inset-0 w-full h-full object-cover" />
                <img data-anim="swipe" src="/landing/demo_after.jpg" alt="True colour, 9 June 2023" className="absolute inset-0 w-full h-full object-cover" />
                <div data-anim="handle" className="absolute inset-y-0 left-0 w-[2px] bg-[#fff] -translate-x-px">
                  <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-6 h-6 rounded-full bg-[#fff] shadow-[0_1px_4px_rgba(0,0,0,.35)] flex items-center justify-center text-[9px] text-[#1a1d23]">
                    ◂▸
                  </div>
                </div>
                <img data-anim="diff" src="/landing/demo_change.jpg" alt="Water change: gained in blue, lost in orange" className="absolute inset-0 w-full h-full object-cover" />
                <Label anim="lblb">before · 2023-06-01</Label>
                <Label anim="lbla">after · 2023-06-09</Label>
                <div data-anim="legend" className="absolute left-2 bottom-2 flex items-center gap-[10px] bg-[rgba(255,255,255,.94)] rounded-[5px] py-[5px] px-2 text-[10.5px] text-[#3a4150]">
                  <span className="font-semibold">difference</span>
                  <span className="flex items-center gap-1">
                    <Swatch className="bg-legend-gained" />
                    water gained
                  </span>
                  <span className="flex items-center gap-1">
                    <Swatch className="bg-legend-lost" />
                    water lost
                  </span>
                </div>
              </div>
              <div data-anim="stats" className="grid grid-cols-3 py-[10px] px-3 gap-2">
                <Stat anim="cntA" label="Water before" value={WATER.before} />
                <Stat anim="cntB" label="Water after" value={WATER.after} />
                <Stat anim="cntC" label="Gained" value={WATER.gained} className="text-legend-gained" />
              </div>
            </div>
            <div data-anim="interp" className="leading-[1.55] text-[13.5px]">
              On the bundled crop, open water grew from 13.9 to 170 km² in eight days: 157 km² gained, 0.6 km² lost.
            </div>
            <div data-anim="follow" className="flex flex-wrap gap-[6px]">
              {['Zoom to the largest change', 'Classify the exposed bed', 'Export GeoTIFF'].map((text) => (
                <span key={text} className="text-[12px] py-[5px] px-[10px] rounded-pill border border-border text-text2">
                  {text}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/** "How it works": three steps beside the self-running demo; each step scrubs the demo to it. */
export function HowItWorks({ note }: { note: string }) {
  const demo = useDemo();
  return (
    <section id="how" aria-labelledby="how-title" className="px-10 max-md:px-4 md:max-lg:px-6 py-14 max-md:py-10 border-t border-border">
      <div className="max-w-[1360px] mx-auto flex flex-col gap-[22px]">
        <h2 id="how-title" className="text-[22px] font-semibold tracking-[-.015em]">
          How it works
        </h2>
        <div className="grid grid-cols-[380px_minmax(0,1fr)] max-lg:grid-cols-1 gap-7 items-start">
          <div className="flex flex-col gap-[10px]">
            {STEPS.map(({ step, title, text }) => {
              const active = demo.state.step === step;
              return (
                <button
                  key={step}
                  type="button"
                  data-testid={`landing-step-${step}`}
                  aria-current={active ? 'step' : undefined}
                  onClick={() => demo.scrub(step)}
                  className={cn(
                    'text-left flex gap-[14px] py-[14px] px-4 rounded-card border focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent focus-visible:outline-offset-2',
                    active ? 'border-accent bg-surface' : 'border-border bg-transparent',
                  )}
                >
                  <div className="w-7 h-7 flex-none rounded-full border-[1.5px] border-border flex items-center justify-center font-mono font-medium text-[12px] text-muted">
                    {step}
                  </div>
                  <div>
                    <div className="font-semibold text-[14.5px] mb-1">{title}</div>
                    <div className="text-text2 leading-[1.5] text-[13px]">{text}</div>
                  </div>
                </button>
              );
            })}
            <div className="text-[13px] text-muted pt-[6px] px-[2px] leading-[1.5]">{note}</div>
          </div>
          <Demo demo={demo} />
        </div>
      </div>
    </section>
  );
}
