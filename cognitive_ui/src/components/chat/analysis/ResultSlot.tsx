/**
 * Where a finished analysis shows its result card. F4 replaces this placeholder
 * with one card per artifact kind, read from GET /v1/jobs/{jobId} `result`.
 */
export function ResultSlot({ jobId, title }: { jobId: string; title: string }) {
  return (
    <div data-testid="result-slot" data-job-id={jobId} className="border border-border rounded-card bg-surface px-3 py-[10px]">
      <div className="text-[13px] font-semibold">{title}</div>
      <p className="text-control text-muted">Result card arrives in F4</p>
    </div>
  );
}
