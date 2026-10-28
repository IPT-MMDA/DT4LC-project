import { Link, useLocation } from 'react-router-dom';

/** An unknown `/app/...` path, and `/app/settings` for anyone but an operator (the API 404s it too). */
export function NotFound() {
  const { pathname } = useLocation();
  return (
    <main
      data-testid="not-found"
      aria-labelledby="not-found-title"
      className="flex-1 min-w-0 h-full overflow-y-auto bg-bg px-10 pt-8 pb-12 flex flex-col gap-2 text-[13px]"
    >
      <p className="font-mono text-[11px] font-medium text-muted">404</p>
      <h1 id="not-found-title" className="text-title tracking-[-0.01em]">
        Page not found
      </h1>
      <p className="text-body text-muted">
        There is nothing at <span className="font-mono text-text2">{pathname}</span> on this server.
      </p>
      <Link to="/app" className="mt-2 self-start text-control font-medium text-accent-ink hover:underline">
        Back to chat
      </Link>
    </main>
  );
}
