import { Link } from 'react-router-dom';

/** A destination that a later run builds (Library, Can do, Settings, Result detail). */
export function Placeholder({ title }: { title: string }) {
  return (
    <main className="flex-1 min-w-0 p-8 bg-bg">
      <h1 className="text-title">{title}</h1>
      <p className="mt-2 text-body text-muted">Arrives in the next run.</p>
      <Link to="/app" className="mt-4 inline-block text-control font-medium text-accent-ink">
        Back to chat
      </Link>
    </main>
  );
}
