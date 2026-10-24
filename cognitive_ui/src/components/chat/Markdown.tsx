import ReactMarkdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';

// Assistant replies are markdown (lists, bold, code, tables via GFM). No
// rehype-raw: HTML in a reply is never rendered as HTML, and react-markdown's
// default URL filter drops javascript: links.

const components: Components = {
  p: ({ children }) => <p className="my-0 [&:not(:first-child)]:mt-2">{children}</p>,
  ul: ({ children }) => <ul className="list-disc pl-5 my-2 flex flex-col gap-[2px]">{children}</ul>,
  ol: ({ children }) => <ol className="list-decimal pl-5 my-2 flex flex-col gap-[2px]">{children}</ol>,
  strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noopener noreferrer" className="text-accent-ink underline underline-offset-2">
      {children}
    </a>
  ),
  h1: ({ children }) => <h3 className="text-card mt-3 mb-1">{children}</h3>,
  h2: ({ children }) => <h3 className="text-card mt-3 mb-1">{children}</h3>,
  h3: ({ children }) => <h4 className="text-body font-semibold mt-3 mb-1">{children}</h4>,
  code: ({ className, children }) => (
    <code className={className ?? 'font-mono text-[12.5px] bg-surface2 rounded-chip px-1 py-[1px]'}>{children}</code>
  ),
  pre: ({ children }) => (
    <pre className="font-mono text-[12.5px] bg-surface2 border border-border2 rounded-control p-3 my-2 overflow-x-auto">
      {children}
    </pre>
  ),
  table: ({ children }) => (
    <div className="my-2 overflow-x-auto">
      <table className="text-control border-collapse">{children}</table>
    </div>
  ),
  th: ({ children }) => <th className="border border-border px-2 py-1 text-left font-semibold bg-surface2">{children}</th>,
  td: ({ children }) => <td className="border border-border px-2 py-1 align-top">{children}</td>,
};

export function Markdown({ text }: { text: string }) {
  return (
    <div className="text-body text-text break-words">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {text}
      </ReactMarkdown>
    </div>
  );
}
