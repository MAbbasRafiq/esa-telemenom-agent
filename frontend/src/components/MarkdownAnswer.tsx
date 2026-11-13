/**
 * Renders assistant answers as Markdown (tables, bold, lists, code).
 */
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { formatAnswer } from "../formatAnswer";

type MarkdownAnswerProps = {
  text: string;
};

export default function MarkdownAnswer({ text }: MarkdownAnswerProps) {
  const cleaned = formatAnswer(text);

  return (
    <div className="markdown-answer text-[15px] leading-relaxed text-[var(--color-ink)]">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          p: ({ children }) => <p className="mb-3 last:mb-0">{children}</p>,
          strong: ({ children }) => (
            <strong className="font-semibold text-[var(--color-ink)]">
              {children}
            </strong>
          ),
          em: ({ children }) => <em className="italic">{children}</em>,
          ul: ({ children }) => (
            <ul className="mb-3 list-disc space-y-1 pl-5 last:mb-0">{children}</ul>
          ),
          ol: ({ children }) => (
            <ol className="mb-3 list-decimal space-y-1 pl-5 last:mb-0">
              {children}
            </ol>
          ),
          li: ({ children }) => <li className="leading-relaxed">{children}</li>,
          code: ({ className, children }) => {
            const isBlock = Boolean(className);
            if (isBlock) {
              return (
                <code className="block overflow-x-auto rounded-lg bg-[var(--color-surface)] p-3 text-[13px]">
                  {children}
                </code>
              );
            }
            return (
              <code className="rounded bg-[var(--color-surface)] px-1.5 py-0.5 font-mono text-[13px] text-[var(--color-accent-dark)]">
                {children}
              </code>
            );
          },
          pre: ({ children }) => (
            <pre className="mb-3 overflow-x-auto rounded-lg last:mb-0">
              {children}
            </pre>
          ),
          table: ({ children }) => (
            <div className="mb-3 overflow-x-auto last:mb-0">
              <table className="w-full min-w-[280px] border-collapse text-left text-sm">
                {children}
              </table>
            </div>
          ),
          thead: ({ children }) => (
            <thead className="border-b border-[var(--color-line)] bg-[var(--color-surface)]">
              {children}
            </thead>
          ),
          th: ({ children }) => (
            <th className="px-3 py-2 font-semibold text-[var(--color-ink)]">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="border-t border-[var(--color-line)]/70 px-3 py-2 text-[var(--color-ink)]">
              {children}
            </td>
          ),
          a: ({ href, children }) => (
            <a
              href={href}
              className="text-[var(--color-accent-dark)] underline underline-offset-2"
              target="_blank"
              rel="noreferrer"
            >
              {children}
            </a>
          ),
          h1: ({ children }) => (
            <h3 className="mb-2 font-[family-name:var(--font-display)] text-xl">
              {children}
            </h3>
          ),
          h2: ({ children }) => (
            <h3 className="mb-2 font-[family-name:var(--font-display)] text-lg">
              {children}
            </h3>
          ),
          h3: ({ children }) => (
            <h4 className="mb-2 text-base font-semibold">{children}</h4>
          ),
        }}
      >
        {cleaned}
      </ReactMarkdown>
    </div>
  );
}
