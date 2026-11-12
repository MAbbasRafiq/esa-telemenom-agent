/**
 * Bottom chat bar (ChatGPT-style):
 *   attach file · text (+ "/" menu) · send
 */
import {
  FormEvent,
  KeyboardEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  activeSlashFilter,
  applySlashCommand,
  filterSlashCommands,
  type SlashCommand,
} from "../slashCommands";

type ChatComposerProps = {
  value: string;
  uploadedFile: File | null;
  disabled?: boolean;
  loading?: boolean;
  onChange: (value: string) => void;
  onUploadFile: (file: File | null) => void;
  onSubmit: () => void;
};

export default function ChatComposer({
  value,
  uploadedFile,
  disabled = false,
  loading = false,
  onChange,
  onUploadFile,
  onSubmit,
}: ChatComposerProps) {
  const [highlight, setHighlight] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const fragment = activeSlashFilter(value);
  const menuOpen = fragment !== null;
  const options = useMemo(
    () => (menuOpen ? filterSlashCommands(fragment ?? "") : []),
    [menuOpen, fragment],
  );

  useEffect(() => {
    setHighlight(0);
  }, [fragment, options.length]);

  // Grow textarea height with content (capped).
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) {
      return;
    }
    el.style.height = "auto";
    const next = Math.min(el.scrollHeight, 160);
    el.style.height = `${next}px`;
  }, [value]);

  function pickCommand(cmd: SlashCommand) {
    const next = applySlashCommand(value, cmd);
    onChange(next.trimStart());
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (menuOpen && options.length > 0) {
      if (event.key === "ArrowDown") {
        event.preventDefault();
        setHighlight((i) => (i + 1) % options.length);
        return;
      }
      if (event.key === "ArrowUp") {
        event.preventDefault();
        setHighlight((i) => (i - 1 + options.length) % options.length);
        return;
      }
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        pickCommand(options[highlight]);
        return;
      }
      if (event.key === "Escape") {
        event.preventDefault();
        onChange(value.replace(/(?:^|\s)\/[a-zA-Z0-9_-]*$/, "").trimEnd());
        return;
      }
    }

    // Enter sends (Shift+Enter = new line), when slash menu is closed.
    if (event.key === "Enter" && !event.shiftKey && !menuOpen) {
      event.preventDefault();
      if (!disabled && !loading && value.trim()) {
        onSubmit();
      }
    }
  }

  function handleFormSubmit(event: FormEvent) {
    event.preventDefault();
    if (!disabled && !loading && value.trim()) {
      onSubmit();
    }
  }

  return (
    <form onSubmit={handleFormSubmit} className="w-full">
      {uploadedFile && (
        <div className="mb-2.5 flex items-center gap-2 px-0.5">
          <span className="inline-flex max-w-full items-center gap-3 rounded-xl border border-[var(--color-line)] bg-white px-3.5 py-2.5 text-sm text-[var(--color-ink)] shadow-sm">
            <span className="text-base leading-none" aria-hidden="true">
              📎
            </span>
            <span className="min-w-0 truncate font-medium">
              {uploadedFile.name}
            </span>
            <button
              type="button"
              disabled={disabled}
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                onUploadFile(null);
                if (fileInputRef.current) {
                  fileInputRef.current.value = "";
                }
              }}
              className="ml-1 inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-lg leading-none text-[var(--color-ink-muted)] transition hover:bg-[var(--color-surface)] hover:text-[var(--color-danger)] disabled:opacity-40"
              aria-label="Remove file"
            >
              ×
            </button>
          </span>
        </div>
      )}

      <div className="relative rounded-2xl border border-[var(--color-line)] bg-white shadow-sm focus-within:border-[var(--color-accent)]">
        {/* Slash menu opens upward above the composer */}
        {menuOpen && options.length > 0 && (
          <ul
            className="absolute bottom-full left-0 right-0 z-20 mb-2 max-h-56 overflow-auto rounded-xl border border-[var(--color-line)] bg-white py-1 shadow-lg"
            role="listbox"
          >
            {options.map((cmd, index) => {
              const active = index === highlight;
              return (
                <li key={cmd.id}>
                  <button
                    type="button"
                    role="option"
                    aria-selected={active}
                    className={
                      "flex w-full items-start gap-3 px-3 py-2 text-left text-sm " +
                      (active ? "bg-teal-50" : "hover:bg-[var(--color-surface)]")
                    }
                    onMouseDown={(e) => {
                      e.preventDefault();
                      pickCommand(cmd);
                    }}
                    onMouseEnter={() => setHighlight(index)}
                  >
                    <span className="shrink-0 font-mono text-xs text-[var(--color-accent-dark)]">
                      /{cmd.slash}
                    </span>
                    <span className="text-[var(--color-ink)]">{cmd.label}</span>
                  </button>
                </li>
              );
            })}
          </ul>
        )}

        <textarea
          ref={textareaRef}
          value={value}
          disabled={disabled || loading}
          rows={1}
          placeholder='Message Telemanom…  (type "/" for samples)'
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          className="max-h-40 min-h-[52px] w-full resize-none bg-transparent px-4 pb-12 pt-3.5 text-[15px] leading-relaxed text-[var(--color-ink)] outline-none placeholder:text-[var(--color-ink-muted)]"
        />

        {/* Bottom row inside the composer: attach + send */}
        <div className="absolute bottom-2 left-2 right-2 flex items-center justify-between gap-2">
          <div>
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,text/csv"
              disabled={disabled || loading}
              className="hidden"
              onChange={(e) => {
                const file =
                  e.target.files && e.target.files[0] ? e.target.files[0] : null;
                onUploadFile(file);
              }}
            />
            <button
              type="button"
              disabled={disabled || loading}
              onClick={() => fileInputRef.current?.click()}
              title="Attach telemetry CSV"
              className="inline-flex h-9 w-9 items-center justify-center rounded-xl text-[var(--color-ink-muted)] transition hover:bg-[var(--color-surface)] hover:text-[var(--color-accent-dark)] disabled:opacity-50"
              aria-label="Attach file"
            >
              {/* Simple paperclip icon (SVG) */}
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48" />
              </svg>
            </button>
          </div>

          <button
            type="submit"
            disabled={disabled || loading || !value.trim()}
            className="inline-flex h-9 items-center rounded-xl bg-[var(--color-accent)] px-4 text-sm font-semibold text-white transition hover:bg-[var(--color-accent-dark)] disabled:cursor-not-allowed disabled:opacity-40"
          >
            {loading ? "Working…" : "Send"}
          </button>
        </div>
      </div>
    </form>
  );
}
