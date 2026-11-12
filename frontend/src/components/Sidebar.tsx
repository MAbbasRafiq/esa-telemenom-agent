/**
 * ChatGPT-style left rail: brand, new chat, session turns, attach status, quick asks.
 */
import { SLASH_COMMANDS } from "../slashCommands";

type SidebarProps = {
  turns: { question: string }[];
  uploadedFileName: string | null;
  open: boolean;
  onClose: () => void;
  onNewChat: () => void;
  onSelectTurn: (index: number) => void;
  onClearFile: () => void;
  onQuickAsk: (question: string) => void;
};

export default function Sidebar({
  turns,
  uploadedFileName,
  open,
  onClose,
  onNewChat,
  onSelectTurn,
  onClearFile,
  onQuickAsk,
}: SidebarProps) {
  return (
    <>
      {/* Mobile scrim */}
      {open && (
        <button
          type="button"
          className="fixed inset-0 z-30 bg-black/30 md:hidden"
          aria-label="Close sidebar"
          onClick={onClose}
        />
      )}

      <aside
        className={[
          "fixed inset-y-0 left-0 z-40 flex h-full w-[260px] shrink-0 flex-col border-r border-[var(--color-line)] bg-[#e8eef6] transition-transform duration-200 md:static md:z-0 md:translate-x-0",
          open ? "translate-x-0" : "-translate-x-full",
        ].join(" ")}
      >
        <div className="shrink-0 border-b border-[var(--color-line)]/70 px-4 py-4">
          <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-[var(--color-accent-dark)]">
            ESA-ADB · Telemanom
          </p>
          <h1 className="mt-1 font-[family-name:var(--font-display)] text-lg text-[var(--color-ink)]">
            Assistant
          </h1>
          <button
            type="button"
            onClick={() => {
              onNewChat();
              onClose();
            }}
            className="mt-4 w-full rounded-xl border border-[var(--color-line)] bg-white px-3 py-2.5 text-left text-sm font-medium text-[var(--color-ink)] shadow-sm transition hover:border-[var(--color-accent)] hover:text-[var(--color-accent-dark)]"
          >
            + New chat
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-3 py-3">
          <p className="mb-2 px-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--color-ink-muted)]">
            This session
          </p>
          {turns.length === 0 ? (
            <p className="px-1 text-xs text-[var(--color-ink-muted)]">
              No messages yet
            </p>
          ) : (
            <ul className="space-y-1">
              {turns.map((turn, index) => (
                <li key={index}>
                  <button
                    type="button"
                    onClick={() => {
                      onSelectTurn(index);
                      onClose();
                    }}
                    className="w-full truncate rounded-lg px-2.5 py-2 text-left text-[13px] text-[var(--color-ink)] transition hover:bg-white/80"
                    title={turn.question}
                  >
                    {turn.question}
                  </button>
                </li>
              ))}
            </ul>
          )}

          <p className="mb-2 mt-5 px-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--color-ink-muted)]">
            Quick asks
          </p>
          <ul className="space-y-1">
            {SLASH_COMMANDS.map((cmd) => (
              <li key={cmd.id}>
                <button
                  type="button"
                  onClick={() => {
                    onQuickAsk(cmd.question);
                    onClose();
                  }}
                  className="w-full truncate rounded-lg px-2.5 py-2 text-left text-[13px] text-[var(--color-ink-muted)] transition hover:bg-white/80 hover:text-[var(--color-ink)]"
                  title={cmd.question}
                >
                  {cmd.label}
                </button>
              </li>
            ))}
          </ul>
        </div>

        <div className="shrink-0 border-t border-[var(--color-line)]/70 px-4 py-3">
          <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--color-ink-muted)]">
            Attached
          </p>
          {uploadedFileName ? (
            <div className="mt-2 flex items-start justify-between gap-2">
              <p
                className="min-w-0 truncate text-xs text-[var(--color-ink)]"
                title={uploadedFileName}
              >
                📎 {uploadedFileName}
              </p>
              <button
                type="button"
                onClick={onClearFile}
                className="shrink-0 text-[11px] text-[var(--color-ink-muted)] underline-offset-2 hover:text-[var(--color-danger)] hover:underline"
              >
                Clear
              </button>
            </div>
          ) : (
            <p className="mt-2 text-xs text-[var(--color-ink-muted)]">
              No CSV attached
            </p>
          )}
        </div>
      </aside>
    </>
  );
}
