/**
 * ChatGPT-style left rail: brand, new chat, past chats, session turns,
 * attach status, quick asks.
 */
import type { ReactNode } from "react";
import { SLASH_COMMANDS } from "../slashCommands";
import type { SessionInfo } from "../types";

type SidebarProps = {
  turns: { question: string }[];
  sessions: SessionInfo[];
  activeThreadId: string | null;
  uploadedFileName: string | null;
  open: boolean;
  onClose: () => void;
  onNewChat: () => void;
  onSelectTurn: (index: number) => void;
  onSelectSession: (threadId: string) => void;
  onDeleteSession: (threadId: string) => void;
  onClearFile: () => void;
  onQuickAsk: (question: string) => void;
};

/** Section heading in the rail, with an optional item count. */
function SectionLabel({
  children,
  count,
  className = "",
}: {
  children: ReactNode;
  count?: number;
  className?: string;
}) {
  return (
    <p
      className={`flex items-center gap-2 px-1 text-[11px] font-bold uppercase tracking-[0.12em] text-[var(--color-ink)] ${className}`}
    >
      {children}
      {count ? (
        <span className="rounded-full bg-white px-1.5 py-0.5 text-[10px] font-semibold text-[var(--color-ink-muted)]">
          {count}
        </span>
      ) : null}
    </p>
  );
}

export default function Sidebar({
  turns,
  sessions,
  activeThreadId,
  uploadedFileName,
  open,
  onClose,
  onNewChat,
  onSelectTurn,
  onSelectSession,
  onDeleteSession,
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
          <SectionLabel count={sessions.length} className="mb-2">
            Chats
          </SectionLabel>
          {sessions.length === 0 ? (
            <p className="px-1 text-xs text-[var(--color-ink-muted)]">
              No saved chats yet
            </p>
          ) : (
            <ul className="space-y-0.5">
              {sessions.map((session) => {
                const isActive = session.thread_id === activeThreadId;
                return (
                  <li key={session.thread_id} className="group flex items-center">
                    <button
                      type="button"
                      onClick={() => {
                        onSelectSession(session.thread_id);
                        onClose();
                      }}
                      className={[
                        "min-w-0 flex-1 truncate rounded-lg px-2.5 py-1.5 text-left text-[13px] transition",
                        isActive
                          ? "bg-white font-semibold text-[var(--color-ink)] shadow-sm"
                          : "text-[var(--color-ink)] hover:bg-white hover:shadow-sm",
                      ].join(" ")}
                      title={session.title ?? session.thread_id}
                    >
                      {session.title || "Untitled chat"}
                    </button>
                    <button
                      type="button"
                      onClick={() => onDeleteSession(session.thread_id)}
                      className="shrink-0 px-1.5 text-xs text-[var(--color-ink-muted)] opacity-0 transition group-hover:opacity-100 hover:text-[var(--color-danger)]"
                      aria-label={`Delete chat: ${session.title ?? session.thread_id}`}
                      title="Delete chat"
                    >
                      ×
                    </button>
                  </li>
                );
              })}
            </ul>
          )}

          <SectionLabel count={turns.length} className="mb-1.5 mt-5">
            This session
          </SectionLabel>
          {turns.length === 0 ? (
            <p className="px-1 text-xs text-[var(--color-ink-muted)]">
              No messages yet
            </p>
          ) : (
            <ul>
              {turns.map((turn, index) => (
                <li key={index}>
                  <button
                    type="button"
                    onClick={() => {
                      onSelectTurn(index);
                      onClose();
                    }}
                    className="flex w-full items-center gap-2 rounded-lg px-2.5 py-1 text-left text-[13px] text-[var(--color-ink-muted)] transition hover:bg-white hover:text-[var(--color-ink)]"
                    title={turn.question}
                  >
                    <span className="shrink-0 text-[10px] text-[var(--color-accent-dark)]">
                      ●
                    </span>
                    <span className="min-w-0 truncate">{turn.question}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}

          <SectionLabel className="mb-1.5 mt-5">Quick asks</SectionLabel>
          <ul className="space-y-0.5">
            {SLASH_COMMANDS.map((cmd) => (
              <li key={cmd.id}>
                <button
                  type="button"
                  onClick={() => {
                    onQuickAsk(cmd.question);
                    onClose();
                  }}
                  className="w-full truncate rounded-lg px-2.5 py-1.5 text-left text-[13px] text-[var(--color-ink-muted)] transition hover:bg-white hover:text-[var(--color-ink)] hover:shadow-sm"
                  title={cmd.question}
                >
                  {cmd.label}
                </button>
              </li>
            ))}
          </ul>
        </div>

        <div className="shrink-0 border-t border-[var(--color-line)]/70 px-3 py-3">
          <SectionLabel>Attached</SectionLabel>
          {uploadedFileName ? (
            <div className="mt-2 flex items-start justify-between gap-2 px-1">
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
            <p className="mt-2 px-1 text-xs text-[var(--color-ink-muted)]">
              No CSV attached
            </p>
          )}
        </div>
      </aside>
    </>
  );
}
