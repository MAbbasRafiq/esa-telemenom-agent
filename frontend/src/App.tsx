/**
 * ChatGPT-style layout:
 *   - Left sidebar (session, quick asks, attach status)
 *   - Clean message area
 *   - Composer pinned at the bottom (attach + text + send)
 */
import { useCallback, useEffect, useRef, useState } from "react";
import {
  checkHealth,
  deleteSession,
  fetchSessionDetail,
  fetchSessions,
  runAgentJob,
} from "./api";
import ChatComposer from "./components/ChatComposer";
import MarkdownAnswer from "./components/MarkdownAnswer";
import Sidebar from "./components/Sidebar";
import ToolSteps from "./components/ToolSteps";
import type { AgentRunResponse, SessionInfo } from "./types";

type ChatTurn = {
  question: string;
  fileName: string | null;
  result: AgentRunResponse | null;
  error: string | null;
};

export default function App() {
  const [question, setQuestion] = useState("");
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState("Working…");
  const [progressSteps, setProgressSteps] = useState<string[]>([]);
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  /** LangGraph thread — same id keeps memory across asks until New chat */
  const [threadId, setThreadId] = useState<string | null>(null);
  /** Past chats from the server (survive reloads and restarts) */
  const [sessions, setSessions] = useState<SessionInfo[]>([]);

  const bottomRef = useRef<HTMLDivElement>(null);
  const mainRef = useRef<HTMLElement>(null);
  const turnRefs = useRef<(HTMLDivElement | null)[]>([]);

  // Scroll the message pane itself (not the window) to the latest turn.
  useEffect(() => {
    const main = mainRef.current;
    if (!main) {
      return;
    }
    main.scrollTo({ top: main.scrollHeight, behavior: "smooth" });
  }, [turns, loading]);

  const refreshSessions = useCallback(async () => {
    setSessions(await fetchSessions());
  }, []);

  useEffect(() => {
    void refreshSessions();
  }, [refreshSessions]);

  function newChat() {
    if (loading) {
      return;
    }
    setTurns([]);
    setQuestion("");
    setUploadedFile(null);
    setThreadId(null);
  }

  /** Reopen a past chat: replay its turns and keep writing to the same thread. */
  async function openSession(id: string) {
    if (loading || id === threadId) {
      return;
    }
    try {
      const detail = await fetchSessionDetail(id);
      setTurns(
        detail.turns.map((turn) => ({
          question: turn.question,
          fileName: null,
          error: null,
          result: turn.answer
            ? {
                provider: "",
                model: "",
                tool_steps: [],
                final_answer: turn.answer,
                thread_id: detail.thread_id,
              }
            : null,
        })),
      );
      setThreadId(detail.thread_id);
      setQuestion("");
      setUploadedFile(null);
    } catch {
      // Chat was deleted or the API restarted — drop it from the list.
      void refreshSessions();
    }
  }

  async function removeSession(id: string) {
    await deleteSession(id);
    if (id === threadId) {
      newChat();
    }
    void refreshSessions();
  }

  function selectTurn(index: number) {
    const el = turnRefs.current[index];
    const main = mainRef.current;
    if (!el || !main) {
      return;
    }
    main.scrollTo({ top: el.offsetTop - 16, behavior: "smooth" });
  }

  async function sendMessage() {
    const trimmed = question.trim();
    if (!trimmed || loading) {
      return;
    }

    const fileName = uploadedFile ? uploadedFile.name : null;
    setLoading(true);
    setStatusMessage("Queued…");
    setProgressSteps(["Queued…"]);

    // Show the user bubble immediately.
    setTurns((prev) => [
      ...prev,
      { question: trimmed, fileName, result: null, error: null },
    ]);
    setQuestion("");

    try {
      const ok = await checkHealth();
      if (!ok) {
        throw new Error(
          "The assistant is not available right now. Please try again in a moment.",
        );
      }

      // Async job: API returns immediately; we poll until scoring/LLM finish.
      const data = await runAgentJob({
        question: trimmed,
        file: uploadedFile,
        sampleId: null,
        threadId,
        onStatus: setStatusMessage,
        onProgressSteps: setProgressSteps,
      });

      if (data.thread_id) {
        setThreadId(data.thread_id);
      }

      setTurns((prev) => {
        const copy = [...prev];
        const last = copy[copy.length - 1];
        copy[copy.length - 1] = { ...last, result: data, error: null };
        return copy;
      });
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Something went wrong. Please try again.";
      setTurns((prev) => {
        const copy = [...prev];
        const last = copy[copy.length - 1];
        copy[copy.length - 1] = { ...last, result: null, error: message };
        return copy;
      });
    } finally {
      setLoading(false);
      setStatusMessage("Working…");
      setProgressSteps([]);
      void refreshSessions();
    }
  }

  const hasMessages = turns.length > 0;

  return (
    <div className="flex h-[100dvh] overflow-hidden">
      <Sidebar
        turns={turns}
        sessions={sessions}
        activeThreadId={threadId}
        uploadedFileName={uploadedFile?.name ?? null}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onNewChat={newChat}
        onSelectTurn={selectTurn}
        onSelectSession={(id) => void openSession(id)}
        onDeleteSession={(id) => void removeSession(id)}
        onClearFile={() => setUploadedFile(null)}
        onQuickAsk={setQuestion}
      />

      <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden bg-[var(--color-surface)]">
        {/* Thin top bar — same surface as chat (no frosted white stripe) */}
        <header className="shrink-0 border-b border-[var(--color-line)]/60 bg-[var(--color-surface)] px-3 py-3 sm:px-6">
          <div className="mx-auto flex max-w-3xl items-center gap-3">
            <button
              type="button"
              className="rounded-lg border border-[var(--color-line)] bg-[var(--color-panel)] px-2.5 py-1.5 text-sm text-[var(--color-ink)] md:hidden"
              aria-label="Open sidebar"
              onClick={() => setSidebarOpen(true)}
            >
              ☰
            </button>
            <h1 className="font-[family-name:var(--font-display)] text-xl text-[var(--color-ink)] sm:text-2xl">
              Telemanom Assistant
            </h1>
          </div>
        </header>

        {/* Scrollable messages — only this pane scrolls */}
        <main
          ref={mainRef}
          className="min-h-0 flex-1 overflow-y-auto overscroll-contain bg-[var(--color-surface)]"
        >
          <div className="mx-auto max-w-3xl px-4 pb-6 pt-6 sm:px-6">
            {!hasMessages && !loading && (
              <div className="flex min-h-[50vh] flex-col items-center justify-center text-center">
                <h2 className="font-[family-name:var(--font-display)] text-3xl text-[var(--color-ink)] sm:text-4xl">
                  How can I help?
                </h2>
                <p className="mt-3 max-w-md text-sm text-[var(--color-ink-muted)]">
                  Ask about the model, or attach a telemetry CSV and ask to score
                  or compare alarms. Type{" "}
                  <span className="font-medium text-[var(--color-ink)]">/</span>{" "}
                  for sample questions.
                </p>
              </div>
            )}

            <div className="space-y-6">
              {turns.map((turn, index) => {
                const isLast = index === turns.length - 1;
                const waiting = loading && isLast && !turn.result && !turn.error;

                return (
                  <div
                    key={index}
                    ref={(el) => {
                      turnRefs.current[index] = el;
                    }}
                    className="space-y-4"
                  >
                    {/* User message */}
                    <div className="flex justify-end">
                      <div className="max-w-[85%] rounded-2xl rounded-br-md bg-[var(--color-ink)] px-4 py-3 text-[15px] leading-relaxed text-white">
                        <p className="whitespace-pre-wrap">{turn.question}</p>
                        {turn.fileName && (
                          <p className="mt-2 truncate text-xs text-white/70">
                            📎 {turn.fileName}
                          </p>
                        )}
                      </div>
                    </div>

                    {/* Assistant message */}
                    <div className="flex justify-start">
                      <div className="max-w-[90%] space-y-3">
                        {waiting && (
                          <div className="rounded-2xl rounded-bl-md border border-[var(--color-line)] bg-white px-4 py-3 text-sm text-[var(--color-ink-muted)]">
                            <p className="font-medium text-[var(--color-ink)]">
                              {statusMessage}
                            </p>
                            {progressSteps.length > 0 && (
                              <ul className="mt-2 space-y-1 border-t border-[var(--color-line)]/70 pt-2 text-xs">
                                {progressSteps.slice(-8).map((step, i, arr) => {
                                  const isCurrent = i === arr.length - 1;
                                  return (
                                    <li
                                      key={`${i}-${step}`}
                                      className={
                                        isCurrent
                                          ? "text-[var(--color-accent-dark)]"
                                          : "text-[var(--color-ink-muted)]"
                                      }
                                    >
                                      {isCurrent ? "● " : "○ "}
                                      {step}
                                    </li>
                                  );
                                })}
                              </ul>
                            )}
                            <span className="mt-2 block text-xs">
                              Anomaly checks can take a minute — you can keep
                              this tab open.
                            </span>
                          </div>
                        )}

                        {turn.error && (
                          <div
                            className="rounded-2xl rounded-bl-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-[var(--color-danger)]"
                            role="alert"
                          >
                            {turn.error}
                          </div>
                        )}

                        {turn.result && (
                          <>
                            <div className="rounded-2xl rounded-bl-md border border-[var(--color-line)] bg-white px-4 py-3 shadow-sm">
                              {turn.result.routed_to && (
                                <span className="mb-2 inline-flex items-center rounded-full border border-[var(--color-line)] bg-[var(--color-surface)] px-2 py-0.5 text-[11px] font-medium uppercase tracking-[0.08em] text-[var(--color-accent-dark)]">
                                  {turn.result.routed_to} agent
                                </span>
                              )}
                              <MarkdownAnswer text={turn.result.final_answer} />
                              {turn.result.unverified_claims &&
                                turn.result.unverified_claims.length > 0 && (
                                  <p className="mt-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
                                    Not found in tool output:{" "}
                                    <span className="font-medium">
                                      {turn.result.unverified_claims.join(", ")}
                                    </span>
                                    . Treat these figures as unverified.
                                  </p>
                                )}
                            </div>

                            {(turn.result.tool_steps.length > 0 ||
                              turn.result.routed_to) && (
                              <details className="rounded-xl border border-[var(--color-line)] bg-white/80 px-4 py-3">
                                <summary className="cursor-pointer text-xs font-medium text-[var(--color-ink-muted)]">
                                  What the system did
                                  {turn.result.routed_to
                                    ? ` · ${turn.result.routed_to} agent`
                                    : ""}
                                  {turn.result.tool_steps.length > 0
                                    ? ` (${turn.result.tool_steps.length})`
                                    : ""}
                                </summary>
                                <div className="mt-3">
                                  {turn.result.routed_to && (
                                    <p className="mb-2 text-xs text-[var(--color-ink-muted)]">
                                      Routed to{" "}
                                      <span className="font-medium text-[var(--color-ink)]">
                                        {turn.result.routed_to}
                                      </span>{" "}
                                      specialist
                                    </p>
                                  )}
                                  {turn.result.tool_steps.length > 0 && (
                                    <ToolSteps steps={turn.result.tool_steps} />
                                  )}
                                </div>
                              </details>
                            )}
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            <div ref={bottomRef} />
          </div>
        </main>

        {/* Bottom composer — same surface as chat */}
        <footer className="shrink-0 bg-[var(--color-surface)] px-4 pb-3 pt-0 sm:px-6 sm:pb-4">
          <div className="mx-auto max-w-3xl">
            <ChatComposer
              value={question}
              uploadedFile={uploadedFile}
              loading={loading}
              onChange={setQuestion}
              onUploadFile={setUploadedFile}
              onSubmit={sendMessage}
            />
            <p className="mt-2 text-center text-[11px] text-[var(--color-ink-muted)]">
              Attach a CSV for anomaly checks · Enter to send · Shift+Enter for a
              new line
            </p>
          </div>
        </footer>
      </div>
    </div>
  );
}
