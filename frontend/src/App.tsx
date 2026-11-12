/**
 * ChatGPT-style layout:
 *   - Left sidebar (session, quick asks, attach status)
 *   - Clean message area
 *   - Composer pinned at the bottom (attach + text + send)
 */
import { useEffect, useRef, useState } from "react";
import { checkHealth, runAgentUpload } from "./api";
import ChatComposer from "./components/ChatComposer";
import Sidebar from "./components/Sidebar";
import ToolSteps from "./components/ToolSteps";
import type { AgentRunResponse } from "./types";

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
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);

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

  function newChat() {
    if (loading) {
      return;
    }
    setTurns([]);
    setQuestion("");
    setUploadedFile(null);
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

      const data = await runAgentUpload({
        question: trimmed,
        file: uploadedFile,
        sampleId: null,
      });

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
    }
  }

  const hasMessages = turns.length > 0;

  return (
    <div className="flex h-[100dvh] overflow-hidden">
      <Sidebar
        turns={turns}
        uploadedFileName={uploadedFile?.name ?? null}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onNewChat={newChat}
        onSelectTurn={selectTurn}
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
                            Working…
                            <span className="mt-1 block text-xs">
                              Anomaly checks can take a minute.
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
                            <div className="rounded-2xl rounded-bl-md border border-[var(--color-line)] bg-white px-4 py-3 text-[15px] leading-relaxed text-[var(--color-ink)] shadow-sm">
                              <p className="whitespace-pre-wrap">
                                {turn.result.final_answer}
                              </p>
                            </div>

                            {turn.result.tool_steps.length > 0 && (
                              <details className="rounded-xl border border-[var(--color-line)] bg-white/80 px-4 py-3">
                                <summary className="cursor-pointer text-xs font-medium text-[var(--color-ink-muted)]">
                                  What the system did (
                                  {turn.result.tool_steps.length})
                                </summary>
                                <div className="mt-3">
                                  <ToolSteps steps={turn.result.tool_steps} />
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
