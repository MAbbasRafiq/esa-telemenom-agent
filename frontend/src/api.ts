/**
 * HTTP helpers for the FastAPI backend.
 *
 * UI uses startAgentJob() + getAgentJob() (async; scoring can take a minute).
 */
import type {
  AgentRunResponse,
  JobAccepted,
  JobStatus,
  SampleInfo,
  SessionDetail,
  SessionInfo,
} from "./types";

/** API base URL (not shown in the analyst UI). */
export function getApiBase(): string {
  const fromEnv = import.meta.env.VITE_API_BASE;
  if (fromEnv && fromEnv.trim().length > 0) {
    return fromEnv.replace(/\/$/, "");
  }
  return "http://127.0.0.1:8100";
}

async function readErrorMessage(response: Response): Promise<string> {
  let message = "Something went wrong. Please try again.";
  try {
    const errBody = (await response.json()) as { detail?: string };
    if (errBody.detail) {
      message = errBody.detail;
    }
  } catch {
    // keep generic message
  }
  if (response.status === 0 || response.type === "error") {
    message =
      "The assistant is not available right now. Please try again later.";
  }
  return message;
}

/** GET /health — used only to fail early with a friendly message. */
export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${getApiBase()}/health`);
    if (!response.ok) {
      return false;
    }
    const data = (await response.json()) as { status?: string };
    return data.status === "ok";
  } catch {
    return false;
  }
}

/** GET /samples — prepared demo windows with friendly titles. */
export async function fetchSamples(): Promise<SampleInfo[]> {
  const response = await fetch(`${getApiBase()}/samples`);
  if (!response.ok) {
    return [];
  }
  const data = (await response.json()) as { samples?: SampleInfo[] };
  return data.samples ?? [];
}

/** GET /sessions — past chats for the sidebar (newest first). */
export async function fetchSessions(): Promise<SessionInfo[]> {
  try {
    const response = await fetch(`${getApiBase()}/sessions`);
    if (!response.ok) {
      return [];
    }
    const data = (await response.json()) as { sessions?: SessionInfo[] };
    return data.sessions ?? [];
  } catch {
    return [];
  }
}

/** GET /sessions/{id} — reopen a past chat with its questions and answers. */
export async function fetchSessionDetail(
  threadId: string,
): Promise<SessionDetail> {
  const response = await fetch(`${getApiBase()}/sessions/${threadId}`);
  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }
  return (await response.json()) as SessionDetail;
}

/** DELETE /sessions/{id} — forget a chat and its history. */
export async function deleteSession(threadId: string): Promise<void> {
  await fetch(`${getApiBase()}/sessions/${threadId}`, { method: "DELETE" });
}

function buildUploadForm(options: {
  question: string;
  file?: File | null;
  sampleId?: string | null;
  threadId?: string | null;
}): FormData {
  const form = new FormData();
  form.append("question", options.question);

  if (options.file) {
    form.append("file", options.file);
  } else if (options.sampleId) {
    form.append("sample_id", options.sampleId);
  }

  if (options.threadId) {
    form.append("thread_id", options.threadId);
  }
  return form;
}

/** POST /agent/jobs — queue a run; returns job_id immediately. */
export async function startAgentJob(options: {
  question: string;
  file?: File | null;
  sampleId?: string | null;
  threadId?: string | null;
}): Promise<JobAccepted> {
  const response = await fetch(`${getApiBase()}/agent/jobs`, {
    method: "POST",
    body: buildUploadForm(options),
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }
  return (await response.json()) as JobAccepted;
}

/** GET /agent/jobs/{id} — poll until done / error. */
export async function getAgentJob(jobId: string): Promise<JobStatus> {
  const response = await fetch(`${getApiBase()}/agent/jobs/${jobId}`);
  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }
  return (await response.json()) as JobStatus;
}

/**
 * Start a job and poll until finished.
 * Calls onStatus with live status_message while waiting.
 */
export async function runAgentJob(options: {
  question: string;
  file?: File | null;
  sampleId?: string | null;
  threadId?: string | null;
  onStatus?: (message: string) => void;
  onProgressSteps?: (steps: string[]) => void;
  pollMs?: number;
}): Promise<AgentRunResponse> {
  const accepted = await startAgentJob(options);
  options.onStatus?.(accepted.status_message || "Queued…");
  options.onProgressSteps?.([accepted.status_message || "Queued…"]);

  const pollMs = options.pollMs ?? 500;
  const deadline = Date.now() + 15 * 60 * 1000; // 15 min safety cap
  let lastStatus = accepted.status_message || "Queued…";

  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, pollMs));
    let status: JobStatus;
    try {
      status = await getAgentJob(accepted.job_id);
    } catch (err) {
      // Job may have been purged after TTL — give a clear message.
      const msg = err instanceof Error ? err.message : "Lost contact with the job.";
      throw new Error(
        msg.includes("Unknown job")
          ? "This run expired on the server. Please send the question again."
          : msg,
      );
    }
    lastStatus = status.status_message || status.status;
    options.onStatus?.(lastStatus);
    if (status.progress_steps && status.progress_steps.length > 0) {
      options.onProgressSteps?.(status.progress_steps);
    }

    if (status.status === "done") {
      if (!status.result) {
        throw new Error("Job finished but returned no result.");
      }
      return {
        ...status.result,
        thread_id: status.result.thread_id || status.thread_id,
      };
    }
    if (status.status === "error") {
      throw new Error(status.error || "Assistant failed.");
    }
  }

  throw new Error(
    `Timed out while: ${lastStatus}. Try a smaller CSV window, or send again.`,
  );
}
