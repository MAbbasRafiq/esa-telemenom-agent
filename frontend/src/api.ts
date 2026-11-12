/**
 * HTTP helpers for the FastAPI backend.
 *
 * The main UI uses runAgentUpload() (multipart form).
 * Health / samples are small JSON GETs.
 */
import type { AgentRunResponse, SampleInfo } from "./types";

/** API base URL (not shown in the analyst UI). */
export function getApiBase(): string {
  const fromEnv = import.meta.env.VITE_API_BASE;
  if (fromEnv && fromEnv.trim().length > 0) {
    return fromEnv.replace(/\/$/, "");
  }
  return "http://127.0.0.1:8100";
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

/**
 * POST /agent/run-upload
 *
 * Sends:
 *   - question (always)
 *   - file (optional browser upload), OR
 *   - sample_id (optional prepared sample)
 */
export async function runAgentUpload(options: {
  question: string;
  file?: File | null;
  sampleId?: string | null;
}): Promise<AgentRunResponse> {
  const form = new FormData();
  form.append("question", options.question);

  if (options.file) {
    form.append("file", options.file);
  } else if (options.sampleId) {
    form.append("sample_id", options.sampleId);
  }

  const response = await fetch(`${getApiBase()}/agent/run-upload`, {
    method: "POST",
    body: form,
    // Do not set Content-Type manually — the browser sets multipart boundaries.
  });

  if (!response.ok) {
    let message = "Something went wrong. Please try again.";
    try {
      const errBody = (await response.json()) as { detail?: string };
      if (errBody.detail) {
        message = errBody.detail;
      }
    } catch {
      // keep generic message
    }
    // Friendlier offline hint
    if (response.status === 0 || response.type === "error") {
      message = "The assistant is not available right now. Please try again later.";
    }
    throw new Error(message);
  }

  return (await response.json()) as AgentRunResponse;
}
