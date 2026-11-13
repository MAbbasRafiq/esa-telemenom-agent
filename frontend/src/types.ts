/**
 * Types shared by the React UI.
 * Mirrors the FastAPI response; upload uses multipart (not this request type).
 */

export type ToolStep = {
  tool: string;
  args: Record<string, unknown>;
  observation: string;
};

export type AgentRunResponse = {
  provider: string;
  model: string;
  tool_steps: ToolStep[];
  final_answer: string;
  /** LangGraph session id — reuse on follow-up asks */
  thread_id?: string | null;
  /** Multi-agent route: knowledge | score | compare */
  routed_to?: string | null;
  /** Numbers in the answer that no tool output backs */
  unverified_claims?: string[];
};

export type JobAccepted = {
  job_id: string;
  thread_id: string;
  status: string;
  status_message: string;
};

export type JobStatus = {
  job_id: string;
  thread_id: string;
  status: "queued" | "running" | "done" | "error" | string;
  status_message: string;
  /** Live trail of steps (load model → inference → reason → …) */
  progress_steps?: string[];
  created_at: string;
  updated_at: string;
  result?: AgentRunResponse | null;
  error?: string | null;
};

/** One past chat from GET /sessions. */
export type SessionInfo = {
  thread_id: string;
  title?: string | null;
  csv_path?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
};

/** One replayed question/answer pair from GET /sessions/{id}. */
export type SessionTurn = {
  question: string;
  answer: string;
  tools: string[];
};

export type SessionDetail = SessionInfo & {
  turns: SessionTurn[];
};

/** One prepared sample from GET /samples (analyst-friendly fields only). */
export type SampleInfo = {
  id: string;
  title: string;
  description: string;
  filename: string;
  exists?: boolean;
};
