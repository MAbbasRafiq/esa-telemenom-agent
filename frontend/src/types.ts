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
};

/** One prepared sample from GET /samples (analyst-friendly fields only). */
export type SampleInfo = {
  id: string;
  title: string;
  description: string;
  filename: string;
  exists?: boolean;
};
