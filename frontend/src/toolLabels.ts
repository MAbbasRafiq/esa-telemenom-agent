/**
 * Map internal tool names to plain language for satellite analysts.
 *
 * Technical tool ids (retrieve_knowledge, …) stay in the API response;
 * the UI shows these labels instead.
 */

const LABELS: Record<string, { title: string; detail: string }> = {
  retrieve_knowledge: {
    title: "Looked up system facts",
    detail: "Searched the project knowledge base for an answer.",
  },
  score_telemetry: {
    title: "Ran anomaly detection",
    detail: "Scored the selected telemetry window with the Telemanom model.",
  },
  compare_alarms_to_labels: {
    title: "Compared detections to known events",
    detail: "Checked model alarms against ground-truth labels on the same window.",
  },
};

/**
 * Return a friendly title + detail for a tool id.
 * Unknown tools fall back to a soft generic label (still readable).
 */
export function labelForTool(toolName: string): { title: string; detail: string } {
  const known = LABELS[toolName];
  if (known) {
    return known;
  }
  return {
    title: "Completed a system step",
    detail: `Internal step: ${toolName}`,
  };
}
