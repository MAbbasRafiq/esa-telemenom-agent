/**
 * Shows what the system did, in plain language.
 *
 * Default view: friendly title + short detail + truncated observation.
 * "Show technical details" reveals tool id + JSON args (for engineers).
 */
import { useState } from "react";
import { labelForTool } from "../toolLabels";
import type { ToolStep } from "../types";

type ToolStepsProps = {
  steps: ToolStep[];
};

export default function ToolSteps({ steps }: ToolStepsProps) {
  // Analyst view by default; technical dump is opt-in.
  const [showTechnical, setShowTechnical] = useState(false);

  if (steps.length === 0) {
    return (
      <p className="text-sm text-[var(--color-ink-muted)]">
        After you run a check, a short list of what the system did will appear
        here.
      </p>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex justify-end">
        <button
          type="button"
          onClick={() => setShowTechnical(!showTechnical)}
          className="text-xs text-[var(--color-ink-muted)] underline-offset-2 hover:text-[var(--color-accent-dark)] hover:underline"
        >
          {showTechnical ? "Hide technical details" : "Show technical details"}
        </button>
      </div>

      {steps.map((step, index) => {
        const label = labelForTool(step.tool);
        // Keep the observation short in the analyst view.
        const shortObs =
          step.observation.length > 400
            ? step.observation.slice(0, 400) + "…"
            : step.observation;

        return (
          <div
            key={`${step.tool}-${index}`}
            className="rounded-xl border border-[var(--color-line)] bg-[var(--color-panel)] px-4 py-3"
          >
            <p className="text-sm font-semibold text-[var(--color-ink)]">
              <span className="mr-2 text-[var(--color-ink-muted)]">
                {index + 1}.
              </span>
              {label.title}
            </p>
            <p className="mt-0.5 text-xs text-[var(--color-ink-muted)]">
              {label.detail}
            </p>

            {/* Soft preview of the tool output (not raw JSON) */}
            <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap rounded-md bg-[var(--color-surface)] p-3 text-xs leading-relaxed text-[var(--color-ink)]">
              {shortObs}
            </pre>

            {showTechnical && (
              <div className="mt-3 space-y-2 border-t border-[var(--color-line)] pt-3">
                <p className="text-[11px] font-medium text-[var(--color-ink-muted)]">
                  Tool id: <code>{step.tool}</code>
                </p>
                <pre className="overflow-x-auto rounded-md bg-[var(--color-surface)] p-3 text-[11px] text-[var(--color-ink)]">
                  {JSON.stringify(step.args, null, 2)}
                </pre>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
