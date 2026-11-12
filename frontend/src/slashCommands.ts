/**
 * Slash shortcuts for the question box.
 * Type "/" to open a menu of sample questions (analyst-friendly labels).
 */
export type SlashCommand = {
  /** Short id */
  id: string;
  /** Text after the slash, e.g. "window" for /window */
  slash: string;
  /** Short label in the menu */
  label: string;
  /** Full question inserted into the box */
  question: string;
};

export const SLASH_COMMANDS: SlashCommand[] = [
  {
    id: "window",
    slash: "window",
    label: "Look-back window size",
    question: "What is the look-back window size?",
  },
  {
    id: "channels",
    slash: "channels",
    label: "Input and target channels",
    question:
      "How many input and target channels does the model use?",
  },
  {
    id: "anomalies",
    slash: "anomalies",
    label: "Check for anomalies",
    question:
      "Score the attached telemetry and summarize which channels show alarms.",
  },
  {
    id: "compare",
    slash: "compare",
    label: "Compare to known events",
    question:
      "Score the attached telemetry and compare alarms to ground-truth labels. Summarize how well detections match known events.",
  },
];

/**
 * If the text ends with a slash command fragment (e.g. "/win"),
 * return the filter string (without "/"). Otherwise null.
 */
export function activeSlashFilter(text: string): string | null {
  // Match a "/" at the start of the last "word" (after newline or space, or whole string).
  const match = text.match(/(?:^|\s)\/([a-zA-Z0-9_-]*)$/);
  if (!match) {
    return null;
  }
  return match[1].toLowerCase();
}

/** Filter commands whose slash starts with the typed fragment. */
export function filterSlashCommands(fragment: string): SlashCommand[] {
  if (!fragment) {
    return SLASH_COMMANDS;
  }
  return SLASH_COMMANDS.filter((cmd) =>
    cmd.slash.startsWith(fragment) || cmd.label.toLowerCase().includes(fragment),
  );
}

/**
 * Replace the trailing /fragment in text with the chosen question.
 */
export function applySlashCommand(text: string, cmd: SlashCommand): string {
  return text.replace(/(?:^|\s)\/[a-zA-Z0-9_-]*$/, (match) => {
    // Keep the leading space if there was one; insert question after it.
    if (match.startsWith(" ") || match.startsWith("\n")) {
      return match[0] + cmd.question;
    }
    return cmd.question;
  });
}
