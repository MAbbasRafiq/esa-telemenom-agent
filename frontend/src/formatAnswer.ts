/**
 * Light cleanup of model answers for analysts.
 * Strips noisy citation markers; keeps the readable sentence text.
 */

/** Patterns like 【1】, [1], [1†source=model_card_3], 【1:2†source】 */
const CITATION_RE =
  /【\d+(?::\d+)?(?:†[^】]*)?】|\[\d+(?:†[^\]]*)?\]/g;

export function formatAnswer(text: string): string {
  if (!text) {
    return "";
  }
  return text
    .replace(CITATION_RE, "")
    .replace(/\s+([,.;:!?])/g, "$1")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/ {2,}/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}
