import type { Element, ElementContent, Root, Text } from "hast";

/**
 * The one shape that becomes a link to the source panel.
 *
 * This regex is the client half of the citation contract; the server half is
 * MARKER_PATTERN in backend/app/agent/markers.py, and the two are deliberately
 * byte-identical. Widening it is not a fix for anything: `[4, Art. 4(5)]` is
 * not a source id, it is a source id and a reference the model welded together,
 * and matching it here would turn the whole bracket into a button labelled with
 * text that is not an index. The backend resolves those into real markers
 * before the text is ever sent — see resolve_citations in app/agent/graph.py.
 *
 * `\d{1,3}` bounds the index; the backend removes any marker whose index was
 * never retrieved, so a marker that arrives here resolves to a card.
 */
export const CITE_PATTERN = /\[(\d{1,3})\]/g;

/** Rehype plugin: turn inline [n] markers into <sup data-cite="n"> elements. */
export function rehypeCitationRefs() {
  return (tree: Root) => visit(tree);
}

function visit(node: Root | Element): void {
  if (
    node.type === "element" &&
    (node.tagName === "code" || node.tagName === "pre" || node.tagName === "a")
  ) {
    return;
  }
  const next: ElementContent[] = [];
  let changed = false;
  for (const child of node.children as ElementContent[]) {
    if (child.type === "text" && CITE_PATTERN.test(child.value)) {
      changed = true;
      next.push(...splitTextNode(child));
    } else {
      if (child.type === "element") visit(child);
      next.push(child);
    }
    CITE_PATTERN.lastIndex = 0;
  }
  if (changed) node.children = next as typeof node.children;
}

function splitTextNode(child: Text): ElementContent[] {
  const parts: ElementContent[] = [];
  let cursor = 0;
  CITE_PATTERN.lastIndex = 0;
  for (const match of child.value.matchAll(CITE_PATTERN)) {
    if (match.index > cursor) {
      parts.push({ type: "text", value: child.value.slice(cursor, match.index) });
    }
    parts.push({
      type: "element",
      tagName: "sup",
      properties: { dataCite: match[1] },
      children: [{ type: "text", value: match[1] }],
    });
    cursor = match.index + match[0].length;
  }
  if (cursor < child.value.length) {
    parts.push({ type: "text", value: child.value.slice(cursor) });
  }
  return parts;
}
