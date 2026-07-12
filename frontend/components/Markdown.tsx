"use client";

import type { Element, ElementContent, Root, Text } from "hast";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const CITE_PATTERN = /\[(\d{1,3})\]/g;

/** Rehype plugin: turn inline [n] markers into <sup data-cite="n"> elements. */
function rehypeCitationRefs() {
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

interface MarkdownProps {
  content: string;
  onCitationClick?: (index: number) => void;
}

export default function Markdown({ content, onCitationClick }: MarkdownProps) {
  return (
    <div className="markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeCitationRefs]}
        components={{
          a: (props) => (
            <a {...props} target="_blank" rel="noopener noreferrer" />
          ),
          sup: (props) => {
            const cite = (props as { "data-cite"?: string })["data-cite"];
            if (!cite) return <sup {...props} />;
            return (
              <sup>
                <button
                  type="button"
                  onClick={() => onCitationClick?.(Number(cite))}
                  className="mx-0.5 rounded bg-blue-50 px-1 py-px text-[11px] font-semibold text-blue-700 hover:bg-blue-100"
                  title={`Citation ${cite}`}
                >
                  {cite}
                </button>
              </sup>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
