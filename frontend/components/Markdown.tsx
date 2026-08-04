"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { rehypeCitationRefs } from "@/lib/citationMarkers";

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
