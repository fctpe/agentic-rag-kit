/**
 * The client half of the citation contract.
 *
 * `CITE_PATTERN` was the only definition of "what counts as a source marker"
 * anywhere in the repo, and it had no test at all — the sole frontend test file
 * parsed SSE. So the shape the backend now guarantees was asserted in exactly
 * one place: a 38-question eval that costs money to run.
 *
 * These run the real `rehypeCitationRefs` from `lib/citationMarkers.ts` through
 * the real react-markdown pipeline the app renders with, and count the elements
 * that actually become clickable buttons.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import remarkGfm from "remark-gfm";
import remarkParse from "remark-parse";
import remarkRehype from "remark-rehype";
import { unified } from "unified";

import { CITE_PATTERN, rehypeCitationRefs } from "../lib/citationMarkers.ts";

/**
 * The same pipeline react-markdown runs: remark-parse -> remark-gfm ->
 * remark-rehype -> our rehype plugin. The hast tree it produces is what
 * react-markdown hands to the renderer, so counting `sup[data-cite]` nodes in
 * it counts exactly the buttons Markdown.tsx will mount.
 */
const processor = unified()
  .use(remarkParse)
  .use(remarkGfm)
  .use(remarkRehype)
  .use(rehypeCitationRefs);

function render(markdown) {
  return processor.runSync(processor.parse(markdown));
}

function walk(node, visit) {
  visit(node);
  for (const child of node.children ?? []) walk(child, visit);
}

/** The indices this markdown turns into source-panel links. */
function linkedIndices(markdown) {
  const found = [];
  walk(render(markdown), (node) => {
    if (node.type === "element" && node.tagName === "sup" && node.properties?.dataCite) {
      found.push(Number(node.properties.dataCite));
    }
  });
  return found;
}

/** Text content of every `code` element, in order. */
function codeSpans(markdown) {
  const found = [];
  walk(render(markdown), (node) => {
    if (node.type === "element" && node.tagName === "code") {
      let text = "";
      walk(node, (inner) => {
        if (inner.type === "text") text += inner.value;
      });
      found.push(text);
    }
  });
  return found;
}

test("a plain marker becomes one source link", () => {
  assert.deepEqual(
    linkedIndices("Risk management runs across the lifecycle (AI Act, Art. 9(2)) [3]."),
    [3],
  );
});

test("a reference welded into the bracket links nothing", () => {
  // This is the defect. The backend rewrites it to `(Art. 4(5)) [4]` before the
  // text is ever sent; if it ever stops doing that, the marker is dead on
  // arrival, and this asserts exactly that consequence.
  assert.deepEqual(
    linkedIndices("…may not be attributed to an identifiable natural person [4, Art. 4(5)]."),
    [],
  );
});

test("a multi-source bracket links nothing", () => {
  assert.deepEqual(linkedIndices("Both instruments apply here [7, 8]."), []);
});

test("a sub-point suffix links nothing", () => {
  assert.deepEqual(linkedIndices("Explicit consent is one basis [2(a)]."), []);
});

test("prose-only citation links nothing", () => {
  assert.deepEqual(linkedIndices("As set out in (AI Act, Art. 50(1)), providers must inform users."), []);
});

test("widening the pattern would NOT rescue the merged shapes", () => {
  // The negative control that matters, because "just widen the regex" is the
  // tempting fix. A pattern loose enough to match `[4, Art. 4(5)]` captures the
  // index but has no way to know the reference is not part of it, and it still
  // cannot decide anything at all about `[7, 8]`: two sources, one bracket, one
  // possible data-cite. Widening trades a visible failure for a wrong link.
  const widened = /\[(\d{1,3})[^\]]*\]/g;
  const merged = "Both instruments apply here [7, 8].";
  const matches = [...merged.matchAll(widened)];
  assert.equal(matches.length, 1);
  assert.equal(matches[0][1], "7"); // source 8 is silently lost
  assert.equal(matches[0][0], "[7, 8]");
});

test("the pattern is anchored to the shape the backend guarantees", () => {
  // Pinned literally: app/agent/markers.py MARKER_PATTERN is the same regex, and
  // backend/tests/test_citation_markers.py reads this file to check it.
  assert.equal(CITE_PATTERN.source, "\\[(\\d{1,3})\\]");
  assert.equal(CITE_PATTERN.flags, "g");
});

test("markers inside code are left alone", () => {
  const markdown = "Use `[1, 2]` as the payload, source [2].";
  assert.deepEqual(codeSpans(markdown), ["[1, 2]"]);
  assert.deepEqual(linkedIndices(markdown), [2]);
});

test("a lastIndex leak cannot make the second marker in a paragraph disappear", () => {
  // CITE_PATTERN is a module-level /g regex shared by `test` and `matchAll`;
  // a stale `lastIndex` would silently drop markers. Two in one text node, then
  // the same text again, is what catches it.
  const text = "First [1] and second [2].";
  assert.deepEqual(linkedIndices(text), [1, 2]);
  assert.deepEqual(linkedIndices(text), [1, 2]);
});
