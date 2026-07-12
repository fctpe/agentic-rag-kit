import assert from "node:assert/strict";
import { test } from "node:test";

import { parseEventBlock, parseSSEStream } from "../lib/sse.ts";

const encoder = new TextEncoder();

function streamOf(chunks) {
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
}

async function collect(stream) {
  const events = [];
  for await (const event of parseSSEStream(stream)) events.push(event);
  return events;
}

test("parses a complete event block", () => {
  assert.deepEqual(parseEventBlock('event: token\ndata: {"text": "hi"}'), {
    event: "token",
    data: '{"text": "hi"}',
  });
});

test("defaults event name to message and joins multi-line data", () => {
  assert.deepEqual(parseEventBlock("data: line1\ndata: line2"), {
    event: "message",
    data: "line1\nline2",
  });
});

test("ignores comment-only blocks", () => {
  assert.equal(parseEventBlock(": keep-alive"), null);
});

test("parses the backend event sequence from one chunk", async () => {
  const events = await collect(
    streamOf([
      'event: token\ndata: {"text": "The AI Act"}\n\n' +
        'event: citations\ndata: {"citations": [{"index": 1}]}\n\n' +
        'event: grounding\ndata: {"grounded": true, "issues": []}\n\n' +
        'event: done\ndata: {"thread_id": "t1", "content": "The AI Act"}\n\n',
    ]),
  );
  assert.deepEqual(
    events.map((e) => e.event),
    ["token", "citations", "grounding", "done"],
  );
  assert.deepEqual(JSON.parse(events[3].data), {
    thread_id: "t1",
    content: "The AI Act",
  });
});

test("handles chunk boundaries mid-line and mid-event", async () => {
  const events = await collect(
    streamOf([
      "event: tok",
      'en\ndata: {"te',
      'xt": "a"}\n',
      '\nevent: done\ndata: {"thread_id": "t2", "content": "a"}\n\n',
    ]),
  );
  assert.deepEqual(events, [
    { event: "token", data: '{"text": "a"}' },
    { event: "done", data: '{"thread_id": "t2", "content": "a"}' },
  ]);
});

test("handles CRLF line endings", async () => {
  const events = await collect(
    streamOf(['event: token\r\ndata: {"text": "b"}\r\n\r\n']),
  );
  assert.deepEqual(events, [{ event: "token", data: '{"text": "b"}' }]);
});

test("handles multi-byte characters split across chunks", async () => {
  const bytes = encoder.encode('event: token\ndata: {"text": "§ Art. 5"}\n\n');
  const mid = 22; // splits inside the two-byte § sequence
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(bytes.slice(0, mid));
      controller.enqueue(bytes.slice(mid));
      controller.close();
    },
  });
  const events = await collect(stream);
  assert.equal(JSON.parse(events[0].data).text, "§ Art. 5");
});

test("emits a trailing event with no final blank line", async () => {
  const events = await collect(
    streamOf(['event: error\ndata: {"message": "boom"}']),
  );
  assert.deepEqual(events, [
    { event: "error", data: '{"message": "boom"}' },
  ]);
});
