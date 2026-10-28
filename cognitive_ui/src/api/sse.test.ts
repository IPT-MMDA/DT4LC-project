import { describe, expect, it } from 'vitest';
import { createSSEParser, readSSE, type SSEEvent } from './sse';

function parse(chunks: string[]): SSEEvent[] {
  const events: SSEEvent[] = [];
  const parser = createSSEParser((e) => events.push(e));
  chunks.forEach((c) => parser.feed(c));
  parser.end();
  return events;
}

// The server's frame format (server/job_events.py sse_event_frame).
const FRAMES =
  'id: 1\nevent: reply_delta\ndata: {"text":"Hello "}\n\n' +
  ': keep-alive\n\n' +
  'id: 2\nevent: reply_done\ndata: {"text":"Hello there"}\n\n';

describe('createSSEParser', () => {
  it('reads id, event and data, and skips keep-alive comments', () => {
    expect(parse([FRAMES])).toEqual([
      { id: '1', event: 'reply_delta', data: '{"text":"Hello "}' },
      { id: '2', event: 'reply_done', data: '{"text":"Hello there"}' },
    ]);
  });

  it('gives the same events however the stream is split, even one character at a time', () => {
    const whole = parse([FRAMES]);
    expect(parse([FRAMES.slice(0, 7), FRAMES.slice(7, 40), FRAMES.slice(40)])).toEqual(whole);
    expect(parse(FRAMES.split(''))).toEqual(whole);
  });

  it('handles CRLF and CR line ends, also when \\r\\n is split across chunks', () => {
    const crlf = FRAMES.replace(/\n/g, '\r\n');
    expect(parse([crlf])).toEqual(parse([FRAMES]));
    const at = crlf.indexOf('\r\n') + 1;
    expect(parse([crlf.slice(0, at), crlf.slice(at)])).toEqual(parse([FRAMES]));
    expect(parse([FRAMES.replace(/\n/g, '\r')])).toEqual(parse([FRAMES]));
  });

  it('joins data lines, defaults the event to "message", keeps the last id, and strips one space', () => {
    expect(parse(['data: a\ndata:b\ndata:  c\n\n', 'id: 7\ndata: x\n\n', 'data: y\n\n'])).toEqual([
      { id: '', event: 'message', data: 'a\nb\n c' },
      { id: '7', event: 'message', data: 'x' },
      { id: '7', event: 'message', data: 'y' },
    ]);
  });

  it('drops frames without data, an unterminated last frame, and a leading BOM', () => {
    expect(parse(['﻿event: ping\n\n', 'id: 3\nevent: done\ndata: {}\n\n', 'event: half\ndata: {'])).toEqual([
      { id: '3', event: 'done', data: '{}' },
    ]);
  });
});

describe('readSSE', () => {
  it('decodes a byte stream with multi-byte characters split between chunks', async () => {
    const bytes = new TextEncoder().encode('event: reply_delta\ndata: {"text":"≤20 % · 412 km²"}\n\n');
    const split = bytes.indexOf(0xe2) + 1;  // inside "≤"
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(bytes.slice(0, split));
        controller.enqueue(bytes.slice(split));
        controller.close();
      },
    });
    const events: SSEEvent[] = [];
    for await (const e of readSSE(body)) events.push(e);
    expect(events).toEqual([{ id: '', event: 'reply_delta', data: '{"text":"≤20 % · 412 km²"}' }]);
  });
});
