// Server-sent events over fetch: EventSource cannot POST (POST /v1/chat) or send
// headers, so the Workspace reads `text/event-stream` bodies with this parser.
// It follows the WHATWG "event stream interpretation" rules: lines end in \n,
// \r\n or \r; `:` lines are comments (the server's `: keep-alive`); `data` lines
// join with \n; a blank line dispatches; `id` persists across events.

export interface SSEEvent {
  /** `event:` field; "message" when the frame has none. */
  event: string;
  /** `data:` lines joined with "\n" (parse JSON yourself). */
  data: string;
  /** The last `id:` seen so far in the stream (resume with `?after=` or Last-Event-ID), or "". */
  id: string;
}

export interface SSEParser {
  /** Feed decoded text; complete events are passed to `onEvent` as they end. */
  feed(text: string): void;
  /** End of stream: a trailing frame without its blank line is dropped, as the spec says. */
  end(): void;
}

export function createSSEParser(onEvent: (event: SSEEvent) => void): SSEParser {
  let buffer = '';
  let first = true;
  let pendingCR = false;  // a chunk ended in \r: a following \n belongs to the same line end
  let eventType = '';
  let data: string[] = [];
  let lastId = '';

  function dispatch() {
    if (data.length > 0) onEvent({ event: eventType || 'message', data: data.join('\n'), id: lastId });
    eventType = '';
    data = [];
  }

  function line(text: string) {
    if (text === '') return dispatch();
    if (text.startsWith(':')) return;  // comment
    const colon = text.indexOf(':');
    const field = colon === -1 ? text : text.slice(0, colon);
    let value = colon === -1 ? '' : text.slice(colon + 1);
    if (value.startsWith(' ')) value = value.slice(1);
    if (field === 'event') eventType = value;
    else if (field === 'data') data.push(value);
    else if (field === 'id' && !value.includes('\0')) lastId = value;
    // `retry` and unknown fields are ignored.
  }

  return {
    feed(text) {
      if (first && text.length > 0) {
        first = false;
        if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);
      }
      if (pendingCR && text.startsWith('\n')) text = text.slice(1);
      pendingCR = false;
      buffer += text;
      let start = 0;
      for (let i = 0; i < buffer.length; i++) {
        const c = buffer[i];
        if (c !== '\n' && c !== '\r') continue;
        line(buffer.slice(start, i));
        if (c === '\r') {
          if (i + 1 === buffer.length) pendingCR = true;
          else if (buffer[i + 1] === '\n') i++;
        }
        start = i + 1;
      }
      buffer = buffer.slice(start);
    },
    end() {
      buffer = '';
      eventType = '';
      data = [];
    },
  };
}

/** Read an SSE response body to its end, yielding each event. */
export async function* readSSE(body: ReadableStream<Uint8Array>): AsyncGenerator<SSEEvent> {
  const queue: SSEEvent[] = [];
  const parser = createSSEParser((e) => queue.push(e));
  const decoder = new TextDecoder();
  const reader = body.getReader();
  let finished = false;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parser.feed(decoder.decode(value, { stream: true }));
      while (queue.length > 0) yield queue.shift() as SSEEvent;
    }
    finished = true;
    parser.feed(decoder.decode());
    parser.end();
    while (queue.length > 0) yield queue.shift() as SSEEvent;
  } finally {
    // The caller stopped early (break, or an exception): close the connection.
    if (!finished) await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
