//! Push framers: bytes in, frames out, no I/O.
//!
//! A [`Wire`](crate::wire::Wire) names how its reply is framed; whoever owns
//! the socket feeds chunks to the framer and hands the frames to the wire's
//! decoder. Chunk boundaries are arbitrary — a frame may arrive over any
//! number of `push` calls.

use bytes::Bytes;
use std::time::Duration;

/// How a reply body is cut into frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Framing {
    /// `text/event-stream`; each `data:` payload is one frame.
    Sse,
    /// Newline-delimited JSON; each line is one frame.
    Ndjson,
    /// The whole body is one payload (a unary reply).
    Whole,
}

/// One event of a `text/event-stream` body, per the WHATWG grammar.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SseEvent {
    /// The `event:` field, or `"message"` when the stream gave none.
    pub event: String,
    /// The `data:` lines, joined by `\n`.
    pub data: String,
    /// The last `id:` seen on the stream, when there was one.
    pub id: Option<String>,
    /// The `retry:` field, when the event carried one.
    pub retry: Option<Duration>,
}

/// A push parser for `text/event-stream`.
///
/// Line terminators are `\r\n`, `\n` or `\r`; a leading BOM is skipped;
/// `:`-prefixed lines are comments; an event is dispatched on a blank line
/// and only if it carries data. An event the body ends in the middle of is
/// never dispatched — the grammar dispatches on the blank line only, and a
/// recorded body that stops after a `data:` line really did lose that event.
/// Its bytes are [`SseFramer::pending`], for truncation diagnostics.
#[derive(Debug, Default)]
pub struct SseFramer {
    /// Bytes of the current, unterminated line.
    line: Vec<u8>,
    /// A `\r` ended the previous chunk; a `\n` at the head of the next one
    /// belongs to it.
    pending_cr: bool,
    /// Whether the BOM check at stream start is still pending.
    at_start: bool,
    event_type: String,
    data: String,
    has_data: bool,
    last_id: Option<String>,
    retry: Option<Duration>,
    ready: Vec<SseEvent>,
}

impl SseFramer {
    /// A framer at the start of a body.
    pub fn new() -> Self {
        Self {
            at_start: true,
            ..Self::default()
        }
    }

    /// Feed one chunk; yields every event completed by it, in order.
    pub fn push(&mut self, chunk: &[u8]) -> impl Iterator<Item = SseEvent> + '_ {
        let mut rest = chunk;
        if self.at_start && !rest.is_empty() {
            // The BOM may itself be split across chunks; only the common
            // whole-BOM case is handled, and a split BOM decodes as text.
            if let Some(after) = rest.strip_prefix(&[0xEF, 0xBB, 0xBF]) {
                rest = after;
            }
            self.at_start = false;
        }
        if std::mem::take(&mut self.pending_cr)
            && let Some(after) = rest.strip_prefix(b"\n")
        {
            rest = after;
        }
        while let Some(at) = rest.iter().position(|byte| matches!(byte, b'\n' | b'\r')) {
            let (head, tail) = rest.split_at(at);
            self.line.extend_from_slice(head);
            self.end_line();
            let is_cr = tail.first() == Some(&b'\r');
            let mut tail = tail.get(1..).unwrap_or_default();
            if is_cr {
                if tail.is_empty() {
                    self.pending_cr = true;
                } else if let Some(after) = tail.strip_prefix(b"\n") {
                    tail = after;
                }
            }
            rest = tail;
        }
        self.line.extend_from_slice(rest);
        self.ready.drain(..)
    }

    /// Bytes of the line in progress, for truncation diagnostics.
    pub fn pending(&self) -> usize {
        self.line.len()
    }

    fn end_line(&mut self) {
        let line = std::mem::take(&mut self.line);
        if line.is_empty() {
            if let Some(event) = self.dispatch() {
                self.ready.push(event);
            }
            return;
        }
        if line.first() == Some(&b':') {
            return;
        }
        let line = String::from_utf8_lossy(&line);
        let (field, value) = match line.find(':') {
            Some(at) => {
                let (k, v) = line.split_at(at);
                let value = v.get(1..).unwrap_or("");
                (k, value.strip_prefix(' ').unwrap_or(value))
            }
            None => (line.as_ref(), ""),
        };
        match field {
            "event" => self.event_type = value.to_owned(),
            "data" => {
                if self.has_data {
                    self.data.push('\n');
                }
                self.data.push_str(value);
                self.has_data = true;
            }
            "id" if !value.contains('\0') => self.last_id = Some(value.to_owned()),
            "retry" if !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()) => {
                self.retry = value.parse().ok().map(Duration::from_millis);
            }
            _ => {}
        }
    }

    fn dispatch(&mut self) -> Option<SseEvent> {
        let event_type = std::mem::take(&mut self.event_type);
        let data = std::mem::take(&mut self.data);
        let retry = self.retry.take();
        if !std::mem::take(&mut self.has_data) {
            return None;
        }
        Some(SseEvent {
            event: if event_type.is_empty() {
                "message".to_owned()
            } else {
                event_type
            },
            data,
            id: self.last_id.clone(),
            retry,
        })
    }
}

/// A push parser for newline-delimited payloads: each `\n`-terminated line
/// (a trailing `\r` trimmed) is one frame; blank lines are skipped.
#[derive(Debug, Default)]
pub struct NdjsonFramer {
    line: Vec<u8>,
    ready: Vec<Bytes>,
}

impl NdjsonFramer {
    /// A framer at the start of a body.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one chunk; yields every line completed by it.
    pub fn push(&mut self, chunk: &[u8]) -> impl Iterator<Item = Bytes> + '_ {
        let mut rest = chunk;
        while let Some(at) = rest.iter().position(|byte| *byte == b'\n') {
            let (head, tail) = rest.split_at(at);
            self.line.extend_from_slice(head);
            self.end_line();
            rest = tail.get(1..).unwrap_or_default();
        }
        self.line.extend_from_slice(rest);
        self.ready.drain(..)
    }

    /// The body ended: the unterminated last line, if any. Unlike SSE, a
    /// final line that no newline terminated is a frame — the transport
    /// closed, the payload was not cut out of the grammar's dispatch.
    pub fn finish(&mut self) -> Option<Bytes> {
        if self.line.is_empty() {
            return None;
        }
        self.end_line();
        self.ready.pop()
    }

    fn end_line(&mut self) {
        let mut line = std::mem::take(&mut self.line);
        if line.last() == Some(&b'\r') {
            line.pop();
        }
        if !line.is_empty() {
            self.ready.push(Bytes::from(line));
        }
    }
}

#[cfg(test)]
mod tests;
