//! Incremental SSE and newline-delimited framing of response bytes.
//! SSE dispatch requires a blank line; incomplete events are not flushed at EOF.
//! NDJSON permits a final unterminated line through [`NdjsonFramer::finish`].
//!
//! ```
//! use rig_core::http_client::framing::NdjsonFramer;
//!
//! let mut framer = NdjsonFramer::new();
//! assert_eq!(framer.push(b"{}\n").collect::<Vec<_>>(), vec![b"{}".to_vec()]);
//! ```

/// How a reply's bytes are split into wire frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Framing {
    /// `text/event-stream`: WHATWG event-stream grammar ([`SseFramer`]).
    Sse,
    /// Newline-delimited JSON ([`NdjsonFramer`]).
    Ndjson,
    /// The whole body is one frame.
    Whole,
}

/// One dispatched `text/event-stream` event.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SseEvent {
    /// The `event:` field, `"message"` when the stream named none.
    pub event: String,
    /// The joined `data:` fields, without the terminating newline.
    pub data: String,
    /// The stream's last event id, empty when none was ever set.
    pub id: String,
    /// Most recent parsed `retry:` value since the previous emitted event.
    /// This value does not trigger automatic reconnection.
    pub retry: Option<u64>,
}

/// The optional leading byte-order mark, which the grammar ignores.
const BOM: [u8; 3] = [0xef, 0xbb, 0xbf];

/// A push parser for the WHATWG `text/event-stream` grammar.
#[derive(Debug, Default)]
pub struct SseFramer {
    /// Bytes not yet consumed as a complete line.
    buffer: Vec<u8>,
    /// Events completed by the current `push`.
    ready: Vec<SseEvent>,
    event_type: String,
    data: String,
    last_event_id: String,
    retry: Option<u64>,
    /// How many BOM bytes have matched so far, while the prefix is undecided.
    bom_prefix: usize,
    bom_done: bool,
    /// Bytes of complete lines consumed since the last blank line.
    since_dispatch: usize,
}

impl SseFramer {
    /// A framer at the start of a stream.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one transport chunk; yields every event it completed.
    pub fn push(&mut self, chunk: &[u8]) -> std::vec::Drain<'_, SseEvent> {
        let chunk = self.strip_bom(chunk);
        self.buffer.extend_from_slice(chunk);
        while let Some((line_len, consumed)) = terminated_line(&self.buffer) {
            let line = String::from_utf8_lossy(self.buffer.get(..line_len).unwrap_or_default())
                .into_owned();
            self.buffer.drain(..consumed);
            self.since_dispatch = self.since_dispatch.saturating_add(consumed);
            self.line(&line);
        }
        self.ready.drain(..)
    }

    /// Saturating count of buffered bytes and complete lines since the last
    /// blank line. Excludes any undecided leading BOM prefix; exposes no frame data.
    pub fn pending(&self) -> usize {
        self.since_dispatch.saturating_add(self.buffer.len())
    }

    /// Consume the stream's optional leading BOM, which may itself be split
    /// across chunks. Returns the chunk with any BOM bytes removed.
    fn strip_bom<'a>(&mut self, chunk: &'a [u8]) -> &'a [u8] {
        if self.bom_done {
            return chunk;
        }
        let mut rest = chunk;
        while let Some(&byte) = rest.first() {
            if BOM.get(self.bom_prefix) != Some(&byte) {
                // Not a BOM after all: replay the bytes that matched so far.
                let matched = std::mem::take(&mut self.bom_prefix);
                self.bom_done = true;
                self.buffer
                    .extend_from_slice(BOM.get(..matched).unwrap_or_default());
                return rest;
            }
            self.bom_prefix += 1;
            rest = rest.get(1..).unwrap_or_default();
            if self.bom_prefix == BOM.len() {
                self.bom_prefix = 0;
                self.bom_done = true;
                return rest;
            }
        }
        rest
    }

    /// Process one complete line of the grammar.
    fn line(&mut self, line: &str) {
        if line.is_empty() {
            self.dispatch();
            return;
        }
        // A line starting with a colon is a comment.
        if let Some(rest) = line.strip_prefix(':') {
            let _ = rest;
            return;
        }
        let (field, value) = match line.split_once(':') {
            Some((field, value)) => (field, value.strip_prefix(' ').unwrap_or(value)),
            None => (line, ""),
        };
        match field {
            "event" => {
                self.event_type.clear();
                self.event_type.push_str(value);
            }
            "data" => {
                self.data.push_str(value);
                self.data.push('\n');
            }
            // A NUL in an id is ignored outright, per the grammar.
            "id" if !value.contains('\0') => {
                self.last_event_id.clear();
                self.last_event_id.push_str(value);
            }
            "retry" if !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit()) => {
                self.retry = value.parse().ok();
            }
            _ => {}
        }
    }

    /// A blank line ends an event. An event with no data is not dispatched:
    /// it only resets the event type, as the grammar says.
    fn dispatch(&mut self) {
        self.since_dispatch = 0;
        if self.data.is_empty() {
            self.event_type.clear();
            return;
        }
        self.data.pop();
        let event = if self.event_type.is_empty() {
            "message".to_owned()
        } else {
            std::mem::take(&mut self.event_type)
        };
        self.ready.push(SseEvent {
            event,
            data: std::mem::take(&mut self.data),
            id: self.last_event_id.clone(),
            retry: self.retry.take(),
        });
        self.event_type.clear();
    }
}

/// Splits nonempty newline-delimited payloads without validating JSON.
/// A nonempty unterminated last line is available through [`Self::finish`].
#[derive(Debug, Default)]
pub struct NdjsonFramer {
    buffer: Vec<u8>,
    ready: Vec<Vec<u8>>,
}

impl NdjsonFramer {
    /// A framer at the start of a stream.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one transport chunk; yields every line it completed. Blank lines
    /// are not frames.
    pub fn push(&mut self, chunk: &[u8]) -> std::vec::Drain<'_, Vec<u8>> {
        self.buffer.extend_from_slice(chunk);
        while let Some(pos) = self.buffer.iter().position(|byte| *byte == b'\n') {
            let mut line: Vec<u8> = self.buffer.drain(..=pos).collect();
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            if !line.is_empty() {
                self.ready.push(line);
            }
        }
        self.ready.drain(..)
    }

    /// The unterminated last line, if the stream ended with one.
    pub fn finish(&mut self) -> Option<Vec<u8>> {
        let line = std::mem::take(&mut self.buffer);
        (!line.is_empty()).then_some(line)
    }

    /// Bytes of an unterminated trailing line, for truncation diagnostics.
    pub fn pending(&self) -> usize {
        self.buffer.len()
    }
}

/// The first complete line in `buffer` as `(line length, bytes consumed)`.
///
/// Defers a trailing CR until another byte arrives so a split CRLF is consumed
/// as one terminator.
fn terminated_line(buffer: &[u8]) -> Option<(usize, usize)> {
    let pos = buffer
        .iter()
        .position(|byte| *byte == b'\n' || *byte == b'\r')?;
    match buffer.get(pos) {
        Some(b'\n') => Some((pos, pos + 1)),
        Some(b'\r') => match buffer.get(pos + 1) {
            Some(b'\n') => Some((pos, pos + 2)),
            Some(_) => Some((pos, pos + 1)),
            None => None,
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests;
