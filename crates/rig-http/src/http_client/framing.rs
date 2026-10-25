//! Incremental SSE and newline-delimited framing of response bytes.
//! SSE dispatch requires a blank line; incomplete events are not flushed at EOF.
//! NDJSON permits a final unterminated line through [`NdjsonFramer::finish`].
//!
//! ```
//! use rig_http::http_client::framing::NdjsonFramer;
//!
//! let mut framer = NdjsonFramer::new();
//! assert_eq!(framer.push(b"{}\n").collect::<Vec<_>>(), vec![b"{}".to_vec()]);
//! ```

use std::borrow::Cow;

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

impl Framing {
    /// The incremental framer of a streamed reply. A whole body is one
    /// frame, so it has none.
    pub fn framer(self) -> Option<Framer> {
        let lines = match self {
            Self::Sse => Lines::Sse(SseFramer::new()),
            Self::Ndjson => Lines::Ndjson(NdjsonFramer::new()),
            Self::Whole => return None,
        };
        Some(Framer { lines })
    }

    /// Split a whole reply body into the frames a decoder reads: one per SSE
    /// event with a non-blank `data` field, one per NDJSON line, or the
    /// whole body.
    ///
    /// ```
    /// use rig_http::http_client::framing::{Framing, WireFrame};
    ///
    /// let frames = Framing::Sse.split(b"data: {}\n\ndata: [DONE]\n\n");
    /// assert_eq!(frames.len(), 2);
    /// assert_eq!(frames[1].as_str(), "[DONE]");
    /// ```
    pub fn split(&self, bytes: &[u8]) -> Vec<WireFrame> {
        let Some(mut framer) = self.framer() else {
            return (!bytes.is_empty())
                .then(|| WireFrame::from_bytes(bytes.to_vec()))
                .into_iter()
                .collect();
        };
        let mut frames: Vec<WireFrame> = framer.push(bytes).collect();
        frames.extend(framer.finish());
        frames
    }
}

/// One transport frame, after framing but before decoding.
///
/// The transport layer (SSE framer, NDJSON splitter, websocket reader) owns
/// byte splitting and yields these; a decoder never splits bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WireFrame {
    /// Text payload, such as an SSE `data:` field or WebSocket message.
    Text(String),
    /// Byte payload, such as an NDJSON line or binary frame.
    Bytes(Vec<u8>),
}

impl WireFrame {
    /// The frame payload as text (lossy for byte frames).
    pub fn as_str(&self) -> Cow<'_, str> {
        match self {
            Self::Text(text) => Cow::Borrowed(text),
            Self::Bytes(bytes) => String::from_utf8_lossy(bytes),
        }
    }

    /// The frame payload's bytes.
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Text(text) => text.as_bytes(),
            Self::Bytes(bytes) => bytes,
        }
    }

    /// A frame of `bytes`: text when they are UTF-8.
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        match String::from_utf8(bytes) {
            Ok(text) => Self::Text(text),
            Err(error) => Self::Bytes(error.into_bytes()),
        }
    }
}

/// The incremental framer of one streamed reply, from [`Framing::framer`].
#[derive(Debug)]
pub struct Framer {
    lines: Lines,
}

#[derive(Debug)]
enum Lines {
    Sse(SseFramer),
    Ndjson(NdjsonFramer),
}

impl Framer {
    /// Feed one transport chunk; yields every frame it completed. An SSE
    /// event whose data is only whitespace is a heartbeat, not a frame.
    pub fn push(&mut self, chunk: &[u8]) -> impl Iterator<Item = WireFrame> + '_ {
        let (sse, ndjson) = match &mut self.lines {
            Lines::Sse(framer) => (Some(framer.push(chunk)), None),
            Lines::Ndjson(framer) => (None, Some(framer.push(chunk))),
        };
        let events = sse.into_iter().flatten();
        let events = events.filter(|event| !event.data.trim().is_empty());
        let lines = ndjson.into_iter().flatten().map(WireFrame::from_bytes);
        events.map(|event| WireFrame::Text(event.data)).chain(lines)
    }

    /// The stream ended: NDJSON's unterminated last line. SSE dispatches
    /// only on a blank line, so an unterminated event is never a frame.
    pub fn finish(&mut self) -> Option<WireFrame> {
        match &mut self.lines {
            Lines::Sse(_) => None,
            Lines::Ndjson(framer) => framer.finish().map(WireFrame::from_bytes),
        }
    }

    /// Bytes pushed but not yet part of a frame, for truncation
    /// diagnostics: [`SseFramer::pending`] or [`NdjsonFramer::pending`].
    pub fn pending(&self) -> usize {
        match &self.lines {
            Lines::Sse(framer) => framer.pending(),
            Lines::Ndjson(framer) => framer.pending(),
        }
    }
}

/// One dispatched `text/event-stream` event.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SseEvent {
    /// The `event:` field, `"message"` when the stream named none.
    pub event: Cow<'static, str>,
    /// The joined `data:` fields, without the terminating newline.
    pub data: String,
    /// The stream's last event id, empty when none was ever set.
    pub id: String,
    /// Most recent parsed `retry:` value since the previous emitted event,
    /// saturating at `u64::MAX`. This value does not trigger automatic
    /// reconnection.
    pub retry: Option<u64>,
}

/// A field recognized by the SSE grammar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SseField {
    /// A payload line.
    Data,
    /// An event type.
    Event,
    /// An accepted event id.
    Id,
    /// A blank dispatch line.
    Blank,
    /// A comment, retry, ignored id or unknown field.
    Other,
}

/// Original byte ranges of one line consumed by [`SseFramer`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseLine {
    /// Whole line, including its line ending, at saturating stream offsets.
    pub span: std::ops::Range<usize>,
    /// Line content without its ending or an initial BOM.
    pub content: std::ops::Range<usize>,
    /// Field value after a colon and optional space, when present.
    pub value: Option<std::ops::Range<usize>>,
    /// The field accepted by the grammar.
    pub field: SseField,
}

/// A parsed event and the byte offset where it became dispatchable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordedSseEvent {
    /// The grammar's parsed event.
    pub event: SseEvent,
    /// Saturating stream offset of its completing line ending or retained EOF.
    pub end: usize,
}

/// Parsed events and source lines consumed by a recording operation.
#[derive(Debug, Default)]
pub struct SseBatch {
    /// Events emitted in order, with their completion offsets.
    pub events: Vec<RecordedSseEvent>,
    /// Source lines in order, including non-dispatching lines.
    pub lines: Vec<SseLine>,
}

/// The optional leading byte-order mark, which the grammar ignores.
const BOM: [u8; 3] = [0xef, 0xbb, 0xbf];

/// A push parser for the WHATWG `text/event-stream` grammar.
#[derive(Debug, Default)]
pub struct SseFramer {
    /// Bytes not yet consumed as a complete line.
    buffer: Vec<u8>,
    /// How many bytes at the start of `buffer` hold no line end.
    scanned: usize,
    /// The last line ended at a CR that was the last byte pushed, so an LF
    /// opening the next chunk completes that CRLF.
    after_cr: bool,
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
    /// Absolute byte position after the last consumed line or BOM.
    offset: usize,
}

impl SseFramer {
    /// A framer at the start of a stream.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one transport chunk; yields every event it completed.
    pub fn push(&mut self, chunk: &[u8]) -> std::vec::Drain<'_, SseEvent> {
        self.push_inner(chunk, None);
        self.ready.drain(..)
    }

    /// Feed a chunk and retain parsed events with source ranges for recording.
    pub fn push_recording(&mut self, chunk: &[u8]) -> SseBatch {
        let mut batch = SseBatch::default();
        self.push_inner(chunk, Some(&mut batch));
        batch
    }

    /// Retain an incomplete final event for a recording, without changing
    /// the normal SSE rule that EOF does not dispatch it.
    pub fn finish_recording(&mut self) -> SseBatch {
        let mut batch = SseBatch::default();
        if self.bom_prefix > 0 {
            self.buffer
                .extend_from_slice(BOM.get(..self.bom_prefix).unwrap_or_default());
            self.bom_prefix = 0;
            self.bom_done = true;
        }
        if !self.buffer.is_empty() {
            let buffer = std::mem::take(&mut self.buffer);
            let start = self.offset;
            self.offset = self.offset.saturating_add(buffer.len());
            let (field, value) = self.line(&String::from_utf8_lossy(&buffer));
            batch.lines.push(SseLine {
                span: start..self.offset,
                content: start..self.offset,
                value: value.map(|at| start.saturating_add(at)..self.offset),
                field,
            });
        }
        self.dispatch();
        if let Some(event) = self.ready.pop() {
            batch.events.push(RecordedSseEvent {
                event,
                end: self.offset,
            });
        }
        batch
    }

    fn push_inner(&mut self, chunk: &[u8], mut recording: Option<&mut SseBatch>) {
        let mut chunk = self.strip_bom(chunk);
        if self.after_cr
            && let Some((&first, rest)) = chunk.split_first()
        {
            self.after_cr = false;
            if first == b'\n' {
                chunk = rest;
                self.offset = self.offset.saturating_add(1);
                // Pending bytes are counted as if the CRLF had arrived whole.
                if self.since_dispatch > 0 {
                    self.since_dispatch = self.since_dispatch.saturating_add(1);
                }
            }
        }
        self.buffer.extend_from_slice(chunk);
        // Taken out, so its lines are read in place while `line` changes the rest.
        let buffer = std::mem::take(&mut self.buffer);
        let mut rest = buffer.as_slice();
        while let Some((line, next)) = terminated_line(rest, self.scanned) {
            self.scanned = 0;
            let start = self.offset;
            self.offset = self.offset.saturating_add(rest.len() - next.len());
            self.since_dispatch = self.since_dispatch.saturating_add(rest.len() - next.len());
            self.after_cr = next.is_empty() && rest.last() == Some(&b'\r');
            let (field, value) = self.line(&String::from_utf8_lossy(line));
            if let Some(batch) = recording.as_deref_mut() {
                batch.lines.push(SseLine {
                    span: start..self.offset,
                    content: start..start.saturating_add(line.len()),
                    value: value
                        .map(|at| start.saturating_add(at)..start.saturating_add(line.len())),
                    field,
                });
                if let Some(event) = self.ready.pop() {
                    batch.events.push(RecordedSseEvent {
                        event,
                        end: self.offset,
                    });
                }
            }
            rest = next;
        }
        self.scanned = rest.len();
        let consumed = buffer.len() - rest.len();
        self.buffer = buffer;
        self.buffer.drain(..consumed);
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
                self.offset = self.offset.saturating_add(BOM.len());
                self.bom_prefix = 0;
                self.bom_done = true;
                return rest;
            }
        }
        rest
    }

    /// Process one complete line of the grammar.
    fn line(&mut self, line: &str) -> (SseField, Option<usize>) {
        if line.is_empty() {
            self.dispatch();
            return (SseField::Blank, None);
        }
        // A line starting with a colon is a comment.
        if line.starts_with(':') {
            return (SseField::Other, None);
        }
        let (field, value, at) = match line.split_once(':') {
            Some((field, value)) => {
                let trimmed = value.strip_prefix(' ').unwrap_or(value);
                (
                    field,
                    trimmed,
                    Some(field.len() + 1 + value.len() - trimmed.len()),
                )
            }
            None => (line, "", None),
        };
        let field = match field {
            "event" => {
                self.event_type.clear();
                self.event_type.push_str(value);
                SseField::Event
            }
            "data" => {
                self.data.push_str(value);
                self.data.push('\n');
                SseField::Data
            }
            // A NUL in an id is ignored outright, per the grammar.
            "id" if !value.contains('\0') => {
                self.last_event_id.clear();
                self.last_event_id.push_str(value);
                SseField::Id
            }
            "retry" if !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit()) => {
                // Digits only is a valid field, so a value past `u64` saturates.
                self.retry = Some(value.parse().unwrap_or(u64::MAX));
                SseField::Other
            }
            _ => SseField::Other,
        };
        (field, at)
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
            Cow::Borrowed("message")
        } else {
            Cow::Owned(std::mem::take(&mut self.event_type))
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
    /// How many bytes at the start of `buffer` hold no line end.
    scanned: usize,
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
        let mut rest = self.buffer.as_slice();
        while let Some((line, end)) = split_line(rest, self.scanned, |byte| *byte == b'\n') {
            self.scanned = 0;
            let line = line.strip_suffix(b"\r").unwrap_or(line);
            if !line.is_empty() {
                self.ready.push(line.to_vec());
            }
            rest = end.get(1..).unwrap_or_default();
        }
        self.scanned = rest.len();
        let consumed = self.buffer.len() - rest.len();
        self.buffer.drain(..consumed);
        self.ready.drain(..)
    }

    /// The unterminated last line, if the stream ended with a nonempty one.
    /// A CR ending it is part of the line end, as in [`Self::push`].
    pub fn finish(&mut self) -> Option<Vec<u8>> {
        let mut line = std::mem::take(&mut self.buffer);
        if line.last() == Some(&b'\r') {
            line.pop();
        }
        (!line.is_empty()).then_some(line)
    }

    /// Bytes of an unterminated trailing line, for truncation diagnostics.
    pub fn pending(&self) -> usize {
        self.buffer.len()
    }
}

/// The first complete line in `buffer`, without its line end, and the
/// bytes after that line end; the first `scanned` bytes hold no line end.
///
/// A CR ends a line at once, as the grammar says; an LF right after it in
/// `buffer` belongs to the same line end.
fn terminated_line(buffer: &[u8], scanned: usize) -> Option<(&[u8], &[u8])> {
    let (line, end) = split_line(buffer, scanned, |byte| *byte == b'\n' || *byte == b'\r')?;
    let rest = end.strip_prefix(b"\r\n").or_else(|| end.get(1..));
    Some((line, rest.unwrap_or_default()))
}

/// `buffer` split before its first byte that `ends` a line, if it has one;
/// the first `scanned` bytes hold none.
fn split_line(buffer: &[u8], scanned: usize, ends: impl Fn(&u8) -> bool) -> Option<(&[u8], &[u8])> {
    buffer.split_at_checked(scanned + buffer.iter().skip(scanned).position(ends)?)
}

#[cfg(test)]
mod tests;
