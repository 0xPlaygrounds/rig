use super::line::{self, LineDecoder};
use futures::{Stream, StreamExt};
use std::fmt::Debug;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SSEDecoderError {
    #[error("Failed to parse SSE: {0}")]
    ParseError(String),

    #[error("Failed to decode UTF-8: {0}")]
    Utf8Error(#[from] std::string::FromUtf8Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Server-Sent Event with event name, data, and raw lines
#[derive(Debug, Clone)]
pub struct ServerSentEvent {
    pub event: Option<String>,
    pub data: String,
    pub raw: Vec<String>,
}

/// SSE Decoder for parsing Server-Sent Events (SSE) format
pub struct SSEDecoder {
    data: Vec<String>,
    event: Option<String>,
    chunks: Vec<String>,
}

impl Default for SSEDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl SSEDecoder {
    /// Create a new SSE decoder
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            event: None,
            chunks: Vec::new(),
        }
    }

    /// Decode a line of SSE text, returning an event if complete
    pub fn decode(&mut self, line: &str) -> Option<ServerSentEvent> {
        let mut line = line.to_string();

        // Handle carriage returns as per TypeScript impl
        if line.ends_with('\r') {
            line = line[0..line.len() - 1].to_string();
        }

        // Empty line signals the end of an event
        if line.is_empty() {
            // If we don't have any data or event, just return None
            if self.event.is_none() && self.data.is_empty() {
                return None;
            }

            // Create the SSE event
            let sse = ServerSentEvent {
                event: self.event.clone(),
                data: self.data.join("\n"),
                raw: self.chunks.clone(),
            };

            // Reset state
            self.event = None;
            self.data.clear();
            self.chunks.clear();

            return Some(sse);
        }

        // Add to raw chunks
        self.chunks.push(line.clone());

        // Ignore comments
        if line.starts_with(':') {
            return None;
        }

        // Parse field:value format
        let parts: Vec<&str> = line.splitn(2, ':').collect();
        let (field_name, value) = match parts.as_slice() {
            [field] => (*field, ""),
            [field, value] => (*field, *value),
            _ => unreachable!(),
        };

        // Trim leading space from value as per SSE spec
        let value = if let Some(stripped) = value.strip_prefix(' ') {
            stripped
        } else {
            value
        };

        // Process fields
        match field_name {
            "event" => self.event = Some(value.to_string()),
            "data" => self.data.push(value.to_string()),
            _ => {} // Ignore other fields
        }

        None
    }
}

/// Process a byte stream to extract SSE messages
pub fn iter_sse_messages<S>(
    mut stream: S,
) -> impl Stream<Item = Result<ServerSentEvent, SSEDecoderError>>
where
    S: Stream<Item = Result<Vec<u8>, std::io::Error>> + Unpin,
{
    let mut sse_decoder = SSEDecoder::new();
    let mut line_decoder = LineDecoder::new();
    let mut buffer = Vec::new();

    async_stream::stream! {
        while let Some(chunk_result) = stream.next().await {
            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => {
                    yield Err(SSEDecoderError::IoError(e));
                    continue;
                }
            };

            // Process bytes through SSE chunking
            buffer.extend_from_slice(&chunk);

            // Extract chunks at double newlines
            while let Some((chunk_data, remaining)) = extract_sse_chunk(&buffer) {
                buffer = remaining;

                // Process the chunk lines
                for line in line_decoder.decode(&chunk_data) {
                    if let Some(sse) = sse_decoder.decode(&line) {
                        yield Ok(sse);
                    }
                }
            }
        }

        // Process any remaining data
        for line in line_decoder.flush() {
            if let Some(sse) = sse_decoder.decode(&line) {
                yield Ok(sse);
            }
        }

        // Force final event if we have pending data
        if !sse_decoder.data.is_empty() || sse_decoder.event.is_some() {
            if let Some(sse) = sse_decoder.decode("") {
                yield Ok(sse);
            }
        }
    }
}

/// Extract an SSE chunk up to a double newline
fn extract_sse_chunk(buffer: &[u8]) -> Option<(Vec<u8>, Vec<u8>)> {
    let pattern_index = line::find_double_newline_index(buffer);

    if pattern_index <= 0 {
        return None;
    }

    let pattern_index = pattern_index as usize;
    let chunk = buffer[0..pattern_index].to_vec();
    let remaining = buffer[pattern_index..].to_vec();

    Some((chunk, remaining))
}

pub fn from_response(
    response: reqwest::Response,
) -> impl Stream<Item = Result<ServerSentEvent, SSEDecoderError>> {
    let stream = response.bytes_stream().map(|result| {
        result
            .map_err(std::io::Error::other)
            .map(|bytes| bytes.to_vec())
    });

    iter_sse_messages(stream)
}
