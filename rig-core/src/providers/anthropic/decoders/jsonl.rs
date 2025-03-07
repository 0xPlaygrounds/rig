//! JSONL is currently not used, it might be used when Anthropic batches beta feature is used.
use crate::providers::anthropic::decoders::line::LineDecoder;
use futures::{Stream, StreamExt};
use serde::de::DeserializeOwned;
use serde::de::Error;
use std::marker::PhantomData;
use std::pin::Pin;
use std::task::{Context, Poll};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum JSONLDecoderError {
    #[error("Failed to parse JSON: {0}")]
    ParseError(#[from] serde_json::Error),

    #[error("Response has no body")]
    NoBodyError,
}

/// Decoder for JSON Lines format, where each line is a separate JSON object.
///
/// This struct allows processing a stream of bytes, decoding them into lines,
/// and then parsing each line as a JSON object of type T.
pub struct JSONLDecoder<T, S>
where
    T: DeserializeOwned + Unpin,
    S: Stream<Item = Result<Vec<u8>, std::io::Error>> + Unpin,
{
    stream: S,
    line_decoder: LineDecoder,
    buffer: Vec<T>,
    _phantom: PhantomData<T>,
}

impl<T, S> JSONLDecoder<T, S>
where
    T: DeserializeOwned + Unpin,
    S: Stream<Item = Result<Vec<u8>, std::io::Error>> + Unpin,
{
    /// Create a new JSONLDecoder from a byte stream
    pub fn new(stream: S) -> Self {
        Self {
            stream,
            line_decoder: LineDecoder::new(),
            buffer: Vec::new(),
            _phantom: PhantomData,
        }
    }

    /// Process a chunk of data, returning a Result with any JSON parsing errors
    fn process_chunk(&mut self, chunk: &[u8]) -> Result<Vec<T>, JSONLDecoderError> {
        let lines = self.line_decoder.decode(chunk);
        let mut results = Vec::with_capacity(lines.len());

        for line in lines {
            // Skip empty lines
            if line.trim().is_empty() {
                continue;
            }

            let value: T = serde_json::from_str(&line)?;
            results.push(value);
        }

        Ok(results)
    }

    /// Flush any remaining data in the line decoder and parse it
    fn flush(&mut self) -> Result<Vec<T>, JSONLDecoderError> {
        let lines = self.line_decoder.flush();
        let mut results = Vec::with_capacity(lines.len());

        for line in lines {
            // Skip empty lines
            if line.trim().is_empty() {
                continue;
            }

            let value: T = serde_json::from_str(&line)?;
            results.push(value);
        }

        Ok(results)
    }
}

impl<T, S> Stream for JSONLDecoder<T, S>
where
    T: DeserializeOwned + Unpin,
    S: Stream<Item = Result<Vec<u8>, std::io::Error>> + Unpin,
{
    type Item = Result<T, JSONLDecoderError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Get a mutable reference to self
        let this = self.get_mut();

        // Return any buffered items first
        if !this.buffer.is_empty() {
            return Poll::Ready(Some(Ok(this.buffer.remove(0))));
        }

        // Poll the underlying stream
        match this.stream.poll_next_unpin(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                // Process the chunk
                match this.process_chunk(&chunk) {
                    Ok(mut parsed) => {
                        // If we got any items, buffer them and return the first one
                        if !parsed.is_empty() {
                            let item = parsed.remove(0);
                            this.buffer.append(&mut parsed);
                            Poll::Ready(Some(Ok(item)))
                        } else {
                            // No items yet, try again
                            Pin::new(this).poll_next(cx)
                        }
                    }
                    Err(e) => Poll::Ready(Some(Err(e))),
                }
            }
            Poll::Ready(Some(Err(e))) => {
                // Propagate stream errors
                Poll::Ready(Some(Err(JSONLDecoderError::ParseError(
                    serde_json::Error::custom(format!("Stream error: {}", e)),
                ))))
            }
            Poll::Ready(None) => {
                // Stream is done, flush any remaining data
                match this.flush() {
                    Ok(mut parsed) => {
                        if !parsed.is_empty() {
                            let item = parsed.remove(0);
                            this.buffer.append(&mut parsed);
                            Poll::Ready(Some(Ok(item)))
                        } else {
                            // Nothing left
                            Poll::Ready(None)
                        }
                    }
                    Err(e) => Poll::Ready(Some(Err(e))),
                }
            }
            Poll::Pending => Poll::Pending,
        }
    }
}
