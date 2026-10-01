//! Completion events, and [`Streamed`], the one stream every reply arrives
//! as.
//!
//! ```
//! use rig_core::streaming::{Item, StreamEvent};
//!
//! fn text(item: &Item<StreamEvent>) -> Option<&str> {
//!     match item {
//!         Item::Event(StreamEvent::Text { text, .. }) => Some(text),
//!         _ => None,
//!     }
//! }
//! # let _ = text;
//! ```

mod event;

use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};

use crate::completion::CompletionResponse;
use crate::driver::{lock, record_request_id};
use crate::error::{ErrorReport, ProviderError};
use crate::message::ToolResult;
use crate::operation::{Completion, Turn};
use crate::wasm_compat::WasmBoxedStream;
use crate::wire::{Operation, Shared};
pub use event::{Item, Part, PartKind, SequenceError, StreamEvent, Transcript};

/// Unmodeled JSON payload with content-redacted `Debug` output.
/// Serialization preserves the payload; [`Self::value`] explicitly exposes it.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct UnknownPayload(serde_json::Value);

impl UnknownPayload {
    /// Wrap a raw unmodeled payload.
    pub fn new(value: serde_json::Value) -> Self {
        Self(value)
    }

    /// The raw payload, for consumers who opt in to the content.
    pub fn value(&self) -> &serde_json::Value {
        &self.0
    }
}

impl std::fmt::Debug for UnknownPayload {
    /// Reports serialized size without exposing payload content.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bytes = serde_json::to_vec(&self.0).map_or(0, |json| json.len());
        write!(f, "UnknownPayload({bytes} bytes redacted)")
    }
}

impl From<serde_json::Value> for UnknownPayload {
    fn from(value: serde_json::Value) -> Self {
        Self(value)
    }
}

#[cfg(test)]
mod unknown_payload_tests;

/// One item of a completion stream relayed over the bus: an item of the
/// reply, or the response the origin folded when the provider ended it.
#[derive(Debug, Clone, PartialEq)]
pub enum Relayed {
    /// An event, or a payload the origin's decoder did not model.
    Item(Item<StreamEvent>),
    /// The provider ended the reply; the origin's response.
    Done(Box<CompletionResponse>),
}

/// A completion stream as the bus carries it.
pub type StreamEvents = WasmBoxedStream<'static, Result<Relayed, ErrorReport>>;

/// One reply as it arrives: its events, and the fold that has seen each of
/// them. It is what [`Model::stream`](crate::Model::stream) returns for
/// every operation, and what a call finishes.
///
/// An error is the last item: the stream ends after it. Stop polling to
/// pause; drop the stream to cancel.
pub struct Streamed<Op: Operation> {
    /// The reading of the reply; `None` once it ended.
    reading: Option<WasmBoxedStream<'static, ()>>,
    shared: Arc<Mutex<Shared<Op>>>,
    span: tracing::Span,
    provider: String,
    /// The error the stream yielded, which ends it.
    failed: Option<ProviderError>,
}

/// A streamed completion.
pub type CompletionStream = Streamed<Completion>;

impl<Op: Operation> Streamed<Op> {
    /// The reply `reading` writes into `shared`, under `span`.
    pub(crate) fn new(
        reading: WasmBoxedStream<'static, ()>,
        shared: Arc<Mutex<Shared<Op>>>,
        span: tracing::Span,
        provider: impl Into<String>,
    ) -> Self {
        Self {
            reading: Some(reading),
            shared,
            span,
            provider: provider.into(),
            failed: None,
        }
    }

    /// The one place an item is taken: an event is absorbed by the fold
    /// before it leaves, and an error, stamped with the request id, ends the
    /// stream.
    fn poll_item(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Item<Op::Event>, ProviderError>>> {
        loop {
            if self.failed.is_some() {
                return Poll::Ready(None);
            }
            {
                let mut shared = lock(&self.shared);
                if let Some(item) = shared.take() {
                    if let Err(error) = &item {
                        record_request_id(&self.span, error.provider_request_id());
                        shared.items.clear();
                        self.failed = Some(error.clone());
                        self.reading = None;
                    }
                    return Poll::Ready(Some(item));
                }
            }
            let Some(reading) = &mut self.reading else {
                return Poll::Ready(None);
            };
            match reading.as_mut().poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(())) => {}
                Poll::Ready(None) => self.reading = None,
            }
        }
    }

    /// Read the rest of the reply, then fold it with the provider's end into
    /// the response: what [`Model::call`](crate::Model::call) returns for
    /// the same reply. An error the stream yielded, now or before, is the
    /// result; a reply the provider did not end is
    /// [`ProviderError::Truncated`].
    pub async fn finish(self) -> Result<Op::Response, ProviderError> {
        self.finish_routed().await.map_err(|(error, _)| error)
    }

    /// [`Self::finish`], with a failure paired with the request path of the
    /// reply that failed.
    pub(crate) async fn finish_routed(mut self) -> Result<Op::Response, (ProviderError, String)> {
        let route = |stream: &Self| lock(&stream.shared).route.clone();
        while let Some(item) = futures::future::poll_fn(|cx| self.poll_item(cx)).await {
            if let Err(error) = item {
                return Err((error, route(&self)));
            }
        }
        if let Some(error) = self.failed.take() {
            return Err((error, route(&self)));
        }
        let path = route(&self);
        let Ok(shared) = Arc::try_unwrap(self.shared) else {
            return Err((
                ProviderError::Response("the reply is still being read".to_owned()),
                path,
            ));
        };
        shared
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .conclude(&self.provider)
            .map_err(|error| (error, path))
    }
}

impl Streamed<Completion> {
    /// A stream relayed over the bus under `label`: its events, then the
    /// response the origin folded. A relay that ends without one was cut
    /// short.
    pub fn relay(label: impl Into<String>, mut events: StreamEvents) -> Self {
        let label = label.into();
        let shared = Arc::new(Mutex::new(Shared::new(Turn::relayed(label.clone()))));
        let writer = Arc::clone(&shared);
        let reading = async_stream::stream! {
            while let Some(item) = events.next().await {
                // The lock is released before the stream yields.
                let ended = {
                    let mut shared = lock(&writer);
                    match item {
                        Ok(Relayed::Item(item)) => {
                            shared.items.push_back(Ok(item));
                            false
                        }
                        Ok(Relayed::Done(response)) => {
                            shared.response = Some(*response);
                            true
                        }
                        Err(report) => {
                            shared
                                .items
                                .push_back(Err(ProviderError::Relayed(Box::new(report))));
                            true
                        }
                    }
                };
                if ended {
                    return;
                }
                yield ();
            }
            lock(&writer).items.push_back(Err(ProviderError::Truncated));
        };
        Self::new(Box::pin(reading), shared, tracing::Span::none(), label)
    }

    /// This stream as the bus carries it: its items, then the response it
    /// folds into, or the error that ended it. A reply cut short closes the
    /// relay without a response, as its transport closed.
    pub fn into_relay(mut self) -> StreamEvents {
        Box::pin(async_stream::stream! {
            while let Some(item) = self.next().await {
                match item {
                    Ok(item) => yield Ok(Relayed::Item(item)),
                    Err(ProviderError::Truncated) => return,
                    Err(error) => {
                        yield Err(ErrorReport::from(&error));
                        return;
                    }
                }
            }
            match self.finish().await {
                Ok(response) => yield Ok(Relayed::Done(Box::new(response))),
                Err(ProviderError::Truncated) => {}
                Err(error) => yield Err(ErrorReport::from(&error)),
            }
        })
    }

    /// What arrived so far: every part that ended, and the provider's end
    /// once it arrived. Valid after an error, and after the caller stopped
    /// polling.
    pub fn partial(&self) -> CompletionResponse {
        let shared = lock(&self.shared);
        if let Some(response) = &shared.response {
            return response.clone();
        }
        shared
            .fold
            .partial(shared.end.as_ref(), &shared.reply(&self.provider))
    }

    /// The assistant message id the reply recorded so far.
    pub fn message_id(&self) -> Option<String> {
        let shared = lock(&self.shared);
        shared
            .response
            .as_ref()
            .and_then(|response| response.message_id.clone())
            .or_else(|| shared.fold.message_id().map(str::to_owned))
    }

    /// The issuer this reply's reasoning is sealed to.
    pub fn reasoning_issuer(&self) -> crate::message::Issuer {
        lock(&self.shared).fold.reasoning_issuer()
    }
}

impl<Op: Operation> Stream for Streamed<Op> {
    type Item = Result<Item<Op::Event>, ProviderError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.get_mut().poll_item(cx)
    }
}

/// Streamed user content. This content is primarily used to represent tool results from tool calls made during a multi-turn/step agent prompt.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum StreamedUserContent {
    /// Tool result emitted during a multi-turn streaming agent loop.
    ToolResult {
        /// The result; `tool_result.call` is the id of the call it answers.
        tool_result: ToolResult,
    },
}

impl StreamedUserContent {
    /// A streamed tool result.
    pub fn tool_result(tool_result: ToolResult) -> Self {
        Self::ToolResult { tool_result }
    }
}

#[cfg(test)]
mod tests;
