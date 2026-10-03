//! The provider's own JSON of a Converse reply, read off the HTTP body the
//! SDK decodes, so a response's `raw` is what Bedrock sent rather than a
//! re-serialization of the SDK's types. A unary body is one JSON document;
//! a stream's body is event-stream messages, each carrying one event's JSON.
//!
//! The SDK's types stay the decoder's input: they are what the SDK checked.
//! A successful unary reply the SDK cannot read is decoded from this JSON.

use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::task::{Context, Poll};

use aws_smithy_eventstream::frame::{DecodedFrame, MessageFrameDecoder};
use aws_smithy_runtime_api::box_error::BoxError;
use aws_smithy_runtime_api::client::interceptors::Intercept;
use aws_smithy_runtime_api::client::interceptors::context::BeforeDeserializationInterceptorContextMut;
use aws_smithy_runtime_api::client::runtime_components::RuntimeComponents;
use aws_smithy_types::body::SdkBody;
use aws_smithy_types::config_bag::ConfigBag;
use aws_smithy_types::event_stream::Message;
use bytes::{Bytes, BytesMut};
use http_body::{Body, Frame, SizeHint};
use serde_json::Value;

/// An interceptor that keeps a copy of each response body the SDK reads.
#[derive(Clone, Debug, Default)]
pub(crate) struct Capture(Arc<Mutex<Vec<u8>>>);

impl Capture {
    /// The body bytes read since the last call.
    pub(crate) fn take(&self) -> Vec<u8> {
        std::mem::take(&mut *lock(&self.0))
    }

    /// The whole body read so far as JSON, when it is JSON.
    pub(crate) fn document(&self) -> Option<Value> {
        serde_json::from_slice(&self.take()).ok()
    }
}

fn lock(bytes: &Mutex<Vec<u8>>) -> MutexGuard<'_, Vec<u8>> {
    bytes.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Intercept for Capture {
    fn name(&self) -> &'static str {
        "rig_bedrock::Capture"
    }

    /// Each attempt's body replaces the copy of the one before it.
    fn modify_before_deserialization(
        &self,
        context: &mut BeforeDeserializationInterceptorContextMut<'_>,
        _runtime_components: &RuntimeComponents,
        _cfg: &mut ConfigBag,
    ) -> Result<(), BoxError> {
        lock(&self.0).clear();
        let response = context.response_mut();
        let body = response.take_body();
        *response.body_mut() = SdkBody::from_body_1_x(Tee {
            body,
            sink: Arc::clone(&self.0),
        });
        Ok(())
    }
}

/// A body that copies each data frame it passes on into `sink`.
struct Tee {
    body: SdkBody,
    sink: Arc<Mutex<Vec<u8>>>,
}

impl Body for Tee {
    type Data = Bytes;
    type Error = aws_smithy_types::body::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Self::Error>>> {
        let this = self.get_mut();
        let polled = Pin::new(&mut this.body).poll_frame(cx);
        if let Poll::Ready(Some(Ok(frame))) = &polled
            && let Some(data) = frame.data_ref()
        {
            lock(&this.sink).extend_from_slice(data);
        }
        polled
    }

    fn is_end_stream(&self) -> bool {
        Body::is_end_stream(&self.body)
    }

    fn size_hint(&self) -> SizeHint {
        Body::size_hint(&self.body)
    }
}

/// Reads a stream's captured bytes into its events' JSON.
#[derive(Default)]
pub(crate) struct Events {
    buffer: BytesMut,
    decoder: MessageFrameDecoder,
}

impl Events {
    /// The JSON of each event `bytes` completes, as `{"<event type>":
    /// <payload>}`. Exception messages are not events: the SDK reports
    /// them as the stream's error.
    pub(crate) fn read(&mut self, bytes: &[u8]) -> Vec<Value> {
        self.buffer.extend_from_slice(bytes);
        let mut events = Vec::new();
        loop {
            match self.decoder.decode_frame(&mut self.buffer) {
                Ok(DecodedFrame::Complete(message)) => events.extend(event(&message)),
                Ok(DecodedFrame::Incomplete) => break,
                Err(error) => {
                    tracing::warn!(%error, "Converse sent bytes that are not an event stream");
                    self.buffer.clear();
                    break;
                }
            }
        }
        events
    }
}

fn event(message: &Message) -> Option<Value> {
    let header = |name: &str| {
        message
            .headers()
            .iter()
            .find(|header| header.name().as_str() == name)
            .and_then(|header| header.value().as_string().ok())
            .map(|value| value.as_str().to_owned())
    };
    if header(":message-type").as_deref() != Some("event") {
        return None;
    }
    let kind = header(":event-type")?;
    // Every event message stands for one SDK event, JSON or not, so the two
    // stay paired.
    let payload = serde_json::from_slice(message.payload()).unwrap_or(Value::Null);
    Some(Value::Object(serde_json::Map::from_iter([(kind, payload)])))
}

#[cfg(test)]
mod tests;
