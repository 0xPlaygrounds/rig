//! The HTTP bodies of one Converse call, at the SDK's boundary: the request
//! body this crate built goes out in place of the SDK's serialization, and
//! a copy of the response body the SDK reads comes back, so a reply is the
//! JSON Bedrock sent rather than a re-serialization of the SDK's types. A
//! unary body is one JSON document; a stream's body is event-stream
//! messages, each carrying one event's or exception's JSON.

use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::task::{Context, Poll};

use aws_smithy_eventstream::frame::{DecodedFrame, MessageFrameDecoder};
use aws_smithy_runtime_api::box_error::BoxError;
use aws_smithy_runtime_api::client::interceptors::Intercept;
use aws_smithy_runtime_api::client::interceptors::context::{
    BeforeDeserializationInterceptorContextMut, BeforeTransmitInterceptorContextMut,
};
use aws_smithy_runtime_api::client::runtime_components::RuntimeComponents;
use aws_smithy_types::body::SdkBody;
use aws_smithy_types::config_bag::ConfigBag;
use aws_smithy_types::event_stream::Message;
use bytes::{Bytes, BytesMut};
use http_body::{Body, Frame, SizeHint};
use serde_json::Value;

/// An interceptor that sends `body` and keeps a copy of each response body
/// the SDK reads.
#[derive(Clone, Debug)]
pub(crate) struct Capture {
    body: Bytes,
    read: Arc<Mutex<Vec<u8>>>,
}

impl Capture {
    /// A capture that sends `body` as the request's JSON.
    pub(crate) fn new(body: Vec<u8>) -> Self {
        Self {
            body: Bytes::from(body),
            read: Arc::default(),
        }
    }

    /// The body bytes read since the last call.
    pub(crate) fn take(&self) -> Vec<u8> {
        std::mem::take(&mut *lock(&self.read))
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

    /// Before the first attempt, so every attempt signs and sends this body.
    fn modify_before_retry_loop(
        &self,
        context: &mut BeforeTransmitInterceptorContextMut<'_>,
        _runtime_components: &RuntimeComponents,
        _cfg: &mut ConfigBag,
    ) -> Result<(), BoxError> {
        let request = context.request_mut();
        request
            .headers_mut()
            .insert("content-length", self.body.len().to_string());
        *request.body_mut() = SdkBody::from(self.body.clone());
        Ok(())
    }

    /// Each attempt's body replaces the copy of the one before it.
    fn modify_before_deserialization(
        &self,
        context: &mut BeforeDeserializationInterceptorContextMut<'_>,
        _runtime_components: &RuntimeComponents,
        _cfg: &mut ConfigBag,
    ) -> Result<(), BoxError> {
        lock(&self.read).clear();
        let response = context.response_mut();
        let body = response.take_body();
        *response.body_mut() = SdkBody::from_body_1_x(Tee {
            body,
            sink: Arc::clone(&self.read),
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

/// Reads a stream's captured bytes into its messages' JSON.
#[derive(Default)]
pub(crate) struct Events {
    buffer: BytesMut,
    decoder: MessageFrameDecoder,
}

impl Events {
    /// The JSON of each event or exception `bytes` completes, as
    /// `{"<type>": <payload>}`.
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
    let kind = match header(":message-type")?.as_str() {
        "event" => header(":event-type")?,
        "exception" => header(":exception-type")?,
        _ => return None,
    };
    let payload = serde_json::from_slice(message.payload()).unwrap_or(Value::Null);
    Some(Value::Object(serde_json::Map::from_iter([(kind, payload)])))
}

#[cfg(test)]
mod tests;
