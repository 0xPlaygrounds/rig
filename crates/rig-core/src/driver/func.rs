//! Models from closures. A closure model is a [`Model`] whose wire passes
//! the request through untouched and whose transport runs the closure, so
//! it goes through the driver like every provider model: same spans, same
//! fold, same `call` and `stream`, and it boxes to a [`BoxedModel`] like any
//! other. A unary closure answers a stream by re-emitting its whole reply
//! as events; a streaming closure answers a call by having its events
//! folded.
//!
//! ```no_run
//! use rig_core::Model;
//! use rig_core::completion::{CompletionRequestBuilder, CompletionResponse, Usage};
//! use rig_core::message::{AssistantContent, Message};
//!
//! # async fn example() -> Result<(), rig_core::error::ProviderError> {
//! let echo = Model::completion_fn("echo", |request| async move {
//!     let text = request.chat_history.last().and_then(Message::rag_text);
//!     let choice = vec![AssistantContent::text(text.unwrap_or_default())];
//!     Ok(CompletionResponse::new(choice, Usage::default(), "echo", serde_json::Value::Null))
//! });
//! let response = echo.call(CompletionRequestBuilder::new("Hello").build()).await?;
//! # let _ = response;
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use futures::{Stream, StreamExt};

use super::{Model, Observation, Opened, Transport};
use crate::completion::{CompletionRequest, CompletionResponse};
use crate::embeddings::{Embedding as Vector, EmbeddingResponse};
use crate::error::{EncodeError, ProviderError};
use crate::operation::{AdapterOutput, Completion, Embedding, EmbeddingCapabilities, ImagePart};
use crate::streaming::StreamEvent;
use crate::wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync};
use crate::wire::{Decoder, Mode, Operation, Output, Sink, Wire, WireEvent};

/// The wire of a closure model: it names the model, states its
/// capabilities, and sends the request itself as the payload.
pub struct FnWire<Op: Operation> {
    name: String,
    capabilities: Op::Capabilities,
}

impl<Op: Operation> FnWire<Op> {
    fn new(name: impl Into<String>, capabilities: Op::Capabilities) -> Self {
        Self {
            name: name.into(),
            capabilities,
        }
    }
}

impl<Op: Operation> Clone for FnWire<Op>
where
    Op::Capabilities: Clone,
{
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            capabilities: self.capabilities.clone(),
        }
    }
}

/// One frame of a closure model's reply: the whole response of a unary
/// closure, or one event of a streaming closure.
pub enum FnFrame<Op: Operation> {
    /// The reply as one response.
    Whole(Op::Response),
    /// One event of a streamed reply.
    Event(Op::Event),
}

/// The decoder of a closure model: a whole response is re-emitted as the
/// events it folds from, and an event is passed through.
pub struct FnDecoder;

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
type SendFn<Op> = dyn Fn(<Op as Operation>::Request) -> Reply<Op> + Send + Sync;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
type SendFn<Op> = dyn Fn(<Op as Operation>::Request) -> Reply<Op>;

type Reply<Op> = WasmBoxedFuture<'static, Opened<<Op as Operation>::Request, FnFrame<Op>>>;

/// The transport of a closure model: the closure, run once per request.
/// Clones share the closure.
pub struct FnTransport<Op: Operation> {
    send: Arc<SendFn<Op>>,
}

impl<Op: Operation> FnTransport<Op> {
    fn new<F, Fut>(send: F) -> Self
    where
        F: Fn(Op::Request) -> Fut + WasmCompatSend + WasmCompatSync + 'static,
        Fut: Future<Output = Opened<Op::Request, FnFrame<Op>>> + WasmCompatSend + 'static,
    {
        Self {
            send: Arc::new(move |request| Box::pin(send(request))),
        }
    }
}

impl<Op: Operation> Clone for FnTransport<Op> {
    fn clone(&self) -> Self {
        Self {
            send: Arc::clone(&self.send),
        }
    }
}

impl<Op> Wire for FnWire<Op>
where
    Op: Operation,
    Op::Capabilities: Clone + WasmCompatSend + WasmCompatSync,
    FnDecoder: Decoder<Op, FnFrame<Op>>,
{
    type Op = Op;
    type Payload = Op::Request;
    type Frame = FnFrame<Op>;
    type Decoder = FnDecoder;

    fn name(&self) -> &str {
        &self.name
    }

    fn encode(&self, request: Op::Request, _mode: Mode) -> Result<Op::Request, EncodeError> {
        Ok(request)
    }

    fn decoder(&self, _mode: Mode) -> FnDecoder {
        FnDecoder
    }

    fn capabilities(&self) -> Op::Capabilities {
        self.capabilities.clone()
    }

    /// The closure is the issuer of every reply it writes.
    fn reasoning_issuer(&self, _model: Option<&str>) -> Option<&str> {
        Some(&self.name)
    }

    /// The closure sees the history exactly as the caller built it.
    fn replay_issuers(&self, _model: Option<&str>) -> Option<Vec<String>> {
        None
    }
}

impl<Op> Transport<FnWire<Op>> for FnTransport<Op>
where
    Op: Operation,
    FnWire<Op>: Wire<Op = Op, Payload = Op::Request, Frame = FnFrame<Op>>,
{
    /// Both modes are accepted: the decoder gives a unary closure's whole
    /// reply to a stream, and the driver folds a streaming closure's
    /// events for a call. Nothing runs until the future is polled.
    fn send(
        &self,
        request: Op::Request,
        _mode: Mode,
        _observation: Option<Observation>,
    ) -> Result<
        impl Future<Output = Opened<Op::Request, FnFrame<Op>>> + WasmCompatSend + 'static + use<Op>,
        ProviderError,
    > {
        Ok((self.send)(request))
    }
}

impl Decoder<Completion, FnFrame<Completion>> for FnDecoder {
    type Event = FnFrame<Completion>;

    fn classify(&self, frame: FnFrame<Completion>) -> WireEvent<FnFrame<Completion>> {
        WireEvent::Known(frame)
    }

    fn interpret(&mut self, frame: FnFrame<Completion>, out: &mut AdapterOutput) {
        match frame {
            FnFrame::Whole(response) => out.response(&response, ImagePart::Block),
            FnFrame::Event(event) => out.push(Ok(event)),
        }
    }
}

impl Decoder<Embedding, FnFrame<Embedding>> for FnDecoder {
    type Event = FnFrame<Embedding>;

    fn classify(&self, frame: FnFrame<Embedding>) -> WireEvent<FnFrame<Embedding>> {
        WireEvent::Known(frame)
    }

    fn interpret(&mut self, frame: FnFrame<Embedding>, out: &mut Output<Embedding>) {
        let (FnFrame::Whole(response) | FnFrame::Event(response)) = frame;
        out.push(Ok(response));
    }
}

/// A whole reply as one frame, or the failure that replaced it.
fn whole<Op: Operation>(
    reply: Result<Op::Response, ProviderError>,
) -> Opened<Op::Request, FnFrame<Op>> {
    match reply {
        Ok(response) => Opened::new(futures::stream::iter([Ok(FnFrame::Whole(response))])),
        Err(error) => Opened::failed(error),
    }
}

impl<Op: Operation> Model<FnWire<Op>, FnTransport<Op>> {
    /// State what a runtime accounts for about this model.
    pub fn with_capabilities(mut self, capabilities: Op::Capabilities) -> Self {
        self.wire.capabilities = capabilities;
        self
    }
}

impl Model<FnWire<Completion>, FnTransport<Completion>> {
    /// A completion model from a closure that answers a request whole. A
    /// stream re-emits the response through
    /// [`AdapterOutput::response`] with [`ImagePart::Block`]. The closure's
    /// error is the call's error.
    ///
    /// Wrappers compose without traits: a fallback holds two erased models
    /// and tries the second when the first fails.
    ///
    /// ```no_run
    /// use rig_core::{BoxedModel, Model};
    /// use rig_core::operation::Completion;
    ///
    /// fn fallback(
    ///     first: BoxedModel<Completion>,
    ///     second: BoxedModel<Completion>,
    /// ) -> BoxedModel<Completion> {
    ///     Model::completion_fn("fallback", move |request| {
    ///         let (first, second) = (first.clone(), second.clone());
    ///         async move {
    ///             match first.call(request.clone()).await {
    ///                 Ok(response) => Ok(response),
    ///                 Err(_) => second.call(request).await,
    ///             }
    ///         }
    ///     })
    ///     .boxed()
    /// }
    /// ```
    pub fn completion_fn<F, Fut>(name: impl Into<String>, f: F) -> Self
    where
        F: Fn(CompletionRequest) -> Fut + WasmCompatSend + WasmCompatSync + 'static,
        Fut: Future<Output = Result<CompletionResponse, ProviderError>> + WasmCompatSend + 'static,
    {
        Self::new(
            FnWire::new(name, Default::default()),
            FnTransport::new(move |request| {
                let reply = f(request);
                async move { whole(reply.await) }
            }),
        )
    }

    /// A completion model from a closure that answers a request as a stream
    /// of events. A call folds the events like any streamed reply. The
    /// closure's error is the call's error; an error item of its stream
    /// arrives after the events before it, and a stream that ends without a
    /// terminal record is truncation.
    pub fn completion_stream_fn<F, Fut, S>(name: impl Into<String>, f: F) -> Self
    where
        F: Fn(CompletionRequest) -> Fut + WasmCompatSend + WasmCompatSync + 'static,
        Fut: Future<Output = Result<S, ProviderError>> + WasmCompatSend + 'static,
        S: Stream<Item = Result<StreamEvent, ProviderError>> + WasmCompatSend + 'static,
    {
        Self::new(
            FnWire::new(name, Default::default()),
            FnTransport::new(move |request| {
                let reply = f(request);
                async move {
                    match reply.await {
                        Ok(events) => Opened::new(events.map(|item| item.map(FnFrame::Event))),
                        Err(error) => Opened::failed(error),
                    }
                }
            }),
        )
    }
}

impl Model<FnWire<Embedding>, FnTransport<Embedding>> {
    /// An embedding model from a closure that embeds a batch of texts, one
    /// vector per text in order, at `ndims` dimensions. The batch limit is
    /// unbounded until [`Self::with_capabilities`] states one.
    pub fn embedding_fn<F, Fut>(name: impl Into<String>, ndims: usize, f: F) -> Self
    where
        F: Fn(Vec<String>) -> Fut + WasmCompatSend + WasmCompatSync + 'static,
        Fut: Future<Output = Result<Vec<Vector>, ProviderError>> + WasmCompatSend + 'static,
    {
        let name = name.into();
        let provider = name.clone();
        Self::new(
            FnWire::new(name, EmbeddingCapabilities::new(usize::MAX, ndims)),
            FnTransport::new(move |texts| {
                let provider = provider.clone();
                let reply = f(texts);
                async move {
                    whole(
                        reply
                            .await
                            .map(|embeddings| EmbeddingResponse::new(embeddings, provider)),
                    )
                }
            }),
        )
    }
}

#[cfg(test)]
mod tests;
