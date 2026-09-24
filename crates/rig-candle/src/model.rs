//! Byte-backed CPU completion models and builders for validated checkpoints.
//! Native inference runs on blocking workers with bounded stream delivery.
//! Dropping a completion or stream signals cancellation between forward passes;
//! admission permits remain held until the worker exits. WASM inference runs
//! synchronously and collects stream events before returning them.
//!
//! ```no_run
//! use rig_candle::{CandleError, CandleModel, ModelData};
//!
//! fn load(data: ModelData) -> Result<CandleModel, CandleError> {
//!     CandleModel::builder(data).temperature(0.0).max_tokens(256).build()
//! }
//! ```

use std::sync::Arc;

#[cfg(not(target_family = "wasm"))]
use futures::Stream;
use futures::StreamExt;
use rig_core::completion::CompletionRequest;
use rig_core::driver::{Observation, Opened, Transport};
use rig_core::error::ProviderError;
#[cfg(test)]
use rig_core::message::{Message, UserContent};
use rig_core::operation::AdapterOutput;
use rig_core::providers::internal::wire::{self, TypedEvent, WireEvent};
use rig_core::streaming::StreamFinal;
use rig_core::wire::Mode;
#[cfg(test)]
use tokenizers::Tokenizer;

use crate::artifacts::{GgufModelData, ModelArtifacts, ModelData};
use crate::generation::{
    GenerationConfig, GenerationEvent, infer, stream_generate, validate_generation,
};
#[cfg(test)]
use crate::generation::{
    IncrementalTextDecoder, effective_generation, effective_output_limit, max_tokens_to_usize,
    next_cache_position, recent_tokens, sampling,
};
#[cfg(test)]
use crate::loader::*;
use crate::loader::{LoadedModel, load_gguf_model, load_model_with_family};
#[cfg(test)]
use crate::profile::{ArtifactFormat, LoaderBackend, definition_for};
#[cfg(test)]
use crate::profile::{BEGIN_OF_TEXT, END_HEADER, END_OF_TURN, IM_END, IM_START, START_HEADER};
use crate::profile::{ConversationProtocol, ModelArchitecture, Quantization};
use crate::runtime::CancellationSignal;
#[cfg(all(test, not(target_family = "wasm")))]
use crate::runtime::TestControl;
#[cfg(not(target_family = "wasm"))]
use crate::runtime::{CancelOnDrop, acquire_concurrency};
use crate::types::*;
#[cfg(test)]
use crate::validation::*;

const DEFAULT_MAX_CONCURRENT_REQUESTS: usize = 1;
#[cfg(not(target_family = "wasm"))]
const STREAM_CHANNEL_CAPACITY: usize = 8;

/// Cloneable CPU completion model sharing validated, loaded weights.
/// Each inference owns its cache and sampler.
#[derive(Clone)]
pub struct CandleModel {
    state: Arc<LoadedModel>,
}

/// Builder for loading a [`CandleModel`] and customizing generation defaults.
pub struct CandleModelBuilder<'a> {
    source: ModelSource<'a>,
    family: Option<ConversationProtocol>,
    generation: GenerationConfig,
    max_concurrent_requests: usize,
}

enum ModelSource<'a> {
    Owned(ModelArtifacts),
    BorrowedGguf(GgufModelData<'a>),
}

impl CandleModel {
    /// Loads a model from config, tokenizer, and one unsharded safetensors buffer.
    pub fn from_safetensors(data: ModelData) -> Result<Self, CandleError> {
        Self::builder(data).build()
    }

    /// Loads a model from config, tokenizer, and a byte-backed GGUF checkpoint.
    pub fn from_gguf(data: ModelData) -> Result<Self, CandleError> {
        Self::builder_from_artifacts(ModelArtifacts::Gguf(data)).build()
    }

    /// Loads GGUF artifacts from borrowed bytes without copying the checkpoint buffer.
    ///
    /// This is intended for `include_bytes!` and other long-lived buffers where
    /// the GGUF bytes are needed only while Candle constructs its owned tensors.
    pub fn from_gguf_bytes(data: GgufModelData<'_>) -> Result<Self, CandleError> {
        Self::builder_from_gguf_bytes(data).build()
    }

    /// Starts a byte-backed model builder.
    pub fn builder(data: ModelData) -> CandleModelBuilder<'static> {
        Self::builder_from_artifacts(ModelArtifacts::Safetensors(data))
    }

    /// Starts a builder from explicitly typed byte-backed artifacts.
    pub fn builder_from_artifacts(artifacts: ModelArtifacts) -> CandleModelBuilder<'static> {
        CandleModelBuilder {
            source: ModelSource::Owned(artifacts),
            family: None,
            generation: GenerationConfig::default(),
            max_concurrent_requests: DEFAULT_MAX_CONCURRENT_REQUESTS,
        }
    }

    /// Starts a GGUF builder without copying any artifact buffer.
    ///
    /// All generation and concurrency settings available to owned artifacts
    /// are also available here. The buffers only need to remain valid until
    /// [`CandleModelBuilder::build`] returns because Candle owns loaded tensors.
    pub fn builder_from_gguf_bytes<'a>(data: GgufModelData<'a>) -> CandleModelBuilder<'a> {
        CandleModelBuilder {
            source: ModelSource::BorrowedGguf(data),
            family: None,
            generation: GenerationConfig::default(),
            max_concurrent_requests: DEFAULT_MAX_CONCURRENT_REQUESTS,
        }
    }

    /// Asynchronously loads owned safetensors artifacts outside the async executor.
    #[cfg(not(target_family = "wasm"))]
    pub async fn from_safetensors_async(data: ModelData) -> Result<Self, CandleError> {
        Self::builder(data).build_async().await
    }

    /// Returns the validated conversation/output protocol.
    pub fn conversation_protocol(&self) -> Option<ConversationProtocol> {
        Some(self.state.profile.definition.protocol)
    }

    /// Returns the validated transformer architecture of the loaded checkpoint.
    pub fn architecture(&self) -> Option<ModelArchitecture> {
        Some(self.state.profile.definition.architecture)
    }

    /// Returns the detected checkpoint quantization, if the model is quantized.
    pub fn quantization(&self) -> Option<Quantization> {
        self.state.profile.definition.quantization
    }
}

impl<'a> CandleModelBuilder<'a> {
    /// Selects a conversation protocol and requires it to match the artifacts.
    pub fn conversation_protocol(mut self, protocol: ConversationProtocol) -> Self {
        self.family = Some(protocol);
        self
    }

    /// Sets the default maximum generated token count.
    pub fn max_tokens(mut self, max_tokens: u64) -> Self {
        self.generation.max_tokens = max_tokens;
        self
    }

    /// Sets the default sampling temperature. Zero enables greedy decoding.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.generation.temperature = temperature;
        self
    }

    /// Sets the default deterministic sampling seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.generation.seed = seed;
        self
    }

    /// Sets or disables the default top-k sampling limit.
    pub fn top_k(mut self, top_k: Option<usize>) -> Self {
        self.generation.top_k = top_k;
        self
    }

    /// Sets or disables the default nucleus-sampling threshold.
    pub fn top_p(mut self, top_p: Option<f64>) -> Self {
        self.generation.top_p = top_p;
        self
    }

    /// Sets the default repeat penalty.
    pub fn repeat_penalty(mut self, repeat_penalty: f32) -> Self {
        self.generation.repeat_penalty = repeat_penalty;
        self
    }

    /// Sets the default number of recent tokens used by the repeat penalty.
    pub fn repeat_last_n(mut self, repeat_last_n: usize) -> Self {
        self.generation.repeat_last_n = repeat_last_n;
        self
    }

    /// Sets the maximum number of native inference requests admitted concurrently.
    ///
    /// The default is one to avoid CPU oversubscription and concurrent KV-cache
    /// memory spikes. WASM inference is synchronous and does not use this limit.
    pub fn max_concurrent_requests(mut self, max_concurrent_requests: usize) -> Self {
        self.max_concurrent_requests = max_concurrent_requests;
        self
    }

    /// Validates all artifacts and loads model tensors onto the CPU.
    pub fn build(self) -> Result<CandleModel, CandleError> {
        validate_generation(&self.generation, None)?;
        if self.max_concurrent_requests == 0 {
            return Err(CandleError::InvalidConcurrencyLimit);
        }
        let loaded = match self.source {
            ModelSource::Owned(artifacts) => load_model_with_family(
                artifacts,
                self.family,
                self.generation,
                self.max_concurrent_requests,
            )?,
            ModelSource::BorrowedGguf(data) => load_gguf_model(
                data,
                self.family,
                self.generation,
                self.max_concurrent_requests,
            )?,
        };
        Ok(CandleModel {
            state: Arc::new(loaded),
        })
    }
}

#[cfg(not(target_family = "wasm"))]
impl CandleModelBuilder<'static> {
    /// Validates and loads model artifacts on Tokio's blocking thread pool.
    ///
    /// Dropping the returned future does not stop a load that has already
    /// started; Tokio keeps admitted blocking work running to completion.
    pub async fn build_async(self) -> Result<CandleModel, CandleError> {
        join_model_load(tokio::task::spawn_blocking(move || self.build())).await
    }
}

#[cfg(not(target_family = "wasm"))]
async fn join_model_load(
    task: tokio::task::JoinHandle<Result<CandleModel, CandleError>>,
) -> Result<CandleModel, CandleError> {
    task.await
        .map_err(|error| CandleError::BlockingTaskJoin(error.to_string()))?
}

#[cfg(test)]
fn render_prompt(request: &CompletionRequest) -> Result<String, CandleError> {
    render_prompt_for(request, ConversationProtocol::Llama3)
}

#[cfg(test)]
fn render_prompt_for(
    request: &CompletionRequest,
    family: ConversationProtocol,
) -> Result<String, CandleError> {
    crate::protocol::render_prompt(request, family)
}

#[cfg(not(target_family = "wasm"))]
type CandleStreamItem = Result<GenerationEvent, ProviderError>;

#[cfg(not(target_family = "wasm"))]
struct CandleReceiverStream {
    receiver: tokio::sync::mpsc::Receiver<CandleStreamItem>,
    cancellation: CancellationSignal,
}

#[cfg(not(target_family = "wasm"))]
impl Stream for CandleReceiverStream {
    type Item = CandleStreamItem;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        context: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.get_mut().receiver.poll_recv(context)
    }
}

#[cfg(not(target_family = "wasm"))]
impl Drop for CandleReceiverStream {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

#[cfg(not(target_family = "wasm"))]
fn stream_infer(
    loaded: &LoadedModel,
    request: &CompletionRequest,
    cancellation: &CancellationSignal,
    sender: &tokio::sync::mpsc::Sender<CandleStreamItem>,
) -> Result<(), CandleError> {
    let response = stream_generate(loaded, request, cancellation, |event| {
        #[cfg(test)]
        if let Some(control) = &loaded.test_control {
            control.record_delivery_attempt();
        }
        sender
            .blocking_send(Ok(event))
            .map_err(|_| CandleError::StreamingChannelClosed)
    })?;
    sender
        .blocking_send(Ok(GenerationEvent::Final(response)))
        .map_err(|_| CandleError::StreamingChannelClosed)
}

/// Local generation: the wire a [`CandleModel`] transport answers. Its
/// payload is the completion request itself; the loaded model renders and
/// runs it.
///
/// ```no_run
/// use rig_candle::{CandleModel, Generation};
/// use rig_core::Model;
///
/// fn model(candle: CandleModel) -> Model<Generation, CandleModel> {
///     Model::new(Generation, candle)
/// }
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Generation;

/// One unit of a local generation's reply.
pub enum CandleFrame {
    /// A whole unary turn: the local response record and its parsed content.
    Whole(crate::generation::InferredCompletion),
    /// One streamed generation event.
    Event(GenerationEvent),
}

impl rig_core::wire::Wire for Generation {
    type Op = rig_core::operation::Completion;
    type Payload = CompletionRequest;
    type Frame = CandleFrame;
    type Decoder = CandleAdapter;

    fn name(&self) -> &str {
        crate::types::PROVIDER_NAME
    }

    fn encode(
        &self,
        request: CompletionRequest,
        _mode: Mode,
    ) -> Result<CompletionRequest, rig_core::error::EncodeError> {
        Ok(request)
    }

    fn decoder(&self, _mode: Mode) -> CandleAdapter {
        CandleAdapter::default()
    }
}

/// Converts typed local generation events through the shared completion driver.
/// Every input is modeled; no byte decoding or unknown-frame classification occurs.
#[derive(Default)]
pub struct CandleAdapter {
    /// The unary turn's local response record, for the response's `raw`.
    document: Option<serde_json::Value>,
}

impl rig_core::wire::Decoder<rig_core::operation::Completion, CandleFrame> for CandleAdapter {
    type Event = CandleFrame;

    fn classify(&self, frame: CandleFrame) -> WireEvent<Self::Event> {
        wire::classify_typed_event(TypedEvent::Modeled(frame))
    }

    fn interpret(&mut self, frame: Self::Event, out: &mut AdapterOutput) {
        let event = match frame {
            CandleFrame::Event(event) => event,
            CandleFrame::Whole(inferred) => {
                match serde_json::to_value(&inferred.response) {
                    Ok(document) => self.document = Some(document),
                    Err(err) => return out.error(err.into()),
                }
                out.content(&inferred.choice);
                GenerationEvent::Final(inferred.response)
            }
        };
        match event {
            GenerationEvent::Text(text) => out.text(text),
            GenerationEvent::ToolCall { id, end } => out.tool_call(id, end),
            GenerationEvent::Reasoning {
                id,
                provider_id,
                content,
            } => out.reasoning_block(id, provider_id, content),
            GenerationEvent::Final(response) => match terminal_record(&response) {
                Ok(record) => out.final_record(record),
                Err(err) => out.error(err.into()),
            },
        }
    }

    fn finish(&mut self, _out: &mut AdapterOutput) {
        // Channel EOF without a `Final` event means the generator failed or
        // was cancelled: truncation, no terminal record.
    }

    fn document(&self) -> Option<serde_json::Value> {
        self.document.clone()
    }
}

/// Map this crate's own terminal record onto rig's [`StreamFinal`],
/// serializing the local record onto [`StreamFinal::raw`].
fn terminal_record(response: &CandleCompletionResponse) -> Result<StreamFinal, serde_json::Error> {
    let usage = response.into();
    Ok(StreamFinal::new(
        crate::types::PROVIDER_NAME,
        usage,
        serde_json::to_value(response)?,
    )
    .with_finish_reason(response.finish_reason.into()))
}

impl Transport<Generation> for CandleModel {
    fn send(
        &self,
        request: CompletionRequest,
        mode: Mode,
        _observation: Option<Observation>,
    ) -> Result<
        impl Future<Output = Opened<CompletionRequest, CandleFrame>>
        + rig_core::wasm_compat::WasmCompatSend
        + 'static
        + use<>,
        ProviderError,
    > {
        // A closed admission controller refuses before anything runs, as
        // opening a completion or a stream always has.
        #[cfg(not(target_family = "wasm"))]
        if self.state.concurrency.is_closed() {
            return Err(CandleError::ConcurrencyControllerClosed.into());
        }
        let model = self.clone();
        Ok(async move {
            match mode {
                Mode::Unary => match model.infer_completion(request).await {
                    Ok(inferred) => {
                        Opened::new(futures::stream::iter([Ok(CandleFrame::Whole(inferred))]))
                    }
                    Err(error) => Opened::failed(error),
                },
                Mode::Streaming => match model.open_stream(request).await {
                    Ok(events) => Opened::new(events.map(|event| event.map(CandleFrame::Event))),
                    Err(error) => Opened::failed(error),
                },
            }
        })
    }
}

impl CandleModel {
    async fn infer_completion(
        &self,
        request: CompletionRequest,
    ) -> Result<crate::generation::InferredCompletion, ProviderError> {
        let loaded = &self.state;

        #[cfg(not(target_family = "wasm"))]
        {
            let cancellation = CancellationSignal::default();
            let mut cancel_on_drop = CancelOnDrop::new(cancellation.clone());
            let permit = acquire_concurrency(Arc::clone(&loaded.concurrency)).await?;
            let loaded = Arc::clone(loaded);
            let result = tokio::task::spawn_blocking(move || {
                let result = loaded
                    .runtime
                    .device()
                    .with_context(|| infer(&loaded, &request, &cancellation));
                drop(permit);
                result
            })
            .await
            .map_err(|error| CandleError::BlockingTaskJoin(error.to_string()));
            cancel_on_drop.disarm();
            result?.map_err(ProviderError::from)
        }

        #[cfg(target_family = "wasm")]
        {
            infer(loaded, &request, &CancellationSignal).map_err(ProviderError::from)
        }
    }

    /// Open a stream of this model's generation events. Dropping the stream
    /// signals cancellation between forward passes.
    async fn open_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<
        rig_core::wasm_compat::WasmBoxedStream<'static, Result<GenerationEvent, ProviderError>>,
        ProviderError,
    > {
        let loaded = &self.state;

        #[cfg(not(target_family = "wasm"))]
        {
            let cancellation = CancellationSignal::default();
            let mut cancel_on_drop = CancelOnDrop::new(cancellation.clone());
            let permit = acquire_concurrency(Arc::clone(&loaded.concurrency)).await?;
            let loaded = Arc::clone(loaded);
            let (sender, receiver) = tokio::sync::mpsc::channel(STREAM_CHANNEL_CAPACITY);
            let producer_sender = sender.clone();
            let producer_cancellation = cancellation.clone();
            let task = tokio::task::spawn_blocking(move || {
                let result = loaded.runtime.device().with_context(|| {
                    stream_infer(&loaded, &request, &producer_cancellation, &producer_sender)
                });
                if let Err(error) = result {
                    let _ = producer_sender.blocking_send(Err(error.into()));
                }
                drop(permit);
            });
            tokio::spawn(async move {
                if let Err(error) = task.await {
                    let error = CandleError::BlockingTaskJoin(error.to_string());
                    let _ = sender.send(Err(error.into())).await;
                }
            });
            // Dropping the receiver signals cancellation.
            cancel_on_drop.disarm();
            Ok(Box::pin(CandleReceiverStream {
                receiver,
                cancellation,
            }))
        }

        #[cfg(target_family = "wasm")]
        {
            let mut events = Vec::new();
            let response = stream_generate(loaded, &request, &CancellationSignal, |event| {
                events.push(Ok(event));
                Ok(())
            })?;
            events.push(Ok(GenerationEvent::Final(response)));
            Ok(Box::pin(futures::stream::iter(events)))
        }
    }
}

#[cfg(test)]
#[allow(clippy::panic_in_result_fn)]
mod tests;
