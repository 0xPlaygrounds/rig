//! Local, CPU-only Llama-compatible and Qwen3 inference for Rig, backed by Candle.
//!
//! Models are loaded entirely from caller-provided owned or borrowed byte
//! buffers. This crate performs no filesystem or network access. On
//! `wasm32-unknown-unknown`, inference runs
//! synchronously inside the completion future; browser applications should own
//! and invoke the model in a Web Worker to avoid blocking the UI thread.
//!
//! Candle models are not expressible as plain provider configuration (they
//! are loaded tensors, not connection details), so they plug into the agent
//! machinery through its sans-IO half: `rig_agent::agent::prepare_request`
//! builds each turn's request and [`crate::functions::complete`] runs it on
//! the loaded model.
//!
//! ```no_run
//! use rig_agent::agent::{AgentConfig, ToolCatalog, prepare_request};
//! use rig_candle::{CandleModel, ModelData};
//! use rig_core::completion::Message;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let data = ModelData {
//!     config: std::fs::read("./model/config.json")?,
//!     tokenizer: std::fs::read("./model/tokenizer.json")?,
//!     weights: std::fs::read("./model/model.safetensors")?,
//! };
//! let model = CandleModel::from_safetensors_async(data).await?;
//! let config = AgentConfig::new()
//!     .with_preamble("You are a helpful assistant.")
//!     .with_temperature(0.7)
//!     .with_max_tokens(256);
//! let prepared = prepare_request(
//!     &config,
//!     &ToolCatalog::default(),
//!     false,
//!     Message::user("Explain Rust ownership briefly."),
//!     &[],
//!     None,
//!     None,
//! )?;
//! let response = rig_candle::functions::complete(&model, prepared.request).await?;
//! println!("{:?}", response.choice);
//! # Ok(())
//! # }
//! ```
//!
//! The validated profiles are unsharded Llama 3 safetensors,
//! SmolLM2-360M-Instruct Q4_K_M GGUF, and (on native targets) the official
//! Qwen3-4B Q4_K_M GGUF. Conversation rendering is explicit; tokenizer-provided
//! templates are validated where necessary but never executed.
//!
//! Qwen3 supports Rig function definitions, all portable `ToolChoice` modes,
//! assistant tool-call history, correlated text/JSON tool results, buffered
//! agent runs, and streaming agent runs. Qwen control markup is buffered for
//! one model turn before complete tool calls are emitted, so partial XML never
//! leaks as assistant text. Tool arguments are checked for JSON object syntax;
//! the registered Rig tool remains responsible for typed/schema validation.
//! Direct `CompletionRequest::output_schema` is rejected because decoding is
//! not grammar constrained. Agent `OutputMode::Tool` is supported through Rig's
//! synthetic final-result tool.
//!
//! Request `max_tokens` and `temperature` override builder defaults. The
//! Candle-specific `additional_params` keys are `top_k`, `top_p`, `seed`,
//! `repeat_penalty`, and `repeat_last_n`; unknown keys are rejected. Output is
//! clamped to the context capacity remaining after tokenizing the prompt.
//!
//! Native inference is admitted asynchronously and runs in `spawn_blocking`.
//! [`CandleModelBuilder::max_concurrent_requests`] defaults to one to control CPU
//! and KV-cache memory pressure. Dropping a native completion future signals
//! cooperative cancellation. Streaming uses an eight-fragment bounded channel;
//! dropping the stream signals the same cancellation while keeping the admission
//! permit until the blocking worker exits. A forward operation already in progress
//! cannot be interrupted, so cancellation is observed at the next generation
//! boundary. WASM does not use native synchronization or threads and collects its
//! synchronously generated events before exposing them as a compatible stream.
//!
//! Multimodal content, accelerators, shards, arbitrary tokenizer chat templates,
//! provider-hosted tools, and in-crate downloads are unsupported.

use std::sync::Arc;

#[cfg(not(target_family = "wasm"))]
use futures::Stream;
use rig_core::completion::{CompletionError, CompletionRequest, CompletionResponse};
#[cfg(test)]
use rig_core::message::{Message, UserContent};
use rig_core::streaming::{RawStreamingChoice, StreamFinal, StreamingCompletionResponse};
#[cfg(test)]
use tokenizers::Tokenizer;

use crate::artifacts::{GgufModelData, ModelArtifacts, ModelData};
use crate::generation::{GenerationConfig, infer, stream_generate, validate_generation};
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
use crate::profile::{ConversationProtocol, ModelArchitecture, ModelFamily, Quantization};
use crate::runtime::CancellationSignal;
#[cfg(all(test, not(target_family = "wasm")))]
use crate::runtime::TestControl;
#[cfg(not(target_family = "wasm"))]
use crate::runtime::{CancelOnDrop, acquire_concurrency};
use crate::types::*;
#[cfg(test)]
use crate::validation::*;

pub(crate) const DEFAULT_MAX_CONCURRENT_REQUESTS: usize = 1;
#[cfg(not(target_family = "wasm"))]
const STREAM_CHANNEL_CAPACITY: usize = 8;

/// A cheaply cloneable, CPU-only Candle completion model.
///
/// This is a *loaded* handle, not a config: constructing one always means
/// validating artifacts and materializing tensors. It is therefore an
/// inherent type with inherent [`complete`](Self::complete) /
/// [`stream`](Self::stream) methods rather than an implementation of any
/// model trait — the documented entry points are
/// [`crate::functions::complete`] and [`crate::functions::open_stream`],
/// which delegate here.
#[derive(Clone)]
pub struct CandleModel {
    loaded: Arc<LoadedModel>,
}

/// Builder for loading a [`CandleModel`] and customizing generation defaults.
pub struct CandleModelBuilder<'a> {
    source: ModelSource<'a>,
    family: Option<ModelFamily>,
    generation: GenerationConfig,
    max_concurrent_requests: usize,
}

enum ModelSource<'a> {
    Owned(ModelArtifacts),
    BorrowedGguf(GgufModelData<'a>),
}

/// Backwards-compatible alias for [`CandleModel`].
///
/// New code should use `CandleModel`, which accurately reflects that the
/// backend also supports validated Qwen3 checkpoints.
pub type LlamaModel = CandleModel;

/// Backwards-compatible alias for [`CandleModelBuilder`].
pub type LlamaModelBuilder<'a> = CandleModelBuilder<'a>;

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

    /// Loads a model from explicitly typed byte-backed artifacts.
    pub fn from_artifacts(artifacts: ModelArtifacts) -> Result<Self, CandleError> {
        Self::builder_from_artifacts(artifacts).build()
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

    /// Asynchronously loads owned GGUF artifacts outside the async executor.
    #[cfg(not(target_family = "wasm"))]
    pub async fn from_gguf_async(data: ModelData) -> Result<Self, CandleError> {
        Self::builder_from_artifacts(ModelArtifacts::Gguf(data))
            .build_async()
            .await
    }

    /// Asynchronously loads borrowed static GGUF artifacts outside the async executor.
    ///
    /// Static buffers such as `include_bytes!` remain zero-copy at the API
    /// boundary and satisfy the blocking task's ownership requirement.
    #[cfg(not(target_family = "wasm"))]
    pub async fn from_gguf_bytes_async(data: GgufModelData<'static>) -> Result<Self, CandleError> {
        Self::builder_from_gguf_bytes(data).build_async().await
    }

    /// Asynchronously loads explicitly typed owned artifacts outside the async executor.
    #[cfg(not(target_family = "wasm"))]
    pub async fn from_artifacts_async(artifacts: ModelArtifacts) -> Result<Self, CandleError> {
        Self::builder_from_artifacts(artifacts).build_async().await
    }

    /// Returns the validated conversation/output protocol.
    pub fn conversation_protocol(&self) -> ConversationProtocol {
        self.loaded.profile.definition.protocol
    }

    /// Backwards-compatible alias for [`Self::conversation_protocol`].
    pub fn model_family(&self) -> ModelFamily {
        self.conversation_protocol()
    }

    /// Returns the validated transformer architecture of the loaded checkpoint.
    pub fn architecture(&self) -> ModelArchitecture {
        self.loaded.profile.definition.architecture
    }

    /// Returns the detected checkpoint quantization, if the model is quantized.
    pub fn quantization(&self) -> Option<Quantization> {
        self.loaded.profile.definition.quantization
    }

    /// Runs a buffered completion against the loaded weights.
    ///
    /// Prefer [`crate::functions::complete`], which is the crate's
    /// documented data-oriented entry point and delegates here.
    pub async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, CompletionError> {
        complete_request(self, request).await
    }

    /// Opens a streaming completion against the loaded weights.
    ///
    /// Prefer [`crate::functions::open_stream`], which is the crate's
    /// documented data-oriented entry point and delegates here.
    pub async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse, CompletionError> {
        open_stream_request(self, request).await
    }
}

impl<'a> CandleModelBuilder<'a> {
    /// Selects a conversation protocol and requires it to match the artifacts.
    pub fn conversation_protocol(mut self, protocol: ConversationProtocol) -> Self {
        self.family = Some(protocol);
        self
    }

    /// Backwards-compatible alias for [`Self::conversation_protocol`].
    pub fn model_family(mut self, family: ModelFamily) -> Self {
        self.family = Some(family);
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
            loaded: Arc::new(loaded),
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
    render_prompt_for(request, ModelFamily::Llama3)
}

#[cfg(test)]
fn render_prompt_for(
    request: &CompletionRequest,
    family: ModelFamily,
) -> Result<String, CandleError> {
    crate::protocol::render_prompt(request, family)
}

#[cfg(not(target_family = "wasm"))]
type CandleStreamItem = Result<RawStreamingChoice, CompletionError>;

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
    request: CompletionRequest,
    cancellation: &CancellationSignal,
    sender: &tokio::sync::mpsc::Sender<CandleStreamItem>,
) -> Result<(), CandleError> {
    let response = stream_generate(loaded, request, cancellation, |choice| {
        #[cfg(test)]
        if let Some(control) = &loaded.test_control {
            control.record_delivery_attempt();
        }
        sender
            .blocking_send(Ok(choice))
            .map_err(|_| CandleError::StreamingChannelClosed)
    })?;
    let final_response = StreamFinal::new("candle", response.token_usage())
        .with_finish_reason(response.finish_reason.normalized());
    sender
        .blocking_send(Ok(RawStreamingChoice::FinalResponse(final_response)))
        .map_err(|_| CandleError::StreamingChannelClosed)
}

/// Buffered completion over a loaded [`CandleModel`], shared by
/// [`CandleModel::complete`] and [`crate::functions::complete`].
pub(crate) async fn complete_request(
    model: &CandleModel,
    request: CompletionRequest,
) -> Result<CompletionResponse, CompletionError> {
    {
        let loaded = &model.loaded;

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
                    .with_context(|| infer(&loaded, request, &cancellation));
                drop(permit);
                result
            })
            .await
            .map_err(|error| CandleError::BlockingTaskJoin(error.to_string()));
            cancel_on_drop.disarm();
            result?
                .map(|(response, _)| response)
                .map_err(CompletionError::from)
        }

        #[cfg(target_family = "wasm")]
        {
            infer(loaded, request, &CancellationSignal)
                .map(|(response, _)| response)
                .map_err(CompletionError::from)
        }
    }
}

/// Streaming completion over a loaded [`CandleModel`], shared by
/// [`CandleModel::stream`] and [`crate::functions::open_stream`].
pub(crate) async fn open_stream_request(
    model: &CandleModel,
    request: CompletionRequest,
) -> Result<StreamingCompletionResponse, CompletionError> {
    {
        let loaded = &model.loaded;

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
                    stream_infer(&loaded, request, &producer_cancellation, &producer_sender)
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
            let stream = CandleReceiverStream {
                receiver,
                cancellation,
            };
            cancel_on_drop.disarm();
            Ok(StreamingCompletionResponse::stream(stream))
        }

        #[cfg(target_family = "wasm")]
        {
            let mut events = Vec::new();
            let response = stream_generate(loaded, request, &CancellationSignal, |choice| {
                events.push(Ok(choice));
                Ok(())
            })?;
            let final_response = StreamFinal::new("candle", response.token_usage())
                .with_finish_reason(response.finish_reason.normalized());
            events.push(Ok(RawStreamingChoice::FinalResponse(final_response)));
            let stream = futures::stream::iter(events);
            Ok(StreamingCompletionResponse::stream(stream))
        }
    }
}

#[cfg(test)]
#[allow(clippy::panic_in_result_fn)]
mod tests;
