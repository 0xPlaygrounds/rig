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

use futures::StreamExt;
use rig_core::completion::CompletionRequest;
use rig_core::completion::options::{FinalBody, RawAt, request_params};
use rig_core::driver::{Exchange, Opened, Opening, Transport};
use rig_core::error::ProviderError;
use rig_core::message::AssistantContent;
#[cfg(test)]
use rig_core::message::{Message, UserContent};
use rig_core::operation::{Block, Completion, Finish};
use rig_core::wire::{Flow, Mode, Out, WireEvent};
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
use crate::runtime::{CancelOnDrop, ReceiverStream, acquire_concurrency};
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
    /// The completion model this runtime serves: the [`Generation`] wire on
    /// this runtime.
    pub fn completion(&self) -> rig_core::Model<Generation, Self> {
        let generation = Generation {
            model: self.state.model_id.clone(),
            protocol: self.state.profile.definition.protocol,
        };
        rig_core::Model::new(generation, self.clone())
    }

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

    /// The id replay names this checkpoint by: its protocol and a digest of
    /// its config and tokenizer, such as `qwen3-1f0c4e5a9b2d7c38`. A turn
    /// from another id replays as another model's.
    pub fn model_id(&self) -> &str {
        &self.state.model_id
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
fn stream_infer(
    loaded: &LoadedModel,
    request: &CandleRequest,
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
///     candle.completion()
/// }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Generation {
    /// The loaded checkpoint's id ([`CandleModel::model_id`]).
    pub model: String,
    /// The protocol its prompts are rendered in.
    pub protocol: ConversationProtocol,
}

impl Generation {
    /// The prompt this wire's protocol renders for a prepared `request`.
    #[doc(hidden)]
    pub fn prompt(&self, request: &CompletionRequest) -> Result<String, CandleError> {
        crate::protocol::render_prompt(request, self.protocol)
    }
}

/// A local generation's payload: the prepared request, and its options and
/// `additional_params` merged into the generation overrides they name
/// (`top_p`, `seed`, `top_k`, `repeat_penalty`, `repeat_last_n`).
#[derive(Clone, Debug)]
pub struct CandleRequest {
    /// The prepared request.
    pub request: CompletionRequest,
    /// The merged generation overrides.
    pub params: FinalBody,
}

/// One unit of a local generation's reply.
pub enum CandleFrame {
    /// A whole unary turn: the local response record and its parsed content.
    Whole(crate::generation::InferredCompletion),
    /// One streamed generation event.
    Event(GenerationEvent),
}

impl rig_core::wire::Wire for Generation {
    type Op = rig_core::operation::Completion;
    type Payload = CandleRequest;
    type Frame = CandleFrame;
    type Decoder<'id> = CandleAdapter<'id>;
    type Reassembler = CandleDocument;

    fn describe(&self) -> rig_core::wire::Descriptor<'_> {
        rig_core::wire::Descriptor::new(crate::types::PROVIDER_NAME)
            .model(self.model.as_str())
            .replay(self)
    }

    /// The prompt protocol decides which reasoning a local model can render
    /// and refuses the rest, so the request passes as it is, beside its
    /// merged generation overrides. A raw `tools` stays in them, for the
    /// generator to refuse.
    fn encode(
        &self,
        request: CompletionRequest,
        _mode: Mode,
    ) -> Result<CandleRequest, rig_core::error::EncodeError> {
        let params = request_params(
            self,
            &request,
            |input| {
                let tools = input.raw_tools()?;
                let mut base = serde_json::Map::new();
                if !tools.is_empty() {
                    base.insert("tools".to_owned(), serde_json::Value::Array(tools));
                }
                Ok(base)
            },
            RawAt::Top,
            &[],
        )?;
        Ok(CandleRequest { request, params })
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        CandleAdapter::default()
    }
}

impl rig_core::completion::ReplayTarget for Generation {
    /// A local generation's settings: `top_p` and `seed` are generation
    /// overrides; thinking is off in every prompt protocol's default
    /// rendering, and nothing is cached across requests.
    fn map_options(
        &self,
        _request: &CompletionRequest,
        fields: rig_core::completion::options::OptionFields<'_>,
    ) -> rig_core::completion::options::OptionMap {
        use rig_core::completion::options::{Mapping, OptionFields, OptionMap};
        use rig_core::completion::{CacheRetention, Reasoning};
        let OptionFields {
            reasoning,
            cache,
            service_tier,
            verbosity,
            parallel_tool_calls,
            top_p,
            seed,
            stop,
        } = fields;
        const LOCAL: &str = "a local generation has no such setting";
        OptionMap {
            reasoning: Mapping::of(reasoning, |reasoning| match reasoning {
                Reasoning::Off => Mapping::Omit(
                    "Qwen3 renders its no-thinking mode by default; other models do not reason",
                ),
                _ => Mapping::unsupported("a local model takes no reasoning level or budget"),
            }),
            cache: Mapping::of(cache, |cache| match cache {
                CacheRetention::None => Mapping::Omit("nothing is cached across requests"),
                _ => Mapping::unsupported("a local generation keeps no prompt cache"),
            }),
            service_tier: Mapping::of(service_tier, |_| Mapping::unsupported(LOCAL)),
            verbosity: Mapping::of(verbosity, |_| Mapping::unsupported(LOCAL)),
            parallel_tool_calls: Mapping::of(parallel_tool_calls, |_| Mapping::unsupported(LOCAL)),
            top_p: Mapping::of(top_p, |top_p| {
                if top_p.is_finite() && 0.0 < top_p && top_p <= 1.0 {
                    Mapping::Send(serde_json::json!({ "top_p": top_p }))
                } else {
                    Mapping::unsupported("`top_p` must be in (0, 1]")
                }
            }),
            seed: Mapping::of(seed, |seed| {
                Mapping::Send(serde_json::json!({ "seed": seed }))
            }),
            stop: Mapping::of_stop(stop, |_| {
                Mapping::unsupported("local generation stops only on end tokens")
            }),
        }
    }

    // A local runtime has no finish vocabulary: its decoder states a stop.
    fn states_finish_reason(&self) -> bool {
        false
    }

    fn api(&self) -> rig_core::message::Api {
        rig_core::message::Api::from_static("candle.generate")
    }

    fn provider(&self) -> &str {
        crate::types::PROVIDER_NAME
    }

    /// A local runtime addresses the one checkpoint it loaded.
    fn model(&self) -> &str {
        &self.model
    }

    /// A local model reads text, and tools only in the Qwen3 protocol, so
    /// another model's tool exchange reaches Llama 3 and SmolLM2 as text.
    fn accepts(&self, _model: &str) -> rig_core::completion::Accepts {
        rig_core::completion::Accepts {
            tools: self.protocol == ConversationProtocol::Qwen3,
            ..rig_core::completion::Accepts::TEXT
        }
    }

    /// The prompt renderers take text only: a text document arrives as its
    /// text, and every other media part as a placeholder.
    fn encodes(&self, _model: &str, _media: rig_core::completion::Media<'_>) -> bool {
        false
    }
}

/// A streamed generation's document: its local response record, as a
/// unary reply's is.
#[derive(Debug, Default)]
pub struct CandleDocument(Option<serde_json::Value>);

impl rig_core::wire::document::Serves<rig_core::operation::Completion> for CandleDocument {}

impl rig_core::wire::document::Reassemble<CandleFrame> for CandleDocument {
    fn absorb(&mut self, frame: &CandleFrame) {
        let response = match frame {
            CandleFrame::Event(GenerationEvent::Final(response)) => response,
            CandleFrame::Whole(inferred) => &inferred.response,
            CandleFrame::Event(_) => return,
        };
        self.0 = serde_json::to_value(response).ok();
    }

    fn finish(self) -> serde_json::Value {
        self.0.unwrap_or(serde_json::Value::Null)
    }
}

/// Writes local generation events into the reply. Every input is modeled;
/// no byte decoding or unknown-frame classification occurs. Channel EOF
/// without a `Final` event means the generator failed or was cancelled: the
/// reply is truncated. Local generation has no provider items: every block
/// is canonical.
#[derive(Default)]
pub struct CandleAdapter<'id> {
    brand: std::marker::PhantomData<fn(&'id ()) -> &'id ()>,
}

impl<'id> rig_core::wire::Decoder<'id, Completion, CandleFrame> for CandleAdapter<'id> {
    type Event = CandleFrame;

    fn classify(&self, frame: CandleFrame) -> WireEvent<Self::Event> {
        WireEvent::Known(frame)
    }

    fn decode(
        &mut self,
        frame: CandleFrame,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        let event = match frame {
            CandleFrame::Event(event) => event,
            CandleFrame::Whole(inferred) => {
                for content in inferred.choice {
                    whole(&mut out, content)?;
                }
                GenerationEvent::Final(inferred.response)
            }
        };
        match event {
            GenerationEvent::Text(text) => {
                out.run(Block::Text, &text)?;
            }
            GenerationEvent::ToolCall(call) => {
                out.end_run()?;
                whole(&mut out, AssistantContent::ToolCall(call))?;
            }
            GenerationEvent::Reasoning(reasoning) => {
                out.end_run()?;
                whole(&mut out, AssistantContent::Reasoning(reasoning))?;
            }
            GenerationEvent::Final(response) => {
                out.end_run()?;
                return Ok(out.end(Finish {
                    usage: (&response).into(),
                    reason: Some(response.finish_reason.into()),
                    ..Finish::default()
                }));
            }
        }
        Ok(Flow::More)
    }
}

/// Write `content`, a block local generation produced whole, at the next
/// index. Local generation has no provider items and makes no images.
fn whole(out: &mut Out<'_, Completion>, content: AssistantContent) -> Result<(), ProviderError> {
    let (block, text) = match content {
        AssistantContent::Text(text) => (Block::Text, text.text),
        AssistantContent::Reasoning(reasoning) => (
            Block::Reasoning {
                redacted: reasoning.redacted,
            },
            reasoning.text,
        ),
        AssistantContent::ToolCall(call) => {
            let arguments = call
                .function
                .invalid_arguments
                .unwrap_or_else(|| serde_json::Value::Object(call.function.arguments).to_string());
            (
                Block::Call {
                    id: call.id,
                    name: call.function.name,
                },
                arguments,
            )
        }
        _ => {
            return Err(ProviderError::Response(
                "local generation produced a block it has no form for".to_owned(),
            ));
        }
    };
    let index = out.fresh_index();
    out.whole(index, block, serde_json::Value::Null, &text)
}

impl Transport<Generation> for CandleModel {
    fn send(&self, request: CandleRequest, exchange: Exchange) -> Opening<CandleFrame> {
        let mode = exchange.mode;
        // A closed admission controller refuses before anything runs, as
        // opening a completion or a stream always has.
        #[cfg(not(target_family = "wasm"))]
        if self.state.concurrency.is_closed() {
            return Opening::failed(CandleError::ConcurrencyControllerClosed.into());
        }
        let model = self.clone();
        Opening::new(async move {
            Ok(match mode {
                // The local response record is the response's `raw`.
                Mode::Unary => match model.infer_completion(request).await {
                    Ok(inferred) => {
                        let document = serde_json::to_value(&inferred.response)?;
                        Opened::new(futures::stream::iter([Ok(CandleFrame::Whole(inferred))]))
                            .with_document(document)
                    }
                    Err(error) => Opened::failed(error),
                },
                Mode::Streaming => match model.open_stream(request).await {
                    Ok(events) => Opened::new(events.map(|event| event.map(CandleFrame::Event))),
                    Err(error) => Opened::failed(error),
                },
            })
        })
    }
}

impl CandleModel {
    async fn infer_completion(
        &self,
        request: CandleRequest,
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
        request: CandleRequest,
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
            Ok(Box::pin(ReceiverStream::new(receiver, cancellation)))
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
