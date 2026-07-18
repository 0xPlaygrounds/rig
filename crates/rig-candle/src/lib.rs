//! Local, CPU-only Llama-compatible and Qwen3 inference for Rig, backed by Candle.
//!
//! Models are loaded entirely from caller-provided owned or borrowed byte
//! buffers. This crate performs no filesystem or network access. On
//! `wasm32-unknown-unknown`, inference runs
//! synchronously inside the completion future; browser applications should own
//! and invoke the model in a Web Worker to avoid blocking the UI thread.
//!
//! ```no_run
//! use rig_agent::{agent::AgentBuilder, completion::Prompt};
//! use rig_candle::{CandleModel, ModelData};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let data = ModelData {
//!     config: std::fs::read("./model/config.json")?,
//!     tokenizer: std::fs::read("./model/tokenizer.json")?,
//!     weights: std::fs::read("./model/model.safetensors")?,
//! };
//! let model = CandleModel::from_safetensors(data)?;
//! let agent = AgentBuilder::new(model)
//!     .preamble("You are a helpful assistant.")
//!     .temperature(0.7)
//!     .max_tokens(256)
//!     .build();
//! let answer = agent.prompt("Explain Rust ownership briefly.").await?;
//! println!("{answer}");
//! # Ok(())
//! # }
//! ```
//!
//! The validated profiles are unsharded Llama 3 safetensors,
//! SmolLM2-360M-Instruct Q4_K_M GGUF, and (on native targets) the official
//! Qwen3-4B-Instruct Q4_K_M GGUF. Conversation rendering is explicit; tokenizer-provided
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

mod protocol;

use std::collections::HashSet;
use std::sync::Arc;

#[cfg(not(target_family = "wasm"))]
use std::sync::atomic::{AtomicBool, Ordering};

use candle_core::quantized::{GgmlDType, gguf_file};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::llama::{
    Cache, Config, Llama, Llama3RopeType, LlamaConfig, LlamaEosToks,
};
use candle_transformers::models::quantized_llama::ModelWeights as QuantizedLlama;
use candle_transformers::models::quantized_qwen3::ModelWeights as QuantizedQwen3;
use candle_transformers::utils::apply_repeat_penalty;
#[cfg(not(target_family = "wasm"))]
use futures::Stream;
use rig_core::OneOrMany;
use rig_core::completion::{
    AssistantContent, CompletionError, CompletionModel, CompletionRequest, CompletionResponse,
    GetTokenUsage, Usage,
};
#[cfg(test)]
use rig_core::message::{Message, UserContent};
use rig_core::streaming::{
    RawStreamingChoice, RawStreamingToolCall, StreamingCompletionResponse, StreamingResult,
};
use safetensors::{Dtype as SafeDtype, SafeTensors};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokenizers::tokenizer::DecodeStream;
use tokenizers::{
    DecoderWrapper, ModelWrapper, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper,
    Tokenizer,
};
use web_time::{Duration, Instant};

const BEGIN_OF_TEXT: &str = "<|begin_of_text|>";
const START_HEADER: &str = "<|start_header_id|>";
const END_HEADER: &str = "<|end_header_id|>";
const END_OF_TURN: &str = "<|eot_id|>";
const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";
const END_OF_TEXT: &str = "<|endoftext|>";
const SMOLLM2_DEFAULT_SYSTEM_PROMPT: &str =
    "You are a helpful AI assistant named SmolLM, trained by Hugging Face";
const QUANTIZED_LLAMA_CONTEXT_LIMIT: usize = 4096;
const DEFAULT_MAX_CONCURRENT_REQUESTS: usize = 1;
#[cfg(not(target_family = "wasm"))]
const STREAM_CHANNEL_CAPACITY: usize = 8;

type TokenDecodeStream<'a> = DecodeStream<
    'a,
    ModelWrapper,
    NormalizerWrapper,
    PreTokenizerWrapper,
    PostProcessorWrapper,
    DecoderWrapper,
>;

/// Sampling and length defaults used when a completion request does not override them.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationConfig {
    /// Maximum number of tokens generated for a request.
    pub max_tokens: u64,
    /// Sampling temperature. Zero selects greedy decoding.
    pub temperature: f64,
    /// Optional number of highest-probability tokens retained during sampling.
    pub top_k: Option<usize>,
    /// Optional nucleus-sampling probability threshold.
    pub top_p: Option<f64>,
    /// Deterministic random seed used by the sampler.
    pub seed: u64,
    /// Penalty applied to tokens repeated in the recent context. `1.0` disables it.
    pub repeat_penalty: f32,
    /// Number of recent tokens considered by the repeat penalty.
    pub repeat_last_n: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.8,
            top_k: None,
            top_p: Some(0.95),
            seed: 299_792_458,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
        }
    }
}

/// Owned model artifacts for exactly one unsharded checkpoint.
#[derive(Debug)]
pub struct ModelData {
    /// Contents of `config.json`.
    pub config: Vec<u8>,
    /// Contents of `tokenizer.json`.
    pub tokenizer: Vec<u8>,
    /// Contents of one safetensors or GGUF checkpoint, as identified by [`ModelArtifacts`].
    pub weights: Vec<u8>,
}

/// Borrowed GGUF artifacts for zero-copy loading from embedded/static bytes.
#[derive(Debug, Clone, Copy)]
pub struct GgufModelData<'a> {
    /// Contents of `config.json`.
    pub config: &'a [u8],
    /// Contents of `tokenizer.json`.
    pub tokenizer: &'a [u8],
    /// Contents of one GGUF checkpoint.
    pub weights: &'a [u8],
}

/// Byte-backed checkpoint format supplied to [`CandleModel`].
#[derive(Debug)]
pub enum ModelArtifacts {
    /// One unsharded Hugging Face safetensors checkpoint.
    Safetensors(ModelData),
    /// A validated SmolLM2 or Qwen3 Q4_K_M GGUF checkpoint.
    Gguf(ModelData),
}

/// Explicit conversation and generated-output protocol selected from validated artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConversationProtocol {
    /// Meta Llama 3 instruct control-token format.
    Llama3,
    /// Hugging Face SmolLM2 instruct (ChatML-style) format.
    SmolLm2,
    /// Qwen3 ChatML/Hermes tool-calling format.
    Qwen3,
}

/// Backwards-compatible name for [`ConversationProtocol`].
pub type ModelFamily = ConversationProtocol;

/// Transformer architecture used to execute a loaded checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelArchitecture {
    /// Candle's Llama implementation, including compatible SmolLM2 checkpoints.
    Llama,
    /// Candle's Qwen3 implementation with per-head query/key normalization.
    Qwen3,
}

/// Quantized tensor encoding detected in a GGUF checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Quantization {
    /// Mixed GGUF tensors whose primary matrix encoding is Q4_K.
    Q4K,
}

/// Why a local Candle completion failed.
#[derive(Debug, Error, Clone)]
#[non_exhaustive]
pub enum CandleError {
    /// A required artifact buffer was empty.
    #[error("the {artifact} buffer is empty")]
    EmptyBuffer { artifact: &'static str },
    /// The Hugging Face configuration could not be parsed or is internally invalid.
    #[error("invalid model configuration: {0}")]
    Configuration(String),
    /// A parsed configuration field is incompatible with Candle's Llama implementation.
    #[error("invalid model configuration field `{field}`: {reason}")]
    InvalidConfigurationValue {
        /// Configuration field name.
        field: &'static str,
        /// Explanation of the invalid value or relationship.
        reason: String,
    },
    /// The tokenizer bytes could not be loaded.
    #[error("invalid tokenizer: {0}")]
    TokenizerLoading(String),
    /// The tokenizer metadata does not identify a supported prompt family.
    #[error("unsupported model family: {0}")]
    UnsupportedModelFamily(String),
    /// An explicitly selected model family disagrees with validated artifacts.
    #[error("selected model family {selected:?} does not match detected family {detected:?}")]
    ModelFamilyMismatch {
        /// Family requested by the caller.
        selected: ModelFamily,
        /// Family detected from the tokenizer.
        detected: ModelFamily,
    },
    /// Independently supplied model artifacts disagree with one another.
    #[error("{artifact} does not match the selected model artifacts: {reason}")]
    ArtifactMismatch {
        /// Artifact or metadata field that disagreed.
        artifact: &'static str,
        /// Human-readable mismatch details.
        reason: String,
    },
    /// The tokenizer vocabulary does not agree with the model configuration.
    #[error("tokenizer vocabulary size {actual} does not match config.vocab_size {expected}")]
    TokenizerVocabularyMismatch {
        /// Vocabulary size required by the model configuration.
        expected: usize,
        /// Vocabulary size reported by the tokenizer, including added tokens.
        actual: usize,
    },
    /// A selected prompt-format token is absent from the tokenizer.
    #[error("tokenizer is missing required prompt-format special token `{token}`")]
    MissingSpecialToken {
        /// Required special-token string.
        token: &'static str,
    },
    /// A prompt-format token exists but is not registered as special.
    #[error("tokenizer token `{token}` must be registered as a special token")]
    SpecialTokenNotMarked {
        /// Formatting token that must be treated atomically by the tokenizer.
        token: &'static str,
    },
    /// A configured or tokenizer-provided token ID lies outside the model vocabulary.
    #[error("token `{token}` has ID {id}, outside vocabulary size {vocab_size}")]
    TokenIdOutOfRange {
        /// Configuration field or special-token name.
        token: String,
        /// Invalid token ID.
        id: u32,
        /// Model vocabulary size.
        vocab_size: usize,
    },
    /// The tokenizer failed to encode the rendered prompt.
    #[error("tokenizer encoding failed: {0}")]
    TokenizerEncoding(String),
    /// The tokenizer failed to decode generated token IDs.
    #[error("tokenizer decoding failed: {0}")]
    TokenizerDecoding(String),
    /// The safetensors checkpoint was malformed or incompatible with the configuration.
    #[error("invalid or incompatible safetensors checkpoint: {0}")]
    InvalidCheckpoint(String),
    /// A GGUF checkpoint was malformed or inconsistent with its configuration.
    #[error("invalid or incompatible GGUF checkpoint: {0}")]
    InvalidQuantizedCheckpoint(String),
    /// The GGUF checkpoint does not use a supported production quantization.
    #[error("unsupported GGUF quantization: {0}")]
    UnsupportedQuantization(String),
    /// A message contains content that the selected text-only prompt renderer cannot represent.
    #[error("unsupported prompt content: {0}")]
    UnsupportedPromptContent(&'static str),
    /// Caller-controlled content contains a delimiter reserved by the selected
    /// chat template and therefore cannot be interpolated safely.
    #[error("{field} contains reserved protocol marker `{marker}`")]
    ReservedProtocolMarker {
        /// Kind of prompt content containing the marker.
        field: &'static str,
        /// Structural delimiter that was rejected.
        marker: &'static str,
    },
    /// A tensor required by the configured architecture was absent.
    #[error("checkpoint is missing expected tensor `{0}`")]
    MissingTensor(String),
    /// A checkpoint tensor has an incompatible shape.
    #[error("tensor `{tensor}` has shape {actual:?}, expected {expected:?}")]
    TensorShapeMismatch {
        /// Tensor name in the checkpoint.
        tensor: String,
        /// Shape required by Candle's Llama implementation.
        expected: Vec<usize>,
        /// Shape stored in the checkpoint.
        actual: Vec<usize>,
    },
    /// A checkpoint tensor uses a dtype outside the portable CPU scope.
    #[error("tensor `{tensor}` uses safetensors dtype `{dtype}`; expected F32, F16, or BF16")]
    UnsupportedTensorDtype {
        /// Tensor name in the checkpoint.
        tensor: String,
        /// Unsupported safetensors dtype.
        dtype: String,
    },
    /// Neither the configuration nor tokenizer supplied a usable stop token.
    #[error("unable to determine a valid EOS or model end-of-turn token")]
    MissingStopToken,
    /// Candle could not load the model tensors.
    #[error("Candle model loading failed: {0}")]
    ModelLoading(String),
    /// Candle inference failed.
    #[error("Candle inference failed: {0}")]
    Inference(String),
    /// A generation setting was invalid.
    #[error("invalid generation setting: {0}")]
    InvalidGeneration(String),
    /// The encoded prompt itself exceeds the model context window.
    #[error("prompt length {prompt_tokens} exceeds the model context limit {context_limit}")]
    PromptTooLong {
        /// Number of prompt tokens.
        prompt_tokens: usize,
        /// Configured model context limit.
        context_limit: usize,
    },
    /// The prompt fills the context window, leaving no capacity for generation.
    #[error(
        "prompt length {prompt_tokens} leaves no generation capacity in context limit {context_limit}"
    )]
    NoGenerationCapacity {
        /// Number of prompt tokens.
        prompt_tokens: usize,
        /// Configured model context limit.
        context_limit: usize,
    },
    /// A numeric generation value cannot be represented on the current platform.
    #[error("generation value `{field}`={value} cannot be represented on this platform")]
    NumericConversion {
        /// Name of the value being converted.
        field: &'static str,
        /// Original portable value.
        value: u64,
    },
    /// The native inference concurrency limit is invalid.
    #[error("max_concurrent_requests must be greater than zero")]
    InvalidConcurrencyLimit,
    /// The native inference admission controller was closed unexpectedly.
    #[cfg(not(target_family = "wasm"))]
    #[error("Candle inference concurrency controller is closed")]
    ConcurrencyControllerClosed,
    /// Local inference was cooperatively cancelled.
    #[error("Candle inference was cancelled")]
    Cancelled,
    /// The request uses a feature outside the selected local-model protocol.
    #[error("unsupported Candle request feature: {0}")]
    UnsupportedFeature(String),
    /// A tool definition cannot be represented safely by the selected protocol.
    #[error("invalid tool definition `{tool}`: {reason}")]
    InvalidToolDefinition {
        /// Tool name, or a placeholder when the name itself is invalid.
        tool: String,
        /// Validation failure.
        reason: String,
    },
    /// A model-generated tool call was malformed or incomplete.
    #[error("malformed Qwen3 tool call: {0}")]
    MalformedToolCall(String),
    /// The generated response violated the requested tool-choice policy.
    #[error("Qwen3 tool-choice violation: {0}")]
    ToolChoiceViolation(String),
    /// A tool result could not be correlated with a preceding assistant call.
    #[error("tool result `{result_id}` does not match a preceding unresolved tool call")]
    UnmatchedToolResult {
        /// Result/call identifier supplied in history.
        result_id: String,
    },
    /// `CompletionModel::make` cannot load a byte-backed model.
    #[error(
        "`CompletionModel::make` is unsupported for rig-candle; use a byte-backed `CandleModel` constructor or builder"
    )]
    UnsupportedMake,
    /// A native blocking inference task could not be joined.
    #[cfg(not(target_family = "wasm"))]
    #[error("Candle blocking task failed: {0}")]
    BlockingTaskJoin(String),
    /// The consumer of a native streaming response closed its bounded channel.
    #[cfg(not(target_family = "wasm"))]
    #[error("Candle streaming response channel was closed")]
    StreamingChannelClosed,
}

impl From<CandleError> for CompletionError {
    fn from(error: CandleError) -> Self {
        CompletionError::ProviderError(error.to_string())
    }
}

/// The reason local generation ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// A configured EOS or family-specific end-of-turn token was sampled.
    Eos,
    /// The configured maximum output length was reached.
    MaxTokens,
}

/// Serializable details returned alongside a Rig completion response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandleCompletionResponse {
    /// Decoded generated text, excluding the prompt and stop token.
    pub text: String,
    /// Number of encoded prompt tokens.
    pub prompt_tokens: u64,
    /// Number of sampled output tokens, including an EOS token when sampled.
    pub generated_tokens: u64,
    /// Maximum output tokens selected by request/default precedence before context clamping.
    pub requested_max_tokens: u64,
    /// Maximum output tokens available after applying the model context limit.
    pub effective_max_tokens: u64,
    /// Why generation ended.
    pub finish_reason: FinishReason,
    /// Time spent preparing the prompt tensor and running its initial forward pass.
    pub prefill_duration_ms: u64,
    /// Time from generation start until the first sampled token, in milliseconds.
    pub time_to_first_token_ms: Option<u64>,
    /// Total time spent in prefill and token generation, in milliseconds.
    pub generation_duration_ms: u64,
    /// Generated tokens per second when the measured duration is nonzero.
    pub tokens_per_second: Option<f64>,
}

impl GetTokenUsage for CandleCompletionResponse {
    fn token_usage(&self) -> Usage {
        Usage {
            input_tokens: self.prompt_tokens,
            output_tokens: self.generated_tokens,
            total_tokens: self.prompt_tokens.saturating_add(self.generated_tokens),
            ..Usage::new()
        }
    }
}

#[derive(Clone)]
enum ModelState {
    Ready(Arc<LoadedModel>),
    UnsupportedMake,
}

struct LoadedModel {
    model: LoadedWeights,
    architecture: ModelArchitecture,
    family: ModelFamily,
    quantization: Option<Quantization>,
    tokenizer: Tokenizer,
    spec: ModelSpec,
    stop_tokens: HashSet<u32>,
    generation: GenerationConfig,
    #[cfg(not(target_family = "wasm"))]
    concurrency: Arc<tokio::sync::Semaphore>,
    #[cfg(all(test, not(target_family = "wasm")))]
    test_control: Option<Arc<TestControl>>,
}

enum LoadedWeights {
    Safetensors { model: Llama, config: Config },
    QuantizedLlama(QuantizedLlama),
    QuantizedQwen3(QuantizedQwen3),
}

#[derive(Debug, Clone, Copy)]
struct ModelSpec {
    vocab_size: usize,
    context_limit: usize,
}

struct PreparedModel {
    architecture: ModelArchitecture,
    family: ModelFamily,
    tokenizer: Tokenizer,
    stop_tokens: HashSet<u32>,
    spec: ModelSpec,
    llama_config: Option<Config>,
    qwen3_config: Option<Qwen3Config>,
}

#[derive(Debug, Clone, Deserialize)]
struct Qwen3Config {
    #[serde(default)]
    architectures: Vec<String>,
    model_type: String,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    max_position_embeddings: usize,
    vocab_size: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    tie_word_embeddings: bool,
    bos_token_id: u32,
    eos_token_id: u32,
    hidden_act: String,
    attention_bias: bool,
}

#[cfg(all(test, not(target_family = "wasm")))]
struct TestControl {
    gate: std::sync::Mutex<bool>,
    gate_changed: std::sync::Condvar,
    entered: AtomicBool,
    entered_notification: tokio::sync::Notify,
    panic_after_gate: AtomicBool,
    delivery_attempts: std::sync::atomic::AtomicUsize,
    delivery_notification: tokio::sync::Notify,
}

#[cfg(all(test, not(target_family = "wasm")))]
impl TestControl {
    fn new(blocked: bool, panic_after_gate: bool) -> Self {
        Self {
            gate: std::sync::Mutex::new(blocked),
            gate_changed: std::sync::Condvar::new(),
            entered: AtomicBool::new(false),
            entered_notification: tokio::sync::Notify::new(),
            panic_after_gate: AtomicBool::new(panic_after_gate),
            delivery_attempts: std::sync::atomic::AtomicUsize::new(0),
            delivery_notification: tokio::sync::Notify::new(),
        }
    }

    fn enter_generation(&self) -> Result<(), CandleError> {
        self.entered.store(true, Ordering::Release);
        self.entered_notification.notify_waiters();
        let mut blocked = self
            .gate
            .lock()
            .map_err(|_| CandleError::Inference("test generation gate was poisoned".to_string()))?;
        while *blocked {
            blocked = self.gate_changed.wait(blocked).map_err(|_| {
                CandleError::Inference("test generation gate was poisoned".to_string())
            })?;
        }
        if self.panic_after_gate.load(Ordering::Acquire) {
            std::panic::resume_unwind(Box::new("intentional blocking-task test panic"));
        }
        Ok(())
    }

    async fn wait_until_entered(&self) {
        loop {
            let notified = self.entered_notification.notified();
            if self.entered.load(Ordering::Acquire) {
                return;
            }
            notified.await;
        }
    }

    fn release(&self) -> Result<(), CandleError> {
        let mut blocked = self
            .gate
            .lock()
            .map_err(|_| CandleError::Inference("test generation gate was poisoned".to_string()))?;
        *blocked = false;
        self.gate_changed.notify_all();
        Ok(())
    }

    fn record_delivery_attempt(&self) {
        self.delivery_attempts.fetch_add(1, Ordering::AcqRel);
        self.delivery_notification.notify_waiters();
    }

    async fn wait_for_delivery_attempts(&self, expected: usize) {
        loop {
            let notified = self.delivery_notification.notified();
            if self.delivery_attempts.load(Ordering::Acquire) >= expected {
                return;
            }
            notified.await;
        }
    }
}

/// A cheaply cloneable, CPU-only Candle completion model.
#[derive(Clone)]
pub struct CandleModel {
    state: ModelState,
}

/// Builder for loading a [`CandleModel`] and customizing generation defaults.
pub struct CandleModelBuilder {
    artifacts: ModelArtifacts,
    family: Option<ModelFamily>,
    generation: GenerationConfig,
    max_concurrent_requests: usize,
}

/// Backwards-compatible alias for [`CandleModel`].
///
/// New code should use `CandleModel`, which accurately reflects that the
/// backend also supports validated Qwen3 checkpoints.
pub type LlamaModel = CandleModel;

/// Backwards-compatible alias for [`CandleModelBuilder`].
pub type LlamaModelBuilder = CandleModelBuilder;

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
        let generation = GenerationConfig::default();
        validate_generation(&generation, None)?;
        let loaded = load_gguf_model(data, None, generation, DEFAULT_MAX_CONCURRENT_REQUESTS)?;
        Ok(Self {
            state: ModelState::Ready(Arc::new(loaded)),
        })
    }

    /// Loads a model from explicitly typed byte-backed artifacts.
    pub fn from_artifacts(artifacts: ModelArtifacts) -> Result<Self, CandleError> {
        Self::builder_from_artifacts(artifacts).build()
    }

    /// Starts a byte-backed model builder.
    pub fn builder(data: ModelData) -> CandleModelBuilder {
        Self::builder_from_artifacts(ModelArtifacts::Safetensors(data))
    }

    /// Starts a builder from explicitly typed byte-backed artifacts.
    pub fn builder_from_artifacts(artifacts: ModelArtifacts) -> CandleModelBuilder {
        CandleModelBuilder {
            artifacts,
            family: None,
            generation: GenerationConfig::default(),
            max_concurrent_requests: DEFAULT_MAX_CONCURRENT_REQUESTS,
        }
    }

    /// Returns the validated conversation/output protocol.
    pub fn conversation_protocol(&self) -> Option<ConversationProtocol> {
        match &self.state {
            ModelState::Ready(loaded) => Some(loaded.family),
            ModelState::UnsupportedMake => None,
        }
    }

    /// Backwards-compatible alias for [`Self::conversation_protocol`].
    pub fn model_family(&self) -> Option<ModelFamily> {
        self.conversation_protocol()
    }

    /// Returns the validated transformer architecture of the loaded checkpoint.
    pub fn architecture(&self) -> Option<ModelArchitecture> {
        match &self.state {
            ModelState::Ready(loaded) => Some(loaded.architecture),
            ModelState::UnsupportedMake => None,
        }
    }

    /// Returns the detected checkpoint quantization, if the model is quantized.
    pub fn quantization(&self) -> Option<Quantization> {
        match &self.state {
            ModelState::Ready(loaded) => loaded.quantization,
            ModelState::UnsupportedMake => None,
        }
    }
}

impl CandleModelBuilder {
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
        let loaded = load_model_with_family(
            self.artifacts,
            self.family,
            self.generation,
            self.max_concurrent_requests,
        )?;
        Ok(CandleModel {
            state: ModelState::Ready(Arc::new(loaded)),
        })
    }
}

fn require_nonempty(bytes: &[u8], artifact: &'static str) -> Result<(), CandleError> {
    if bytes.is_empty() {
        Err(CandleError::EmptyBuffer { artifact })
    } else {
        Ok(())
    }
}

fn load_model_with_family(
    artifacts: ModelArtifacts,
    selected_family: Option<ModelFamily>,
    generation: GenerationConfig,
    _max_concurrent_requests: usize,
) -> Result<LoadedModel, CandleError> {
    let data = match artifacts {
        ModelArtifacts::Safetensors(data) => data,
        ModelArtifacts::Gguf(data) => {
            return load_gguf_model(
                GgufModelData {
                    config: &data.config,
                    tokenizer: &data.tokenizer,
                    weights: &data.weights,
                },
                selected_family,
                generation,
                _max_concurrent_requests,
            );
        }
    };
    require_nonempty(&data.config, "config")?;
    require_nonempty(&data.tokenizer, "tokenizer")?;
    require_nonempty(&data.weights, "weights")?;

    let mut prepared = prepare_model(&data.config, &data.tokenizer, selected_family)?;
    if prepared.architecture != ModelArchitecture::Llama {
        return Err(CandleError::UnsupportedModelFamily(
            "Qwen3 is currently supported through validated GGUF checkpoints only".to_string(),
        ));
    }
    let config = prepared.llama_config.take().ok_or_else(|| {
        CandleError::Configuration("prepared Llama model omitted its configuration".to_string())
    })?;

    let device = Device::Cpu;
    validate_checkpoint(&data.weights, &config)?;
    let builder = VarBuilder::from_buffered_safetensors(data.weights, DType::F32, &device)
        .map_err(|error| CandleError::InvalidCheckpoint(error.to_string()))?;
    let model = Llama::load(builder, &config)
        .map_err(|error| CandleError::ModelLoading(error.to_string()))?;

    Ok(LoadedModel {
        model: LoadedWeights::Safetensors { model, config },
        architecture: prepared.architecture,
        family: prepared.family,
        quantization: None,
        tokenizer: prepared.tokenizer,
        spec: prepared.spec,
        stop_tokens: prepared.stop_tokens,
        generation,
        #[cfg(not(target_family = "wasm"))]
        concurrency: Arc::new(tokio::sync::Semaphore::new(_max_concurrent_requests)),
        #[cfg(all(test, not(target_family = "wasm")))]
        test_control: None,
    })
}

fn load_gguf_model(
    data: GgufModelData<'_>,
    selected_family: Option<ModelFamily>,
    generation: GenerationConfig,
    _max_concurrent_requests: usize,
) -> Result<LoadedModel, CandleError> {
    require_nonempty(data.config, "config")?;
    require_nonempty(data.tokenizer, "tokenizer")?;
    require_nonempty(data.weights, "weights")?;
    let mut prepared = prepare_model(data.config, data.tokenizer, selected_family)?;
    #[cfg(target_family = "wasm")]
    if prepared.architecture == ModelArchitecture::Qwen3 {
        return Err(CandleError::UnsupportedModelFamily(
            "the validated Qwen3-4B profile is native-only because its runtime memory exceeds wasm32 linear-memory capacity; use SmolLM2 for WASM"
                .to_string(),
        ));
    }
    let device = Device::Cpu;
    let model = load_gguf(data.weights, &prepared, &device)?;
    prepared.spec.context_limit = prepared
        .spec
        .context_limit
        .min(QUANTIZED_LLAMA_CONTEXT_LIMIT);
    Ok(LoadedModel {
        model,
        architecture: prepared.architecture,
        family: prepared.family,
        quantization: Some(Quantization::Q4K),
        tokenizer: prepared.tokenizer,
        spec: prepared.spec,
        stop_tokens: prepared.stop_tokens,
        generation,
        #[cfg(not(target_family = "wasm"))]
        concurrency: Arc::new(tokio::sync::Semaphore::new(_max_concurrent_requests)),
        #[cfg(all(test, not(target_family = "wasm")))]
        test_control: None,
    })
}

fn prepare_model(
    config_bytes: &[u8],
    tokenizer_bytes: &[u8],
    selected_family: Option<ModelFamily>,
) -> Result<PreparedModel, CandleError> {
    let identity: ModelIdentity = serde_json::from_slice(config_bytes)
        .map_err(|error| CandleError::Configuration(error.to_string()))?;
    let tokenizer = Tokenizer::from_bytes(tokenizer_bytes)
        .map_err(|error| CandleError::TokenizerLoading(error.to_string()))?;
    let is_qwen3 = identity.model_type.as_deref() == Some("qwen3")
        || identity
            .architectures
            .iter()
            .any(|architecture| architecture == "Qwen3ForCausalLM");
    if is_qwen3 {
        let config: Qwen3Config = serde_json::from_slice(config_bytes)
            .map_err(|error| CandleError::Configuration(error.to_string()))?;
        validate_qwen3_config(&config)?;
        let detected_family = ModelFamily::Qwen3;
        if let Some(selected) = selected_family
            && selected != detected_family
        {
            return Err(CandleError::ModelFamilyMismatch {
                selected,
                detected: detected_family,
            });
        }
        validate_qwen3_tokenizer(&config, &tokenizer)?;
        let mut stop_tokens = HashSet::new();
        stop_tokens.insert(config.eos_token_id);
        return Ok(PreparedModel {
            architecture: ModelArchitecture::Qwen3,
            family: detected_family,
            tokenizer,
            stop_tokens,
            spec: ModelSpec {
                vocab_size: config.vocab_size,
                context_limit: config.max_position_embeddings,
            },
            llama_config: None,
            qwen3_config: Some(config),
        });
    }

    let llama_config: LlamaConfig = serde_json::from_slice(config_bytes)
        .map_err(|error| CandleError::Configuration(error.to_string()))?;
    let config = llama_config.into_config(false);
    validate_model_config(&config)?;
    let detected_family = detect_model_family(&tokenizer)?;
    if let Some(selected) = selected_family
        && selected != detected_family
    {
        return Err(CandleError::ModelFamilyMismatch {
            selected,
            detected: detected_family,
        });
    }
    validate_family_config(config_bytes, &config, detected_family)?;
    validate_tokenizer(&config, &tokenizer, detected_family)?;
    let stop_tokens = resolve_stop_tokens(&config, &tokenizer, detected_family)?;
    let spec = ModelSpec {
        vocab_size: config.vocab_size,
        context_limit: config.max_position_embeddings,
    };
    Ok(PreparedModel {
        architecture: ModelArchitecture::Llama,
        family: detected_family,
        tokenizer,
        stop_tokens,
        spec,
        llama_config: Some(config),
        qwen3_config: None,
    })
}

#[cfg(test)]
fn load_model(
    data: ModelData,
    generation: GenerationConfig,
    max_concurrent_requests: usize,
) -> Result<LoadedModel, CandleError> {
    load_model_with_family(
        ModelArtifacts::Safetensors(data),
        None,
        generation,
        max_concurrent_requests,
    )
}

fn validate_model_config(config: &Config) -> Result<(), CandleError> {
    for (field, value) in [
        ("hidden_size", config.hidden_size),
        ("intermediate_size", config.intermediate_size),
        ("vocab_size", config.vocab_size),
        ("num_hidden_layers", config.num_hidden_layers),
        ("num_attention_heads", config.num_attention_heads),
        ("num_key_value_heads", config.num_key_value_heads),
        ("max_position_embeddings", config.max_position_embeddings),
    ] {
        if value == 0 {
            return Err(CandleError::InvalidConfigurationValue {
                field,
                reason: "value 0 must be greater than zero".to_string(),
            });
        }
    }
    if !config
        .hidden_size
        .is_multiple_of(config.num_attention_heads)
    {
        return Err(CandleError::InvalidConfigurationValue {
            field: "hidden_size",
            reason: format!(
                "value {} must be divisible by num_attention_heads {}",
                config.hidden_size, config.num_attention_heads
            ),
        });
    }
    if !config
        .num_attention_heads
        .is_multiple_of(config.num_key_value_heads)
    {
        return Err(CandleError::InvalidConfigurationValue {
            field: "num_attention_heads",
            reason: format!(
                "value {} must be divisible by num_key_value_heads {}",
                config.num_attention_heads, config.num_key_value_heads
            ),
        });
    }
    let head_dim = config.hidden_size / config.num_attention_heads;
    if !head_dim.is_multiple_of(2) {
        return Err(CandleError::InvalidConfigurationValue {
            field: "hidden_size",
            reason: format!(
                "attention head dimension {head_dim} must be even for rotary embeddings"
            ),
        });
    }
    if !config.rms_norm_eps.is_finite() || config.rms_norm_eps <= 0.0 {
        return Err(CandleError::InvalidConfigurationValue {
            field: "rms_norm_eps",
            reason: format!(
                "value {} must be finite and greater than zero",
                config.rms_norm_eps
            ),
        });
    }
    if !config.rope_theta.is_finite() || config.rope_theta <= 0.0 {
        return Err(CandleError::InvalidConfigurationValue {
            field: "rope_theta",
            reason: format!(
                "value {} must be finite and greater than zero",
                config.rope_theta
            ),
        });
    }
    if config.max_position_embeddings > u32::MAX as usize {
        return Err(CandleError::InvalidConfigurationValue {
            field: "max_position_embeddings",
            reason: format!(
                "value {} exceeds Candle's maximum representable value {}",
                config.max_position_embeddings,
                u32::MAX
            ),
        });
    }
    if let Some(rope_scaling) = &config.rope_scaling
        && matches!(rope_scaling.rope_type, Llama3RopeType::Llama3)
    {
        validate_positive_finite("rope_scaling.factor", rope_scaling.factor)?;
        validate_positive_finite("rope_scaling.low_freq_factor", rope_scaling.low_freq_factor)?;
        validate_positive_finite(
            "rope_scaling.high_freq_factor",
            rope_scaling.high_freq_factor,
        )?;
        if rope_scaling.high_freq_factor <= rope_scaling.low_freq_factor {
            return Err(CandleError::InvalidConfigurationValue {
                field: "rope_scaling.high_freq_factor",
                reason: format!(
                    "value {} must be greater than low_freq_factor {}",
                    rope_scaling.high_freq_factor, rope_scaling.low_freq_factor
                ),
            });
        }
        if rope_scaling.original_max_position_embeddings == 0 {
            return Err(CandleError::InvalidConfigurationValue {
                field: "rope_scaling.original_max_position_embeddings",
                reason: "value 0 must be greater than zero".to_string(),
            });
        }
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
struct ModelIdentity {
    #[serde(default)]
    architectures: Vec<String>,
    model_type: Option<String>,
    hidden_act: Option<String>,
    attention_bias: Option<bool>,
    mlp_bias: Option<bool>,
    rope_interleaved: Option<bool>,
}

fn validate_family_config(
    config_bytes: &[u8],
    config: &Config,
    family: ModelFamily,
) -> Result<(), CandleError> {
    let identity: ModelIdentity = serde_json::from_slice(config_bytes)
        .map_err(|error| CandleError::Configuration(error.to_string()))?;
    if identity
        .model_type
        .as_deref()
        .is_some_and(|model_type| model_type != "llama")
        || (!identity.architectures.is_empty()
            && !identity
                .architectures
                .iter()
                .any(|architecture| architecture == "LlamaForCausalLM"))
    {
        return Err(CandleError::UnsupportedModelFamily(
            "configuration must declare the LlamaForCausalLM architecture".to_string(),
        ));
    }
    if family == ModelFamily::SmolLm2 {
        if identity.model_type.as_deref() != Some("llama")
            || !identity
                .architectures
                .iter()
                .any(|architecture| architecture == "LlamaForCausalLM")
            || identity.hidden_act.as_deref() != Some("silu")
            || identity.attention_bias != Some(false)
            || identity.mlp_bias != Some(false)
            || identity.rope_interleaved != Some(false)
        {
            return Err(CandleError::ArtifactMismatch {
                artifact: "config.json",
                reason: "SmolLM2 requires LlamaForCausalLM with SiLU, bias-free projections, and non-interleaved RoPE metadata".to_string(),
            });
        }
        let expected = [
            ("hidden_size", config.hidden_size, 960),
            ("intermediate_size", config.intermediate_size, 2560),
            ("vocab_size", config.vocab_size, 49152),
            ("num_hidden_layers", config.num_hidden_layers, 32),
            ("num_attention_heads", config.num_attention_heads, 15),
            ("num_key_value_heads", config.num_key_value_heads, 5),
            (
                "max_position_embeddings",
                config.max_position_embeddings,
                8192,
            ),
        ];
        for (field, actual, required) in expected {
            if actual != required {
                return Err(CandleError::ArtifactMismatch {
                    artifact: "config.json",
                    reason: format!(
                        "SmolLM2-360M-Instruct requires {field}={required}, found {actual}"
                    ),
                });
            }
        }
        if !config.tie_word_embeddings
            || config.rms_norm_eps != 1e-5
            || config.rope_theta != 100_000.0
        {
            return Err(CandleError::ArtifactMismatch {
                artifact: "config.json",
                reason: "SmolLM2-360M-Instruct requires tied embeddings, rms_norm_eps=1e-5, and rope_theta=100000".to_string(),
            });
        }
    }
    Ok(())
}

fn validate_qwen3_config(config: &Qwen3Config) -> Result<(), CandleError> {
    if config.model_type != "qwen3"
        || !config
            .architectures
            .iter()
            .any(|architecture| architecture == "Qwen3ForCausalLM")
    {
        return Err(CandleError::UnsupportedModelFamily(
            "configuration must declare Qwen3ForCausalLM with model_type `qwen3`".to_string(),
        ));
    }
    let expected = [
        ("hidden_size", config.hidden_size, 2560),
        ("intermediate_size", config.intermediate_size, 9728),
        ("num_hidden_layers", config.num_hidden_layers, 36),
        ("num_attention_heads", config.num_attention_heads, 32),
        ("num_key_value_heads", config.num_key_value_heads, 8),
        ("head_dim", config.head_dim, 128),
        (
            "max_position_embeddings",
            config.max_position_embeddings,
            40960,
        ),
        ("vocab_size", config.vocab_size, 151936),
    ];
    for (field, actual, required) in expected {
        if actual != required {
            return Err(CandleError::ArtifactMismatch {
                artifact: "config.json",
                reason: format!("Qwen3-4B requires {field}={required}, found {actual}"),
            });
        }
    }
    if !config
        .num_attention_heads
        .is_multiple_of(config.num_key_value_heads)
        || !config.head_dim.is_multiple_of(2)
    {
        return Err(CandleError::InvalidConfigurationValue {
            field: "attention dimensions",
            reason: "Qwen3 attention heads must divide KV heads and head_dim must be even"
                .to_string(),
        });
    }
    if config.hidden_act != "silu"
        || config.attention_bias
        || !config.tie_word_embeddings
        || (config.rms_norm_eps - 1e-6).abs() > f64::EPSILON
        || (config.rope_theta - 1_000_000.0).abs() > f64::EPSILON
        || config.bos_token_id != 151643
        || config.eos_token_id != 151645
    {
        return Err(CandleError::ArtifactMismatch {
            artifact: "config.json",
            reason: "Qwen3-4B requires SiLU, bias-free attention, tied embeddings, rms_norm_eps=1e-6, rope_theta=1000000, bos_token_id=151643, and eos_token_id=151645".to_string(),
        });
    }
    validate_token_id("bos_token_id", config.bos_token_id, config.vocab_size)?;
    validate_token_id("eos_token_id", config.eos_token_id, config.vocab_size)?;
    Ok(())
}

fn validate_qwen3_tokenizer(
    config: &Qwen3Config,
    tokenizer: &Tokenizer,
) -> Result<(), CandleError> {
    let actual = tokenizer.get_vocab_size(true);
    // Qwen3-4B reserves the tail of its 151,936-row embedding matrix for
    // explicit GGUF padding tokens. The upstream tokenizer defines 151,669
    // usable IDs; equating tokenizer cardinality with model capacity rejects
    // the official artifacts.
    if actual != 151_669 || actual > config.vocab_size {
        return Err(CandleError::ArtifactMismatch {
            artifact: "tokenizer.json",
            reason: format!(
                "Qwen3-4B requires 151669 defined tokenizer IDs within model capacity {}, found {actual}",
                config.vocab_size
            ),
        });
    }
    for token in [END_OF_TEXT, IM_START, IM_END] {
        let id = tokenizer
            .token_to_id(token)
            .ok_or(CandleError::MissingSpecialToken { token })?;
        validate_token_id(token, id, config.vocab_size)?;
        if !tokenizer.get_added_vocabulary().is_special_token(token) {
            return Err(CandleError::SpecialTokenNotMarked { token });
        }
    }
    for (field, token, configured) in [
        ("bos_token_id", END_OF_TEXT, config.bos_token_id),
        ("eos_token_id", IM_END, config.eos_token_id),
    ] {
        let actual = tokenizer
            .token_to_id(token)
            .ok_or(CandleError::MissingSpecialToken { token })?;
        if actual != configured {
            return Err(CandleError::ArtifactMismatch {
                artifact: field,
                reason: format!(
                    "configured ID {configured} does not match special token `{token}` ID {actual}"
                ),
            });
        }
    }
    Ok(())
}

fn validate_positive_finite(field: &'static str, value: f32) -> Result<(), CandleError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(CandleError::InvalidConfigurationValue {
            field,
            reason: format!("value {value} must be finite and greater than zero"),
        });
    }
    Ok(())
}

fn detect_model_family(tokenizer: &Tokenizer) -> Result<ModelFamily, CandleError> {
    let llama3 = [BEGIN_OF_TEXT, START_HEADER, END_HEADER, END_OF_TURN]
        .iter()
        .all(|token| tokenizer.token_to_id(token).is_some());
    let smollm2 = [IM_START, IM_END]
        .iter()
        .all(|token| tokenizer.token_to_id(token).is_some());
    match (llama3, smollm2) {
        (true, false) => Ok(ModelFamily::Llama3),
        (false, true) => Ok(ModelFamily::SmolLm2),
        (true, true) => Err(CandleError::UnsupportedModelFamily(
            "tokenizer ambiguously contains both Llama 3 and SmolLM2 control tokens".to_string(),
        )),
        (false, false) => Err(CandleError::UnsupportedModelFamily(
            "tokenizer contains neither the Llama 3 nor SmolLM2 control-token set".to_string(),
        )),
    }
}

fn validate_tokenizer(
    config: &Config,
    tokenizer: &Tokenizer,
    family: ModelFamily,
) -> Result<(), CandleError> {
    let actual = tokenizer.get_vocab_size(true);
    if actual != config.vocab_size {
        return Err(CandleError::TokenizerVocabularyMismatch {
            expected: config.vocab_size,
            actual,
        });
    }
    if let Some(token) = config.bos_token_id {
        validate_token_id("bos_token_id", token, config.vocab_size)?;
    }
    match &config.eos_token_id {
        Some(LlamaEosToks::Single(token)) => {
            validate_token_id("eos_token_id", *token, config.vocab_size)?;
        }
        Some(LlamaEosToks::Multiple(tokens)) => {
            if tokens.is_empty() {
                return Err(CandleError::InvalidConfigurationValue {
                    field: "eos_token_id",
                    reason: "must contain at least one token ID when present".to_string(),
                });
            }
            for token in tokens {
                validate_token_id("eos_token_id", *token, config.vocab_size)?;
            }
        }
        None => {}
    }
    let required: &[&'static str] = match family {
        ModelFamily::Llama3 => &[BEGIN_OF_TEXT, START_HEADER, END_HEADER, END_OF_TURN],
        ModelFamily::SmolLm2 => &[IM_START, IM_END],
        ModelFamily::Qwen3 => &[END_OF_TEXT, IM_START, IM_END],
    };
    for &token in required {
        let id = tokenizer
            .token_to_id(token)
            .ok_or(CandleError::MissingSpecialToken { token })?;
        validate_token_id(token, id, config.vocab_size)?;
        if !tokenizer.get_added_vocabulary().is_special_token(token) {
            return Err(CandleError::SpecialTokenNotMarked { token });
        }
    }
    let (start_token, end_token) = match family {
        ModelFamily::Llama3 => (BEGIN_OF_TEXT, END_OF_TURN),
        ModelFamily::SmolLm2 => (IM_START, IM_END),
        ModelFamily::Qwen3 => (END_OF_TEXT, IM_END),
    };
    let start_id = tokenizer
        .token_to_id(start_token)
        .ok_or(CandleError::MissingSpecialToken { token: start_token })?;
    if let Some(configured) = config.bos_token_id
        && configured != start_id
    {
        return Err(CandleError::ArtifactMismatch {
            artifact: "bos_token_id",
            reason: format!(
                "configured ID {configured} does not match {start_token} ID {start_id}"
            ),
        });
    }
    let end_id = tokenizer
        .token_to_id(end_token)
        .ok_or(CandleError::MissingSpecialToken { token: end_token })?;
    let configured_end_matches = match &config.eos_token_id {
        Some(LlamaEosToks::Single(configured)) => *configured == end_id,
        Some(LlamaEosToks::Multiple(configured)) => configured.contains(&end_id),
        None => true,
    };
    if !configured_end_matches {
        return Err(CandleError::ArtifactMismatch {
            artifact: "eos_token_id",
            reason: format!("configured EOS IDs do not contain {end_token} ID {end_id}"),
        });
    }
    Ok(())
}

fn validate_token_id(token: &str, id: u32, vocab_size: usize) -> Result<(), CandleError> {
    if (id as usize) >= vocab_size {
        return Err(CandleError::TokenIdOutOfRange {
            token: token.to_string(),
            id,
            vocab_size,
        });
    }
    Ok(())
}

fn resolve_stop_tokens(
    config: &Config,
    tokenizer: &Tokenizer,
    family: ModelFamily,
) -> Result<HashSet<u32>, CandleError> {
    let mut tokens = HashSet::new();
    match &config.eos_token_id {
        Some(LlamaEosToks::Single(token)) => {
            tokens.insert(*token);
        }
        Some(LlamaEosToks::Multiple(items)) => {
            tokens.extend(items.iter().copied());
        }
        None => {}
    }
    let end_of_turn = match family {
        ModelFamily::Llama3 => END_OF_TURN,
        ModelFamily::SmolLm2 => IM_END,
        ModelFamily::Qwen3 => IM_END,
    };
    if let Some(token) = tokenizer.token_to_id(end_of_turn) {
        tokens.insert(token);
    }
    if tokens.is_empty() {
        Err(CandleError::MissingStopToken)
    } else {
        Ok(tokens)
    }
}

fn load_gguf(
    weights: &[u8],
    prepared: &PreparedModel,
    device: &Device,
) -> Result<LoadedWeights, CandleError> {
    let mut reader = std::io::Cursor::new(weights);
    let content = gguf_file::Content::read(&mut reader)
        .map_err(|error| CandleError::InvalidQuantizedCheckpoint(error.to_string()))?;
    let expected_architecture = match prepared.architecture {
        ModelArchitecture::Llama => "llama",
        ModelArchitecture::Qwen3 => "qwen3",
    };
    match content.metadata.get("general.architecture") {
        Some(gguf_file::Value::String(architecture)) if architecture == expected_architecture => {}
        Some(value) => {
            return Err(CandleError::InvalidQuantizedCheckpoint(format!(
                "general.architecture must be `{expected_architecture}`, found {value:?}"
            )));
        }
        None => {
            return Err(CandleError::InvalidQuantizedCheckpoint(
                "missing general.architecture metadata".to_string(),
            ));
        }
    }
    match content.metadata.get("general.file_type") {
        // GGML file type 15 is Q4_K_M. Small matrices legitimately use
        // auxiliary F32/Q5/Q6/Q8 encodings in this mixed quantization.
        Some(gguf_file::Value::U32(15)) => {}
        Some(value) => {
            return Err(CandleError::UnsupportedQuantization(format!(
                "general.file_type must identify Q4_K_M (15), found {value:?}"
            )));
        }
        None => {
            return Err(CandleError::InvalidQuantizedCheckpoint(
                "missing general.file_type metadata".to_string(),
            ));
        }
    }
    if metadata_usize(&content, "general.quantization_version")? != 2 {
        return Err(CandleError::UnsupportedQuantization(
            "Q4_K_M checkpoint must use GGML quantization version 2".to_string(),
        ));
    }
    match prepared.architecture {
        ModelArchitecture::Llama => {
            if prepared.family != ModelFamily::SmolLm2 {
                return Err(CandleError::UnsupportedModelFamily(
                    "Q4_K_M Llama GGUF loading currently supports SmolLM2-360M-Instruct only"
                        .to_string(),
                ));
            }
            let config = prepared.llama_config.as_ref().ok_or_else(|| {
                CandleError::Configuration(
                    "prepared Llama GGUF omitted its Llama configuration".to_string(),
                )
            })?;
            validate_gguf_metadata(&content, config, &prepared.tokenizer)?;
            validate_gguf_tensors(&content, config)?;
            QuantizedLlama::from_gguf(content, &mut reader, device)
                .map(LoadedWeights::QuantizedLlama)
                .map_err(|error| CandleError::ModelLoading(error.to_string()))
        }
        ModelArchitecture::Qwen3 => {
            let config = prepared.qwen3_config.as_ref().ok_or_else(|| {
                CandleError::Configuration(
                    "prepared Qwen3 GGUF omitted its Qwen3 configuration".to_string(),
                )
            })?;
            validate_qwen3_gguf_metadata(&content, config, &prepared.tokenizer)?;
            validate_qwen3_gguf_tensors(&content, config)?;
            QuantizedQwen3::from_gguf(content, &mut reader, device)
                .map(LoadedWeights::QuantizedQwen3)
                .map_err(|error| CandleError::ModelLoading(error.to_string()))
        }
    }
}

fn validate_gguf_metadata(
    content: &gguf_file::Content,
    config: &Config,
    tokenizer: &Tokenizer,
) -> Result<(), CandleError> {
    let expected = [
        ("llama.vocab_size", config.vocab_size),
        ("llama.embedding_length", config.hidden_size),
        ("llama.feed_forward_length", config.intermediate_size),
        ("llama.block_count", config.num_hidden_layers),
        ("llama.attention.head_count", config.num_attention_heads),
        ("llama.attention.head_count_kv", config.num_key_value_heads),
        ("llama.context_length", config.max_position_embeddings),
        (
            "llama.rope.dimension_count",
            config.hidden_size / config.num_attention_heads,
        ),
    ];
    for (key, expected) in expected {
        let actual = metadata_usize(content, key)?;
        if actual != expected {
            return Err(CandleError::ArtifactMismatch {
                artifact: "model.gguf",
                reason: format!("metadata `{key}` is {actual}, but config requires {expected}"),
            });
        }
    }
    for (key, actual, expected) in [
        (
            "llama.rope.freq_base",
            metadata_f64(content, "llama.rope.freq_base")?,
            config.rope_theta as f64,
        ),
        (
            "llama.attention.layer_norm_rms_epsilon",
            metadata_f64(content, "llama.attention.layer_norm_rms_epsilon")?,
            config.rms_norm_eps,
        ),
    ] {
        if (actual - expected).abs() > 1e-5 * expected.abs().max(f64::MIN_POSITIVE) {
            return Err(CandleError::ArtifactMismatch {
                artifact: "model.gguf",
                reason: format!("metadata `{key}` is {actual}, but config requires {expected}"),
            });
        }
    }
    require_metadata_string(content, "general.basename", "smollm2")?;
    require_metadata_string(content, "tokenizer.ggml.model", "gpt2")?;
    require_metadata_string(content, "tokenizer.ggml.pre", "smollm")?;
    let tokens = match content.metadata.get("tokenizer.ggml.tokens") {
        Some(gguf_file::Value::Array(tokens)) => tokens,
        Some(value) => {
            return Err(CandleError::InvalidQuantizedCheckpoint(format!(
                "metadata `tokenizer.ggml.tokens` must be an array, found {value:?}"
            )));
        }
        None => {
            return Err(CandleError::InvalidQuantizedCheckpoint(
                "missing `tokenizer.ggml.tokens` metadata".to_string(),
            ));
        }
    };
    if tokens.len() != config.vocab_size {
        return Err(CandleError::ArtifactMismatch {
            artifact: "model.gguf",
            reason: format!(
                "GGUF tokenizer has {} tokens, but config requires {}",
                tokens.len(),
                config.vocab_size
            ),
        });
    }
    for (id, value) in tokens.iter().enumerate() {
        let gguf_token = match value {
            gguf_file::Value::String(token) => token,
            value => {
                return Err(CandleError::InvalidQuantizedCheckpoint(format!(
                    "tokenizer.ggml.tokens[{id}] must be a string, found {value:?}"
                )));
            }
        };
        let id = u32::try_from(id).map_err(|_| {
            CandleError::InvalidQuantizedCheckpoint(
                "GGUF tokenizer index does not fit in u32".to_string(),
            )
        })?;
        if tokenizer.id_to_token(id).as_deref() != Some(gguf_token.as_str()) {
            return Err(CandleError::ArtifactMismatch {
                artifact: "tokenizer.json",
                reason: format!("token ID {id} does not match the GGUF tokenizer vocabulary"),
            });
        }
    }
    for (key, token) in [
        ("tokenizer.ggml.bos_token_id", IM_START),
        ("tokenizer.ggml.eos_token_id", IM_END),
    ] {
        let actual = metadata_usize(content, key)?;
        let expected = tokenizer
            .token_to_id(token)
            .ok_or(CandleError::MissingSpecialToken { token })? as usize;
        if actual != expected {
            return Err(CandleError::ArtifactMismatch {
                artifact: "model.gguf",
                reason: format!("metadata `{key}` is {actual}, but tokenizer requires {expected}"),
            });
        }
    }
    Ok(())
}

fn validate_qwen3_gguf_metadata(
    content: &gguf_file::Content,
    config: &Qwen3Config,
    tokenizer: &Tokenizer,
) -> Result<(), CandleError> {
    let expected = [
        ("qwen3.embedding_length", config.hidden_size),
        ("qwen3.feed_forward_length", config.intermediate_size),
        ("qwen3.block_count", config.num_hidden_layers),
        ("qwen3.attention.head_count", config.num_attention_heads),
        ("qwen3.attention.head_count_kv", config.num_key_value_heads),
        ("qwen3.attention.key_length", config.head_dim),
        ("qwen3.attention.value_length", config.head_dim),
        ("qwen3.context_length", config.max_position_embeddings),
    ];
    for (key, expected) in expected {
        let actual = metadata_usize(content, key)?;
        if actual != expected {
            return Err(CandleError::ArtifactMismatch {
                artifact: "model.gguf",
                reason: format!("metadata `{key}` is {actual}, but config requires {expected}"),
            });
        }
    }
    for (key, actual, expected) in [
        (
            "qwen3.rope.freq_base",
            metadata_f64(content, "qwen3.rope.freq_base")?,
            config.rope_theta,
        ),
        (
            "qwen3.attention.layer_norm_rms_epsilon",
            metadata_f64(content, "qwen3.attention.layer_norm_rms_epsilon")?,
            config.rms_norm_eps,
        ),
    ] {
        if (actual - expected).abs() > 1e-5 * expected.abs().max(f64::MIN_POSITIVE) {
            return Err(CandleError::ArtifactMismatch {
                artifact: "model.gguf",
                reason: format!("metadata `{key}` is {actual}, but config requires {expected}"),
            });
        }
    }
    require_metadata_string(content, "general.basename", "qwen3")?;
    require_metadata_string(content, "general.size_label", "4b")?;
    require_metadata_string(content, "general.finetune", "instruct-awq")?;
    require_metadata_string(content, "tokenizer.ggml.model", "gpt2")?;
    require_metadata_string(content, "tokenizer.ggml.pre", "qwen2")?;
    validate_gguf_tokenizer_vocabulary(content, config.vocab_size, tokenizer)?;
    for (key, token) in [
        ("tokenizer.ggml.bos_token_id", END_OF_TEXT),
        ("tokenizer.ggml.eos_token_id", IM_END),
    ] {
        let actual = metadata_usize(content, key)?;
        let expected = tokenizer
            .token_to_id(token)
            .ok_or(CandleError::MissingSpecialToken { token })? as usize;
        if actual != expected {
            return Err(CandleError::ArtifactMismatch {
                artifact: "model.gguf",
                reason: format!("metadata `{key}` is {actual}, but tokenizer requires {expected}"),
            });
        }
    }
    match content.metadata.get("tokenizer.chat_template") {
        Some(gguf_file::Value::String(template))
            if [
                "# Tools",
                "<tools></tools>",
                "<tool_call>",
                "<tool_response>",
                "enable_thinking",
            ]
            .iter()
            .all(|required| template.contains(required)) => {}
        Some(_) => {
            return Err(CandleError::ArtifactMismatch {
                artifact: "model.gguf",
                reason: "Qwen3 tokenizer.chat_template does not contain the official Hermes tool and no-thinking protocol markers".to_string(),
            });
        }
        None => {
            return Err(CandleError::InvalidQuantizedCheckpoint(
                "missing `tokenizer.chat_template` metadata".to_string(),
            ));
        }
    }
    Ok(())
}

fn validate_gguf_tokenizer_vocabulary(
    content: &gguf_file::Content,
    vocab_size: usize,
    tokenizer: &Tokenizer,
) -> Result<(), CandleError> {
    let tokens = match content.metadata.get("tokenizer.ggml.tokens") {
        Some(gguf_file::Value::Array(tokens)) => tokens,
        Some(value) => {
            return Err(CandleError::InvalidQuantizedCheckpoint(format!(
                "metadata `tokenizer.ggml.tokens` must be an array, found {value:?}"
            )));
        }
        None => {
            return Err(CandleError::InvalidQuantizedCheckpoint(
                "missing `tokenizer.ggml.tokens` metadata".to_string(),
            ));
        }
    };
    if tokens.len() != vocab_size {
        return Err(CandleError::ArtifactMismatch {
            artifact: "model.gguf",
            reason: format!(
                "GGUF tokenizer has {} tokens, but config requires {vocab_size}",
                tokens.len()
            ),
        });
    }
    for (id, value) in tokens.iter().enumerate() {
        let gguf_token = match value {
            gguf_file::Value::String(token) => token,
            value => {
                return Err(CandleError::InvalidQuantizedCheckpoint(format!(
                    "tokenizer.ggml.tokens[{id}] must be a string, found {value:?}"
                )));
            }
        };
        let id = u32::try_from(id).map_err(|_| {
            CandleError::InvalidQuantizedCheckpoint(
                "GGUF tokenizer index does not fit in u32".to_string(),
            )
        })?;
        let matches = tokenizer.id_to_token(id).map_or_else(
            || *gguf_token == format!("[PAD{id}]"),
            |token| token == gguf_token.as_str(),
        );
        if !matches {
            return Err(CandleError::ArtifactMismatch {
                artifact: "tokenizer.json",
                reason: format!(
                    "token ID {id} does not match the GGUF tokenizer vocabulary or its required padding token"
                ),
            });
        }
    }
    Ok(())
}

fn metadata_usize(content: &gguf_file::Content, key: &str) -> Result<usize, CandleError> {
    let value = content.metadata.get(key).ok_or_else(|| {
        CandleError::InvalidQuantizedCheckpoint(format!("missing `{key}` metadata"))
    })?;
    match value {
        gguf_file::Value::U32(value) => Ok(*value as usize),
        gguf_file::Value::U64(value) => usize::try_from(*value).map_err(|_| {
            CandleError::InvalidQuantizedCheckpoint(format!(
                "metadata `{key}` does not fit in usize"
            ))
        }),
        value => Err(CandleError::InvalidQuantizedCheckpoint(format!(
            "metadata `{key}` must be an unsigned integer, found {value:?}"
        ))),
    }
}

fn metadata_f64(content: &gguf_file::Content, key: &str) -> Result<f64, CandleError> {
    match content.metadata.get(key) {
        Some(gguf_file::Value::F32(value)) => Ok(f64::from(*value)),
        Some(gguf_file::Value::F64(value)) => Ok(*value),
        Some(value) => Err(CandleError::InvalidQuantizedCheckpoint(format!(
            "metadata `{key}` must be floating point, found {value:?}"
        ))),
        None => Err(CandleError::InvalidQuantizedCheckpoint(format!(
            "missing `{key}` metadata"
        ))),
    }
}

fn require_metadata_string(
    content: &gguf_file::Content,
    key: &str,
    expected: &str,
) -> Result<(), CandleError> {
    match content.metadata.get(key) {
        Some(gguf_file::Value::String(actual)) if actual.eq_ignore_ascii_case(expected) => Ok(()),
        Some(value) => Err(CandleError::ArtifactMismatch {
            artifact: "model.gguf",
            reason: format!("metadata `{key}` must be `{expected}`, found {value:?}"),
        }),
        None => Err(CandleError::InvalidQuantizedCheckpoint(format!(
            "missing `{key}` metadata"
        ))),
    }
}

fn validate_gguf_tensors(content: &gguf_file::Content, config: &Config) -> Result<(), CandleError> {
    let allowed = |dtype| {
        matches!(
            dtype,
            GgmlDType::F32 | GgmlDType::Q4K | GgmlDType::Q5_0 | GgmlDType::Q6K | GgmlDType::Q8_0
        )
    };
    for (name, tensor) in &content.tensor_infos {
        if !allowed(tensor.ggml_dtype) {
            return Err(CandleError::UnsupportedQuantization(format!(
                "tensor `{name}` uses unsupported {:?} in a Q4_K_M checkpoint",
                tensor.ggml_dtype
            )));
        }
    }
    if !content
        .tensor_infos
        .values()
        .any(|tensor| tensor.ggml_dtype == GgmlDType::Q4K)
    {
        return Err(CandleError::UnsupportedQuantization(
            "checkpoint contains no Q4_K tensors; rig-candle supports the Q4_K_M tensor mix"
                .to_string(),
        ));
    }
    validate_gguf_tensor(
        content,
        "token_embd.weight",
        &[config.vocab_size, config.hidden_size],
    )?;
    validate_gguf_tensor(content, "output_norm.weight", &[config.hidden_size])?;
    if !config.tie_word_embeddings || content.tensor_infos.contains_key("output.weight") {
        validate_gguf_tensor(
            content,
            "output.weight",
            &[config.vocab_size, config.hidden_size],
        )?;
    }
    let head_dim = config.hidden_size / config.num_attention_heads;
    let kv_size = head_dim * config.num_key_value_heads;
    for layer in 0..config.num_hidden_layers {
        let prefix = format!("blk.{layer}");
        for (suffix, shape) in [
            (
                "attn_q.weight",
                vec![config.hidden_size, config.hidden_size],
            ),
            ("attn_k.weight", vec![kv_size, config.hidden_size]),
            ("attn_v.weight", vec![kv_size, config.hidden_size]),
            (
                "attn_output.weight",
                vec![config.hidden_size, config.hidden_size],
            ),
            (
                "ffn_gate.weight",
                vec![config.intermediate_size, config.hidden_size],
            ),
            (
                "ffn_down.weight",
                vec![config.hidden_size, config.intermediate_size],
            ),
            (
                "ffn_up.weight",
                vec![config.intermediate_size, config.hidden_size],
            ),
            ("attn_norm.weight", vec![config.hidden_size]),
            ("ffn_norm.weight", vec![config.hidden_size]),
        ] {
            validate_gguf_tensor(content, &format!("{prefix}.{suffix}"), &shape)?;
        }
    }
    Ok(())
}

fn validate_qwen3_gguf_tensors(
    content: &gguf_file::Content,
    config: &Qwen3Config,
) -> Result<(), CandleError> {
    for (name, tensor) in &content.tensor_infos {
        if !matches!(
            tensor.ggml_dtype,
            GgmlDType::F32 | GgmlDType::Q4K | GgmlDType::Q6K
        ) {
            return Err(CandleError::UnsupportedQuantization(format!(
                "Qwen3 tensor `{name}` uses unsupported {:?}; pinned Q4_K_M permits F32, Q4_K, and Q6_K",
                tensor.ggml_dtype
            )));
        }
    }
    let expected_count = 2usize
        .checked_add(config.num_hidden_layers.checked_mul(11).ok_or_else(|| {
            CandleError::InvalidQuantizedCheckpoint(
                "Qwen3 expected tensor count overflowed usize".to_string(),
            )
        })?)
        .ok_or_else(|| {
            CandleError::InvalidQuantizedCheckpoint(
                "Qwen3 expected tensor count overflowed usize".to_string(),
            )
        })?;
    if content.tensor_infos.len() != expected_count {
        return Err(CandleError::InvalidQuantizedCheckpoint(format!(
            "Qwen3-4B Q4_K_M contains {} tensors, expected exactly {expected_count}",
            content.tensor_infos.len()
        )));
    }
    validate_gguf_tensor_dtype(
        content,
        "token_embd.weight",
        &[config.vocab_size, config.hidden_size],
        &[GgmlDType::Q6K],
    )?;
    validate_gguf_tensor_dtype(
        content,
        "output_norm.weight",
        &[config.hidden_size],
        &[GgmlDType::F32],
    )?;
    let query_size = config.num_attention_heads * config.head_dim;
    let kv_size = config.num_key_value_heads * config.head_dim;
    for layer in 0..config.num_hidden_layers {
        let prefix = format!("blk.{layer}");
        for (suffix, shape, dtypes) in [
            (
                "attn_q.weight",
                vec![query_size, config.hidden_size],
                &[GgmlDType::Q4K][..],
            ),
            (
                "attn_k.weight",
                vec![kv_size, config.hidden_size],
                &[GgmlDType::Q4K],
            ),
            (
                "attn_v.weight",
                vec![kv_size, config.hidden_size],
                &[GgmlDType::Q4K, GgmlDType::Q6K],
            ),
            (
                "attn_output.weight",
                vec![config.hidden_size, query_size],
                &[GgmlDType::Q4K],
            ),
            (
                "attn_q_norm.weight",
                vec![config.head_dim],
                &[GgmlDType::F32],
            ),
            (
                "attn_k_norm.weight",
                vec![config.head_dim],
                &[GgmlDType::F32],
            ),
            (
                "ffn_gate.weight",
                vec![config.intermediate_size, config.hidden_size],
                &[GgmlDType::Q4K],
            ),
            (
                "ffn_down.weight",
                vec![config.hidden_size, config.intermediate_size],
                &[GgmlDType::Q4K, GgmlDType::Q6K],
            ),
            (
                "ffn_up.weight",
                vec![config.intermediate_size, config.hidden_size],
                &[GgmlDType::Q4K],
            ),
            (
                "attn_norm.weight",
                vec![config.hidden_size],
                &[GgmlDType::F32],
            ),
            (
                "ffn_norm.weight",
                vec![config.hidden_size],
                &[GgmlDType::F32],
            ),
        ] {
            validate_gguf_tensor_dtype(content, &format!("{prefix}.{suffix}"), &shape, dtypes)?;
        }
    }
    Ok(())
}

fn validate_gguf_tensor_dtype(
    content: &gguf_file::Content,
    name: &str,
    expected_shape: &[usize],
    allowed_dtypes: &[GgmlDType],
) -> Result<(), CandleError> {
    validate_gguf_tensor(content, name, expected_shape)?;
    let tensor = content.tensor_infos.get(name).ok_or_else(|| {
        CandleError::InvalidQuantizedCheckpoint(format!("missing expected tensor `{name}`"))
    })?;
    if !allowed_dtypes.contains(&tensor.ggml_dtype) {
        return Err(CandleError::UnsupportedQuantization(format!(
            "tensor `{name}` uses {:?}, expected one of {allowed_dtypes:?}",
            tensor.ggml_dtype
        )));
    }
    Ok(())
}

fn validate_gguf_tensor(
    content: &gguf_file::Content,
    name: &str,
    expected: &[usize],
) -> Result<(), CandleError> {
    let tensor = content.tensor_infos.get(name).ok_or_else(|| {
        CandleError::InvalidQuantizedCheckpoint(format!("missing expected tensor `{name}`"))
    })?;
    if tensor.shape.dims() != expected {
        return Err(CandleError::InvalidQuantizedCheckpoint(format!(
            "tensor `{name}` has shape {:?}, expected {expected:?}",
            tensor.shape.dims()
        )));
    }
    Ok(())
}

fn validate_checkpoint(weights: &[u8], config: &Config) -> Result<(), CandleError> {
    let tensors = SafeTensors::deserialize(weights)
        .map_err(|error| CandleError::InvalidCheckpoint(error.to_string()))?;
    let head_dim = config.hidden_size / config.num_attention_heads;
    let kv_size = head_dim * config.num_key_value_heads;

    validate_tensor(
        &tensors,
        "model.embed_tokens.weight",
        &[config.vocab_size, config.hidden_size],
    )?;
    validate_tensor(&tensors, "model.norm.weight", &[config.hidden_size])?;
    if !config.tie_word_embeddings {
        validate_tensor(
            &tensors,
            "lm_head.weight",
            &[config.vocab_size, config.hidden_size],
        )?;
    }
    for layer in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{layer}");
        for (suffix, shape) in [
            (
                "self_attn.q_proj.weight",
                vec![config.hidden_size, config.hidden_size],
            ),
            ("self_attn.k_proj.weight", vec![kv_size, config.hidden_size]),
            ("self_attn.v_proj.weight", vec![kv_size, config.hidden_size]),
            (
                "self_attn.o_proj.weight",
                vec![config.hidden_size, config.hidden_size],
            ),
            (
                "mlp.gate_proj.weight",
                vec![config.intermediate_size, config.hidden_size],
            ),
            (
                "mlp.up_proj.weight",
                vec![config.intermediate_size, config.hidden_size],
            ),
            (
                "mlp.down_proj.weight",
                vec![config.hidden_size, config.intermediate_size],
            ),
            ("input_layernorm.weight", vec![config.hidden_size]),
            ("post_attention_layernorm.weight", vec![config.hidden_size]),
        ] {
            validate_tensor(&tensors, &format!("{prefix}.{suffix}"), &shape)?;
        }
    }
    Ok(())
}

fn validate_tensor(
    tensors: &SafeTensors<'_>,
    name: &str,
    expected_shape: &[usize],
) -> Result<(), CandleError> {
    let tensor = tensors
        .tensor(name)
        .map_err(|_| CandleError::MissingTensor(name.to_string()))?;
    if tensor.shape() != expected_shape {
        return Err(CandleError::TensorShapeMismatch {
            tensor: name.to_string(),
            expected: expected_shape.to_vec(),
            actual: tensor.shape().to_vec(),
        });
    }
    if !matches!(
        tensor.dtype(),
        SafeDtype::F32 | SafeDtype::F16 | SafeDtype::BF16
    ) {
        return Err(CandleError::UnsupportedTensorDtype {
            tensor: name.to_string(),
            dtype: format!("{:?}", tensor.dtype()),
        });
    }
    Ok(())
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RequestGenerationOverrides {
    top_k: Option<usize>,
    top_p: Option<f64>,
    seed: Option<u64>,
    repeat_penalty: Option<f32>,
    repeat_last_n: Option<usize>,
}

fn override_or<T>(value: Option<T>, default: T) -> T {
    match value {
        Some(value) => value,
        None => default,
    }
}

fn effective_generation(
    request: &CompletionRequest,
    defaults: &GenerationConfig,
    vocab_size: usize,
) -> Result<GenerationConfig, CandleError> {
    let overrides = match &request.additional_params {
        Some(value) => serde_json::from_value::<RequestGenerationOverrides>(value.clone())
            .map_err(|error| CandleError::InvalidGeneration(error.to_string()))?,
        None => RequestGenerationOverrides::default(),
    };
    let generation = GenerationConfig {
        max_tokens: override_or(request.max_tokens, defaults.max_tokens),
        temperature: override_or(request.temperature, defaults.temperature),
        top_k: overrides.top_k.or(defaults.top_k),
        top_p: overrides.top_p.or(defaults.top_p),
        seed: override_or(overrides.seed, defaults.seed),
        repeat_penalty: override_or(overrides.repeat_penalty, defaults.repeat_penalty),
        repeat_last_n: override_or(overrides.repeat_last_n, defaults.repeat_last_n),
    };
    validate_generation(&generation, Some(vocab_size))?;
    Ok(generation)
}

fn validate_generation(
    generation: &GenerationConfig,
    vocab_size: Option<usize>,
) -> Result<(), CandleError> {
    if generation.max_tokens == 0 {
        return Err(CandleError::InvalidGeneration(
            "max_tokens must be greater than zero".to_string(),
        ));
    }
    if !generation.temperature.is_finite() || generation.temperature < 0.0 {
        return Err(CandleError::InvalidGeneration(
            "temperature must be finite and non-negative".to_string(),
        ));
    }
    if let Some(top_k) = generation.top_k
        && (top_k == 0 || vocab_size.is_some_and(|size| top_k > size))
    {
        return Err(CandleError::InvalidGeneration(
            "top_k must be greater than zero and no larger than the vocabulary".to_string(),
        ));
    }
    if let Some(top_p) = generation.top_p
        && !(top_p.is_finite() && 0.0 < top_p && top_p <= 1.0)
    {
        return Err(CandleError::InvalidGeneration(
            "top_p must be finite and in (0, 1]".to_string(),
        ));
    }
    if !generation.repeat_penalty.is_finite() || generation.repeat_penalty <= 0.0 {
        return Err(CandleError::InvalidGeneration(
            "repeat_penalty must be finite and greater than zero".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
fn render_prompt(request: &CompletionRequest) -> Result<String, CandleError> {
    render_prompt_for(request, ModelFamily::Llama3)
}

fn render_prompt_for(
    request: &CompletionRequest,
    family: ModelFamily,
) -> Result<String, CandleError> {
    protocol::render_prompt(request, family)
}

fn sampling(config: &GenerationConfig) -> Sampling {
    if config.temperature == 0.0 {
        Sampling::ArgMax
    } else {
        match (config.top_k, config.top_p) {
            (Some(k), Some(p)) => Sampling::TopKThenTopP {
                k,
                p,
                temperature: config.temperature,
            },
            (Some(k), None) => Sampling::TopK {
                k,
                temperature: config.temperature,
            },
            (None, Some(p)) => Sampling::TopP {
                p,
                temperature: config.temperature,
            },
            (None, None) => Sampling::All {
                temperature: config.temperature,
            },
        }
    }
}

fn effective_output_limit(
    prompt_tokens: usize,
    requested_max_tokens: u64,
    context_limit: usize,
) -> Result<usize, CandleError> {
    if prompt_tokens > context_limit {
        return Err(CandleError::PromptTooLong {
            prompt_tokens,
            context_limit,
        });
    }
    let remaining = context_limit - prompt_tokens;
    if remaining == 0 {
        return Err(CandleError::NoGenerationCapacity {
            prompt_tokens,
            context_limit,
        });
    }
    let remaining = u64::try_from(remaining).map_err(|_| CandleError::NumericConversion {
        field: "remaining_context_tokens",
        value: u64::MAX,
    })?;
    max_tokens_to_usize(requested_max_tokens.min(remaining), usize::MAX as u64)
}

fn max_tokens_to_usize(value: u64, platform_max: u64) -> Result<usize, CandleError> {
    if value > platform_max {
        return Err(CandleError::NumericConversion {
            field: "max_tokens",
            value,
        });
    }
    usize::try_from(value).map_err(|_| CandleError::NumericConversion {
        field: "max_tokens",
        value,
    })
}

fn recent_tokens(tokens: &[u32], repeat_last_n: usize) -> &[u32] {
    tokens
        .get(tokens.len().saturating_sub(repeat_last_n)..)
        .map_or(&[], |recent| recent)
}

fn next_cache_position(prompt_tokens: usize, generated_index: usize) -> Result<usize, CandleError> {
    prompt_tokens
        .checked_add(generated_index)
        .ok_or_else(|| CandleError::Inference("KV-cache position overflowed usize".to_string()))
}

#[cfg(not(target_family = "wasm"))]
#[derive(Clone, Default)]
struct CancellationSignal(Arc<AtomicBool>);

#[cfg(not(target_family = "wasm"))]
impl CancellationSignal {
    fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

#[cfg(target_family = "wasm")]
#[derive(Clone, Default)]
struct CancellationSignal;

#[cfg(target_family = "wasm")]
impl CancellationSignal {
    fn is_cancelled(&self) -> bool {
        false
    }
}

#[cfg(not(target_family = "wasm"))]
struct CancelOnDrop {
    signal: CancellationSignal,
    armed: bool,
}

#[cfg(not(target_family = "wasm"))]
impl CancelOnDrop {
    fn new(signal: CancellationSignal) -> Self {
        Self {
            signal,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

#[cfg(not(target_family = "wasm"))]
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.signal.cancel();
        }
    }
}

fn check_cancellation(signal: &CancellationSignal) -> Result<(), CandleError> {
    if signal.is_cancelled() {
        Err(CandleError::Cancelled)
    } else {
        Ok(())
    }
}

#[cfg(not(target_family = "wasm"))]
async fn acquire_concurrency(
    semaphore: Arc<tokio::sync::Semaphore>,
) -> Result<tokio::sync::OwnedSemaphorePermit, CandleError> {
    semaphore
        .acquire_owned()
        .await
        .map_err(|_| CandleError::ConcurrencyControllerClosed)
}

enum GenerationStep {
    /// A token was sampled. Some token sequences need more IDs before they decode to valid UTF-8.
    Token(Option<String>),
    /// Generation and incremental decoding are complete.
    Finished(CandleCompletionResponse),
}

struct IncrementalTextDecoder<'a> {
    tokenizer: &'a Tokenizer,
    stream: TokenDecodeStream<'a>,
    token_ids: Vec<u32>,
    text: String,
    flushed: bool,
}

impl<'a> IncrementalTextDecoder<'a> {
    fn new(tokenizer: &'a Tokenizer) -> Self {
        Self {
            tokenizer,
            stream: tokenizer.decode_stream(true),
            token_ids: Vec::new(),
            text: String::new(),
            flushed: false,
        }
    }

    fn push(&mut self, token: u32) -> Result<Option<String>, CandleError> {
        self.token_ids.push(token);
        let fragment = self
            .stream
            .step(token)
            .map_err(|error| CandleError::TokenizerDecoding(error.to_string()))?;
        if let Some(fragment) = &fragment {
            self.text.push_str(fragment);
        }
        Ok(fragment)
    }

    fn finish(&mut self) -> Result<Option<String>, CandleError> {
        if self.flushed {
            return Ok(None);
        }
        self.flushed = true;
        let fully_decoded = self
            .tokenizer
            .decode(&self.token_ids, true)
            .map_err(|error| CandleError::TokenizerDecoding(error.to_string()))?;
        let suffix = fully_decoded.strip_prefix(&self.text).ok_or_else(|| {
            CandleError::TokenizerDecoding(
                "incremental decoding did not match complete decoding".to_string(),
            )
        })?;
        if suffix.is_empty() {
            Ok(None)
        } else {
            let suffix = suffix.to_string();
            self.text.push_str(&suffix);
            Ok(Some(suffix))
        }
    }

    fn text(&self) -> &str {
        &self.text
    }
}

struct GenerationSession<'a> {
    loaded: &'a LoadedModel,
    generation: GenerationConfig,
    cancellation: &'a CancellationSignal,
    decoder: IncrementalTextDecoder<'a>,
    weights: SessionWeights<'a>,
    logits: Tensor,
    processor: LogitsProcessor,
    prompt_tokens: usize,
    max_tokens: usize,
    effective_max_tokens: u64,
    all_tokens: Vec<u32>,
    generated_tokens: usize,
    finish_reason: Option<FinishReason>,
    started: Instant,
    prefill_duration: Duration,
    time_to_first_token: Option<Duration>,
    delivery_duration: Duration,
}

enum SessionWeights<'a> {
    Safetensors { model: &'a Llama, cache: Cache },
    QuantizedLlama(QuantizedLlama),
    QuantizedQwen3(QuantizedQwen3),
}

impl SessionWeights<'_> {
    fn forward(&mut self, input: &Tensor, position: usize) -> Result<Tensor, CandleError> {
        match self {
            Self::Safetensors { model, cache } => model
                .forward(input, position, cache)
                .and_then(|tensor| tensor.squeeze(0)),
            Self::QuantizedLlama(model) => model
                .forward(input, position)
                .and_then(|tensor| tensor.squeeze(0)),
            Self::QuantizedQwen3(model) => model
                .forward(input, position)
                .and_then(|tensor| tensor.squeeze(0)),
        }
        .map_err(|error| CandleError::Inference(error.to_string()))
    }
}

impl<'a> GenerationSession<'a> {
    fn new(
        loaded: &'a LoadedModel,
        request: CompletionRequest,
        cancellation: &'a CancellationSignal,
    ) -> Result<Self, CandleError> {
        let prompt = render_prompt_for(&request, loaded.family)?;
        let generation =
            effective_generation(&request, &loaded.generation, loaded.spec.vocab_size)?;
        let encoding = loaded
            .tokenizer
            .encode(prompt, false)
            .map_err(|error| CandleError::TokenizerEncoding(error.to_string()))?;
        let prompt_ids = encoding.get_ids();
        if prompt_ids.is_empty() {
            return Err(CandleError::TokenizerEncoding(
                "the rendered prompt produced no tokens".to_string(),
            ));
        }
        let max_tokens = effective_output_limit(
            prompt_ids.len(),
            generation.max_tokens,
            loaded.spec.context_limit,
        )?;
        let effective_max_tokens =
            u64::try_from(max_tokens).map_err(|_| CandleError::NumericConversion {
                field: "effective_max_tokens",
                value: u64::MAX,
            })?;

        check_cancellation(cancellation)?;
        let started = Instant::now();
        let device = Device::Cpu;
        let input = Tensor::new(prompt_ids, &device)
            .and_then(|tensor| tensor.unsqueeze(0))
            .map_err(|error| CandleError::Inference(error.to_string()))?;
        let mut weights = match &loaded.model {
            LoadedWeights::Safetensors { model, config } => SessionWeights::Safetensors {
                model,
                cache: Cache::new(true, DType::F32, config, &device)
                    .map_err(|error| CandleError::Inference(error.to_string()))?,
            },
            LoadedWeights::QuantizedLlama(model) => SessionWeights::QuantizedLlama(model.clone()),
            LoadedWeights::QuantizedQwen3(model) => SessionWeights::QuantizedQwen3(model.clone()),
        };
        check_cancellation(cancellation)?;
        let logits = weights.forward(&input, 0)?;
        let prefill_duration = started.elapsed();
        let processor = LogitsProcessor::from_sampling(generation.seed, sampling(&generation));

        Ok(Self {
            loaded,
            processor,
            decoder: IncrementalTextDecoder::new(&loaded.tokenizer),
            weights,
            logits,
            prompt_tokens: prompt_ids.len(),
            max_tokens,
            effective_max_tokens,
            all_tokens: prompt_ids.to_vec(),
            generated_tokens: 0,
            finish_reason: None,
            started,
            prefill_duration,
            time_to_first_token: None,
            delivery_duration: Duration::ZERO,
            generation,
            cancellation,
        })
    }

    fn next_token(&mut self) -> Result<GenerationStep, CandleError> {
        check_cancellation(self.cancellation)?;

        if self.finish_reason.is_some() {
            return self.finish();
        }

        if self.generation.repeat_penalty != 1.0 && self.generation.repeat_last_n > 0 {
            let recent = recent_tokens(&self.all_tokens, self.generation.repeat_last_n);
            self.logits =
                apply_repeat_penalty(&self.logits, self.generation.repeat_penalty, recent)
                    .map_err(|error| CandleError::Inference(error.to_string()))?;
        }
        let token = self
            .processor
            .sample(&self.logits)
            .map_err(|error| CandleError::Inference(error.to_string()))?;
        self.generated_tokens = self.generated_tokens.checked_add(1).ok_or_else(|| {
            CandleError::Inference("generated token count overflowed usize".to_string())
        })?;
        if self.time_to_first_token.is_none() {
            self.time_to_first_token = Some(self.started.elapsed());
        }

        if self.loaded.stop_tokens.contains(&token) {
            self.finish_reason = Some(FinishReason::Eos);
            return Ok(GenerationStep::Token(None));
        }

        self.all_tokens.push(token);
        let fragment = self.decoder.push(token)?;

        if self.generated_tokens >= self.max_tokens {
            self.finish_reason = Some(FinishReason::MaxTokens);
        } else {
            check_cancellation(self.cancellation)?;
            let generated_index = self.generated_tokens.saturating_sub(1);
            let position = next_cache_position(self.prompt_tokens, generated_index)?;
            let device = Device::Cpu;
            let next = Tensor::new(&[token], &device)
                .and_then(|tensor| tensor.unsqueeze(0))
                .map_err(|error| CandleError::Inference(error.to_string()))?;
            self.logits = self.weights.forward(&next, position)?;
        }

        Ok(GenerationStep::Token(fragment))
    }

    fn finish(&mut self) -> Result<GenerationStep, CandleError> {
        if let Some(suffix) = self.decoder.finish()? {
            return Ok(GenerationStep::Token(Some(suffix)));
        }

        let finish_reason = self.finish_reason.ok_or_else(|| {
            CandleError::Inference("generation finished without a finish reason".to_string())
        })?;
        let prompt_tokens = u64::try_from(self.prompt_tokens).map_err(|_| {
            CandleError::Inference("prompt token count does not fit in u64".to_string())
        })?;
        let generated_tokens = u64::try_from(self.generated_tokens).map_err(|_| {
            CandleError::Inference("generated token count does not fit in u64".to_string())
        })?;
        let generation_duration = self
            .started
            .elapsed()
            .saturating_sub(self.delivery_duration);
        let tokens_per_second = if generation_duration.is_zero() {
            None
        } else {
            Some(generated_tokens as f64 / generation_duration.as_secs_f64())
        };
        Ok(GenerationStep::Finished(CandleCompletionResponse {
            text: self.decoder.text().to_string(),
            prompt_tokens,
            generated_tokens,
            requested_max_tokens: self.generation.max_tokens,
            effective_max_tokens: self.effective_max_tokens,
            finish_reason,
            prefill_duration_ms: duration_millis(self.prefill_duration),
            time_to_first_token_ms: self.time_to_first_token.map(duration_millis),
            generation_duration_ms: duration_millis(generation_duration),
            tokens_per_second,
        }))
    }

    fn record_delivery_duration(&mut self, duration: Duration) {
        self.delivery_duration = self.delivery_duration.saturating_add(duration);
    }
}

fn duration_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).map_or(u64::MAX, |value| value)
}

fn generate(
    loaded: &LoadedModel,
    request: CompletionRequest,
    cancellation: &CancellationSignal,
    mut emit: impl FnMut(String) -> Result<(), CandleError>,
) -> Result<CandleCompletionResponse, CandleError> {
    #[cfg(all(test, not(target_family = "wasm")))]
    if let Some(control) = &loaded.test_control {
        control.enter_generation()?;
    }
    let mut session = GenerationSession::new(loaded, request, cancellation)?;
    loop {
        match session.next_token()? {
            GenerationStep::Token(Some(fragment)) if !fragment.is_empty() => {
                let delivery_started = Instant::now();
                let result = emit(fragment);
                session.record_delivery_duration(delivery_started.elapsed());
                result?;
            }
            GenerationStep::Token(_) => {}
            GenerationStep::Finished(response) => return Ok(response),
        }
    }
}

fn infer(
    loaded: &LoadedModel,
    request: CompletionRequest,
    cancellation: &CancellationSignal,
) -> Result<CompletionResponse<CandleCompletionResponse>, CandleError> {
    let parse_request = request.clone();
    let mut raw_response = generate(loaded, request, cancellation, |_| Ok(()))?;
    let parsed = protocol::parse_assistant(&raw_response.text, &parse_request, loaded.family)?;
    raw_response.text = parsed.visible_text;
    let choice = OneOrMany::many(parsed.items).map_err(|_| {
        CandleError::Inference("output protocol produced no assistant content".to_string())
    })?;
    let usage = raw_response.token_usage();
    Ok(CompletionResponse {
        choice,
        usage,
        raw_response,
        message_id: None,
    })
}

fn stream_generate(
    loaded: &LoadedModel,
    request: CompletionRequest,
    cancellation: &CancellationSignal,
    mut emit: impl FnMut(RawStreamingChoice<CandleCompletionResponse>) -> Result<(), CandleError>,
) -> Result<CandleCompletionResponse, CandleError> {
    if loaded.family != ModelFamily::Qwen3 {
        return generate(loaded, request, cancellation, |fragment| {
            emit(RawStreamingChoice::Message(fragment))
        });
    }

    let parse_request = request.clone();
    // Qwen tool syntax can straddle arbitrary token boundaries. Buffer one
    // model turn so control markup is never leaked as assistant text; complete
    // tool calls are still delivered through Rig's streaming agent driver.
    let mut response = generate(loaded, request, cancellation, |_| Ok(()))?;
    let parsed = protocol::parse_assistant(&response.text, &parse_request, loaded.family)?;
    response.text = parsed.visible_text;
    for item in parsed.items {
        match item {
            AssistantContent::Text(text) => emit(RawStreamingChoice::Message(text.text))?,
            AssistantContent::ToolCall(call) => {
                let mut raw =
                    RawStreamingToolCall::new(call.id, call.function.name, call.function.arguments);
                raw.call_id = call.call_id;
                raw.signature = call.signature;
                raw.additional_params = call.additional_params;
                emit(RawStreamingChoice::ToolCall(raw))?;
            }
            AssistantContent::Reasoning(reasoning) => {
                for content in reasoning.content {
                    emit(RawStreamingChoice::Reasoning {
                        id: reasoning.id.clone(),
                        content,
                    })?;
                }
            }
            AssistantContent::Image(_) => {
                return Err(CandleError::Inference(
                    "text-only Qwen output parser produced image content".to_string(),
                ));
            }
        }
    }
    Ok(response)
}

#[cfg(not(target_family = "wasm"))]
type CandleStreamItem = Result<RawStreamingChoice<CandleCompletionResponse>, CompletionError>;

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
    sender
        .blocking_send(Ok(RawStreamingChoice::FinalResponse(response)))
        .map_err(|_| CandleError::StreamingChannelClosed)
}

impl CompletionModel for CandleModel {
    type Response = CandleCompletionResponse;
    type StreamingResponse = CandleCompletionResponse;
    type Client = ();

    fn make(_: &Self::Client, _: impl Into<String>) -> Self {
        Self {
            state: ModelState::UnsupportedMake,
        }
    }

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        let ModelState::Ready(loaded) = &self.state else {
            return Err(CandleError::UnsupportedMake.into());
        };

        #[cfg(not(target_family = "wasm"))]
        {
            let cancellation = CancellationSignal::default();
            let mut cancel_on_drop = CancelOnDrop::new(cancellation.clone());
            let permit = acquire_concurrency(Arc::clone(&loaded.concurrency)).await?;
            let loaded = Arc::clone(loaded);
            let result = tokio::task::spawn_blocking(move || {
                let result = infer(&loaded, request, &cancellation);
                drop(permit);
                result
            })
            .await
            .map_err(|error| CandleError::BlockingTaskJoin(error.to_string()));
            cancel_on_drop.disarm();
            result?.map_err(CompletionError::from)
        }

        #[cfg(target_family = "wasm")]
        {
            infer(loaded, request, &CancellationSignal).map_err(CompletionError::from)
        }
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        let ModelState::Ready(loaded) = &self.state else {
            return Err(CandleError::UnsupportedMake.into());
        };

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
                let result =
                    stream_infer(&loaded, request, &producer_cancellation, &producer_sender);
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
            let stream: StreamingResult<CandleCompletionResponse> =
                Box::pin(CandleReceiverStream {
                    receiver,
                    cancellation,
                });
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
            events.push(Ok(RawStreamingChoice::FinalResponse(response)));
            let stream: StreamingResult<CandleCompletionResponse> =
                Box::pin(futures::stream::iter(events));
            Ok(StreamingCompletionResponse::stream(stream))
        }
    }
}

#[cfg(test)]
#[allow(clippy::panic_in_result_fn)]
mod tests {
    use std::borrow::Cow;
    use std::collections::HashMap;

    #[cfg(not(target_family = "wasm"))]
    use futures::StreamExt;
    use rig_core::completion::{CompletionModel, Document, ToolDefinition};
    use rig_core::message::{AudioMediaType, ImageDetail, ImageMediaType, ToolChoice};
    #[cfg(not(target_family = "wasm"))]
    use rig_core::streaming::StreamedAssistantContent;
    use safetensors::tensor::{Dtype, View, serialize};
    use tokenizers::decoders::byte_fallback::ByteFallback;
    use tokenizers::models::bpe::{BPE, Vocab};
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::normalizers::unicode::NFC;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use tokenizers::{AddedToken, TokenizerBuilder};

    use super::*;

    #[cfg(not(target_family = "wasm"))]
    type ControlledModel = (LlamaModel, Arc<TestControl>, Arc<tokio::sync::Semaphore>);

    struct TestTensor {
        dtype: Dtype,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    }

    impl View for TestTensor {
        fn dtype(&self) -> Dtype {
            self.dtype
        }

        fn shape(&self) -> &[usize] {
            &self.shape
        }

        fn data(&self) -> Cow<'_, [u8]> {
            Cow::Borrowed(&self.bytes)
        }

        fn data_len(&self) -> usize {
            self.bytes.len()
        }
    }

    fn tiny_config() -> Vec<u8> {
        br#"{
            "hidden_size": 4,
            "intermediate_size": 8,
            "vocab_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "rms_norm_eps": 0.00001,
            "max_position_embeddings": 128,
            "bos_token_id": 2,
            "eos_token_id": [1, 3],
            "tie_word_embeddings": false
        }"#
        .to_vec()
    }

    fn tiny_tokenizer() -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        tiny_tokenizer_with_end_header(END_HEADER, true)
    }

    fn tiny_tokenizer_with_end_header(
        end_header: &str,
        mark_end_header_special: bool,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        let vocab = [
            ("<unk>".to_string(), 0),
            (END_OF_TURN.to_string(), 1),
            (BEGIN_OF_TEXT.to_string(), 2),
            ("<eos>".to_string(), 3),
            (START_HEADER.to_string(), 4),
            (end_header.to_string(), 5),
            ("assistant".to_string(), 6),
            ("hello".to_string(), 7),
        ]
        .into_iter()
        .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".to_string())
            .build()?;
        let mut tokenizer = Tokenizer::new(model);
        let mut special_tokens = vec![
            AddedToken::from(END_OF_TURN, true),
            AddedToken::from(BEGIN_OF_TEXT, true),
            AddedToken::from(START_HEADER, true),
        ];
        if mark_end_header_special {
            special_tokens.push(AddedToken::from(end_header, true));
        }
        tokenizer.add_special_tokens(&special_tokens);
        Ok(tokenizer.to_string(false)?.into_bytes())
    }

    fn tiny_smollm2_tokenizer(
        include_end: bool,
        mark_end_special: bool,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        let end = if include_end { IM_END } else { "<other>" };
        let vocab = [
            ("<unk>".to_string(), 0),
            (IM_START.to_string(), 1),
            (end.to_string(), 2),
            ("<eos>".to_string(), 3),
            ("system".to_string(), 4),
            ("user".to_string(), 5),
            ("assistant".to_string(), 6),
            ("hello".to_string(), 7),
        ]
        .into_iter()
        .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".to_string())
            .build()?;
        let mut tokenizer = Tokenizer::new(model);
        let mut special = vec![AddedToken::from(IM_START, true)];
        if mark_end_special {
            special.push(AddedToken::from(end, true));
        }
        tokenizer.add_special_tokens(&special);
        Ok(tokenizer.to_string(false)?.into_bytes())
    }

    fn tensor(shape: &[usize]) -> TestTensor {
        tensor_with_dtype(shape, Dtype::F32)
    }

    fn tensor_with_dtype(shape: &[usize], dtype: Dtype) -> TestTensor {
        let elements = shape.iter().product::<usize>();
        let element_size = match dtype {
            Dtype::F64 | Dtype::I64 | Dtype::U64 => 8,
            Dtype::F32 | Dtype::I32 | Dtype::U32 => 4,
            Dtype::F16 | Dtype::BF16 | Dtype::I16 | Dtype::U16 => 2,
            _ => 1,
        };
        TestTensor {
            dtype,
            shape: shape.to_vec(),
            bytes: vec![0; elements * element_size],
        }
    }

    fn checkpoint(include_all: bool) -> Result<Vec<u8>, safetensors::SafeTensorError> {
        checkpoint_custom(include_all, tensor(&[8, 4]), true)
    }

    fn checkpoint_custom(
        include_all: bool,
        embedding: TestTensor,
        include_lm_head: bool,
    ) -> Result<Vec<u8>, safetensors::SafeTensorError> {
        let mut tensors = vec![
            ("model.embed_tokens.weight".to_string(), embedding),
            ("model.norm.weight".to_string(), tensor(&[4])),
            (
                "model.layers.0.self_attn.q_proj.weight".to_string(),
                tensor(&[4, 4]),
            ),
            (
                "model.layers.0.self_attn.k_proj.weight".to_string(),
                tensor(&[4, 4]),
            ),
            (
                "model.layers.0.self_attn.v_proj.weight".to_string(),
                tensor(&[4, 4]),
            ),
            (
                "model.layers.0.self_attn.o_proj.weight".to_string(),
                tensor(&[4, 4]),
            ),
            (
                "model.layers.0.mlp.gate_proj.weight".to_string(),
                tensor(&[8, 4]),
            ),
            (
                "model.layers.0.mlp.up_proj.weight".to_string(),
                tensor(&[8, 4]),
            ),
            (
                "model.layers.0.mlp.down_proj.weight".to_string(),
                tensor(&[4, 8]),
            ),
            (
                "model.layers.0.input_layernorm.weight".to_string(),
                tensor(&[4]),
            ),
            (
                "model.layers.0.post_attention_layernorm.weight".to_string(),
                tensor(&[4]),
            ),
        ];
        if include_lm_head {
            tensors.push(("lm_head.weight".to_string(), tensor(&[8, 4])));
        }
        if !include_all {
            tensors.retain(|(name, _)| name != "model.layers.0.self_attn.q_proj.weight");
        }
        serialize(tensors, None)
    }

    fn model_data() -> Result<ModelData, Box<dyn std::error::Error + Send + Sync>> {
        Ok(ModelData {
            config: tiny_config(),
            tokenizer: tiny_tokenizer()?,
            weights: checkpoint(true)?,
        })
    }

    fn config_with(
        field: &str,
        value: serde_json::Value,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        let mut config: serde_json::Value = serde_json::from_slice(&tiny_config())?;
        config
            .as_object_mut()
            .ok_or("test config must be a JSON object")?
            .insert(field.to_string(), value);
        Ok(serde_json::to_vec(&config)?)
    }

    fn request(messages: Vec<Message>) -> CompletionRequest {
        CompletionRequest {
            model: None,
            preamble: None,
            chat_history: match OneOrMany::many(messages) {
                Ok(messages) => messages,
                Err(_) => OneOrMany::one(Message::user("hello")),
            },
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        }
    }

    #[cfg(not(target_family = "wasm"))]
    async fn collect_stream(
        model: &LlamaModel,
        request: CompletionRequest,
    ) -> Result<(String, CandleCompletionResponse), Box<dyn std::error::Error + Send + Sync>> {
        let mut response = model.stream(request).await?;
        let mut text = String::new();
        let mut final_response = None;
        while let Some(item) = response.next().await {
            match item? {
                StreamedAssistantContent::Text(fragment) => text.push_str(&fragment.text),
                StreamedAssistantContent::Final(raw) => final_response = Some(raw),
                _ => {}
            }
        }
        let raw = final_response.ok_or("stream did not emit a final response")?;
        Ok((text, raw))
    }

    #[cfg(not(target_family = "wasm"))]
    fn controlled_model(
        blocked: bool,
        panic_after_gate: bool,
        max_tokens: u64,
    ) -> Result<ControlledModel, Box<dyn std::error::Error + Send + Sync>> {
        let generation = GenerationConfig {
            temperature: 0.0,
            max_tokens,
            ..GenerationConfig::default()
        };
        let mut loaded = load_model(model_data()?, generation, 1)?;
        let control = Arc::new(TestControl::new(blocked, panic_after_gate));
        let concurrency = Arc::clone(&loaded.concurrency);
        loaded.test_control = Some(Arc::clone(&control));
        Ok((
            LlamaModel {
                state: ModelState::Ready(Arc::new(loaded)),
            },
            control,
            concurrency,
        ))
    }

    #[test]
    fn rejects_empty_and_malformed_artifacts()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        for (artifact, data) in [
            (
                "config",
                ModelData {
                    config: Vec::new(),
                    tokenizer: vec![1],
                    weights: vec![1],
                },
            ),
            (
                "tokenizer",
                ModelData {
                    config: tiny_config(),
                    tokenizer: Vec::new(),
                    weights: vec![1],
                },
            ),
            (
                "weights",
                ModelData {
                    config: tiny_config(),
                    tokenizer: tiny_tokenizer()?,
                    weights: Vec::new(),
                },
            ),
        ] {
            let error = LlamaModel::from_safetensors(data)
                .err()
                .ok_or("expected empty-buffer error")?;
            assert!(
                matches!(error, CandleError::EmptyBuffer { artifact: actual } if actual == artifact)
            );
        }

        let mut data = model_data()?;
        data.config = b"not json".to_vec();
        assert!(matches!(
            LlamaModel::from_safetensors(data),
            Err(CandleError::Configuration(_))
        ));
        let mut data = model_data()?;
        data.tokenizer = b"not json".to_vec();
        assert!(matches!(
            LlamaModel::from_safetensors(data),
            Err(CandleError::TokenizerLoading(_))
        ));
        let mut data = model_data()?;
        data.weights = b"not safetensors".to_vec();
        assert!(matches!(
            LlamaModel::from_safetensors(data),
            Err(CandleError::InvalidCheckpoint(_))
        ));
        Ok(())
    }

    #[test]
    fn validates_checkpoint_metadata_before_model_loading()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let config: LlamaConfig = serde_json::from_slice(&tiny_config())?;
        let config = config.into_config(false);
        validate_checkpoint(&checkpoint(true)?, &config)?;
        let error = validate_checkpoint(&checkpoint(false)?, &config)
            .err()
            .ok_or("expected missing tensor")?;
        assert!(
            matches!(error, CandleError::MissingTensor(name) if name == "model.layers.0.self_attn.q_proj.weight")
        );
        Ok(())
    }

    #[test]
    fn validates_tensor_shapes_dtypes_and_tied_embeddings()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let config: LlamaConfig = serde_json::from_slice(&tiny_config())?;
        let config = config.into_config(false);

        let shape_error =
            validate_checkpoint(&checkpoint_custom(true, tensor(&[7, 4]), true)?, &config)
                .err()
                .ok_or("expected shape error")?;
        assert!(matches!(
            shape_error,
            CandleError::TensorShapeMismatch { tensor, expected, actual }
                if tensor == "model.embed_tokens.weight"
                    && expected == vec![8, 4]
                    && actual == vec![7, 4]
        ));

        let dtype_error = validate_checkpoint(
            &checkpoint_custom(true, tensor_with_dtype(&[8, 4], Dtype::U8), true)?,
            &config,
        )
        .err()
        .ok_or("expected dtype error")?;
        assert!(matches!(
            dtype_error,
            CandleError::UnsupportedTensorDtype { tensor, dtype }
                if tensor == "model.embed_tokens.weight" && dtype == "U8"
        ));
        validate_checkpoint(
            &checkpoint_custom(true, tensor_with_dtype(&[8, 4], Dtype::F16), true)?,
            &config,
        )?;
        validate_checkpoint(
            &checkpoint_custom(true, tensor_with_dtype(&[8, 4], Dtype::BF16), true)?,
            &config,
        )?;
        assert!(matches!(
            validate_checkpoint(
                &checkpoint_custom(true, tensor(&[8, 4]), false)?,
                &config
            ),
            Err(CandleError::MissingTensor(name)) if name == "lm_head.weight"
        ));

        let tied_config: LlamaConfig =
            serde_json::from_slice(&config_with("tie_word_embeddings", true.into())?)?;
        let tied_config = tied_config.into_config(false);
        validate_checkpoint(
            &checkpoint_custom(true, tensor(&[8, 4]), false)?,
            &tied_config,
        )?;
        let model = LlamaModel::from_safetensors(ModelData {
            config: config_with("tie_word_embeddings", true.into())?,
            tokenizer: tiny_tokenizer()?,
            weights: checkpoint_custom(true, tensor(&[8, 4]), false)?,
        })?;
        assert!(matches!(model.state, ModelState::Ready(_)));
        Ok(())
    }

    #[test]
    fn validates_tokenizer_vocabulary_special_tokens_and_configured_ids()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let data = ModelData {
            config: config_with("vocab_size", 9.into())?,
            tokenizer: tiny_tokenizer()?,
            weights: checkpoint(true)?,
        };
        assert!(matches!(
            LlamaModel::from_safetensors(data),
            Err(CandleError::TokenizerVocabularyMismatch {
                expected: 9,
                actual: 8
            })
        ));

        let config: LlamaConfig = serde_json::from_slice(&tiny_config())?;
        let config = config.into_config(false);
        let tokenizer = Tokenizer::from_bytes(tiny_tokenizer_with_end_header("<other>", true)?)?;
        assert!(matches!(
            validate_tokenizer(&config, &tokenizer, ModelFamily::Llama3),
            Err(CandleError::MissingSpecialToken { token: END_HEADER })
        ));
        let tokenizer = Tokenizer::from_bytes(tiny_tokenizer_with_end_header(END_HEADER, false)?)?;
        assert!(matches!(
            validate_tokenizer(&config, &tokenizer, ModelFamily::Llama3),
            Err(CandleError::SpecialTokenNotMarked { token: END_HEADER })
        ));

        let tokenizer = Tokenizer::from_bytes(tiny_smollm2_tokenizer(false, true)?)?;
        assert!(matches!(
            validate_tokenizer(&config, &tokenizer, ModelFamily::SmolLm2),
            Err(CandleError::MissingSpecialToken { token: IM_END })
        ));
        let tokenizer = Tokenizer::from_bytes(tiny_smollm2_tokenizer(true, false)?)?;
        assert!(matches!(
            validate_tokenizer(&config, &tokenizer, ModelFamily::SmolLm2),
            Err(CandleError::SpecialTokenNotMarked { token: IM_END })
        ));

        let data = ModelData {
            config: config_with("bos_token_id", 8.into())?,
            tokenizer: tiny_tokenizer()?,
            weights: checkpoint(true)?,
        };
        assert!(matches!(
            LlamaModel::from_safetensors(data),
            Err(CandleError::TokenIdOutOfRange { token, id: 8, .. }) if token == "bos_token_id"
        ));
        let data = ModelData {
            config: config_with("eos_token_id", serde_json::json!([1, 9]))?,
            tokenizer: tiny_tokenizer()?,
            weights: checkpoint(true)?,
        };
        assert!(matches!(
            LlamaModel::from_safetensors(data),
            Err(CandleError::TokenIdOutOfRange { token, id: 9, .. }) if token == "eos_token_id"
        ));
        for (field, value) in [("bos_token_id", 7.into()), ("eos_token_id", 3.into())] {
            let data = ModelData {
                config: config_with(field, value)?,
                tokenizer: tiny_tokenizer()?,
                weights: checkpoint(true)?,
            };
            assert!(matches!(
                LlamaModel::from_safetensors(data),
                Err(CandleError::ArtifactMismatch { artifact, .. }) if artifact == field
            ));
        }
        let data = ModelData {
            config: config_with("eos_token_id", serde_json::json!([]))?,
            tokenizer: tiny_tokenizer()?,
            weights: checkpoint(true)?,
        };
        assert!(matches!(
            LlamaModel::from_safetensors(data),
            Err(CandleError::InvalidConfigurationValue {
                field: "eos_token_id",
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn validates_model_dimension_relationships()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        for (field, value) in [
            ("hidden_size", 0),
            ("num_attention_heads", 0),
            ("num_key_value_heads", 0),
            ("max_position_embeddings", 0),
        ] {
            let config: LlamaConfig = serde_json::from_slice(&config_with(field, value.into())?)?;
            assert!(matches!(
                validate_model_config(&config.into_config(false)),
                Err(CandleError::InvalidConfigurationValue { field: actual, .. }) if actual == field
            ));
        }
        let config: LlamaConfig =
            serde_json::from_slice(&config_with("num_attention_heads", 3.into())?)?;
        assert!(matches!(
            validate_model_config(&config.into_config(false)),
            Err(CandleError::InvalidConfigurationValue {
                field: "hidden_size",
                ..
            })
        ));
        let mut odd_head_config: serde_json::Value = serde_json::from_slice(&tiny_config())?;
        let object = odd_head_config
            .as_object_mut()
            .ok_or("test config must be a JSON object")?;
        object.insert("hidden_size".to_string(), 6.into());
        object.insert("num_attention_heads".to_string(), 2.into());
        let config: LlamaConfig = serde_json::from_value(odd_head_config)?;
        assert!(matches!(
            validate_model_config(&config.into_config(false)),
            Err(CandleError::InvalidConfigurationValue {
                field: "hidden_size",
                ..
            })
        ));

        #[cfg(target_pointer_width = "64")]
        {
            let oversized_context = u64::from(u32::MAX) + 1;
            let config: LlamaConfig = serde_json::from_slice(&config_with(
                "max_position_embeddings",
                oversized_context.into(),
            )?)?;
            assert!(matches!(
                validate_model_config(&config.into_config(false)),
                Err(CandleError::InvalidConfigurationValue {
                    field: "max_position_embeddings",
                    ..
                })
            ));
        }

        let mut rope_config: serde_json::Value = serde_json::from_slice(&tiny_config())?;
        rope_config
            .as_object_mut()
            .ok_or("test config must be a JSON object")?
            .insert(
                "rope_scaling".to_string(),
                serde_json::json!({
                    "factor": 0.0,
                    "low_freq_factor": 1.0,
                    "high_freq_factor": 4.0,
                    "original_max_position_embeddings": 128,
                    "rope_type": "llama3"
                }),
            );
        let config: LlamaConfig = serde_json::from_value(rope_config)?;
        assert!(matches!(
            validate_model_config(&config.into_config(false)),
            Err(CandleError::InvalidConfigurationValue {
                field: "rope_scaling.factor",
                ..
            })
        ));

        let mut rope_config: serde_json::Value = serde_json::from_slice(&tiny_config())?;
        rope_config
            .as_object_mut()
            .ok_or("test config must be a JSON object")?
            .insert(
                "rope_scaling".to_string(),
                serde_json::json!({
                    "factor": 8.0,
                    "low_freq_factor": 4.0,
                    "high_freq_factor": 4.0,
                    "original_max_position_embeddings": 128,
                    "rope_type": "llama3"
                }),
            );
        let config: LlamaConfig = serde_json::from_value(rope_config)?;
        assert!(matches!(
            validate_model_config(&config.into_config(false)),
            Err(CandleError::InvalidConfigurationValue {
                field: "rope_scaling.high_freq_factor",
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn context_limit_boundaries_clamp_and_detect_conversion_overflow() {
        assert!(matches!(
            effective_output_limit(9, 1, 8),
            Err(CandleError::PromptTooLong {
                prompt_tokens: 9,
                context_limit: 8
            })
        ));
        assert!(matches!(
            effective_output_limit(8, 1, 8),
            Err(CandleError::NoGenerationCapacity {
                prompt_tokens: 8,
                context_limit: 8
            })
        ));
        assert!(matches!(effective_output_limit(6, 10, 8), Ok(2)));
        assert!(matches!(effective_output_limit(6, 1, 8), Ok(1)));
        assert!(matches!(
            max_tokens_to_usize(256, 255),
            Err(CandleError::NumericConversion {
                field: "max_tokens",
                value: 256
            })
        ));
    }

    #[test]
    fn loads_entirely_from_owned_bytes() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let model = LlamaModel::from_safetensors(model_data()?)?;
        assert!(matches!(model.state, ModelState::Ready(_)));
        assert_eq!(model.model_family(), Some(ModelFamily::Llama3));
        assert_eq!(model.quantization(), None);
        Ok(())
    }

    #[test]
    fn typed_gguf_and_family_errors_preserve_the_failure_kind()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let data = model_data()?;
        assert!(matches!(
            LlamaModel::builder(data)
                .model_family(ModelFamily::SmolLm2)
                .build(),
            Err(CandleError::ModelFamilyMismatch {
                selected: ModelFamily::SmolLm2,
                detected: ModelFamily::Llama3,
            })
        ));

        let mut malformed = model_data()?;
        malformed.weights = b"not a gguf checkpoint".to_vec();
        assert!(matches!(
            LlamaModel::from_gguf(malformed),
            Err(CandleError::InvalidQuantizedCheckpoint(_))
        ));
        Ok(())
    }

    #[test]
    fn gguf_metadata_shapes_and_tensor_encodings_are_validated_before_loading()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let config: LlamaConfig = serde_json::from_slice(&tiny_config())?;
        let config = config.into_config(false);
        let mut content = gguf_file::Content {
            magic: gguf_file::VersionedMagic::GgufV3,
            metadata: HashMap::from([("llama.vocab_size".to_string(), gguf_file::Value::U32(9))]),
            tensor_infos: HashMap::new(),
            tensor_data_offset: 0,
        };
        let tokenizer = Tokenizer::from_bytes(tiny_tokenizer()?)?;
        assert!(matches!(
            validate_gguf_metadata(&content, &config, &tokenizer),
            Err(CandleError::ArtifactMismatch {
                artifact: "model.gguf",
                ..
            })
        ));

        content.tensor_infos.insert(
            "token_embd.weight".to_string(),
            gguf_file::TensorInfo {
                ggml_dtype: GgmlDType::Q4K,
                shape: candle_core::Shape::from(vec![7, 4]),
                offset: 0,
            },
        );
        assert!(matches!(
            validate_gguf_tensors(&content, &config),
            Err(CandleError::InvalidQuantizedCheckpoint(message))
                if message.contains("token_embd.weight") && message.contains("expected")
        ));

        content
            .tensor_infos
            .get_mut("token_embd.weight")
            .ok_or("synthetic GGUF tensor disappeared")?
            .ggml_dtype = GgmlDType::Q2K;
        assert!(matches!(
            validate_gguf_tensors(&content, &config),
            Err(CandleError::UnsupportedQuantization(message))
                if message.contains("token_embd.weight")
        ));
        Ok(())
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn loaded_model_works_with_agent_builder()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use rig_agent::agent::AgentBuilder;
        use rig_agent::completion::Prompt;

        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        runtime.block_on(async {
            let model = LlamaModel::builder(model_data()?)
                .temperature(0.0)
                .max_tokens(1)
                .build()?;
            let agent = AgentBuilder::new(model).preamble("Be brief.").build();
            let _answer = agent.prompt("hello").await?;
            Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
        })?;
        Ok(())
    }

    #[cfg(not(target_family = "wasm"))]
    #[tokio::test(flavor = "current_thread")]
    async fn buffered_and_streaming_generation_are_equivalent()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let model = LlamaModel::builder(model_data()?)
            .temperature(0.0)
            .max_tokens(3)
            .build()?;
        let completion_request = request(vec![Message::user("hello")]);
        let buffered = model.completion(completion_request.clone()).await?;
        let (streamed_text, streamed) = collect_stream(&model, completion_request).await?;

        assert_eq!(streamed_text, buffered.raw_response.text);
        assert_eq!(streamed.text, buffered.raw_response.text);
        assert_eq!(streamed.prompt_tokens, buffered.raw_response.prompt_tokens);
        assert_eq!(
            streamed.generated_tokens,
            buffered.raw_response.generated_tokens
        );
        assert_eq!(streamed.finish_reason, buffered.raw_response.finish_reason);
        assert_eq!(
            streamed.requested_max_tokens,
            buffered.raw_response.requested_max_tokens
        );
        assert_eq!(
            streamed.effective_max_tokens,
            buffered.raw_response.effective_max_tokens
        );
        assert_eq!(streamed.token_usage(), buffered.usage);
        assert!(streamed.time_to_first_token_ms.is_some());
        assert!(streamed.prefill_duration_ms <= streamed.generation_duration_ms);
        Ok(())
    }

    #[cfg(not(target_family = "wasm"))]
    #[tokio::test(flavor = "current_thread")]
    async fn streaming_reports_eos_and_excludes_the_stop_token()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut loaded = load_model(model_data()?, GenerationConfig::default(), 1)?;
        loaded.generation.temperature = 0.0;
        loaded.stop_tokens.insert(0);
        let model = LlamaModel {
            state: ModelState::Ready(Arc::new(loaded)),
        };
        let (text, raw) = collect_stream(&model, request(vec![Message::user("hello")])).await?;
        assert!(text.is_empty());
        assert!(raw.text.is_empty());
        assert_eq!(raw.finish_reason, FinishReason::Eos);
        assert_eq!(raw.generated_tokens, 1);
        assert_eq!(raw.token_usage().output_tokens, 1);
        Ok(())
    }

    #[cfg(not(target_family = "wasm"))]
    #[tokio::test(flavor = "current_thread")]
    async fn streaming_clamps_context_and_rejects_bad_request_options()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut loaded = load_model(model_data()?, GenerationConfig::default(), 1)?;
        let mut completion_request = request(vec![Message::user("hello")]);
        completion_request.max_tokens = Some(10);
        completion_request.temperature = Some(0.0);
        let prompt = render_prompt(&completion_request)?;
        let prompt_tokens = loaded.tokenizer.encode(prompt, false)?.len();
        loaded.spec.context_limit = prompt_tokens + 2;
        let model = LlamaModel {
            state: ModelState::Ready(Arc::new(loaded)),
        };
        let (_, raw) = collect_stream(&model, completion_request).await?;
        assert_eq!(raw.requested_max_tokens, 10);
        assert_eq!(raw.effective_max_tokens, 2);
        assert_eq!(raw.generated_tokens, 2);

        for additional_params in [
            serde_json::json!({"unknown": true}),
            serde_json::json!({"top_k": "four"}),
        ] {
            let mut bad_request = request(vec![Message::user("hello")]);
            bad_request.additional_params = Some(additional_params);
            let mut stream = model.stream(bad_request).await?;
            let item = stream
                .next()
                .await
                .ok_or("bad streaming request produced no error item")?;
            assert!(item.is_err());
        }
        Ok(())
    }

    #[test]
    fn incremental_decoder_preserves_token_boundaries() -> Result<(), CandleError> {
        let tokenizer = Tokenizer::from_bytes(
            tiny_tokenizer().map_err(|error| CandleError::TokenizerLoading(error.to_string()))?,
        )
        .map_err(|error| CandleError::TokenizerLoading(error.to_string()))?;
        let ids = [0, 0, 7];
        let independently_decoded = ids
            .iter()
            .map(|id| tokenizer.decode(&[*id], true))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| CandleError::TokenizerDecoding(error.to_string()))?
            .join("");
        let complete = tokenizer
            .decode(&ids, true)
            .map_err(|error| CandleError::TokenizerDecoding(error.to_string()))?;
        assert_ne!(independently_decoded, complete);

        let mut decoder = IncrementalTextDecoder::new(&tokenizer);
        let mut streamed = String::new();
        for id in ids {
            if let Some(fragment) = decoder.push(id)? {
                streamed.push_str(&fragment);
            }
        }
        if let Some(fragment) = decoder.finish()? {
            streamed.push_str(&fragment);
        }
        assert_eq!(streamed, complete);
        assert_eq!(streamed, decoder.text());
        Ok(())
    }

    #[test]
    fn incremental_decoder_waits_for_complete_unicode_bytes()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let vocab: Vocab = [
            ("<0x20>".to_string(), 0),
            ("<0xC3>".to_string(), 1),
            ("<0xA9>".to_string(), 2),
        ]
        .into_iter()
        .collect();
        let tokenizer: Tokenizer = TokenizerBuilder::default()
            .with_model(
                BPE::builder()
                    .vocab_and_merges(vocab, Vec::new())
                    .byte_fallback(true)
                    .build()?,
            )
            .with_decoder(Some(ByteFallback::default()))
            .with_normalizer(Some(NFC))
            .with_pre_tokenizer(Some(ByteLevel::default()))
            .with_post_processor(Some(ByteLevel::default()))
            .build()?
            .into();
        let mut decoder = IncrementalTextDecoder::new(&tokenizer);
        assert!(decoder.push(1)?.is_none());
        assert_eq!(decoder.push(2)?.as_deref(), Some("é"));
        assert!(decoder.finish()?.is_none());
        assert_eq!(decoder.text(), "é");
        Ok(())
    }

    #[test]
    fn inference_clamps_context_and_uses_fresh_generation_state()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut loaded = load_model(model_data()?, GenerationConfig::default(), 1)?;
        let mut completion_request = request(vec![Message::user("hello")]);
        completion_request.max_tokens = Some(10);
        completion_request.temperature = Some(0.0);
        let prompt = render_prompt(&completion_request)?;
        let prompt_tokens = loaded.tokenizer.encode(prompt, false)?.len();
        loaded.spec.context_limit = prompt_tokens + 2;

        let first = infer(
            &loaded,
            completion_request.clone(),
            &CancellationSignal::default(),
        )?;
        let second = infer(&loaded, completion_request, &CancellationSignal::default())?;
        assert_eq!(first.raw_response.text, second.raw_response.text);
        assert_eq!(first.raw_response.generated_tokens, 2);
        assert_eq!(first.raw_response.requested_max_tokens, 10);
        assert_eq!(first.raw_response.effective_max_tokens, 2);
        assert_eq!(first.raw_response.finish_reason, FinishReason::MaxTokens);
        assert_eq!(first.usage.output_tokens, 2);
        assert!(!first.raw_response.text.contains("hello"));
        Ok(())
    }

    #[test]
    fn eos_is_counted_but_excluded_from_decoded_text()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut loaded = load_model(model_data()?, GenerationConfig::default(), 1)?;
        loaded.stop_tokens.insert(0);
        let mut completion_request = request(vec![Message::user("hello")]);
        completion_request.temperature = Some(0.0);
        let response = infer(&loaded, completion_request, &CancellationSignal::default())?;
        assert_eq!(response.raw_response.finish_reason, FinishReason::Eos);
        assert_eq!(response.raw_response.generated_tokens, 1);
        assert_eq!(response.usage.output_tokens, 1);
        assert!(response.raw_response.text.is_empty());
        Ok(())
    }

    #[test]
    fn sampling_modes_and_repeat_window_are_exact() {
        let mut config = GenerationConfig {
            temperature: 0.0,
            ..GenerationConfig::default()
        };
        assert!(matches!(sampling(&config), Sampling::ArgMax));
        config.temperature = 0.5;
        config.top_k = Some(3);
        config.top_p = None;
        assert!(matches!(sampling(&config), Sampling::TopK { k: 3, .. }));
        config.top_k = None;
        config.top_p = Some(0.8);
        assert!(matches!(sampling(&config), Sampling::TopP { p: 0.8, .. }));
        config.top_k = Some(2);
        assert!(matches!(
            sampling(&config),
            Sampling::TopKThenTopP { k: 2, p: 0.8, .. }
        ));
        assert_eq!(recent_tokens(&[1, 2, 3, 4], 2), &[3, 4]);
        assert_eq!(recent_tokens(&[1, 2], 8), &[1, 2]);
        assert_eq!(recent_tokens(&[1, 2], 0), &[] as &[u32]);
        assert!(matches!(next_cache_position(12, 0), Ok(12)));
        assert!(matches!(next_cache_position(12, 1), Ok(13)));
        assert!(next_cache_position(usize::MAX, 1).is_err());
    }

    #[test]
    fn qwen3_4b_configuration_is_exactly_scoped() -> Result<(), CandleError> {
        let mut config: Qwen3Config = serde_json::from_str(
            r#"{
                "architectures":["Qwen3ForCausalLM"],
                "model_type":"qwen3",
                "hidden_size":2560,
                "intermediate_size":9728,
                "num_hidden_layers":36,
                "num_attention_heads":32,
                "num_key_value_heads":8,
                "head_dim":128,
                "max_position_embeddings":40960,
                "vocab_size":151936,
                "rms_norm_eps":0.000001,
                "rope_theta":1000000,
                "tie_word_embeddings":true,
                "bos_token_id":151643,
                "eos_token_id":151645,
                "hidden_act":"silu",
                "attention_bias":false
            }"#,
        )
        .map_err(|error| CandleError::Configuration(error.to_string()))?;
        validate_qwen3_config(&config)?;

        config.model_type = "qwen2".to_string();
        assert!(matches!(
            validate_qwen3_config(&config),
            Err(CandleError::UnsupportedModelFamily(_))
        ));
        config.model_type = "qwen3".to_string();
        config.hidden_size = 4096;
        assert!(matches!(
            validate_qwen3_config(&config),
            Err(CandleError::ArtifactMismatch {
                artifact: "config.json",
                ..
            })
        ));
        Ok(())
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn concurrency_limit_and_cancellation_are_deterministic()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        assert!(matches!(
            LlamaModel::builder(model_data()?)
                .max_concurrent_requests(0)
                .build(),
            Err(CandleError::InvalidConcurrencyLimit)
        ));

        let loaded = load_model(model_data()?, GenerationConfig::default(), 1)?;
        let permit = Arc::clone(&loaded.concurrency).try_acquire_owned()?;
        assert!(Arc::clone(&loaded.concurrency).try_acquire_owned().is_err());
        drop(permit);
        assert!(Arc::clone(&loaded.concurrency).try_acquire_owned().is_ok());

        let signal = CancellationSignal::default();
        {
            let _guard = CancelOnDrop::new(signal.clone());
        }
        assert!(signal.is_cancelled());

        let signal = CancellationSignal::default();
        signal.cancel();
        assert!(matches!(
            infer(&loaded, request(vec![Message::user("hello")]), &signal),
            Err(CandleError::Cancelled)
        ));
        Ok(())
    }

    #[cfg(not(target_family = "wasm"))]
    #[tokio::test(flavor = "current_thread")]
    async fn concurrent_completions_have_independent_caches_and_samplers()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let model = LlamaModel::builder(model_data()?)
            .temperature(0.0)
            .max_tokens(2)
            .max_concurrent_requests(2)
            .build()?;
        let first = model.completion(request(vec![Message::user("hello")]));
        let second = model.completion(request(vec![Message::user("hello")]));
        let (first, second) = tokio::join!(first, second);
        let first = first?;
        let second = second?;
        assert_eq!(first.raw_response.text, second.raw_response.text);
        assert_eq!(first.raw_response.generated_tokens, 2);
        assert_eq!(second.raw_response.generated_tokens, 2);

        let first_stream = collect_stream(&model, request(vec![Message::user("hello")]));
        let second_stream = collect_stream(&model, request(vec![Message::user("hello")]));
        let (first_stream, second_stream) = tokio::join!(first_stream, second_stream);
        let (first_text, first_raw) = first_stream?;
        let (second_text, second_raw) = second_stream?;
        assert_eq!(first_text, second_text);
        assert_eq!(first_raw.text, second_raw.text);
        assert_eq!(first_raw.generated_tokens, 2);
        assert_eq!(second_raw.generated_tokens, 2);

        let semaphore = Arc::new(tokio::sync::Semaphore::new(1));
        semaphore.close();
        assert!(matches!(
            acquire_concurrency(semaphore).await,
            Err(CandleError::ConcurrencyControllerClosed)
        ));
        Ok(())
    }

    #[cfg(not(target_family = "wasm"))]
    #[tokio::test(flavor = "current_thread")]
    async fn closed_admission_controller_fails_public_operations()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let model = LlamaModel::builder(model_data()?).build()?;
        let ModelState::Ready(loaded) = &model.state else {
            return Err("loaded model was not ready".into());
        };
        loaded.concurrency.close();
        let completion_error = model
            .completion(request(vec![Message::user("hello")]))
            .await
            .err()
            .ok_or("closed completion admission unexpectedly succeeded")?;
        assert!(
            completion_error
                .to_string()
                .contains("concurrency controller is closed")
        );
        let stream_error = model
            .stream(request(vec![Message::user("hello")]))
            .await
            .err()
            .ok_or("closed stream admission unexpectedly succeeded")?;
        assert!(
            stream_error
                .to_string()
                .contains("concurrency controller is closed")
        );
        Ok(())
    }

    #[cfg(not(target_family = "wasm"))]
    #[tokio::test(flavor = "current_thread")]
    async fn dropping_buffered_completion_retains_permit_until_worker_exits()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let (model, control, concurrency) = controlled_model(true, false, 2)?;
        let first_model = model.clone();
        let first = tokio::spawn(async move {
            first_model
                .completion(request(vec![Message::user("hello")]))
                .await
        });
        control.wait_until_entered().await;

        let second = model.completion(request(vec![Message::user("hello")]));
        futures::pin_mut!(second);
        assert!(futures::poll!(&mut second).is_pending());

        first.abort();
        assert!(first.await.is_err());
        assert!(Arc::clone(&concurrency).try_acquire_owned().is_err());
        control.release()?;

        let second = second.await?;
        assert_eq!(second.raw_response.generated_tokens, 2);
        Ok(())
    }

    #[cfg(not(target_family = "wasm"))]
    #[tokio::test(flavor = "current_thread")]
    async fn dropping_stream_cancels_worker_before_queued_request_runs()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let (model, control, concurrency) = controlled_model(true, false, 2)?;
        let stream = model.stream(request(vec![Message::user("hello")])).await?;
        control.wait_until_entered().await;

        let queued = model.completion(request(vec![Message::user("hello")]));
        futures::pin_mut!(queued);
        assert!(futures::poll!(&mut queued).is_pending());

        drop(stream);
        assert!(Arc::clone(&concurrency).try_acquire_owned().is_err());
        control.release()?;

        let queued = queued.await?;
        assert_eq!(queued.raw_response.generated_tokens, 2);
        Ok(())
    }

    #[cfg(not(target_family = "wasm"))]
    #[tokio::test(flavor = "current_thread")]
    async fn public_stream_cancel_stops_worker_without_dropping_response()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let (model, control, concurrency) = controlled_model(true, false, 2)?;
        let mut stream = model.stream(request(vec![Message::user("hello")])).await?;
        control.wait_until_entered().await;

        stream.cancel();
        assert!(stream.next().await.is_none());
        let queued = model.completion(request(vec![Message::user("hello")]));
        futures::pin_mut!(queued);
        assert!(futures::poll!(&mut queued).is_pending());
        assert!(Arc::clone(&concurrency).try_acquire_owned().is_err());

        control.release()?;
        let queued = queued.await?;
        assert_eq!(queued.raw_response.generated_tokens, 2);
        assert!(stream.next().await.is_none());
        Ok(())
    }

    #[cfg(not(target_family = "wasm"))]
    #[tokio::test(flavor = "current_thread")]
    async fn streaming_channel_applies_bounded_backpressure()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let (model, control, concurrency) =
            controlled_model(false, false, (STREAM_CHANNEL_CAPACITY + 4) as u64)?;
        let stream = model.stream(request(vec![Message::user("hello")])).await?;
        control
            .wait_for_delivery_attempts(STREAM_CHANNEL_CAPACITY + 1)
            .await;
        assert_eq!(
            control.delivery_attempts.load(Ordering::Acquire),
            STREAM_CHANNEL_CAPACITY + 1
        );

        drop(stream);
        let permit = Arc::clone(&concurrency).acquire_owned().await?;
        drop(permit);
        Ok(())
    }

    #[cfg(not(target_family = "wasm"))]
    #[tokio::test(flavor = "current_thread")]
    async fn blocking_task_panic_maps_to_typed_completion_error()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let (model, _, _) = controlled_model(false, true, 1)?;
        let error = model
            .completion(request(vec![Message::user("hello")]))
            .await
            .err()
            .ok_or("blocking task panic unexpectedly succeeded")?;
        assert!(error.to_string().contains("Candle blocking task failed"));

        let (model, _, _) = controlled_model(false, true, 1)?;
        let mut stream = model.stream(request(vec![Message::user("hello")])).await?;
        let error = stream
            .next()
            .await
            .ok_or("panicked streaming task produced no error")?
            .err()
            .ok_or("panicked streaming task unexpectedly produced content")?;
        assert!(error.to_string().contains("Candle blocking task failed"));
        Ok(())
    }

    #[test]
    fn builder_rejects_invalid_generation_defaults() {
        assert!(matches!(
            LlamaModel::builder(ModelData {
                config: Vec::new(),
                tokenizer: Vec::new(),
                weights: Vec::new(),
            })
            .max_tokens(0)
            .build(),
            Err(CandleError::InvalidGeneration(_))
        ));
        assert!(matches!(
            LlamaModel::builder(ModelData {
                config: Vec::new(),
                tokenizer: Vec::new(),
                weights: Vec::new(),
            })
            .temperature(f64::INFINITY)
            .build(),
            Err(CandleError::InvalidGeneration(_))
        ));
        assert!(matches!(
            LlamaModel::builder(ModelData {
                config: Vec::new(),
                tokenizer: Vec::new(),
                weights: Vec::new(),
            })
            .top_p(Some(0.0))
            .build(),
            Err(CandleError::InvalidGeneration(_))
        ));
        assert!(matches!(
            LlamaModel::builder(ModelData {
                config: Vec::new(),
                tokenizer: Vec::new(),
                weights: Vec::new(),
            })
            .repeat_penalty(0.0)
            .build(),
            Err(CandleError::InvalidGeneration(_))
        ));
    }

    #[test]
    fn renders_llama3_history_and_documents() -> Result<(), Box<dyn std::error::Error + Send + Sync>>
    {
        let history_request = request(vec![
            Message::system("rules"),
            Message::user("question"),
            Message::assistant("answer"),
            Message::user("follow-up"),
        ]);
        assert_eq!(
            render_prompt(&history_request)?,
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nrules<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nquestion<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nanswer<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nfollow-up<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );

        let mut request = request(vec![Message::system("rules"), Message::user("question")]);
        request.documents.push(Document {
            id: "doc-1".to_string(),
            text: "context".to_string(),
            additional_props: HashMap::new(),
        });
        let rendered = render_prompt(&request)?;
        assert!(rendered.contains("<file id: doc-1>\ncontext\n</file>"));
        assert!(rendered.find("<file id: doc-1>") < rendered.find("question"));
        Ok(())
    }

    #[test]
    fn renders_smollm2_history_default_system_and_generation_suffix()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let with_system = request(vec![
            Message::system("rules"),
            Message::user("question"),
            Message::assistant("answer"),
            Message::user("follow-up"),
        ]);
        assert_eq!(
            render_prompt_for(&with_system, ModelFamily::SmolLm2)?,
            "<|im_start|>system\nrules<|im_end|>\n<|im_start|>user\nquestion<|im_end|>\n<|im_start|>assistant\nanswer<|im_end|>\n<|im_start|>user\nfollow-up<|im_end|>\n<|im_start|>assistant\n"
        );

        let without_system = request(vec![Message::user("hello")]);
        assert_eq!(
            render_prompt_for(&without_system, ModelFamily::SmolLm2)?,
            "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
        );
        Ok(())
    }

    #[test]
    fn rejects_unsupported_request_features() -> Result<(), Box<dyn std::error::Error + Send + Sync>>
    {
        let mut tools = request(vec![Message::user("hello")]);
        tools.tools.push(ToolDefinition {
            name: "tool".to_string(),
            description: "tool".to_string(),
            parameters: serde_json::json!({}),
        });
        assert!(
            matches!(render_prompt(&tools), Err(CandleError::UnsupportedFeature(feature)) if feature.contains("Qwen3"))
        );

        let mut choice = request(vec![Message::user("hello")]);
        choice.tool_choice = Some(ToolChoice::Auto);
        assert!(render_prompt(&choice).is_err());

        let mut schema = request(vec![Message::user("hello")]);
        schema.output_schema = Some(serde_json::from_value(
            serde_json::json!({"type": "string"}),
        )?);
        assert!(render_prompt(&schema).is_err());

        let mut override_request = request(vec![Message::user("hello")]);
        override_request.model = Some("other".to_string());
        assert!(render_prompt(&override_request).is_err());

        let tool_result = request(vec![Message::tool_result("id", "result")]);
        assert!(render_prompt(&tool_result).is_err());

        let image = Message::User {
            content: OneOrMany::one(UserContent::image_base64(
                "data",
                Some(ImageMediaType::PNG),
                Some(ImageDetail::Auto),
            )),
        };
        assert!(render_prompt(&request(vec![image])).is_err());

        let audio = Message::User {
            content: OneOrMany::one(UserContent::audio("data", Some(AudioMediaType::WAV))),
        };
        assert!(render_prompt(&request(vec![audio])).is_err());
        Ok(())
    }

    #[test]
    fn request_generation_overrides_defaults_and_validates() -> Result<(), CandleError> {
        let defaults = GenerationConfig::default();
        let mut request = request(vec![Message::user("hello")]);
        request.max_tokens = Some(12);
        request.temperature = Some(0.0);
        request.additional_params = Some(serde_json::json!({
            "top_k": 4,
            "top_p": 0.7,
            "seed": 7,
            "repeat_penalty": 1.2,
            "repeat_last_n": 9
        }));
        let effective = effective_generation(&request, &defaults, 8)?;
        assert_eq!(effective.max_tokens, 12);
        assert_eq!(effective.temperature, 0.0);
        assert_eq!(effective.top_k, Some(4));
        assert_eq!(effective.seed, 7);

        request.additional_params = Some(serde_json::json!({"unknown": true}));
        assert!(effective_generation(&request, &defaults, 8).is_err());
        request.additional_params = Some(serde_json::json!({"top_k": "four"}));
        assert!(effective_generation(&request, &defaults, 8).is_err());
        request.additional_params = None;
        request.max_tokens = Some(0);
        assert!(effective_generation(&request, &defaults, 8).is_err());
        request.max_tokens = Some(1);
        request.temperature = Some(f64::NAN);
        assert!(effective_generation(&request, &defaults, 8).is_err());
        Ok(())
    }

    #[test]
    fn converts_finish_reason_and_usage() -> Result<(), CandleError> {
        let response = CandleCompletionResponse {
            text: "done".to_string(),
            prompt_tokens: 5,
            generated_tokens: 2,
            requested_max_tokens: 4,
            effective_max_tokens: 3,
            finish_reason: FinishReason::Eos,
            prefill_duration_ms: 8,
            time_to_first_token_ms: Some(10),
            generation_duration_ms: 20,
            tokens_per_second: Some(100.0),
        };
        let usage = response.token_usage();
        assert_eq!(usage.input_tokens, 5);
        assert_eq!(usage.output_tokens, 2);
        assert_eq!(usage.total_tokens, 7);
        assert_eq!(response.finish_reason, FinishReason::Eos);
        assert_eq!(response.text, "done");
        assert_eq!(response.requested_max_tokens, 4);
        assert_eq!(response.effective_max_tokens, 3);
        assert_eq!(response.prefill_duration_ms, 8);
        assert_eq!(response.time_to_first_token_ms, Some(10));
        assert_eq!(response.generation_duration_ms, 20);
        assert_eq!(response.tokens_per_second, Some(100.0));
        Ok(())
    }

    #[test]
    fn unsupported_make_fails_for_buffered_and_streaming()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        runtime.block_on(async {
            let model = <LlamaModel as CompletionModel>::make(&(), "llama");
            let completion_error = model
                .completion(request(vec![Message::user("hello")]))
                .await
                .err()
                .ok_or("expected unsupported make")?;
            assert!(
                completion_error
                    .to_string()
                    .contains("CompletionModel::make")
            );
            let stream_error = model
                .stream(request(vec![Message::user("hello")]))
                .await
                .err()
                .ok_or("expected unsupported make")?;
            assert!(stream_error.to_string().contains("CompletionModel::make"));
            Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
        })?;
        Ok(())
    }
}
