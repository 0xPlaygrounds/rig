//! Public errors and response metadata.

use rig_core::completion::{CompletionError, Usage};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::profile::ModelFamily;

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
    /// Caller-controlled content contains a delimiter reserved by the selected chat template.
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
        /// Shape required by Candle's implementation.
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

/// Stable descriptor name reported on normalized responses from this crate.
pub const PROVIDER_NAME: &str = "candle";

impl From<&CandleCompletionResponse> for Usage {
    fn from(response: &CandleCompletionResponse) -> Self {
        Usage {
            input_tokens: response.prompt_tokens,
            output_tokens: response.generated_tokens,
            total_tokens: response
                .prompt_tokens
                .saturating_add(response.generated_tokens),
            ..Usage::new()
        }
    }
}

impl From<FinishReason> for rig_core::completion::FinishReason {
    fn from(reason: FinishReason) -> Self {
        match reason {
            // A sampled EOS token is the local equivalent of a natural stop;
            // `reconcile_with_output` promotes it to `ToolCalls` when the
            // protocol parsed tool calls out of the generated text.
            FinishReason::Eos => rig_core::completion::FinishReason::Stop,
            FinishReason::MaxTokens => rig_core::completion::FinishReason::Length,
        }
    }
}
