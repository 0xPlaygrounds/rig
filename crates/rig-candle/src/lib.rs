//! Local, CPU-only Llama-family inference for Rig, backed by Candle.
//!
//! Models are loaded entirely from owned byte buffers. This crate performs no
//! filesystem or network access. On `wasm32-unknown-unknown`, inference runs
//! synchronously inside the completion future; browser applications should own
//! and invoke the model in a Web Worker to avoid blocking the UI thread.
//!
//! ```no_run
//! use rig_core::{agent::AgentBuilder, completion::Prompt};
//! use rig_candle::{LlamaModel, ModelData};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let data = ModelData {
//!     config: std::fs::read("./model/config.json")?,
//!     tokenizer: std::fs::read("./model/tokenizer.json")?,
//!     weights: std::fs::read("./model/model.safetensors")?,
//! };
//! let model = LlamaModel::from_safetensors(data)?;
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
//! The crate accepts one unsharded, non-quantized Hugging Face safetensors
//! checkpoint and a Llama 3 instruct tokenizer. Loading validates the complete
//! tensor layout, supported floating-point dtypes, model dimensions, vocabulary,
//! configured token IDs, and Llama 3 formatting tokens before Candle is invoked.
//!
//! Request `max_tokens` and `temperature` override builder defaults. The
//! Candle-specific `additional_params` keys are `top_k`, `top_p`, `seed`,
//! `repeat_penalty`, and `repeat_last_n`; unknown keys are rejected. Output is
//! clamped to the context capacity remaining after tokenizing the prompt.
//!
//! Native inference is admitted asynchronously and runs in `spawn_blocking`.
//! [`LlamaModelBuilder::max_concurrent_requests`] defaults to one to control CPU
//! and KV-cache memory pressure. Dropping a native completion future signals
//! cooperative cancellation; a forward operation already in progress cannot be
//! interrupted, so cancellation is observed at the next generation boundary.
//! WASM does not use native synchronization or threads.
//!
//! Tools, structured output, multimodal content, streaming, quantized formats,
//! accelerators, shards, tokenizer chat templates, and downloads are unsupported.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(not(target_family = "wasm"))]
use std::sync::atomic::{AtomicBool, Ordering};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::llama::{
    Cache, Config, Llama, Llama3RopeType, LlamaConfig, LlamaEosToks,
};
use candle_transformers::utils::apply_repeat_penalty;
use rig_core::OneOrMany;
use rig_core::completion::{
    AssistantContent, CompletionError, CompletionModel, CompletionRequest, CompletionResponse,
    Usage,
};
use rig_core::message::{Message, UserContent};
use rig_core::streaming::StreamingCompletionResponse;
use safetensors::{Dtype as SafeDtype, SafeTensors};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokenizers::Tokenizer;

const BEGIN_OF_TEXT: &str = "<|begin_of_text|>";
const START_HEADER: &str = "<|start_header_id|>";
const END_HEADER: &str = "<|end_header_id|>";
const END_OF_TURN: &str = "<|eot_id|>";
const DEFAULT_MAX_CONCURRENT_REQUESTS: usize = 1;

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

/// Owned Hugging Face model artifacts for exactly one unsharded checkpoint.
#[derive(Debug)]
pub struct ModelData {
    /// Contents of `config.json`.
    pub config: Vec<u8>,
    /// Contents of `tokenizer.json`.
    pub tokenizer: Vec<u8>,
    /// Contents of one `model.safetensors` file.
    pub weights: Vec<u8>,
}

/// Why a local Candle completion failed.
#[derive(Debug, Error, Clone)]
#[non_exhaustive]
pub enum CandleError {
    /// A required artifact buffer was empty.
    #[error("the {artifact} buffer is empty")]
    EmptyBuffer { artifact: &'static str },
    /// The Hugging Face configuration could not be parsed or is internally invalid.
    #[error("invalid Llama configuration: {0}")]
    Configuration(String),
    /// A parsed configuration field is incompatible with Candle's Llama implementation.
    #[error("invalid Llama configuration field `{field}`: {reason}")]
    InvalidConfigurationValue {
        /// Configuration field name.
        field: &'static str,
        /// Explanation of the invalid value or relationship.
        reason: String,
    },
    /// The tokenizer bytes could not be loaded.
    #[error("invalid tokenizer: {0}")]
    TokenizerLoading(String),
    /// The tokenizer vocabulary does not agree with the model configuration.
    #[error("tokenizer vocabulary size {actual} does not match config.vocab_size {expected}")]
    TokenizerVocabularyMismatch {
        /// Vocabulary size required by the model configuration.
        expected: usize,
        /// Vocabulary size reported by the tokenizer, including added tokens.
        actual: usize,
    },
    /// A Llama 3 prompt-format token is absent from the tokenizer.
    #[error("tokenizer is missing required Llama 3 special token `{token}`")]
    MissingSpecialToken {
        /// Required special-token string.
        token: &'static str,
    },
    /// A Llama 3 formatting token exists but is not registered as special.
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
    /// A tensor required by the configured Llama architecture was absent.
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
    #[error("unable to determine a valid EOS or Llama 3 end-of-turn token")]
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
    /// The request uses a feature outside the text-only MVP.
    #[error("unsupported Candle request feature: {0}")]
    UnsupportedFeature(String),
    /// `CompletionModel::make` cannot load a byte-backed model.
    #[error(
        "`CompletionModel::make` is unsupported for rig-candle; use `LlamaModel::from_safetensors` or `LlamaModel::builder`"
    )]
    UnsupportedMake,
    /// A native blocking inference task could not be joined.
    #[cfg(not(target_family = "wasm"))]
    #[error("Candle blocking task failed: {0}")]
    BlockingTaskJoin(String),
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
    /// A configured EOS or Llama 3 end-of-turn token was sampled.
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
    /// Time spent in model prefill and token generation, in milliseconds.
    pub generation_duration_ms: u64,
    /// Generated tokens per second when the measured duration is nonzero.
    pub tokens_per_second: Option<f64>,
}

#[derive(Clone)]
enum ModelState {
    Ready(Arc<LoadedModel>),
    UnsupportedMake,
}

struct LoadedModel {
    model: Llama,
    tokenizer: Tokenizer,
    config: Config,
    stop_tokens: HashSet<u32>,
    generation: GenerationConfig,
    #[cfg(not(target_family = "wasm"))]
    concurrency: Arc<tokio::sync::Semaphore>,
}

/// A cheaply cloneable, CPU-only Llama completion model.
#[derive(Clone)]
pub struct LlamaModel {
    state: ModelState,
}

/// Builder for loading a [`LlamaModel`] and customizing generation defaults.
pub struct LlamaModelBuilder {
    data: ModelData,
    generation: GenerationConfig,
    max_concurrent_requests: usize,
}

impl LlamaModel {
    /// Loads a model from config, tokenizer, and one unsharded safetensors buffer.
    pub fn from_safetensors(data: ModelData) -> Result<Self, CandleError> {
        Self::builder(data).build()
    }

    /// Starts a byte-backed model builder.
    pub fn builder(data: ModelData) -> LlamaModelBuilder {
        LlamaModelBuilder {
            data,
            generation: GenerationConfig::default(),
            max_concurrent_requests: DEFAULT_MAX_CONCURRENT_REQUESTS,
        }
    }
}

impl LlamaModelBuilder {
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
    pub fn build(self) -> Result<LlamaModel, CandleError> {
        validate_generation(&self.generation, None)?;
        if self.max_concurrent_requests == 0 {
            return Err(CandleError::InvalidConcurrencyLimit);
        }
        let loaded = load_model(self.data, self.generation, self.max_concurrent_requests)?;
        Ok(LlamaModel {
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

fn load_model(
    data: ModelData,
    generation: GenerationConfig,
    _max_concurrent_requests: usize,
) -> Result<LoadedModel, CandleError> {
    require_nonempty(&data.config, "config")?;
    require_nonempty(&data.tokenizer, "tokenizer")?;
    require_nonempty(&data.weights, "weights")?;

    let llama_config: LlamaConfig = serde_json::from_slice(&data.config)
        .map_err(|error| CandleError::Configuration(error.to_string()))?;
    let config = llama_config.into_config(false);
    validate_model_config(&config)?;

    let tokenizer = Tokenizer::from_bytes(&data.tokenizer)
        .map_err(|error| CandleError::TokenizerLoading(error.to_string()))?;
    validate_tokenizer(&config, &tokenizer)?;
    let stop_tokens = resolve_stop_tokens(&config, &tokenizer)?;
    validate_checkpoint(&data.weights, &config)?;

    let device = Device::Cpu;
    let builder = VarBuilder::from_buffered_safetensors(data.weights, DType::F32, &device)
        .map_err(|error| CandleError::InvalidCheckpoint(error.to_string()))?;
    let model = Llama::load(builder, &config)
        .map_err(|error| CandleError::ModelLoading(error.to_string()))?;

    Ok(LoadedModel {
        model,
        tokenizer,
        config,
        stop_tokens,
        generation,
        #[cfg(not(target_family = "wasm"))]
        concurrency: Arc::new(tokio::sync::Semaphore::new(_max_concurrent_requests)),
    })
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

fn validate_positive_finite(field: &'static str, value: f32) -> Result<(), CandleError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(CandleError::InvalidConfigurationValue {
            field,
            reason: format!("value {value} must be finite and greater than zero"),
        });
    }
    Ok(())
}

fn validate_tokenizer(config: &Config, tokenizer: &Tokenizer) -> Result<(), CandleError> {
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
    for token in [BEGIN_OF_TEXT, START_HEADER, END_HEADER, END_OF_TURN] {
        let id = tokenizer
            .token_to_id(token)
            .ok_or(CandleError::MissingSpecialToken { token })?;
        validate_token_id(token, id, config.vocab_size)?;
        if !tokenizer.get_added_vocabulary().is_special_token(token) {
            return Err(CandleError::SpecialTokenNotMarked { token });
        }
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
    if let Some(token) = tokenizer.token_to_id(END_OF_TURN) {
        tokens.insert(token);
    }
    if tokens.is_empty() {
        Err(CandleError::MissingStopToken)
    } else {
        Ok(tokens)
    }
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

fn validate_request(request: &CompletionRequest) -> Result<(), CandleError> {
    if !request.tools.is_empty() {
        return Err(CandleError::UnsupportedFeature("tools".to_string()));
    }
    if request.tool_choice.is_some() {
        return Err(CandleError::UnsupportedFeature("tool_choice".to_string()));
    }
    if request.output_schema.is_some() {
        return Err(CandleError::UnsupportedFeature("output_schema".to_string()));
    }
    if let Some(model) = &request.model {
        return Err(CandleError::UnsupportedFeature(format!(
            "model override `{model}`; byte-loaded models do not support request-time model selection"
        )));
    }
    Ok(())
}

fn render_prompt(request: &CompletionRequest) -> Result<String, CandleError> {
    validate_request(request)?;
    let mut messages = Vec::new();
    if let Some(preamble) = &request.preamble {
        messages.push(Message::system(preamble.clone()));
    }
    messages.extend(request.chat_history.iter().cloned());
    if !request.documents.is_empty() {
        let context = request
            .documents
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        let insertion = match messages
            .iter()
            .position(|message| !matches!(message, Message::System { .. }))
        {
            Some(index) => index,
            None => messages.len(),
        };
        messages.insert(insertion, Message::user(context));
    }

    let mut rendered = String::from(BEGIN_OF_TEXT);
    for message in messages {
        let (role, content) = render_message(&message)?;
        rendered.push_str(START_HEADER);
        rendered.push_str(role);
        rendered.push_str(END_HEADER);
        rendered.push_str("\n\n");
        rendered.push_str(&content);
        rendered.push_str(END_OF_TURN);
    }
    rendered.push_str(START_HEADER);
    rendered.push_str("assistant");
    rendered.push_str(END_HEADER);
    rendered.push_str("\n\n");
    Ok(rendered)
}

fn render_message(message: &Message) -> Result<(&'static str, String), CandleError> {
    match message {
        Message::System { content } => Ok(("system", content.clone())),
        Message::User { content } => {
            let mut parts = Vec::new();
            for item in content.iter() {
                match item {
                    UserContent::Text(text) => parts.push(text.text.clone()),
                    UserContent::ToolResult(_) => {
                        return Err(CandleError::UnsupportedFeature("tool results".to_string()));
                    }
                    UserContent::Image(_) => {
                        return Err(CandleError::UnsupportedFeature("image content".to_string()));
                    }
                    UserContent::Audio(_) => {
                        return Err(CandleError::UnsupportedFeature("audio content".to_string()));
                    }
                    UserContent::Video(_) => {
                        return Err(CandleError::UnsupportedFeature("video content".to_string()));
                    }
                    UserContent::Document(_) => {
                        return Err(CandleError::UnsupportedFeature(
                            "message document content".to_string(),
                        ));
                    }
                }
            }
            Ok(("user", parts.join("\n")))
        }
        Message::Assistant { content, .. } => {
            let mut parts = Vec::new();
            for item in content.iter() {
                match item {
                    AssistantContent::Text(text) => parts.push(text.text.clone()),
                    AssistantContent::ToolCall(_) => {
                        return Err(CandleError::UnsupportedFeature("tool calls".to_string()));
                    }
                    AssistantContent::Reasoning(_) => {
                        return Err(CandleError::UnsupportedFeature(
                            "structured reasoning".to_string(),
                        ));
                    }
                    AssistantContent::Image(_) => {
                        return Err(CandleError::UnsupportedFeature("image content".to_string()));
                    }
                }
            }
            Ok(("assistant", parts.join("\n")))
        }
    }
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

fn infer(
    loaded: &LoadedModel,
    request: CompletionRequest,
    cancellation: &CancellationSignal,
) -> Result<CompletionResponse<CandleCompletionResponse>, CandleError> {
    let prompt = render_prompt(&request)?;
    let generation = effective_generation(&request, &loaded.generation, loaded.config.vocab_size)?;
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
        loaded.config.max_position_embeddings,
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
    let mut cache = Cache::new(true, DType::F32, &loaded.config, &device)
        .map_err(|error| CandleError::Inference(error.to_string()))?;
    check_cancellation(cancellation)?;
    let mut logits = loaded
        .model
        .forward(&input, 0, &mut cache)
        .and_then(|tensor| tensor.squeeze(0))
        .map_err(|error| CandleError::Inference(error.to_string()))?;
    let mut processor = LogitsProcessor::from_sampling(generation.seed, sampling(&generation));
    let mut all_tokens = prompt_ids.to_vec();
    let mut decoded_tokens = Vec::new();
    let mut generated_count = 0usize;
    let mut finish_reason = FinishReason::MaxTokens;

    for index in 0..max_tokens {
        check_cancellation(cancellation)?;
        if generation.repeat_penalty != 1.0 && generation.repeat_last_n > 0 {
            let recent = recent_tokens(&all_tokens, generation.repeat_last_n);
            logits = apply_repeat_penalty(&logits, generation.repeat_penalty, recent)
                .map_err(|error| CandleError::Inference(error.to_string()))?;
        }
        let token = processor
            .sample(&logits)
            .map_err(|error| CandleError::Inference(error.to_string()))?;
        generated_count += 1;
        if loaded.stop_tokens.contains(&token) {
            finish_reason = FinishReason::Eos;
            break;
        }
        decoded_tokens.push(token);
        all_tokens.push(token);
        if index + 1 < max_tokens {
            check_cancellation(cancellation)?;
            let position = next_cache_position(prompt_ids.len(), index)?;
            let next = Tensor::new(&[token], &device)
                .and_then(|tensor| tensor.unsqueeze(0))
                .map_err(|error| CandleError::Inference(error.to_string()))?;
            logits = loaded
                .model
                .forward(&next, position, &mut cache)
                .and_then(|tensor| tensor.squeeze(0))
                .map_err(|error| CandleError::Inference(error.to_string()))?;
        }
    }

    let text = loaded
        .tokenizer
        .decode(&decoded_tokens, true)
        .map_err(|error| CandleError::TokenizerDecoding(error.to_string()))?;
    completion_response(
        text,
        prompt_ids.len(),
        generated_count,
        generation.max_tokens,
        effective_max_tokens,
        finish_reason,
        started.elapsed(),
    )
}

fn completion_response(
    text: String,
    prompt_tokens: usize,
    generated_tokens: usize,
    requested_max_tokens: u64,
    effective_max_tokens: u64,
    finish_reason: FinishReason,
    generation_duration: Duration,
) -> Result<CompletionResponse<CandleCompletionResponse>, CandleError> {
    let prompt_tokens = u64::try_from(prompt_tokens).map_err(|_| {
        CandleError::Inference("prompt token count does not fit in u64".to_string())
    })?;
    let generated_tokens = u64::try_from(generated_tokens).map_err(|_| {
        CandleError::Inference("generated token count does not fit in u64".to_string())
    })?;
    let generation_duration_ms =
        u64::try_from(generation_duration.as_millis()).map_or(u64::MAX, |value| value);
    let tokens_per_second = if generation_duration.is_zero() {
        None
    } else {
        Some(generated_tokens as f64 / generation_duration.as_secs_f64())
    };
    let raw_response = CandleCompletionResponse {
        text: text.clone(),
        prompt_tokens,
        generated_tokens,
        requested_max_tokens,
        effective_max_tokens,
        finish_reason,
        generation_duration_ms,
        tokens_per_second,
    };
    Ok(CompletionResponse {
        choice: OneOrMany::one(AssistantContent::text(text)),
        usage: Usage {
            input_tokens: prompt_tokens,
            output_tokens: generated_tokens,
            total_tokens: prompt_tokens + generated_tokens,
            ..Usage::new()
        },
        raw_response,
        message_id: None,
    })
}

impl CompletionModel for LlamaModel {
    type Response = CandleCompletionResponse;
    type StreamingResponse = ();
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
        _request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        Err(CandleError::UnsupportedFeature("streaming".to_string()).into())
    }
}

#[cfg(test)]
#[allow(clippy::panic_in_result_fn)]
mod tests {
    use std::borrow::Cow;
    use std::collections::HashMap;

    use rig_core::completion::{CompletionModel, Document, ToolDefinition};
    use rig_core::message::{AudioMediaType, ImageDetail, ImageMediaType, ToolChoice};
    use safetensors::tensor::{Dtype, View, serialize};
    use tokenizers::AddedToken;
    use tokenizers::models::wordlevel::WordLevel;

    use super::*;

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
            validate_tokenizer(&config, &tokenizer),
            Err(CandleError::MissingSpecialToken { token: END_HEADER })
        ));
        let tokenizer = Tokenizer::from_bytes(tiny_tokenizer_with_end_header(END_HEADER, false)?)?;
        assert!(matches!(
            validate_tokenizer(&config, &tokenizer),
            Err(CandleError::SpecialTokenNotMarked { token: END_HEADER })
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
        Ok(())
    }

    #[test]
    fn loaded_model_works_with_agent_builder()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use rig_core::agent::AgentBuilder;
        use rig_core::completion::Prompt;

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

    #[test]
    fn inference_clamps_context_and_uses_fresh_generation_state()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut loaded = load_model(model_data()?, GenerationConfig::default(), 1)?;
        let mut completion_request = request(vec![Message::user("hello")]);
        completion_request.max_tokens = Some(10);
        completion_request.temperature = Some(0.0);
        let prompt = render_prompt(&completion_request)?;
        let prompt_tokens = loaded.tokenizer.encode(prompt, false)?.len();
        loaded.config.max_position_embeddings = prompt_tokens + 2;

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
        let loaded = load_model(
            ModelData {
                config: config_with("eos_token_id", 0.into())?,
                tokenizer: tiny_tokenizer()?,
                weights: checkpoint(true)?,
            },
            GenerationConfig::default(),
            1,
        )?;
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

        let semaphore = Arc::new(tokio::sync::Semaphore::new(1));
        semaphore.close();
        assert!(matches!(
            acquire_concurrency(semaphore).await,
            Err(CandleError::ConcurrencyControllerClosed)
        ));
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
    fn rejects_unsupported_request_features() -> Result<(), Box<dyn std::error::Error + Send + Sync>>
    {
        let mut tools = request(vec![Message::user("hello")]);
        tools.tools.push(ToolDefinition {
            name: "tool".to_string(),
            description: "tool".to_string(),
            parameters: serde_json::json!({}),
        });
        assert!(
            matches!(render_prompt(&tools), Err(CandleError::UnsupportedFeature(feature)) if feature == "tools")
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
        let response = completion_response(
            "done".to_string(),
            5,
            2,
            4,
            3,
            FinishReason::Eos,
            Duration::from_millis(20),
        )?;
        assert_eq!(response.usage.input_tokens, 5);
        assert_eq!(response.usage.output_tokens, 2);
        assert_eq!(response.usage.total_tokens, 7);
        assert_eq!(response.raw_response.finish_reason, FinishReason::Eos);
        assert_eq!(response.raw_response.text, "done");
        assert_eq!(response.raw_response.requested_max_tokens, 4);
        assert_eq!(response.raw_response.effective_max_tokens, 3);
        assert_eq!(response.raw_response.generation_duration_ms, 20);
        assert_eq!(response.raw_response.tokens_per_second, Some(100.0));
        Ok(())
    }

    #[test]
    fn unsupported_make_and_stream_fail_explicitly()
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
                .ok_or("expected unsupported streaming")?;
            assert!(stream_error.to_string().contains("streaming"));
            Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
        })?;
        Ok(())
    }
}
