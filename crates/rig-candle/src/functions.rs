//! Local Candle inference as config + free functions (data-oriented face).
//!
//! Candle is not a network transport at all: "building the client" means
//! loading model tensors from caller-supplied byte buffers. The serde
//! [`Config`] therefore describes *how* to load and generate (conversation
//! protocol, generation defaults, concurrency) as plain data; the artifact
//! bytes themselves are passed separately to [`model_from_config`], keeping
//! the crate's no-filesystem/no-network invariant. Free
//! [`complete`]/[`open_stream`] functions run over the loaded
//! [`CandleModel`] handle; the existing `CompletionModel` trait impl is
//! rewired through the same extracted implementations.

use rig_core::completion::{CompletionError, CompletionRequest, CompletionResponse};
use rig_core::providers::descriptor::ProviderDescriptor;
use rig_core::streaming::StreamingCompletionResponse;
use serde::{Deserialize, Serialize};

use crate::artifacts::ModelArtifacts;
use crate::generation::GenerationConfig;
use crate::model::{CandleModel, CandleModelBuilder, DEFAULT_MAX_CONCURRENT_REQUESTS};
use crate::profile::ConversationProtocol;
use crate::types::CandleError;

/// Candle's capability sheet.
///
/// `supports_tools` is true because the validated Qwen3 protocol supports
/// Rig function definitions and all portable tool-choice modes (Llama 3 and
/// SmolLM2 protocols reject tools per-request). `supports_response_format`
/// is false because decoding is not grammar constrained and
/// `output_schema` is rejected. There is no embeddings API.
pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor::named("candle")
    .with_tools(true);

/// Plain-data description of how to load a [`CandleModel`] and what
/// generation defaults to use.
///
/// Mirrors the [`crate::CandleModelBuilder`] inputs: an optional
/// conversation protocol pin, the [`GenerationConfig`] fields, and the
/// native concurrency limit. Artifact bytes are deliberately not part of
/// the config (this crate performs no filesystem or network access); pass
/// them to [`model_from_config`] alongside the config.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Config {
    /// Required conversation protocol; `None` autodetects from artifacts.
    pub protocol: Option<ConversationProtocol>,
    /// Default maximum generated token count.
    pub max_tokens: u64,
    /// Default sampling temperature. Zero selects greedy decoding.
    pub temperature: f64,
    /// Optional top-k sampling limit.
    pub top_k: Option<usize>,
    /// Optional nucleus-sampling threshold.
    pub top_p: Option<f64>,
    /// Deterministic sampling seed.
    pub seed: u64,
    /// Repeat penalty. `1.0` disables it.
    pub repeat_penalty: f32,
    /// Number of recent tokens considered by the repeat penalty.
    pub repeat_last_n: usize,
    /// Maximum native inference requests admitted concurrently.
    pub max_concurrent_requests: usize,
}

impl Default for Config {
    fn default() -> Self {
        let generation = GenerationConfig::default();
        Self {
            protocol: None,
            max_tokens: generation.max_tokens,
            temperature: generation.temperature,
            top_k: generation.top_k,
            top_p: generation.top_p,
            seed: generation.seed,
            repeat_penalty: generation.repeat_penalty,
            repeat_last_n: generation.repeat_last_n,
            max_concurrent_requests: DEFAULT_MAX_CONCURRENT_REQUESTS,
        }
    }
}

fn configured_builder(cfg: &Config, artifacts: ModelArtifacts) -> CandleModelBuilder<'static> {
    let mut builder = CandleModel::builder_from_artifacts(artifacts)
        .max_tokens(cfg.max_tokens)
        .temperature(cfg.temperature)
        .top_k(cfg.top_k)
        .top_p(cfg.top_p)
        .seed(cfg.seed)
        .repeat_penalty(cfg.repeat_penalty)
        .repeat_last_n(cfg.repeat_last_n)
        .max_concurrent_requests(cfg.max_concurrent_requests);
    if let Some(protocol) = cfg.protocol {
        builder = builder.conversation_protocol(protocol);
    }
    builder
}

/// Validate `artifacts` and load model tensors onto the CPU as described by
/// `cfg`. Synchronous because loading is synchronous; see
/// [`model_from_config_async`] for the native off-executor variant.
pub fn model_from_config(
    cfg: &Config,
    artifacts: ModelArtifacts,
) -> Result<CandleModel, CandleError> {
    configured_builder(cfg, artifacts).build()
}

/// [`model_from_config`] on Tokio's blocking thread pool (native targets).
#[cfg(not(target_family = "wasm"))]
pub async fn model_from_config_async(
    cfg: &Config,
    artifacts: ModelArtifacts,
) -> Result<CandleModel, CandleError> {
    configured_builder(cfg, artifacts).build_async().await
}

/// Run a buffered completion on the loaded `model`.
///
/// Extracted from the `CompletionModel::completion` trait impl, which is
/// rewired through the same implementation; behavior is unchanged.
pub async fn complete(
    model: &CandleModel,
    request: CompletionRequest,
) -> Result<CompletionResponse, CompletionError> {
    crate::model::complete_request(model, request).await
}

/// Open a streaming completion on the loaded `model`, returning the
/// concrete [`StreamingCompletionResponse`].
///
/// Extracted from the `CompletionModel::stream` trait impl, which is
/// rewired through the same implementation; behavior is unchanged.
pub async fn open_stream(
    model: &CandleModel,
    request: CompletionRequest,
) -> Result<StreamingCompletionResponse, CompletionError> {
    crate::model::open_stream_request(model, request).await
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::artifacts::ModelData;

    #[test]
    fn config_default_mirrors_generation_defaults() {
        let cfg = Config::default();
        let generation = GenerationConfig::default();
        assert_eq!(cfg.max_tokens, generation.max_tokens);
        assert_eq!(cfg.temperature, generation.temperature);
        assert_eq!(cfg.top_k, generation.top_k);
        assert_eq!(cfg.top_p, generation.top_p);
        assert_eq!(cfg.seed, generation.seed);
        assert_eq!(cfg.repeat_penalty, generation.repeat_penalty);
        assert_eq!(cfg.repeat_last_n, generation.repeat_last_n);
        assert_eq!(cfg.max_concurrent_requests, 1);
        assert_eq!(cfg.protocol, None);
    }

    #[test]
    fn config_serde_round_trip() {
        let cfg = Config {
            protocol: Some(ConversationProtocol::Qwen3),
            max_tokens: 128,
            seed: 7,
            ..Config::default()
        };
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: Config = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, cfg);
    }

    #[test]
    fn model_from_config_rejects_zero_concurrency_before_loading() {
        let cfg = Config {
            max_concurrent_requests: 0,
            ..Config::default()
        };
        let artifacts = ModelArtifacts::Safetensors(ModelData {
            config: Vec::new(),
            tokenizer: Vec::new(),
            weights: Vec::new(),
        });
        // The concurrency check runs before any artifact validation, so even
        // empty buffers surface the concurrency error first.
        assert!(matches!(
            model_from_config(&cfg, artifacts),
            Err(CandleError::InvalidConcurrencyLimit)
        ));
    }

    #[test]
    fn model_from_config_validates_generation_settings() {
        let cfg = Config {
            temperature: -1.0,
            ..Config::default()
        };
        let artifacts = ModelArtifacts::Safetensors(ModelData {
            config: Vec::new(),
            tokenizer: Vec::new(),
            weights: Vec::new(),
        });
        assert!(matches!(
            model_from_config(&cfg, artifacts),
            Err(CandleError::InvalidGeneration(_))
        ));
    }
}
