//! The closed, exhaustively-matched provider set.
//!
//! [`ProviderConfig`] is one serde value per bundled provider — plain
//! configuration, never a live handle — and [`complete`]/[`open_stream`]
//! fulfil a [`CompletionRequest`] for any arm with an exhaustive `match`.
//! Adding a provider fails to compile until every fulfilment site handles
//! it: that is the feature, which is also why the enum is deliberately
//! **not** `#[non_exhaustive]` — external hosts matching provider configs
//! get the same compile-time guarantee. Cargo features never change that
//! vocabulary: Bedrock and Gemini gRPC keep lightweight configs in
//! `rig-core`, and missing transport features fail only when fulfillment is
//! attempted.
//!
//! The deterministic [`ProviderConfig::Mock`] and [`EmbedderConfig::Mock`]
//! variants are part of that stable production-visible serde vocabulary.
//! Hosts that accept serialized provider configuration across an untrusted
//! boundary must validate or allowlist variants and fields before fulfillment;
//! a [`MockScript`] can deliberately leave selected operations pending until
//! their futures are cancelled.
//!
//! Live transports live in [`Runtime`], not in configs: a serialized
//! `ProviderConfig` resumes anywhere, and handles (HTTP client, AWS client,
//! gRPC channel) are rebuilt on first use per process.
//!
//! Out-of-tree providers cannot add arms; they drive the public
//! [`AgentRun`](crate::agent::run::AgentRun) +
//! [`prepare_request`](crate::agent::prepare::prepare_request) protocol
//! directly instead.
//!
//! [`PROVIDER_SURFACES`] and both config enums are generated from one
//! registry containing descriptors, completion/embedding fulfillment paths,
//! and feature requirements. FastEmbed is deliberately absent because loaded
//! local weights are runtime state, not resumable serde configuration.

use rig_core::completion::{CompletionError, CompletionRequest, CompletionResponse};
use rig_core::embeddings::EmbeddingError;
use rig_core::http_runtime::HttpRuntime;
use rig_core::providers::descriptor::ProviderDescriptor;
use rig_core::streaming::CompletionStream;

/// A compile-time description of one bundled provider surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderSurface {
    /// Public enum variant name.
    pub variant: &'static str,
    /// Provider capability sheet.
    pub descriptor: &'static ProviderDescriptor,
    /// Whether the provider fulfills completion requests.
    pub completion: bool,
    /// Whether the provider fulfills embedding requests.
    pub embedding: bool,
    /// Cargo feature required to fulfill non-HTTP operations, if any.
    pub fulfillment_feature: Option<&'static str>,
}

#[cfg(not(all(feature = "bedrock", feature = "gemini-grpc")))]
#[derive(Debug, thiserror::Error)]
#[error("provider `{provider}` fulfillment requires the rig-agent `{feature}` Cargo feature")]
struct ProviderFeatureUnavailable {
    provider: &'static str,
    feature: &'static str,
}

#[cfg(not(all(feature = "bedrock", feature = "gemini-grpc")))]
fn completion_feature_unavailable(
    provider: &'static str,
    feature: &'static str,
) -> CompletionError {
    CompletionError::RequestError(Box::new(ProviderFeatureUnavailable { provider, feature }))
}

#[cfg(not(all(feature = "bedrock", feature = "gemini-grpc")))]
fn embedding_feature_unavailable(provider: &'static str, feature: &'static str) -> EmbeddingError {
    EmbeddingError::ProviderError(ProviderFeatureUnavailable { provider, feature }.to_string())
}

macro_rules! capability_present {
    () => {
        false
    };
    ($config:path) => {
        true
    };
}

macro_rules! config_model {
    (Http, $cfg:expr) => {
        &$cfg.model
    };
    (Bedrock, $cfg:expr) => {
        &$cfg.model
    };
    (GeminiGrpc, $cfg:expr) => {
        &$cfg.model
    };
    (Mock, $cfg:expr) => {
        "mock"
    };
}

macro_rules! complete_dispatch {
    (Http, $complete:path, $cfg:expr, $rt:expr, $request:expr) => {
        $complete($cfg, &$rt.http, $request).await
    };
    (Bedrock, $complete:path, $cfg:expr, $rt:expr, $request:expr) => {{
        #[cfg(feature = "bedrock")]
        {
            let client = $rt.bedrock_client($cfg).await;
            $complete(&client, &$cfg.model, $cfg.prompt_caching, $request).await
        }
        #[cfg(not(feature = "bedrock"))]
        {
            let _ = ($cfg, $rt, $request);
            Err(completion_feature_unavailable("aws_bedrock", "bedrock"))
        }
    }};
    (GeminiGrpc, $complete:path, $cfg:expr, $rt:expr, $request:expr) => {{
        #[cfg(feature = "gemini-grpc")]
        {
            let client = $rt.gemini_grpc_client($cfg).await?;
            $complete(&client, &$cfg.model, $request).await
        }
        #[cfg(not(feature = "gemini-grpc"))]
        {
            let _ = ($cfg, $rt, $request);
            Err(completion_feature_unavailable("gemini-grpc", "gemini-grpc"))
        }
    }};
    (Mock, $complete:path, $cfg:expr, $rt:expr, $request:expr) => {{
        let _ = $rt;
        $complete($cfg, &$request).await
    }};
}

macro_rules! stream_dispatch {
    (Http, $open_stream:path, $cfg:expr, $rt:expr, $request:expr) => {
        $open_stream($cfg, &$rt.http, $request).await
    };
    (Bedrock, $open_stream:path, $cfg:expr, $rt:expr, $request:expr) => {{
        #[cfg(feature = "bedrock")]
        {
            let client = $rt.bedrock_client($cfg).await;
            $open_stream(&client, &$cfg.model, $cfg.prompt_caching, $request).await
        }
        #[cfg(not(feature = "bedrock"))]
        {
            let _ = ($cfg, $rt, $request);
            Err(completion_feature_unavailable("aws_bedrock", "bedrock"))
        }
    }};
    (GeminiGrpc, $open_stream:path, $cfg:expr, $rt:expr, $request:expr) => {{
        #[cfg(feature = "gemini-grpc")]
        {
            let client = $rt.gemini_grpc_client($cfg).await?;
            $open_stream(&client, &$cfg.model, $request).await
        }
        #[cfg(not(feature = "gemini-grpc"))]
        {
            let _ = ($cfg, $rt, $request);
            Err(completion_feature_unavailable("gemini-grpc", "gemini-grpc"))
        }
    }};
    (Mock, $open_stream:path, $cfg:expr, $rt:expr, $request:expr) => {{
        let _ = $rt;
        $open_stream($cfg, &$request).await
    }};
}

macro_rules! embed_dispatch {
    (Http, $embed:path, $cfg:expr, $rt:expr, $texts:expr) => {
        $embed($cfg, &$rt.http, $texts).await
    };
    (Bedrock, $embed:path, $cfg:expr, $rt:expr, $texts:expr) => {{
        #[cfg(feature = "bedrock")]
        {
            let client = $rt.bedrock_client(&$cfg.client_config()).await;
            $embed(&client, &$cfg.model, $cfg.ndims, $texts).await
        }
        #[cfg(not(feature = "bedrock"))]
        {
            let _ = ($cfg, $rt, $texts);
            Err(embedding_feature_unavailable("aws_bedrock", "bedrock"))
        }
    }};
    (GeminiGrpc, $embed:path, $cfg:expr, $rt:expr, $texts:expr) => {{
        #[cfg(feature = "gemini-grpc")]
        {
            let client = $rt
                .gemini_grpc_client(&$cfg.client_config())
                .await
                .map_err(|error| EmbeddingError::ProviderError(error.to_string()))?;
            $embed(&client, &$cfg.model, $cfg.ndims, $texts).await
        }
        #[cfg(not(feature = "gemini-grpc"))]
        {
            let _ = ($cfg, $rt, $texts);
            Err(embedding_feature_unavailable("gemini-grpc", "gemini-grpc"))
        }
    }};
    (Mock, $embed:path, $cfg:expr, $rt:expr, $texts:expr) => {{
        let _ = $rt;
        $embed($cfg, &$texts)
    }};
}

macro_rules! embed_batches_dispatch {
    (Http, $embed_batches:path, $cfg:expr, $rt:expr, $texts:expr) => {
        $embed_batches($cfg, &$rt.http, $texts).await
    };
    (Bedrock, $embed_batches:path, $cfg:expr, $rt:expr, $texts:expr) => {{
        #[cfg(feature = "bedrock")]
        {
            let client = $rt.bedrock_client(&$cfg.client_config()).await;
            $embed_batches(&client, &$cfg.model, $cfg.ndims, $texts).await
        }
        #[cfg(not(feature = "bedrock"))]
        {
            let _ = ($cfg, $rt, $texts);
            Err(embedding_feature_unavailable("aws_bedrock", "bedrock"))
        }
    }};
    (GeminiGrpc, $embed_batches:path, $cfg:expr, $rt:expr, $texts:expr) => {{
        #[cfg(feature = "gemini-grpc")]
        {
            let client = $rt
                .gemini_grpc_client(&$cfg.client_config())
                .await
                .map_err(|error| EmbeddingError::ProviderError(error.to_string()))?;
            $embed_batches(&client, &$cfg.model, $cfg.ndims, $texts).await
        }
        #[cfg(not(feature = "gemini-grpc"))]
        {
            let _ = ($cfg, $rt, $texts);
            Err(embedding_feature_unavailable("gemini-grpc", "gemini-grpc"))
        }
    }};
    (Mock, $embed_batches:path, $cfg:expr, $rt:expr, $texts:expr) => {{
        let _ = $rt;
        let counts: Vec<usize> = $texts.iter().map(Vec::len).collect();
        let flat: Vec<String> = $texts.into_iter().flatten().collect();
        let response = $embed_batches($cfg, &flat)?;
        let groups = rig_core::embeddings::batching::group_batches(&counts, response.embeddings)?;
        Ok((groups, response.usage))
    }};
}

/// The single provider-capability registry.
///
/// Each row declares the stable public variant, capability descriptor,
/// completion and embedding config/fulfillment paths, and any Cargo feature
/// needed for fulfillment. Empty capability tuples omit that provider from
/// the corresponding enum without creating a second hand-maintained list.
macro_rules! provider_surface_registry {
    ($apply:ident) => {
        $apply! {
            Anthropic {
                descriptor: rig_core::providers::anthropic::functions::DESCRIPTOR,
                feature: None,
                completion: (
                    rig_core::providers::anthropic::functions::Config,
                    rig_core::providers::anthropic::functions::complete,
                    rig_core::providers::anthropic::functions::open_stream,
                    Http
                ),
                embedding: ()
            },
            Azure {
                descriptor: rig_core::providers::azure::functions::DESCRIPTOR,
                feature: None,
                completion: (
                    rig_core::providers::azure::functions::Config,
                    rig_core::providers::azure::functions::complete,
                    rig_core::providers::azure::functions::open_stream,
                    Http
                ),
                embedding: (
                    rig_core::providers::azure::functions::EmbeddingConfig,
                    rig_core::providers::azure::functions::embed,
                    rig_core::providers::azure::functions::embed_batches,
                    Http
                )
            },
            Bedrock {
                descriptor: rig_core::providers::bedrock::DESCRIPTOR,
                feature: Some("bedrock"),
                completion: (
                    rig_core::providers::bedrock::Config,
                    rig_bedrock::functions::complete_with_options,
                    rig_bedrock::functions::open_stream_with_options,
                    Bedrock
                ),
                embedding: (
                    rig_core::providers::bedrock::EmbeddingConfig,
                    rig_bedrock::functions::embed,
                    rig_bedrock::functions::embed_batches,
                    Bedrock
                )
            },
            ChatGpt {
                descriptor: rig_core::providers::chatgpt::functions::DESCRIPTOR,
                feature: None,
                completion: (
                    rig_core::providers::chatgpt::functions::Config,
                    rig_core::providers::chatgpt::functions::complete,
                    rig_core::providers::chatgpt::functions::open_stream,
                    Http
                ),
                embedding: ()
            },
            Cohere {
                descriptor: rig_core::providers::cohere::functions::DESCRIPTOR,
                feature: None,
                completion: (
                    rig_core::providers::cohere::functions::Config,
                    rig_core::providers::cohere::functions::complete,
                    rig_core::providers::cohere::functions::open_stream,
                    Http
                ),
                embedding: (
                    rig_core::providers::cohere::functions::EmbeddingConfig,
                    rig_core::providers::cohere::functions::embed,
                    rig_core::providers::cohere::functions::embed_batches,
                    Http
                )
            },
            Copilot {
                descriptor: rig_core::providers::copilot::functions::DESCRIPTOR,
                feature: None,
                completion: (
                    rig_core::providers::copilot::functions::Config,
                    rig_core::providers::copilot::functions::complete,
                    rig_core::providers::copilot::functions::open_stream,
                    Http
                ),
                embedding: (
                    rig_core::providers::copilot::functions::EmbeddingConfig,
                    rig_core::providers::copilot::functions::embed,
                    rig_core::providers::copilot::functions::embed_batches,
                    Http
                )
            },
            DeepSeek {
                descriptor: rig_core::providers::deepseek::functions::DESCRIPTOR,
                feature: None,
                completion: (
                    rig_core::providers::deepseek::functions::Config,
                    rig_core::providers::deepseek::functions::complete,
                    rig_core::providers::deepseek::functions::open_stream,
                    Http
                ),
                embedding: ()
            },
            Doubleword {
                descriptor: rig_core::providers::doubleword::functions::DESCRIPTOR,
                feature: None,
                completion: (
                    rig_core::providers::doubleword::functions::Config,
                    rig_core::providers::doubleword::functions::complete,
                    rig_core::providers::doubleword::functions::open_stream,
                    Http
                ),
                embedding: (
                    rig_core::providers::doubleword::functions::EmbeddingConfig,
                    rig_core::providers::doubleword::functions::embed,
                    rig_core::providers::doubleword::functions::embed_batches,
                    Http
                )
            },
            Gemini {
                descriptor: rig_core::providers::gemini::functions::DESCRIPTOR,
                feature: None,
                completion: (
                    rig_core::providers::gemini::functions::Config,
                    rig_core::providers::gemini::functions::complete,
                    rig_core::providers::gemini::functions::open_stream,
                    Http
                ),
                embedding: (
                    rig_core::providers::gemini::functions::EmbeddingConfig,
                    rig_core::providers::gemini::functions::embed,
                    rig_core::providers::gemini::functions::embed_batches,
                    Http
                )
            },
            GeminiInteractions {
                descriptor: rig_core::providers::gemini::interactions_api::functions::DESCRIPTOR,
                feature: None,
                completion: (
                    rig_core::providers::gemini::interactions_api::functions::Config,
                    rig_core::providers::gemini::interactions_api::functions::complete,
                    rig_core::providers::gemini::interactions_api::functions::open_stream,
                    Http
                ),
                embedding: ()
            },
            GeminiGrpc {
                descriptor: rig_core::providers::gemini_grpc::DESCRIPTOR,
                feature: Some("gemini-grpc"),
                completion: (
                    rig_core::providers::gemini_grpc::Config,
                    rig_gemini_grpc::functions::complete,
                    rig_gemini_grpc::functions::open_stream,
                    GeminiGrpc
                ),
                embedding: (
                    rig_core::providers::gemini_grpc::EmbeddingConfig,
                    rig_gemini_grpc::functions::embed,
                    rig_gemini_grpc::functions::embed_batches,
                    GeminiGrpc
                )
            },
            Groq {
                descriptor: rig_core::providers::groq::functions::DESCRIPTOR,
                feature: None,
                completion: (
                    rig_core::providers::groq::functions::Config,
                    rig_core::providers::groq::functions::complete,
                    rig_core::providers::groq::functions::open_stream,
                    Http
                ),
                embedding: ()
            },
            HuggingFace {
                descriptor: rig_core::providers::huggingface::functions::DESCRIPTOR,
                feature: None,
                completion: (
                    rig_core::providers::huggingface::functions::Config,
                    rig_core::providers::huggingface::functions::complete,
                    rig_core::providers::huggingface::functions::open_stream,
                    Http
                ),
                embedding: ()
            },
            Hyperbolic {
                descriptor: rig_core::providers::hyperbolic::functions::DESCRIPTOR,
                feature: None,
                completion: (
                    rig_core::providers::hyperbolic::functions::Config,
                    rig_core::providers::hyperbolic::functions::complete,
                    rig_core::providers::hyperbolic::functions::open_stream,
                    Http
                ),
                embedding: ()
            },
            Llamafile {
                descriptor: rig_core::providers::llamafile::functions::DESCRIPTOR,
                feature: None,
                completion: (
                    rig_core::providers::llamafile::functions::Config,
                    rig_core::providers::llamafile::functions::complete,
                    rig_core::providers::llamafile::functions::open_stream,
                    Http
                ),
                embedding: (
                    rig_core::providers::llamafile::functions::EmbeddingConfig,
                    rig_core::providers::llamafile::functions::embed,
                    rig_core::providers::llamafile::functions::embed_batches,
                    Http
                )
            },
            Minimax {
                descriptor: rig_core::providers::minimax::functions::DESCRIPTOR,
                feature: None,
                completion: (
                    rig_core::providers::minimax::functions::Config,
                    rig_core::providers::minimax::functions::complete,
                    rig_core::providers::minimax::functions::open_stream,
                    Http
                ),
                embedding: ()
            },
            Mira {
                descriptor: rig_core::providers::mira::functions::DESCRIPTOR,
                feature: None,
                completion: (
                    rig_core::providers::mira::functions::Config,
                    rig_core::providers::mira::functions::complete,
                    rig_core::providers::mira::functions::open_stream,
                    Http
                ),
                embedding: ()
            },
            Mistral {
                descriptor: rig_core::providers::mistral::functions::DESCRIPTOR,
                feature: None,
                completion: (
                    rig_core::providers::mistral::functions::Config,
                    rig_core::providers::mistral::functions::complete,
                    rig_core::providers::mistral::functions::open_stream,
                    Http
                ),
                embedding: (
                    rig_core::providers::mistral::functions::EmbeddingConfig,
                    rig_core::providers::mistral::functions::embed,
                    rig_core::providers::mistral::functions::embed_batches,
                    Http
                )
            },
            Mock {
                descriptor: MOCK_DESCRIPTOR,
                feature: None,
                completion: (
                    MockScript,
                    MockScript::next_response,
                    MockScript::next_stream,
                    Mock
                ),
                embedding: (
                    MockEmbedder,
                    MockEmbedder::next_response,
                    MockEmbedder::next_response,
                    Mock
                )
            },
            Moonshot {
                descriptor: rig_core::providers::moonshot::functions::DESCRIPTOR,
                feature: None,
                completion: (
                    rig_core::providers::moonshot::functions::Config,
                    rig_core::providers::moonshot::functions::complete,
                    rig_core::providers::moonshot::functions::open_stream,
                    Http
                ),
                embedding: ()
            },
            Ollama {
                descriptor: rig_core::providers::ollama::functions::DESCRIPTOR,
                feature: None,
                completion: (
                    rig_core::providers::ollama::functions::Config,
                    rig_core::providers::ollama::functions::complete,
                    rig_core::providers::ollama::functions::open_stream,
                    Http
                ),
                embedding: (
                    rig_core::providers::ollama::functions::EmbeddingConfig,
                    rig_core::providers::ollama::functions::embed,
                    rig_core::providers::ollama::functions::embed_batches,
                    Http
                )
            },
            OpenAi {
                descriptor: rig_core::providers::openai::functions::DESCRIPTOR,
                feature: None,
                completion: (
                    rig_core::providers::openai::functions::Config,
                    rig_core::providers::openai::functions::complete,
                    rig_core::providers::openai::functions::open_stream,
                    Http
                ),
                embedding: (
                    rig_core::providers::openai::functions::EmbeddingConfig,
                    rig_core::providers::openai::functions::embed,
                    rig_core::providers::openai::functions::embed_batches,
                    Http
                )
            },
            OpenAiResponses {
                descriptor: rig_core::providers::openai::responses_api::functions::DESCRIPTOR,
                feature: None,
                completion: (
                    rig_core::providers::openai::responses_api::functions::Config,
                    rig_core::providers::openai::responses_api::functions::complete,
                    rig_core::providers::openai::responses_api::functions::open_stream,
                    Http
                ),
                embedding: ()
            },
            OpenRouter {
                descriptor: rig_core::providers::openrouter::functions::DESCRIPTOR,
                feature: None,
                completion: (
                    rig_core::providers::openrouter::functions::Config,
                    rig_core::providers::openrouter::functions::complete,
                    rig_core::providers::openrouter::functions::open_stream,
                    Http
                ),
                embedding: (
                    rig_core::providers::openrouter::functions::EmbeddingConfig,
                    rig_core::providers::openrouter::functions::embed,
                    rig_core::providers::openrouter::functions::embed_batches,
                    Http
                )
            },
            Perplexity {
                descriptor: rig_core::providers::perplexity::functions::DESCRIPTOR,
                feature: None,
                completion: (
                    rig_core::providers::perplexity::functions::Config,
                    rig_core::providers::perplexity::functions::complete,
                    rig_core::providers::perplexity::functions::open_stream,
                    Http
                ),
                embedding: ()
            },
            Together {
                descriptor: rig_core::providers::together::functions::DESCRIPTOR,
                feature: None,
                completion: (
                    rig_core::providers::together::functions::Config,
                    rig_core::providers::together::functions::complete,
                    rig_core::providers::together::functions::open_stream,
                    Http
                ),
                embedding: (
                    rig_core::providers::together::functions::EmbeddingConfig,
                    rig_core::providers::together::functions::embed,
                    rig_core::providers::together::functions::embed_batches,
                    Http
                )
            },
            VoyageAi {
                descriptor: rig_core::providers::voyageai::functions::DESCRIPTOR,
                feature: None,
                completion: (),
                embedding: (
                    rig_core::providers::voyageai::functions::EmbeddingConfig,
                    rig_core::providers::voyageai::functions::embed,
                    rig_core::providers::voyageai::functions::embed_batches,
                    Http
                )
            },
            Xai {
                descriptor: rig_core::providers::xai::functions::DESCRIPTOR,
                feature: None,
                completion: (
                    rig_core::providers::xai::functions::Config,
                    rig_core::providers::xai::functions::complete,
                    rig_core::providers::xai::functions::open_stream,
                    Http
                ),
                embedding: ()
            },
            XiaomiMimo {
                descriptor: rig_core::providers::xiaomimimo::functions::DESCRIPTOR,
                feature: None,
                completion: (
                    rig_core::providers::xiaomimimo::functions::Config,
                    rig_core::providers::xiaomimimo::functions::complete,
                    rig_core::providers::xiaomimimo::functions::open_stream,
                    Http
                ),
                embedding: ()
            },
            Zai {
                descriptor: rig_core::providers::zai::functions::DESCRIPTOR,
                feature: None,
                completion: (
                    rig_core::providers::zai::functions::Config,
                    rig_core::providers::zai::functions::complete,
                    rig_core::providers::zai::functions::open_stream,
                    Http
                ),
                embedding: ()
            }
        }
    };
}

macro_rules! define_provider_surfaces {
    (
        $(
            $variant:ident {
                descriptor: $descriptor:path,
                feature: $feature:expr,
                completion: (
                    $(
                        $completion_config:path,
                        $complete:path,
                        $open_stream:path,
                        $completion_kind:ident
                    )?
                ),
                embedding: (
                    $(
                        $embedding_config:path,
                        $embed:path,
                        $embed_batches:path,
                        $embedding_kind:ident
                    )?
                )
            }
        ),* $(,)?
    ) => {
        /// All bundled provider surfaces, generated from the registry.
        pub const PROVIDER_SURFACES: &[ProviderSurface] = &[
            $(
                ProviderSurface {
                    variant: stringify!($variant),
                    descriptor: &$descriptor,
                    completion: capability_present!($($completion_config)?),
                    embedding: capability_present!($($embedding_config)?),
                    fulfillment_feature: $feature,
                },
            )*
        ];

        /// A bundled completion-provider selection as plain serde config.
        ///
        /// The closed enum is feature-stable: variants whose transport needs
        /// an optional Cargo feature remain serializable and matchable without
        /// it; fulfillment returns a clear boundary error until enabled.
        /// The always-present [`ProviderConfig::Mock`] variant is likewise
        /// production-visible and must be handled by exhaustive matches.
        #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
        pub enum ProviderConfig {
            $(
                $(
                    #[doc = concat!("The `", stringify!($variant), "` completion provider.")]
                    $variant($completion_config),
                )?
            )*
        }

        impl ProviderConfig {
            /// The provider's capability sheet.
            pub fn descriptor(&self) -> &'static ProviderDescriptor {
                match self {
                    $(
                        $(
                            Self::$variant(cfg) => {
                                let _: &$completion_config = cfg;
                                &$descriptor
                            },
                        )?
                    )*
                }
            }

            /// The model identifier this configuration targets.
            pub fn model(&self) -> &str {
                match self {
                    $(
                        $(
                            Self::$variant(cfg) => {
                                let _: &$completion_config = cfg;
                                config_model!($completion_kind, cfg)
                            },
                        )?
                    )*
                }
            }
        }

        /// Fulfill a completion request for any bundled provider.
        pub async fn complete(
            provider: &ProviderConfig,
            rt: &Runtime,
            request: CompletionRequest,
        ) -> Result<CompletionResponse, CompletionError> {
            match provider {
                $(
                    $(
                        ProviderConfig::$variant(cfg) => {
                            complete_dispatch!(
                                $completion_kind,
                                $complete,
                                cfg,
                                rt,
                                request
                            )
                        },
                    )?
                )*
            }
        }

        /// Open a streaming completion for any bundled provider.
        pub async fn open_stream(
            provider: &ProviderConfig,
            rt: &Runtime,
            request: CompletionRequest,
        ) -> Result<CompletionStream, CompletionError> {
            match provider {
                $(
                    $(
                        ProviderConfig::$variant(cfg) => {
                            stream_dispatch!(
                                $completion_kind,
                                $open_stream,
                                cfg,
                                rt,
                                request
                            )
                        },
                    )?
                )*
            }
        }

        $(
            $(
                impl From<$completion_config> for ProviderConfig {
                    fn from(config: $completion_config) -> Self {
                        Self::$variant(config)
                    }
                }
            )?
        )*

        /// A bundled embedding-provider selection as plain serde config.
        ///
        /// FastEmbed remains an explicit exception: loaded local weights are
        /// runtime state rather than resumable serde configuration.
        /// The deterministic [`EmbedderConfig::Mock`] variant remains present
        /// in production builds and must be handled by exhaustive matches.
        #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
        pub enum EmbedderConfig {
            $(
                $(
                    #[doc = concat!("The `", stringify!($variant), "` embedding provider.")]
                    $variant($embedding_config),
                )?
            )*
        }

        impl EmbedderConfig {
            /// The provider's capability sheet.
            pub fn descriptor(&self) -> &'static ProviderDescriptor {
                match self {
                    $(
                        $(
                            Self::$variant(cfg) => {
                                let _: &$embedding_config = cfg;
                                &$descriptor
                            },
                        )?
                    )*
                }
            }

            /// The embedding model identifier this configuration targets.
            pub fn model(&self) -> &str {
                match self {
                    $(
                        $(
                            Self::$variant(cfg) => {
                                let _: &$embedding_config = cfg;
                                config_model!($embedding_kind, cfg)
                            },
                        )?
                    )*
                }
            }
        }

        /// Embed `texts` with any bundled embedding provider.
        pub async fn embed(
            provider: &EmbedderConfig,
            rt: &Runtime,
            texts: Vec<String>,
        ) -> Result<rig_core::embeddings::EmbeddingResponse, EmbeddingError> {
            match provider {
                $(
                    $(
                        EmbedderConfig::$variant(cfg) => {
                            embed_dispatch!($embedding_kind, $embed, cfg, rt, texts)
                        },
                    )?
                )*
            }
        }

        /// Embed caller-defined batches while preserving batch order.
        pub async fn embed_batches(
            provider: &EmbedderConfig,
            rt: &Runtime,
            texts: Vec<Vec<String>>,
        ) -> Result<
            (
                Vec<rig_core::OneOrMany<rig_core::embeddings::Embedding>>,
                rig_core::completion::Usage,
            ),
            EmbeddingError,
        > {
            match provider {
                $(
                    $(
                        EmbedderConfig::$variant(cfg) => {
                            embed_batches_dispatch!(
                                $embedding_kind,
                                $embed_batches,
                                cfg,
                                rt,
                                texts
                            )
                        },
                    )?
                )*
            }
        }

        $(
            $(
                impl From<$embedding_config> for EmbedderConfig {
                    fn from(config: $embedding_config) -> Self {
                        Self::$variant(config)
                    }
                }
            )?
        )*
    };
}

provider_surface_registry!(define_provider_surfaces);

/// List the models available to `provider`'s credentials.
///
/// Model listing is an optional provider capability: only arms whose upstream
/// API exposes a listing endpoint are dispatched. Every other provider returns
/// a [`ModelListingError::RequestError`](rig_core::model::ModelListingError)
/// naming itself. Unlike completion and embedding fulfillment, the wildcard
/// is deliberate: listing support is outside the capability registry and a new
/// provider defaults to "unsupported".
pub async fn list_models(
    provider: &ProviderConfig,
    rt: &Runtime,
) -> Result<rig_core::model::ModelList, rig_core::model::ModelListingError> {
    use rig_core::providers as p;
    match provider {
        ProviderConfig::Anthropic(cfg) => p::anthropic::functions::list_models(cfg, &rt.http).await,
        ProviderConfig::Copilot(cfg) => p::copilot::functions::list_models(cfg, &rt.http).await,
        ProviderConfig::DeepSeek(cfg) => p::deepseek::functions::list_models(cfg, &rt.http).await,
        ProviderConfig::Gemini(cfg) => p::gemini::functions::list_models(cfg, &rt.http).await,
        ProviderConfig::GeminiInteractions(cfg) => {
            p::gemini::interactions_api::functions::list_models(cfg, &rt.http).await
        }
        ProviderConfig::Mistral(cfg) => p::mistral::functions::list_models(cfg, &rt.http).await,
        ProviderConfig::Ollama(cfg) => p::ollama::functions::list_models(cfg, &rt.http).await,
        ProviderConfig::OpenAi(cfg) => p::openai::functions::list_models(cfg, &rt.http).await,
        ProviderConfig::OpenRouter(cfg) => {
            p::openrouter::functions::list_models(cfg, &rt.http).await
        }
        ProviderConfig::XiaomiMimo(cfg) => {
            p::xiaomimimo::functions::list_models(cfg, &rt.http).await
        }
        other => Err(rig_core::model::ModelListingError::request_error(format!(
            "provider `{}` does not support model listing",
            other.descriptor().name
        ))),
    }
}

/// Scripted embedding responses for deterministic tests and host simulation —
/// the embeddings sibling of [`MockScript`].
///
/// Plain data plus an interior-mutable call cursor: `clone` SHARES the
/// cursor (so a session and a test observing it stay in step), and
/// deserialize RESETS it (`#[serde(skip)]`).
///
/// This type is the payload of the production-visible
/// [`EmbedderConfig::Mock`] serde variant. Hosts that accept provider
/// configuration across an untrusted boundary should validate or allowlist
/// variants and fields before fulfillment.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct MockEmbedder {
    /// One response per expected embed call, in order: the vectors for that
    /// call's texts (index-aligned with the texts).
    pub responses: Vec<Vec<Vec<f64>>>,
    #[serde(skip)]
    cursor: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    /// Text batches observed so far. Like the cursor, `clone` SHARES the
    /// record so a test probe stays in step with the script under test.
    #[serde(skip)]
    requests: std::sync::Arc<std::sync::Mutex<Vec<Vec<String>>>>,
}

impl MockEmbedder {
    /// A script answering each embed call with the next vector set.
    pub fn from_responses(responses: Vec<Vec<Vec<f64>>>) -> Self {
        Self {
            responses,
            cursor: std::sync::Arc::default(),
            requests: std::sync::Arc::default(),
        }
    }

    /// Calls served so far.
    pub fn calls(&self) -> usize {
        self.cursor.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Text batches served so far, in call order. Clones share this record.
    pub fn requests(&self) -> Vec<Vec<String>> {
        self.requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    fn next_response(
        &self,
        texts: &[String],
    ) -> Result<rig_core::embeddings::EmbeddingResponse, EmbeddingError> {
        // Claim the call index and record the request under the same lock so
        // `requests()[i]` always pairs with response `i`, even when clones
        // sharing the cursor are driven concurrently.
        let index = {
            let mut requests = self
                .requests
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let index = self
                .cursor
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            requests.push(texts.to_vec());
            index
        };
        let vectors = self.responses.get(index).cloned().ok_or_else(|| {
            EmbeddingError::ProviderError(format!(
                "mock embedder exhausted: call {index} has no scripted response"
            ))
        })?;
        if vectors.len() != texts.len() {
            return Err(EmbeddingError::ResponseError(format!(
                "mock embedder call {index} scripted {} vectors for {} texts",
                vectors.len(),
                texts.len()
            )));
        }
        let embeddings = texts
            .iter()
            .cloned()
            .zip(vectors)
            .map(|(document, vec)| rig_core::embeddings::Embedding { document, vec })
            .collect();
        Ok(rig_core::embeddings::EmbeddingResponse {
            embeddings,
            usage: rig_core::completion::Usage::new(),
        })
    }
}

/// The mock provider's capability sheet: everything on, so scripted tests
/// exercise every request shape.
pub static MOCK_DESCRIPTOR: ProviderDescriptor = ProviderDescriptor::named("mock")
    .with_tools(true)
    .with_response_format(true)
    .with_composes_native_output_with_tools(true);

/// Live transport handles, rebuilt per process — never serialized.
///
/// Configuration is what persists ([`ProviderConfig`]); the runtime holds
/// what cannot honestly be data: the HTTP client and, feature-gated, the
/// lazily-built AWS client and gRPC channel. Each cache is a monomorphic
/// struct with one concrete accessor — no generic cache, no stored build
/// closures.
#[derive(Debug, Default, Clone)]
pub struct Runtime {
    /// rig-core's HTTP executor (serves every in-core provider arm).
    pub http: HttpRuntime,
    #[cfg(feature = "bedrock")]
    bedrock: BedrockCache,
    #[cfg(feature = "gemini-grpc")]
    gemini_grpc: GeminiGrpcCache,
}

impl Runtime {
    /// A runtime with default (empty) caches over a fresh HTTP client.
    pub fn new() -> Self {
        Self::default()
    }

    /// A runtime reusing an existing [`HttpRuntime`].
    pub fn with_http(http: HttpRuntime) -> Self {
        Self {
            http,
            #[cfg(feature = "bedrock")]
            bedrock: BedrockCache::default(),
            #[cfg(feature = "gemini-grpc")]
            gemini_grpc: GeminiGrpcCache::default(),
        }
    }

    /// A runtime pre-seeded with a caller-built Bedrock SDK client.
    #[cfg(feature = "bedrock")]
    pub fn with_bedrock_client(
        connection: rig_bedrock::functions::ConnectionConfig,
        client: aws_sdk_bedrockruntime::Client,
    ) -> Self {
        Self {
            http: HttpRuntime::new(),
            bedrock: BedrockCache {
                provider_client: None,
                slot: std::sync::Arc::new(tokio::sync::Mutex::new(Some((
                    bedrock_connection_key_from_connection(&connection),
                    client,
                )))),
            },
            #[cfg(feature = "gemini-grpc")]
            gemini_grpc: GeminiGrpcCache::default(),
        }
    }

    /// A runtime sharing the lazy or seeded SDK client owned by a concrete
    /// Bedrock provider client.
    #[cfg(feature = "bedrock")]
    pub fn with_bedrock_provider_client(client: rig_bedrock::Client) -> Self {
        Self {
            http: HttpRuntime::new(),
            bedrock: BedrockCache {
                provider_client: Some(client),
                slot: std::sync::Arc::default(),
            },
            #[cfg(feature = "gemini-grpc")]
            gemini_grpc: GeminiGrpcCache::default(),
        }
    }

    /// A runtime pre-seeded with a connected Gemini gRPC client.
    #[cfg(feature = "gemini-grpc")]
    pub fn with_gemini_grpc_client(
        connection: rig_gemini_grpc::functions::ConnectionConfig,
        client: rig_gemini_grpc::Client,
    ) -> Self {
        Self {
            http: HttpRuntime::new(),
            #[cfg(feature = "bedrock")]
            bedrock: BedrockCache::default(),
            gemini_grpc: GeminiGrpcCache {
                slot: std::sync::Arc::new(tokio::sync::Mutex::new(Some((connection, client)))),
            },
        }
    }

    /// The Bedrock client for `cfg`, built on first use and rebuilt only
    /// when the *connection* projection of the configuration changes.
    ///
    /// The cache is keyed on `bedrock_connection_key` — the inputs
    /// `rig_bedrock::functions::client_from_config` actually consumes — so
    /// per-request knobs (model, `prompt_caching`) never rebuild the AWS
    /// client or evict a seeded one.
    #[cfg(feature = "bedrock")]
    pub async fn bedrock_client(
        &self,
        cfg: &rig_bedrock::functions::Config,
    ) -> aws_sdk_bedrockruntime::Client {
        let key = bedrock_connection_key(cfg);
        {
            let slot = self.bedrock.slot.lock().await;
            if let Some((cached_key, client)) = slot.as_ref()
                && *cached_key == key
            {
                return client.clone();
            }
        }

        if let Some(client) = &self.bedrock.provider_client
            && client.connection_config() == &cfg.connection
        {
            return client.get_inner().await;
        }

        let mut slot = self.bedrock.slot.lock().await;
        if let Some((cached_key, client)) = slot.as_ref()
            && *cached_key == key
        {
            return client.clone();
        }
        let client = rig_bedrock::functions::client_from_config(cfg).await;
        *slot = Some((key, client.clone()));
        client
    }

    /// Seed the Bedrock cache with a caller-built AWS client for `cfg`.
    ///
    /// Escape hatch for AWS clients that cannot be rebuilt from plain
    /// configuration (custom endpoints, credential providers, or HTTP
    /// connectors — e.g. recording transports in tests). Runs whose config
    /// shares `cfg`'s connection details (region / profile / endpoint URL)
    /// use `client` instead of building one from the config, regardless of
    /// model or prompt-caching settings.
    #[cfg(feature = "bedrock")]
    pub async fn seed_bedrock_client(
        &self,
        cfg: rig_bedrock::functions::Config,
        client: aws_sdk_bedrockruntime::Client,
    ) {
        let mut slot = self.bedrock.slot.lock().await;
        *slot = Some((bedrock_connection_key(&cfg), client));
    }

    /// The Gemini gRPC client for `cfg`, built (channel connected) on first
    /// use and rebuilt when the configuration changes.
    #[cfg(feature = "gemini-grpc")]
    pub async fn gemini_grpc_client(
        &self,
        cfg: &rig_gemini_grpc::functions::Config,
    ) -> Result<rig_gemini_grpc::Client, CompletionError> {
        let mut slot = self.gemini_grpc.slot.lock().await;
        if let Some((cached_connection, client)) = slot.as_ref()
            && cached_connection == &cfg.connection
        {
            return Ok(client.clone());
        }
        let client = rig_gemini_grpc::functions::client_from_config(cfg)
            .await
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?;
        *slot = Some((cfg.connection.clone(), client.clone()));
        Ok(client)
    }
}

/// The connection-defining projection of a Bedrock config: exactly the
/// fields `rig_bedrock::functions::client_from_config` reads when building
/// an AWS client — `(region, profile, endpoint_url)`.
#[cfg(feature = "bedrock")]
type BedrockConnectionKey = (Option<String>, Option<String>, Option<String>);

#[cfg(feature = "bedrock")]
fn bedrock_connection_key(cfg: &rig_bedrock::functions::Config) -> BedrockConnectionKey {
    bedrock_connection_key_from_connection(&cfg.connection)
}

#[cfg(feature = "bedrock")]
fn bedrock_connection_key_from_connection(
    connection: &rig_bedrock::functions::ConnectionConfig,
) -> BedrockConnectionKey {
    (
        connection.region.clone(),
        connection.profile.clone(),
        connection.endpoint_url.clone(),
    )
}

#[cfg(feature = "bedrock")]
#[derive(Debug, Default, Clone)]
struct BedrockCache {
    provider_client: Option<rig_bedrock::Client>,
    slot: std::sync::Arc<
        tokio::sync::Mutex<Option<(BedrockConnectionKey, aws_sdk_bedrockruntime::Client)>>,
    >,
}

#[cfg(feature = "gemini-grpc")]
#[derive(Debug, Default, Clone)]
struct GeminiGrpcCache {
    slot: std::sync::Arc<
        tokio::sync::Mutex<
            Option<(
                rig_gemini_grpc::functions::ConnectionConfig,
                rig_gemini_grpc::Client,
            )>,
        >,
    >,
}

/// Scripted provider responses for tests.
///
/// Plain data plus an interior-mutable turn cursor: `clone` SHARES the
/// cursor (so a session and a test observing it stay in step), and
/// deserialize RESETS it (`#[serde(skip)]`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MockStreamError {
    /// Number of converted provider items yielded before the error.
    pub after_items: usize,
    /// Provider-error message yielded by the stream.
    pub message: String,
}

impl MockStreamError {
    /// Create a scripted midstream provider error.
    pub fn new(after_items: usize, message: impl Into<String>) -> Self {
        Self {
            after_items,
            message: message.into(),
        }
    }
}

/// Scripted completion responses for deterministic tests and host simulation.
///
/// This type is the payload of the always-present, production-visible
/// [`ProviderConfig::Mock`] serde variant. Its [`Self::pending`] entries can
/// deliberately keep selected provider operations pending until the caller
/// cancels their futures. Hosts that accept provider configuration across an
/// untrusted boundary should validate or allowlist variants and fields before
/// fulfillment.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct MockScript {
    /// One normalized response per expected model call, in order.
    pub responses: Vec<CompletionResponse>,
    /// Streamed item scripts per expected model call, in order. When a turn
    /// has no stream script, its `responses` entry is converted into a
    /// terminal-only stream.
    #[serde(default)]
    pub streams: Vec<Vec<rig_core::streaming::StreamedAssistantContent>>,
    /// Per-call scripted transport errors, index-aligned with model calls.
    /// `Some(message)` at index `i` makes call `i` fail with
    /// [`CompletionError::ProviderError`] before any response lookup, so a
    /// failed attempt can be scripted ahead of a successful retry.
    #[serde(default)]
    pub errors: Vec<Option<String>>,
    /// Per-call errors yielded after a configured number of provider stream
    /// items. Unlike [`Self::errors`], these occur after a stream opened and
    /// can exercise rollback after partial output.
    #[serde(default)]
    pub stream_errors: Vec<Option<MockStreamError>>,
    /// Calls that remain pending until their provider future is cancelled,
    /// index-aligned with model calls.
    ///
    /// A `true` entry intentionally prevents that call from completing on its
    /// own. Validate this field before fulfillment when the script came from
    /// an untrusted serialized source.
    #[serde(default)]
    pub pending: Vec<bool>,
    #[serde(skip)]
    cursor: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    /// Requests observed so far. Like the cursor, `clone` SHARES the record
    /// so a test probe stays in step with the script under test.
    #[serde(skip)]
    requests: std::sync::Arc<std::sync::Mutex<Vec<CompletionRequest>>>,
}

impl MockScript {
    /// A script answering each model call with the next response.
    pub fn from_responses(responses: Vec<CompletionResponse>) -> Self {
        Self {
            responses,
            streams: Vec::new(),
            errors: Vec::new(),
            stream_errors: Vec::new(),
            pending: Vec::new(),
            cursor: std::sync::Arc::default(),
            requests: std::sync::Arc::default(),
        }
    }

    /// Attach streamed item scripts (index-aligned with `responses`).
    pub fn with_streams(
        mut self,
        streams: Vec<Vec<rig_core::streaming::StreamedAssistantContent>>,
    ) -> Self {
        self.streams = streams;
        self
    }

    /// Attach per-call transport errors (index-aligned with model calls).
    /// A `Some(message)` slot fails that call with
    /// [`CompletionError::ProviderError`]; `None` slots fall through to the
    /// scripted response or stream for that index.
    pub fn with_errors(mut self, errors: Vec<Option<String>>) -> Self {
        self.errors = errors;
        self
    }

    /// Attach midstream provider errors, index-aligned with model calls.
    pub fn with_stream_errors(mut self, errors: Vec<Option<MockStreamError>>) -> Self {
        self.stream_errors = errors;
        self
    }

    /// Mark selected provider operations as pending until their future is
    /// dropped, index-aligned with model calls.
    ///
    /// A selected operation deliberately has no natural completion. The host
    /// must cancel or drop its future.
    pub fn with_pending(mut self, pending: Vec<bool>) -> Self {
        self.pending = pending;
        self
    }

    /// Calls served so far.
    pub fn calls(&self) -> usize {
        self.cursor.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Requests served so far, in call order. Clones share this record.
    pub fn requests(&self) -> Vec<CompletionRequest> {
        self.requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Claim the next call index and record its request as one atomic step
    /// (both under the requests lock), so `requests()[i]` always pairs with
    /// scripted response `i` even when clones sharing the cursor are driven
    /// concurrently.
    fn record_call(&self, request: &CompletionRequest) -> usize {
        let mut requests = self
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let index = self
            .cursor
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        requests.push(request.clone());
        index
    }

    fn scripted_error(&self, index: usize) -> Option<CompletionError> {
        self.errors
            .get(index)
            .and_then(Option::as_ref)
            .map(|message| CompletionError::ProviderError(message.clone()))
    }

    async fn next_response(
        &self,
        request: &CompletionRequest,
    ) -> Result<CompletionResponse, CompletionError> {
        let index = self.record_call(request);
        if self.pending.get(index).copied().unwrap_or(false) {
            return futures::future::pending().await;
        }
        if let Some(error) = self.scripted_error(index) {
            return Err(error);
        }
        self.responses.get(index).cloned().ok_or_else(|| {
            CompletionError::ProviderError(format!(
                "mock script exhausted: call {index} has no scripted response"
            ))
        })
    }

    async fn next_stream(
        &self,
        request: &CompletionRequest,
    ) -> Result<CompletionStream, CompletionError> {
        use rig_core::streaming::{RawStreamingChoice, RawStreamingToolCall, StreamFinal};

        let index = self.record_call(request);
        if self.pending.get(index).copied().unwrap_or(false) {
            return futures::future::pending().await;
        }
        if let Some(error) = self.scripted_error(index) {
            return Err(error);
        }
        let items: Vec<RawStreamingChoice> = if let Some(script) = self.streams.get(index) {
            let mut items: Vec<RawStreamingChoice> = Vec::with_capacity(script.len());
            for item in script.iter() {
                let raw = match item.clone() {
                    rig_core::streaming::StreamedAssistantContent::Text(text) => {
                        // Preserve text-block metadata boundaries: a Text item
                        // carrying additional_params opens a fresh block with
                        // that metadata before its (possibly empty) delta.
                        if let Some(params) = text.additional_params {
                            items.push(RawStreamingChoice::TextStart {
                                additional_params: Some(params),
                            });
                        }
                        RawStreamingChoice::Message(text.text)
                    }
                    rig_core::streaming::StreamedAssistantContent::ToolCall {
                        tool_call,
                        internal_call_id,
                    } => {
                        let mut raw = RawStreamingToolCall::new(
                            tool_call.id,
                            tool_call.function.name,
                            tool_call.function.arguments,
                        )
                        .with_internal_call_id(internal_call_id)
                        .with_additional_params(tool_call.additional_params);
                        if let Some(call_id) = tool_call.call_id {
                            raw = raw.with_call_id(call_id);
                        }
                        RawStreamingChoice::ToolCall(raw)
                    }
                    rig_core::streaming::StreamedAssistantContent::ToolCallDelta {
                        id,
                        internal_call_id,
                        content,
                    } => RawStreamingChoice::ToolCallDelta {
                        id,
                        internal_call_id,
                        content,
                    },
                    rig_core::streaming::StreamedAssistantContent::Reasoning(reasoning) => {
                        RawStreamingChoice::Reasoning {
                            id: reasoning.id.clone(),
                            content: reasoning.content.into_iter().next().unwrap_or_else(|| {
                                rig_core::message::ReasoningContent::Text {
                                    text: String::new(),
                                    signature: None,
                                }
                            }),
                        }
                    }
                    rig_core::streaming::StreamedAssistantContent::ReasoningDelta {
                        id,
                        reasoning,
                    } => RawStreamingChoice::ReasoningDelta { id, reasoning },
                    rig_core::streaming::StreamedAssistantContent::Final(final_record) => {
                        // Real providers surface the message id as its own raw
                        // event; the assembler reads only the stream-level id.
                        if let Some(id) = final_record.message_id.clone() {
                            items.push(RawStreamingChoice::MessageId(id));
                        }
                        RawStreamingChoice::FinalResponse(final_record)
                    }
                    rig_core::streaming::StreamedAssistantContent::Unknown(value) => {
                        RawStreamingChoice::Unknown(value)
                    }
                };
                items.push(raw);
            }
            // Scripts that don't hand-author a terminal record inherit one
            // from the paired `responses` entry, like the unary branch.
            if !items
                .iter()
                .any(|item| matches!(item, RawStreamingChoice::FinalResponse(_)))
                && let Some(response) = self.responses.get(index).cloned()
            {
                let mut final_record = StreamFinal::new("mock", response.usage);
                if let Some(reason) = response.finish_reason {
                    final_record = final_record.with_finish_reason(reason);
                }
                if let Some(id) = response.message_id {
                    items.push(RawStreamingChoice::MessageId(id.clone()));
                    final_record = final_record.with_message_id(id);
                }
                items.push(RawStreamingChoice::FinalResponse(final_record));
            }
            items
        } else {
            // Derive a terminal-only stream from the unary script entry.
            let response = self.responses.get(index).cloned().ok_or_else(|| {
                CompletionError::ProviderError(format!(
                    "mock script exhausted: call {index} has no scripted stream"
                ))
            })?;
            let mut items: Vec<RawStreamingChoice> = response
                .choice
                .iter()
                .filter_map(|content| match content {
                    rig_core::message::AssistantContent::Text(text) => {
                        Some(RawStreamingChoice::Message(text.text.clone()))
                    }
                    rig_core::message::AssistantContent::ToolCall(tool_call) => {
                        Some(RawStreamingChoice::ToolCall(RawStreamingToolCall::new(
                            tool_call.id.clone(),
                            tool_call.function.name.clone(),
                            tool_call.function.arguments.clone(),
                        )))
                    }
                    _ => None,
                })
                .collect();
            let mut final_record = StreamFinal::new("mock", response.usage);
            if let Some(reason) = response.finish_reason {
                final_record = final_record.with_finish_reason(reason);
            }
            if let Some(id) = response.message_id {
                items.push(RawStreamingChoice::MessageId(id.clone()));
                final_record = final_record.with_message_id(id);
            }
            items.push(RawStreamingChoice::FinalResponse(final_record));
            items
        };
        let mut emitted: Vec<Result<RawStreamingChoice, CompletionError>> =
            items.into_iter().map(Ok).collect();
        if let Some(failure) = self.stream_errors.get(index).and_then(Option::as_ref) {
            emitted.insert(
                failure.after_items.min(emitted.len()),
                Err(CompletionError::ProviderError(failure.message.clone())),
            );
        }
        Ok(CompletionStream::from_stream(futures::stream::iter(
            emitted,
        )))
    }
}

#[cfg(test)]
mod conversion_tests {
    use super::*;

    /// `From` must select the same variant an explicit wrap would, for each
    /// distinct provider config type — including the two providers that expose
    /// a second, incompatible API surface.
    #[test]
    fn from_config_selects_the_same_variant_as_explicit_wrapping() {
        let openai = rig_core::providers::openai::functions::Config::new("gpt-4o");
        assert!(matches!(
            ProviderConfig::from(openai.clone()),
            ProviderConfig::OpenAi(_)
        ));
        assert_eq!(
            ProviderConfig::from(openai.clone()).model(),
            ProviderConfig::OpenAi(openai).model()
        );

        let anthropic = rig_core::providers::anthropic::functions::Config::new("claude-sonnet-4-5");
        assert!(matches!(
            ProviderConfig::from(anthropic),
            ProviderConfig::Anthropic(_)
        ));

        // The second surface on the same provider stays distinct: the chat
        // config maps to `OpenAi`, the responses config to `OpenAiResponses`.
        let responses =
            rig_core::providers::openai::responses_api::functions::Config::new("gpt-4o");
        assert!(matches!(
            ProviderConfig::from(responses),
            ProviderConfig::OpenAiResponses(_)
        ));

        let interactions = rig_core::providers::gemini::interactions_api::functions::Config::new(
            "gemini-2.5-flash",
        );
        assert!(matches!(
            ProviderConfig::from(interactions),
            ProviderConfig::GeminiInteractions(_)
        ));

        let bedrock = rig_core::providers::bedrock::Config::new("bedrock-model");
        assert!(matches!(
            ProviderConfig::from(bedrock),
            ProviderConfig::Bedrock(_)
        ));

        let grpc = rig_core::providers::gemini_grpc::Config::new("grpc-model");
        assert!(matches!(
            ProviderConfig::from(grpc),
            ProviderConfig::GeminiGrpc(_)
        ));
    }

    #[test]
    fn missing_in_core_embedders_are_generated_with_from_conversions() {
        assert!(matches!(
            EmbedderConfig::from(
                rig_core::providers::llamafile::functions::EmbeddingConfig::new("llama")
            ),
            EmbedderConfig::Llamafile(_)
        ));
        assert!(matches!(
            EmbedderConfig::from(
                rig_core::providers::mistral::functions::EmbeddingConfig::new("mistral")
            ),
            EmbedderConfig::Mistral(_)
        ));
        assert!(matches!(
            EmbedderConfig::from(
                rig_core::providers::openrouter::functions::EmbeddingConfig::new("openrouter")
            ),
            EmbedderConfig::OpenRouter(_)
        ));
        assert!(matches!(
            EmbedderConfig::from(
                rig_core::providers::together::functions::EmbeddingConfig::new("together")
            ),
            EmbedderConfig::Together(_)
        ));
    }

    #[test]
    fn generated_surface_census_is_unique_and_complete() {
        let variants: std::collections::HashSet<_> = PROVIDER_SURFACES
            .iter()
            .map(|surface| surface.variant)
            .collect();
        assert_eq!(variants.len(), PROVIDER_SURFACES.len());
        assert_eq!(
            PROVIDER_SURFACES
                .iter()
                .filter(|surface| surface.completion)
                .count(),
            29
        );
        assert_eq!(
            PROVIDER_SURFACES
                .iter()
                .filter(|surface| surface.embedding)
                .count(),
            15
        );
        assert_eq!(
            PROVIDER_SURFACES
                .iter()
                .find(|surface| surface.variant == "Bedrock")
                .and_then(|surface| surface.fulfillment_feature),
            Some("bedrock")
        );
        assert_eq!(
            PROVIDER_SURFACES
                .iter()
                .find(|surface| surface.variant == "GeminiGrpc")
                .and_then(|surface| surface.fulfillment_feature),
            Some("gemini-grpc")
        );
    }

    #[test]
    fn feature_backed_variants_round_trip_without_transport_features() {
        for provider in [
            ProviderConfig::Bedrock(rig_core::providers::bedrock::Config::new("bedrock-model")),
            ProviderConfig::GeminiGrpc(rig_core::providers::gemini_grpc::Config::new("grpc-model")),
        ] {
            let model = provider.model().to_string();
            let descriptor = provider.descriptor().name;
            let json = serde_json::to_string(&provider).expect("serialize stable provider variant");
            let round_trip: ProviderConfig =
                serde_json::from_str(&json).expect("deserialize stable provider variant");
            assert_eq!(round_trip.model(), model);
            assert_eq!(round_trip.descriptor().name, descriptor);
        }

        for provider in [
            EmbedderConfig::Bedrock(rig_core::providers::bedrock::EmbeddingConfig::new(
                "bedrock-embed",
            )),
            EmbedderConfig::GeminiGrpc(rig_core::providers::gemini_grpc::EmbeddingConfig::new(
                "grpc-embed",
            )),
        ] {
            let model = provider.model().to_string();
            let json = serde_json::to_string(&provider).expect("serialize stable embedder variant");
            let round_trip: EmbedderConfig =
                serde_json::from_str(&json).expect("deserialize stable embedder variant");
            assert_eq!(round_trip.model(), model);
        }
    }

    #[test]
    fn voyage_outer_embedder_debug_redacts_connection_secrets() {
        let mut config = rig_core::providers::voyageai::functions::EmbeddingConfig::new("voyage-3")
            .with_api_key("outer-api-secret");
        config.extra_headers.push((
            "x-private-token".to_string(),
            "outer-header-secret".to_string(),
        ));
        let provider = EmbedderConfig::from(config);
        let debug = format!("{provider:?}");

        assert!(debug.contains("x-private-token"));
        assert!(!debug.contains("outer-api-secret"));
        assert!(!debug.contains("outer-header-secret"));
    }

    /// The `impl Into<ProviderConfig>` seam must not change the built agent:
    /// passing a bare config and passing the wrapped enum agree.
    #[test]
    fn agent_builder_accepts_a_bare_config_and_an_explicit_variant_alike() {
        let cfg = rig_core::providers::openai::functions::Config::new("gpt-4o");

        let from_bare = crate::AgentBuilder::new(cfg.clone()).build();
        let from_variant = crate::AgentBuilder::new(ProviderConfig::OpenAi(cfg)).build();

        assert_eq!(from_bare.provider.model(), from_variant.provider.model());
        assert_eq!(
            from_bare.provider.descriptor().name,
            from_variant.provider.descriptor().name
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EMBEDDING_RESPONSE: &str = r#"{
        "object": "list",
        "model": "test-embedder",
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
        "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}]
    }"#;

    const MODEL_LIST_RESPONSE: &str = r#"{
        "data": [{"id": "gpt-test", "created": 123, "owned_by": "test-org"}]
    }"#;

    #[tokio::test]
    async fn model_listing_dispatch_remains_available_outside_the_registry() {
        let client = rig_core::test_utils::RecordingHttpClient::new(MODEL_LIST_RESPONSE);
        let runtime = Runtime::with_http(HttpRuntime::recording(client.clone()));
        let provider = ProviderConfig::OpenAi(
            rig_core::providers::openai::functions::Config::new("gpt-test")
                .with_api_key("test-key"),
        );

        let models = list_models(&provider, &runtime)
            .await
            .expect("OpenAI model-list dispatch should succeed");

        assert_eq!(models.data.len(), 1);
        assert_eq!(models.data[0].id, "gpt-test");
        let requests = client.requests();
        assert_eq!(requests.len(), 1);
        assert!(
            requests[0].uri.ends_with("/models"),
            "unexpected URI: {}",
            requests[0].uri
        );

        let unsupported = ProviderConfig::Mock(MockScript::default());
        let error = list_models(&unsupported, &Runtime::new())
            .await
            .expect_err("mock has no model-listing capability");

        assert!(error.to_string().contains("mock"));
    }

    async fn assert_no_network_embedding_dispatch(
        provider: EmbedderConfig,
        expected_uri_suffix: &str,
    ) {
        let client = rig_core::test_utils::RecordingHttpClient::new(EMBEDDING_RESPONSE);
        let runtime = Runtime::with_http(HttpRuntime::recording(client.clone()));

        let response = embed(&provider, &runtime, vec!["hello".to_string()])
            .await
            .expect("generated embedding dispatch should succeed");

        assert_eq!(response.embeddings.len(), 1);
        let requests = client.requests();
        assert_eq!(requests.len(), 1);
        assert!(
            requests[0].uri.ends_with(expected_uri_suffix),
            "unexpected URI: {}",
            requests[0].uri
        );
    }

    #[tokio::test]
    async fn generated_missing_embedder_dispatch_reaches_each_request_builder() {
        assert_no_network_embedding_dispatch(
            rig_core::providers::llamafile::functions::EmbeddingConfig::new("llama").into(),
            "/v1/embeddings",
        )
        .await;
        assert_no_network_embedding_dispatch(
            rig_core::providers::mistral::functions::EmbeddingConfig::new("mistral")
                .with_api_key("test-key")
                .into(),
            "/v1/embeddings",
        )
        .await;
        assert_no_network_embedding_dispatch(
            rig_core::providers::openrouter::functions::EmbeddingConfig::new("openrouter")
                .with_api_key("test-key")
                .into(),
            "/api/v1/embeddings",
        )
        .await;
        assert_no_network_embedding_dispatch(
            rig_core::providers::together::functions::EmbeddingConfig::new("together")
                .with_api_key("test-key")
                .into(),
            "/v1/embeddings",
        )
        .await;
    }

    #[cfg(not(feature = "bedrock"))]
    #[tokio::test]
    async fn bedrock_variant_fails_at_fulfillment_boundary_when_feature_is_disabled() {
        let provider = ProviderConfig::Bedrock(rig_core::providers::bedrock::Config::new("model"));
        let error = complete(
            &provider,
            &Runtime::new(),
            CompletionRequest::from_prompt("hello"),
        )
        .await
        .expect_err("disabled transport must fail before fulfillment");
        assert!(matches!(error, CompletionError::RequestError(_)));
        assert!(error.to_string().contains("bedrock"));

        let embedder = EmbedderConfig::Bedrock(rig_core::providers::bedrock::EmbeddingConfig::new(
            "embed-model",
        ));
        let error = embed(&embedder, &Runtime::new(), vec!["hello".to_string()])
            .await
            .expect_err("disabled transport must fail before fulfillment");
        assert!(error.to_string().contains("bedrock"));
    }

    #[cfg(not(feature = "gemini-grpc"))]
    #[tokio::test]
    async fn gemini_grpc_variant_fails_at_fulfillment_boundary_when_feature_is_disabled() {
        let provider =
            ProviderConfig::GeminiGrpc(rig_core::providers::gemini_grpc::Config::new("model"));
        let error = open_stream(
            &provider,
            &Runtime::new(),
            CompletionRequest::from_prompt("hello"),
        )
        .await
        .err()
        .expect("disabled transport must fail before fulfillment");
        assert!(matches!(error, CompletionError::RequestError(_)));
        assert!(error.to_string().contains("gemini-grpc"));

        let embedder = EmbedderConfig::GeminiGrpc(
            rig_core::providers::gemini_grpc::EmbeddingConfig::new("embed-model"),
        );
        let error = embed(&embedder, &Runtime::new(), vec!["hello".to_string()])
            .await
            .expect_err("disabled transport must fail before fulfillment");
        assert!(error.to_string().contains("gemini-grpc"));
    }

    /// A seeded Bedrock client must survive requests for configs that share
    /// its connection details but differ in per-request knobs (model,
    /// prompt caching) — the cache is keyed on the connection projection,
    /// not full config equality.
    #[cfg(feature = "bedrock")]
    #[tokio::test]
    async fn bedrock_cache_keeps_seeded_client_across_model_changes() {
        use aws_sdk_bedrockruntime::config::{BehaviorVersion, Region};

        // A region string no config in this test uses — a rebuild through
        // `client_from_config` could never produce a client carrying it, so
        // seeing it on the returned client proves the seeded client survived.
        const SEEDED_MARKER_REGION: &str = "seeded-marker";

        let seeded = aws_sdk_bedrockruntime::Client::from_conf(
            aws_sdk_bedrockruntime::config::Builder::new()
                .behavior_version(BehaviorVersion::latest())
                .region(Region::new(SEEDED_MARKER_REGION))
                .endpoint_url("http://seeded.invalid")
                .build(),
        );

        let cfg_a = rig_bedrock::functions::Config::new("model-a").with_region("us-east-1");
        let mut cfg_b = rig_bedrock::functions::Config::new("model-b").with_region("us-east-1");
        cfg_b.prompt_caching = true;

        let rt = Runtime::new();
        rt.seed_bedrock_client(cfg_a, seeded).await;

        // Same connection (region/profile/endpoint_url), different model and
        // prompt-caching flag: the seeded client must be returned, not a
        // freshly built one (which would carry no custom endpoint).
        let client = rt.bedrock_client(&cfg_b).await;
        assert_eq!(
            client.config().region().map(|region| region.as_ref()),
            Some(SEEDED_MARKER_REGION)
        );
    }

    #[tokio::test]
    async fn mock_embedder_scripts_responses_in_order() {
        let script =
            MockEmbedder::from_responses(vec![vec![vec![0.1], vec![0.2]], vec![vec![0.3]]]);
        let cfg = EmbedderConfig::Mock(script.clone());
        let rt = Runtime::new();

        let response = embed(&cfg, &rt, vec!["a".to_string(), "b".to_string()])
            .await
            .expect("first call");
        let documents: Vec<_> = response
            .embeddings
            .iter()
            .map(|e| e.document.clone())
            .collect();
        assert_eq!(documents, ["a", "b"]);

        // Clone shares the cursor: the second call advances the script.
        let response = embed(&cfg, &rt, vec!["c".to_string()])
            .await
            .expect("second call");
        assert_eq!(
            response.embeddings.first().map(|e| e.vec.clone()),
            Some(vec![0.3])
        );
        assert_eq!(script.calls(), 2);
        assert!(embed(&cfg, &rt, vec!["d".to_string()]).await.is_err());
    }

    #[tokio::test]
    async fn embed_batches_regroups_order_aligned() {
        let script = MockEmbedder::from_responses(vec![vec![vec![0.1], vec![0.2], vec![0.3]]]);
        let cfg = EmbedderConfig::Mock(script.clone());
        let rt = Runtime::new();

        let (groups, _usage) = embed_batches(
            &cfg,
            &rt,
            vec![
                vec!["a".to_string(), "b".to_string()],
                vec!["c".to_string()],
            ],
        )
        .await
        .expect("embed batches");

        // The mock saw one flattened call, regrouped 2 + 1 in input order.
        assert_eq!(script.requests(), vec![vec!["a", "b", "c"]]);
        assert_eq!(groups.len(), 2);
        let first: Vec<_> = groups
            .first()
            .expect("first group")
            .iter()
            .map(|e| e.document.clone())
            .collect();
        assert_eq!(first, ["a", "b"]);
        let second: Vec<_> = groups
            .get(1)
            .expect("second group")
            .iter()
            .map(|e| e.document.clone())
            .collect();
        assert_eq!(second, ["c"]);
    }

    #[test]
    fn mock_script_pairs_requests_with_responses_under_concurrency() {
        use rig_core::OneOrMany;
        use rig_core::message::{AssistantContent, Message, UserContent};

        let count = 16usize;
        let responses: Vec<CompletionResponse> = (0..count)
            .map(|i| {
                CompletionResponse::new(
                    OneOrMany::one(AssistantContent::text(format!("r{i}"))),
                    rig_core::completion::Usage::new(),
                    "mock",
                )
            })
            .collect();
        let script = MockScript::from_responses(responses);

        let handles: Vec<_> = (0..count)
            .map(|thread| {
                let script = script.clone();
                std::thread::spawn(move || {
                    let request = CompletionRequest::from_prompt(format!("q{thread}"));
                    let response = futures::executor::block_on(script.next_response(&request))
                        .expect("scripted response");
                    let text = match response.choice.first() {
                        AssistantContent::Text(text) => text.text,
                        other => panic!("expected text, got {other:?}"),
                    };
                    (format!("q{thread}"), text)
                })
            })
            .collect();
        let served: Vec<(String, String)> = handles
            .into_iter()
            .map(|handle| handle.join().expect("thread"))
            .collect();

        // requests()[i] must pair with response i: the response each caller
        // received is the one scripted at its request's recorded index.
        let recorded_prompts: Vec<String> = script
            .requests()
            .into_iter()
            .map(|request| match request.chat_history.first() {
                Message::User { content } => match content.first() {
                    UserContent::Text(text) => text.text,
                    other => panic!("expected text prompt, got {other:?}"),
                },
                other => panic!("expected user prompt, got {other:?}"),
            })
            .collect();
        assert_eq!(recorded_prompts.len(), count);
        for (prompt, response_text) in served {
            let index = recorded_prompts
                .iter()
                .position(|recorded| *recorded == prompt)
                .expect("every request must be recorded");
            assert_eq!(response_text, format!("r{index}"));
        }
    }

    #[test]
    fn mock_embedder_deserialize_resets_cursor() {
        let script = MockEmbedder::from_responses(vec![vec![vec![0.5]]]);
        let json = serde_json::to_string(&script).expect("serialize");
        let back: MockEmbedder = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.calls(), 0);
        assert_eq!(back.responses, vec![vec![vec![0.5]]]);
    }
}
