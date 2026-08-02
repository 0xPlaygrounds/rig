//! The closed, exhaustively-matched provider set.
//!
//! [`ProviderConfig`] is one serde value per bundled provider — plain
//! configuration, never a live handle — and [`complete`]/[`open_stream`]
//! fulfil a [`CompletionRequest`] for any arm with an exhaustive `match`.
//! Adding a provider fails to compile until every fulfilment site handles
//! it: that is the feature, which is also why the enum is deliberately
//! **not** `#[non_exhaustive]` — external hosts matching provider configs
//! get the same compile-time guarantee.
//!
//! Live transports live in [`Runtime`], not in configs: a serialized
//! `ProviderConfig` resumes anywhere, and handles (HTTP client, AWS client,
//! gRPC channel) are rebuilt on first use per process.
//!
//! Out-of-tree providers cannot add arms; they drive the public
//! [`AgentRun`](crate::agent::run::AgentRun) +
//! [`prepare_request`](crate::agent::prepare::prepare_request) protocol
//! directly instead.

use rig_core::completion::{CompletionError, CompletionRequest, CompletionResponse};
use rig_core::embeddings::EmbeddingError;
use rig_core::http_runtime::HttpRuntime;
use rig_core::providers::descriptor::ProviderDescriptor;
use rig_core::streaming::CompletionStream;

/// One row per bundled in-core provider: `(Variant, module, feature-less)`.
/// Adding a provider is one row here plus its `functions` module; the
/// compiler walks you through every match this macro does not generate.
macro_rules! for_each_builtin_provider {
    ($apply:ident) => {
        $apply! {
            (Anthropic, anthropic),
            (Azure, azure),
            (ChatGpt, chatgpt),
            (Cohere, cohere),
            (Copilot, copilot),
            (DeepSeek, deepseek),
            (Doubleword, doubleword),
            (Gemini, gemini),
            (Groq, groq),
            (HuggingFace, huggingface),
            (Hyperbolic, hyperbolic),
            (Llamafile, llamafile),
            (Minimax, minimax),
            (Mira, mira),
            (Mistral, mistral),
            (Moonshot, moonshot),
            (Ollama, ollama),
            (OpenAi, openai),
            (OpenRouter, openrouter),
            (Perplexity, perplexity),
            (Together, together),
            (Xai, xai),
            (XiaomiMimo, xiaomimimo),
            (Zai, zai),
        }
    };
}

macro_rules! define_provider_config {
    ($(($variant:ident, $module:ident),)*) => {
        /// A bundled provider selection as plain serde configuration.
        ///
        /// Deliberately exhaustive (no `#[non_exhaustive]`): a new provider
        /// must fail to compile in every matching host rather than fall
        /// through a wildcard arm.
        ///
        /// # Configs may carry credentials
        ///
        /// A provider config is connection data, and connection data
        /// includes secrets: `api_key` may be
        /// [`ApiKeyLocation::Inline`](rig_core::providers::descriptor::ApiKeyLocation),
        /// and `extra_headers` can include an `authorization` /
        /// `x-api-key` / `api-key` header. `Debug` redacts inline API keys and
        /// every extra-header value, while serde stays faithful by design
        /// (resuming a serialized config requires the real values) — treat
        /// serialized configs as secrets.
        #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
        pub enum ProviderConfig {
            $(
                #[doc = concat!("The `", stringify!($module), "` provider.")]
                $variant(rig_core::providers::$module::functions::Config),
            )*
            /// The canonical OpenAI client speaking the Responses API
            /// (`openai::responses_api::functions`). Hand-written because its
            /// module path doesn't fit the one-module-per-row macro shape;
            /// the chat-completions face stays on [`Self::OpenAi`].
            OpenAiResponses(rig_core::providers::openai::responses_api::functions::Config),
            /// Gemini's Interactions API
            /// (`gemini::interactions_api::functions`). Hand-written for the
            /// same reason as [`Self::OpenAiResponses`]: it is a second,
            /// incompatible surface on a provider whose macro row already
            /// carries the `generateContent` config, and it authenticates with
            /// an `x-goog-api-key` header rather than a `?key=` query
            /// parameter. The `generateContent` face stays on [`Self::Gemini`].
            GeminiInteractions(
                rig_core::providers::gemini::interactions_api::functions::Config,
            ),
            /// AWS Bedrock (Converse API over the AWS SDK).
            #[cfg(feature = "bedrock")]
            Bedrock(rig_bedrock::functions::Config),
            /// Gemini over gRPC (tonic).
            #[cfg(feature = "gemini-grpc")]
            GeminiGrpc(rig_gemini_grpc::functions::Config),
            /// Scripted responses for tests — the successor to the deleted
            /// `MockCompletionModel`. Clone SHARES the turn cursor;
            /// deserialize resets it.
            #[cfg(any(test, feature = "test-utils"))]
            Mock(MockScript),
        }

        impl ProviderConfig {
            /// The provider's capability sheet.
            pub fn descriptor(&self) -> &'static ProviderDescriptor {
                match self {
                    $(Self::$variant(_) => &rig_core::providers::$module::functions::DESCRIPTOR,)*
                    Self::OpenAiResponses(_) => {
                        &rig_core::providers::openai::responses_api::functions::DESCRIPTOR
                    }
                    Self::GeminiInteractions(_) => {
                        &rig_core::providers::gemini::interactions_api::functions::DESCRIPTOR
                    }
                    #[cfg(feature = "bedrock")]
                    Self::Bedrock(_) => &rig_bedrock::functions::DESCRIPTOR,
                    #[cfg(feature = "gemini-grpc")]
                    Self::GeminiGrpc(_) => &rig_gemini_grpc::functions::DESCRIPTOR,
                    #[cfg(any(test, feature = "test-utils"))]
                    Self::Mock(_) => &MOCK_DESCRIPTOR,
                }
            }

            /// The model identifier this configuration targets.
            pub fn model(&self) -> &str {
                match self {
                    $(Self::$variant(cfg) => &cfg.model,)*
                    Self::OpenAiResponses(cfg) => &cfg.model,
                    Self::GeminiInteractions(cfg) => &cfg.model,
                    #[cfg(feature = "bedrock")]
                    Self::Bedrock(cfg) => &cfg.model,
                    #[cfg(feature = "gemini-grpc")]
                    Self::GeminiGrpc(cfg) => &cfg.model,
                    #[cfg(any(test, feature = "test-utils"))]
                    Self::Mock(_) => "mock",
                }
            }
        }

        /// Fulfil a completion request for any bundled provider.
        pub async fn complete(
            provider: &ProviderConfig,
            rt: &Runtime,
            request: CompletionRequest,
        ) -> Result<CompletionResponse, CompletionError> {
            match provider {
                $(
                    ProviderConfig::$variant(cfg) => {
                        rig_core::providers::$module::functions::complete(cfg, &rt.http, request)
                            .await
                    }
                )*
                ProviderConfig::OpenAiResponses(cfg) => {
                    rig_core::providers::openai::responses_api::functions::complete(
                        cfg, &rt.http, request,
                    )
                    .await
                }
                ProviderConfig::GeminiInteractions(cfg) => {
                    rig_core::providers::gemini::interactions_api::functions::complete(
                        cfg, &rt.http, request,
                    )
                    .await
                }
                #[cfg(feature = "bedrock")]
                ProviderConfig::Bedrock(cfg) => {
                    let client = rt.bedrock_client(cfg).await;
                    rig_bedrock::functions::complete_with_options(
                        &client,
                        &cfg.model,
                        cfg.prompt_caching,
                        request,
                    )
                    .await
                }
                #[cfg(feature = "gemini-grpc")]
                ProviderConfig::GeminiGrpc(cfg) => {
                    let client = rt.gemini_grpc_client(cfg).await?;
                    rig_gemini_grpc::functions::complete(&client, &cfg.model, request).await
                }
                #[cfg(any(test, feature = "test-utils"))]
                ProviderConfig::Mock(script) => script.next_response(&request).await,
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
                    ProviderConfig::$variant(cfg) => {
                        rig_core::providers::$module::functions::open_stream(
                            cfg, &rt.http, request,
                        )
                        .await
                    }
                )*
                ProviderConfig::OpenAiResponses(cfg) => {
                    rig_core::providers::openai::responses_api::functions::open_stream(
                        cfg, &rt.http, request,
                    )
                    .await
                }
                ProviderConfig::GeminiInteractions(cfg) => {
                    rig_core::providers::gemini::interactions_api::functions::open_stream(
                        cfg, &rt.http, request,
                    )
                    .await
                }
                #[cfg(feature = "bedrock")]
                ProviderConfig::Bedrock(cfg) => {
                    let client = rt.bedrock_client(cfg).await;
                    rig_bedrock::functions::open_stream_with_options(
                        &client,
                        &cfg.model,
                        cfg.prompt_caching,
                        request,
                    )
                    .await
                }
                #[cfg(feature = "gemini-grpc")]
                ProviderConfig::GeminiGrpc(cfg) => {
                    let client = rt.gemini_grpc_client(cfg).await?;
                    rig_gemini_grpc::functions::open_stream(&client, &cfg.model, request).await
                }
                #[cfg(any(test, feature = "test-utils"))]
                ProviderConfig::Mock(script) => script.next_stream(&request).await,
            }
        }

        // One `From` per provider config, so callers hand a config straight to
        // `AgentBuilder::new` without naming the variant. Each provider owns a
        // distinct `Config` type, so the impls cannot overlap; a provider that
        // ever aliased another's config would fail to compile here, which is
        // the right place to find out.
        $(
            impl From<rig_core::providers::$module::functions::Config> for ProviderConfig {
                fn from(config: rig_core::providers::$module::functions::Config) -> Self {
                    Self::$variant(config)
                }
            }
        )*
    };
}

for_each_builtin_provider!(define_provider_config);

impl From<rig_core::providers::openai::responses_api::functions::Config> for ProviderConfig {
    fn from(config: rig_core::providers::openai::responses_api::functions::Config) -> Self {
        Self::OpenAiResponses(config)
    }
}

impl From<rig_core::providers::gemini::interactions_api::functions::Config> for ProviderConfig {
    fn from(config: rig_core::providers::gemini::interactions_api::functions::Config) -> Self {
        Self::GeminiInteractions(config)
    }
}

#[cfg(feature = "bedrock")]
impl From<rig_bedrock::functions::Config> for ProviderConfig {
    fn from(config: rig_bedrock::functions::Config) -> Self {
        Self::Bedrock(config)
    }
}

#[cfg(feature = "gemini-grpc")]
impl From<rig_gemini_grpc::functions::Config> for ProviderConfig {
    fn from(config: rig_gemini_grpc::functions::Config) -> Self {
        Self::GeminiGrpc(config)
    }
}

#[cfg(any(test, feature = "test-utils"))]
impl From<MockScript> for ProviderConfig {
    fn from(script: MockScript) -> Self {
        Self::Mock(script)
    }
}

/// List the models available to `provider`'s credentials.
///
/// Model listing is an *optional* provider capability: only the arms whose
/// upstream API exposes a listing endpoint (and whose `functions` face has a
/// `list_models` free function) are dispatched; every other provider returns
/// a [`ModelListingError::RequestError`](rig_core::model::ModelListingError)
/// naming itself. Unlike [`complete`]/[`open_stream`], the wildcard arm is
/// deliberate — a newly added provider defaults to "listing unsupported"
/// rather than failing to compile.
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

/// One row per bundled in-core embedding provider:
/// `(Variant, module)` where the module's `functions` face carries an
/// `EmbeddingConfig` plus free `embed`/`embed_batches` functions.
macro_rules! for_each_builtin_embedder {
    ($apply:ident) => {
        $apply! {
            (Azure, azure),
            (Cohere, cohere),
            (Copilot, copilot),
            (Doubleword, doubleword),
            (Gemini, gemini),
            (Ollama, ollama),
            (OpenAi, openai),
            (VoyageAi, voyageai),
        }
    };
}

macro_rules! define_embedder_config {
    ($(($variant:ident, $module:ident),)*) => {
        /// A bundled embedding-provider selection as plain serde
        /// configuration — the embeddings sibling of [`ProviderConfig`],
        /// with the same exhaustive-match contract (deliberately not
        /// `#[non_exhaustive]`).
        ///
        /// FastEmbed (`rig-fastembed`) deliberately has no arm: it runs
        /// local model weights whose loaded handle cannot honestly be
        /// serde configuration, and giving it an arm would require a new
        /// weights cache in [`Runtime`]. Drive
        /// `rig_fastembed::functions::embed` directly instead.
        #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
        pub enum EmbedderConfig {
            $(
                #[doc = concat!("The `", stringify!($module), "` embedding provider.")]
                $variant(rig_core::providers::$module::functions::EmbeddingConfig),
            )*
            /// AWS Bedrock (InvokeModel over the AWS SDK).
            #[cfg(feature = "bedrock")]
            Bedrock(rig_bedrock::functions::EmbeddingConfig),
            /// Gemini over gRPC (tonic).
            #[cfg(feature = "gemini-grpc")]
            GeminiGrpc(rig_gemini_grpc::functions::EmbeddingConfig),
            /// Scripted embeddings for tests. Clone SHARES the call
            /// cursor; deserialize resets it.
            #[cfg(any(test, feature = "test-utils"))]
            Mock(MockEmbedder),
        }

        impl EmbedderConfig {
            /// The provider's capability sheet.
            pub fn descriptor(&self) -> &'static ProviderDescriptor {
                match self {
                    $(Self::$variant(_) => &rig_core::providers::$module::functions::DESCRIPTOR,)*
                    #[cfg(feature = "bedrock")]
                    Self::Bedrock(_) => &rig_bedrock::functions::DESCRIPTOR,
                    #[cfg(feature = "gemini-grpc")]
                    Self::GeminiGrpc(_) => &rig_gemini_grpc::functions::DESCRIPTOR,
                    #[cfg(any(test, feature = "test-utils"))]
                    Self::Mock(_) => &MOCK_DESCRIPTOR,
                }
            }

            /// The embedding model identifier this configuration targets.
            pub fn model(&self) -> &str {
                match self {
                    $(Self::$variant(cfg) => &cfg.model,)*
                    #[cfg(feature = "bedrock")]
                    Self::Bedrock(cfg) => &cfg.model,
                    #[cfg(feature = "gemini-grpc")]
                    Self::GeminiGrpc(cfg) => &cfg.model,
                    #[cfg(any(test, feature = "test-utils"))]
                    Self::Mock(_) => "mock",
                }
            }
        }

        /// Embed `texts` with any bundled embedding provider.
        ///
        /// Chunking to each provider's `max_embedding_documents` happens
        /// inside the provider's free function; embeddings come back in
        /// input order with summed usage.
        pub async fn embed(
            provider: &EmbedderConfig,
            rt: &Runtime,
            texts: Vec<String>,
        ) -> Result<rig_core::embeddings::EmbeddingResponse, EmbeddingError> {
            match provider {
                $(
                    EmbedderConfig::$variant(cfg) => {
                        rig_core::providers::$module::functions::embed(cfg, &rt.http, texts).await
                    }
                )*
                #[cfg(feature = "bedrock")]
                EmbedderConfig::Bedrock(cfg) => {
                    let client = rt.bedrock_client(&cfg.client_config()).await;
                    rig_bedrock::functions::embed(&client, &cfg.model, cfg.ndims, texts).await
                }
                #[cfg(feature = "gemini-grpc")]
                EmbedderConfig::GeminiGrpc(cfg) => {
                    let client = rt
                        .gemini_grpc_client(&cfg.client_config())
                        .await
                        .map_err(|e| EmbeddingError::ProviderError(e.to_string()))?;
                    rig_gemini_grpc::functions::embed(&client, &cfg.model, cfg.ndims, texts).await
                }
                #[cfg(any(test, feature = "test-utils"))]
                EmbedderConfig::Mock(script) => script.next_response(&texts),
            }
        }

        /// Embed caller-defined batches with any bundled embedding
        /// provider, returning one order-aligned
        /// [`rig_core::OneOrMany`] group per input batch plus summed usage.
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
            let counts: Vec<usize> = texts.iter().map(Vec::len).collect();
            let flat: Vec<String> = texts.into_iter().flatten().collect();
            let response = embed(provider, rt, flat).await?;
            let groups =
                rig_core::embeddings::batching::group_batches(&counts, response.embeddings)?;
            Ok((groups, response.usage))
        }
    };
}

for_each_builtin_embedder!(define_embedder_config);

/// Scripted embedding responses for tests — the embeddings sibling of
/// [`MockScript`].
///
/// Plain data plus an interior-mutable call cursor: `clone` SHARES the
/// cursor (so a session and a test observing it stay in step), and
/// deserialize RESETS it (`#[serde(skip)]`).
#[cfg(any(test, feature = "test-utils"))]
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

#[cfg(any(test, feature = "test-utils"))]
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
#[cfg(any(test, feature = "test-utils"))]
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
#[cfg(any(test, feature = "test-utils"))]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MockStreamError {
    /// Number of converted provider items yielded before the error.
    pub after_items: usize,
    /// Provider-error message yielded by the stream.
    pub message: String,
}

#[cfg(any(test, feature = "test-utils"))]
impl MockStreamError {
    /// Create a scripted midstream provider error.
    pub fn new(after_items: usize, message: impl Into<String>) -> Self {
        Self {
            after_items,
            message: message.into(),
        }
    }
}

#[cfg(any(test, feature = "test-utils"))]
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
    /// index-aligned with model calls. Used to verify driver cancellation
    /// ownership without wall-clock-dependent transport behavior.
    #[serde(default)]
    pub pending: Vec<bool>,
    #[serde(skip)]
    cursor: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    /// Requests observed so far. Like the cursor, `clone` SHARES the record
    /// so a test probe stays in step with the script under test.
    #[serde(skip)]
    requests: std::sync::Arc<std::sync::Mutex<Vec<CompletionRequest>>>,
}

#[cfg(any(test, feature = "test-utils"))]
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
