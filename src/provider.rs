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
//! [`prepare_request`](rig_agent::agent::prepare::prepare_request) protocol
//! directly instead.

use rig_core::completion::{CompletionError, CompletionRequest, CompletionResponse};
use rig_core::http_runtime::HttpRuntime;
use rig_core::providers::descriptor::ProviderDescriptor;
use rig_core::streaming::StreamingCompletionResponse;

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
        #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
        pub enum ProviderConfig {
            $(
                #[doc = concat!("The `", stringify!($module), "` provider.")]
                $variant(rig_core::providers::$module::functions::Config),
            )*
            /// AWS Bedrock (Converse API over the AWS SDK).
            #[cfg(feature = "bedrock")]
            Bedrock(rig_bedrock::functions::Config),
            /// Gemini over gRPC (tonic).
            #[cfg(feature = "gemini-grpc")]
            GeminiGrpc(rig_gemini_grpc::functions::Config),
            /// Scripted responses for tests — the successor to the deleted
            /// `MockCompletionModel`. Clone SHARES the turn cursor;
            /// deserialize resets it.
            #[cfg(feature = "test-utils")]
            Mock(MockScript),
        }

        impl ProviderConfig {
            /// The provider's capability sheet.
            pub fn descriptor(&self) -> &'static ProviderDescriptor {
                match self {
                    $(Self::$variant(_) => &rig_core::providers::$module::functions::DESCRIPTOR,)*
                    #[cfg(feature = "bedrock")]
                    Self::Bedrock(_) => &rig_bedrock::functions::DESCRIPTOR,
                    #[cfg(feature = "gemini-grpc")]
                    Self::GeminiGrpc(_) => &rig_gemini_grpc::functions::DESCRIPTOR,
                    #[cfg(feature = "test-utils")]
                    Self::Mock(_) => &MOCK_DESCRIPTOR,
                }
            }

            /// The model identifier this configuration targets.
            pub fn model(&self) -> &str {
                match self {
                    $(Self::$variant(cfg) => &cfg.model,)*
                    #[cfg(feature = "bedrock")]
                    Self::Bedrock(cfg) => &cfg.model,
                    #[cfg(feature = "gemini-grpc")]
                    Self::GeminiGrpc(cfg) => &cfg.model,
                    #[cfg(feature = "test-utils")]
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
                #[cfg(feature = "bedrock")]
                ProviderConfig::Bedrock(cfg) => {
                    let client = rt.bedrock_client(cfg).await;
                    rig_bedrock::functions::complete(&client, &cfg.model, request).await
                }
                #[cfg(feature = "gemini-grpc")]
                ProviderConfig::GeminiGrpc(cfg) => {
                    let client = rt.gemini_grpc_client(cfg).await?;
                    rig_gemini_grpc::functions::complete(&client, &cfg.model, request).await
                }
                #[cfg(feature = "test-utils")]
                ProviderConfig::Mock(script) => script.next_response(&request),
            }
        }

        /// Open a streaming completion for any bundled provider.
        pub async fn open_stream(
            provider: &ProviderConfig,
            rt: &Runtime,
            request: CompletionRequest,
        ) -> Result<StreamingCompletionResponse, CompletionError> {
            match provider {
                $(
                    ProviderConfig::$variant(cfg) => {
                        rig_core::providers::$module::functions::open_stream(
                            cfg, &rt.http, request,
                        )
                        .await
                    }
                )*
                #[cfg(feature = "bedrock")]
                ProviderConfig::Bedrock(cfg) => {
                    let client = rt.bedrock_client(cfg).await;
                    rig_bedrock::functions::open_stream(&client, &cfg.model, request).await
                }
                #[cfg(feature = "gemini-grpc")]
                ProviderConfig::GeminiGrpc(cfg) => {
                    let client = rt.gemini_grpc_client(cfg).await?;
                    rig_gemini_grpc::functions::open_stream(&client, &cfg.model, request).await
                }
                #[cfg(feature = "test-utils")]
                ProviderConfig::Mock(script) => script.next_stream(&request),
            }
        }
    };
}

for_each_builtin_provider!(define_provider_config);

/// The mock provider's capability sheet: everything on, so scripted tests
/// exercise every request shape.
#[cfg(feature = "test-utils")]
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

    /// The Bedrock client for `cfg`, built on first use and rebuilt when
    /// the configuration changes.
    #[cfg(feature = "bedrock")]
    pub async fn bedrock_client(
        &self,
        cfg: &rig_bedrock::functions::Config,
    ) -> aws_sdk_bedrockruntime::Client {
        let mut slot = self.bedrock.slot.lock().await;
        if let Some((cached_cfg, client)) = slot.as_ref()
            && cached_cfg == cfg
        {
            return client.clone();
        }
        let client = rig_bedrock::functions::client_from_config(cfg).await;
        *slot = Some((cfg.clone(), client.clone()));
        client
    }

    /// The Gemini gRPC client for `cfg`, built (channel connected) on first
    /// use and rebuilt when the configuration changes.
    #[cfg(feature = "gemini-grpc")]
    pub async fn gemini_grpc_client(
        &self,
        cfg: &rig_gemini_grpc::functions::Config,
    ) -> Result<rig_gemini_grpc::Client, CompletionError> {
        let mut slot = self.gemini_grpc.slot.lock().await;
        if let Some((cached_cfg, client)) = slot.as_ref()
            && cached_cfg == cfg
        {
            return Ok(client.clone());
        }
        let client = rig_gemini_grpc::functions::client_from_config(cfg)
            .await
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?;
        *slot = Some((cfg.clone(), client.clone()));
        Ok(client)
    }
}

#[cfg(feature = "bedrock")]
#[derive(Debug, Default, Clone)]
struct BedrockCache {
    slot: std::sync::Arc<
        tokio::sync::Mutex<Option<(rig_bedrock::functions::Config, aws_sdk_bedrockruntime::Client)>>,
    >,
}

#[cfg(feature = "gemini-grpc")]
#[derive(Debug, Default, Clone)]
struct GeminiGrpcCache {
    slot: std::sync::Arc<
        tokio::sync::Mutex<Option<(rig_gemini_grpc::functions::Config, rig_gemini_grpc::Client)>>,
    >,
}

/// Scripted provider responses for tests.
///
/// Plain data plus an interior-mutable turn cursor: `clone` SHARES the
/// cursor (so a session and a test observing it stay in step), and
/// deserialize RESETS it (`#[serde(skip)]`).
#[cfg(feature = "test-utils")]
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct MockScript {
    /// One normalized response per expected model call, in order.
    pub responses: Vec<CompletionResponse>,
    /// Streamed item scripts per expected model call, in order. When a turn
    /// has no stream script, its `responses` entry is converted into a
    /// terminal-only stream.
    #[serde(default)]
    pub streams: Vec<Vec<rig_core::streaming::StreamedAssistantContent>>,
    #[serde(skip)]
    cursor: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

#[cfg(feature = "test-utils")]
impl MockScript {
    /// A script answering each model call with the next response.
    pub fn from_responses(responses: Vec<CompletionResponse>) -> Self {
        Self {
            responses,
            streams: Vec::new(),
            cursor: std::sync::Arc::default(),
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

    /// Calls served so far.
    pub fn calls(&self) -> usize {
        self.cursor.load(std::sync::atomic::Ordering::SeqCst)
    }

    fn take_index(&self) -> usize {
        self.cursor
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    }

    fn next_response(
        &self,
        _request: &CompletionRequest,
    ) -> Result<CompletionResponse, CompletionError> {
        let index = self.take_index();
        self.responses.get(index).cloned().ok_or_else(|| {
            CompletionError::ProviderError(format!(
                "mock script exhausted: call {index} has no scripted response"
            ))
        })
    }

    fn next_stream(
        &self,
        request: &CompletionRequest,
    ) -> Result<StreamingCompletionResponse, CompletionError> {
        use rig_core::streaming::{RawStreamingChoice, RawStreamingToolCall, StreamFinal};

        let index = self.take_index();
        let items: Vec<RawStreamingChoice> = if let Some(script) = self.streams.get(index) {
            let mut items: Vec<RawStreamingChoice> = script
                .iter()
                .map(|item| match item.clone() {
                    rig_core::streaming::StreamedAssistantContent::Text(text) => {
                        RawStreamingChoice::Message(text.text)
                    }
                    rig_core::streaming::StreamedAssistantContent::ToolCall {
                        tool_call,
                        internal_call_id,
                    } => RawStreamingChoice::ToolCall(
                        RawStreamingToolCall::new(
                            tool_call.id,
                            tool_call.function.name,
                            tool_call.function.arguments,
                        )
                        .with_internal_call_id(internal_call_id),
                    ),
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
                        RawStreamingChoice::FinalResponse(final_record)
                    }
                    rig_core::streaming::StreamedAssistantContent::Unknown(value) => {
                        RawStreamingChoice::Unknown(value)
                    }
                })
                .collect();
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
                        Some(RawStreamingChoice::ToolCall(
                            RawStreamingToolCall::new(
                                tool_call.id.clone(),
                                tool_call.function.name.clone(),
                                tool_call.function.arguments.clone(),
                            ),
                        ))
                    }
                    _ => None,
                })
                .collect();
            let mut final_record = StreamFinal::new("mock", response.usage);
            if let Some(reason) = response.finish_reason {
                final_record = final_record.with_finish_reason(reason);
            }
            if let Some(id) = response.message_id {
                final_record = final_record.with_message_id(id);
            }
            items.push(RawStreamingChoice::FinalResponse(final_record));
            items
        };
        let _ = request;
        Ok(StreamingCompletionResponse::stream(Box::pin(
            futures::stream::iter(items.into_iter().map(Ok)),
        )))
    }
}
