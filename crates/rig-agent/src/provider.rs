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
            /// The canonical OpenAI client speaking the Responses API
            /// (`openai::responses_api::functions`). Hand-written because its
            /// module path doesn't fit the one-module-per-row macro shape;
            /// the chat-completions face stays on [`Self::OpenAi`].
            OpenAiResponses(rig_core::providers::openai::responses_api::functions::Config),
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
                #[cfg(any(test, feature = "test-utils"))]
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
                ProviderConfig::OpenAiResponses(cfg) => {
                    rig_core::providers::openai::responses_api::functions::open_stream(
                        cfg, &rt.http, request,
                    )
                    .await
                }
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
                #[cfg(any(test, feature = "test-utils"))]
                ProviderConfig::Mock(script) => script.next_stream(&request),
            }
        }
    };
}

for_each_builtin_provider!(define_provider_config);

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
        tokio::sync::Mutex<
            Option<(
                rig_bedrock::functions::Config,
                aws_sdk_bedrockruntime::Client,
            )>,
        >,
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

    fn record_request(&self, request: &CompletionRequest) {
        self.requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(request.clone());
    }

    fn scripted_error(&self, index: usize) -> Option<CompletionError> {
        self.errors
            .get(index)
            .and_then(Option::as_ref)
            .map(|message| CompletionError::ProviderError(message.clone()))
    }

    fn take_index(&self) -> usize {
        self.cursor
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    }

    fn next_response(
        &self,
        request: &CompletionRequest,
    ) -> Result<CompletionResponse, CompletionError> {
        self.record_request(request);
        let index = self.take_index();
        if let Some(error) = self.scripted_error(index) {
            return Err(error);
        }
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

        self.record_request(request);
        let index = self.take_index();
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
        Ok(StreamingCompletionResponse::stream(Box::pin(
            futures::stream::iter(items.into_iter().map(Ok)),
        )))
    }
}
