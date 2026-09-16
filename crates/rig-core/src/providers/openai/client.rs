use super::responses_api::{
    ConfigurableSystemInstructionsPlacement, ResponsesProviderExt, SystemInstructionsPlacement,
};
#[cfg(feature = "audio")]
use crate::client::HasAudioGeneration;
#[cfg(feature = "image")]
use crate::client::HasImageGeneration;
#[cfg(feature = "audio")]
use crate::operation::AudioGeneration as AudioGenOp;
#[cfg(feature = "image")]
use crate::operation::ImageGeneration as ImageGenOp;
use crate::{
    client::{
        self, BearerAuth, HasCompletion, HasEmbeddings, HasModelListing, HasTranscription,
        ModelTransport, Provider, ProviderClientResult,
    },
    driver::{Bound, HasCompletion as DriverHasCompletion},
    http_client::{self, HeaderMap, HeaderValue, HttpClientExt},
    operation::{
        Completion, Embedding as EmbeddingOp, ModelListing as ModelListingOp,
        Transcription as TranscriptionOp, Verify as VerifyOp,
    },
    wasm_compat::{WasmCompatSend, WasmCompatSync},
    wire::{Body, Decoder, Encoded, Framing, Secret, Wire, WireEvent, WireFrame},
};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
// ================================================================
// Main OpenAI Client and Dialect
// ================================================================
pub const OPENAI_API_BASE_URL: &str = "https://api.openai.com/v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Quirks {
    pub emits_complete_single_chunk_tool_calls: bool,
    pub supports_tools: bool,
    pub supports_response_format: bool,
    pub stream_include_usage: bool,
    pub supports_image_tool_results: bool,
    pub string_content_parts: bool,
    #[serde(skip)]
    pub modern_output_cap_models: &'static [&'static str],
}

impl Quirks {
    pub fn requires_modern_output_cap(&self, model: &str) -> bool {
        self.modern_output_cap_models
            .iter()
            .any(|m| model.starts_with(m))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Dialect {
    pub name: &'static str,
    pub base_url: &'static str,
    pub api_key_env: &'static str,
    pub base_url_env: Option<&'static str>,
    pub request_id_header: Option<&'static str>,
    pub default_max_tokens: Option<u64>,
    pub quirks: Quirks,
}

impl<'de> Deserialize<'de> for Dialect {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct DialectHelper {
            name: String,
        }
        let helper = DialectHelper::deserialize(deserializer)?;
        let dialect = match helper.name.as_str() {
            "azure" => AZURE,
            "deepseek" => DEEPSEEK,
            "groq" => GROQ,
            "hyperbolic" => HYPERBOLIC,
            "mira" => MIRA,
            "perplexity" => PERPLEXITY,
            "together" => TOGETHER,
            "huggingface" => HUGGINGFACE,
            "llamacpp" => LLAMACPP,
            "mistral" => MISTRAL,
            "openrouter" => OPENROUTER,
            "venice" => VENICE,
            "doubleword" => DOUBLEWORD,
            "mistralrs" => MISTRALRS,
            "xai" => XAI,
            _ => OPENAI,
        };
        Ok(dialect)
    }
}

pub const OPENAI: Dialect = Dialect {
    name: "openai",
    base_url: "https://api.openai.com/v1",
    api_key_env: "OPENAI_API_KEY",
    base_url_env: Some("OPENAI_BASE_URL"),
    request_id_header: Some("x-request-id"),
    default_max_tokens: None,
    quirks: Quirks {
        emits_complete_single_chunk_tool_calls: false,
        supports_tools: true,
        supports_response_format: true,
        stream_include_usage: true,
        supports_image_tool_results: false,
        string_content_parts: false,
        modern_output_cap_models: &[
            "o1",
            "o1-mini",
            "o1-preview",
            "o3",
            "o3-mini",
            "o4-mini",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-5.1",
            "gpt-5.2",
        ],
    },
};

pub const AZURE: Dialect = Dialect {
    name: "azure",
    base_url: "https://api.cognitive.microsoft.com/sts/v1.0",
    api_key_env: "AZURE_OPENAI_API_KEY",
    base_url_env: Some("AZURE_OPENAI_ENDPOINT"),
    request_id_header: Some("x-ms-request-id"),
    default_max_tokens: None,
    quirks: Quirks {
        emits_complete_single_chunk_tool_calls: false,
        supports_tools: true,
        supports_response_format: true,
        stream_include_usage: true,
        supports_image_tool_results: false,
        string_content_parts: false,
        modern_output_cap_models: &[],
    },
};

pub const DEEPSEEK: Dialect = Dialect {
    name: "deepseek",
    base_url: "https://api.deepseek.com",
    api_key_env: "DEEPSEEK_API_KEY",
    base_url_env: None,
    request_id_header: None,
    default_max_tokens: None,
    quirks: Quirks {
        emits_complete_single_chunk_tool_calls: true,
        supports_tools: true,
        supports_response_format: false,
        stream_include_usage: true,
        supports_image_tool_results: false,
        string_content_parts: true,
        modern_output_cap_models: &[],
    },
};

pub const GROQ: Dialect = Dialect {
    name: "groq",
    base_url: "https://api.groq.com/openai/v1",
    api_key_env: "GROQ_API_KEY",
    base_url_env: None,
    request_id_header: Some("x-request-id"),
    default_max_tokens: None,
    quirks: Quirks {
        emits_complete_single_chunk_tool_calls: false,
        supports_tools: true,
        supports_response_format: true,
        stream_include_usage: true,
        supports_image_tool_results: false,
        string_content_parts: false,
        modern_output_cap_models: &[],
    },
};

pub const HYPERBOLIC: Dialect = Dialect {
    name: "hyperbolic",
    base_url: "https://api.hyperbolic.xyz/v1",
    api_key_env: "HYPERBOLIC_API_KEY",
    base_url_env: None,
    request_id_header: None,
    default_max_tokens: None,
    quirks: Quirks {
        emits_complete_single_chunk_tool_calls: false,
        supports_tools: true,
        supports_response_format: true,
        stream_include_usage: true,
        supports_image_tool_results: false,
        string_content_parts: false,
        modern_output_cap_models: &[],
    },
};

pub const MIRA: Dialect = Dialect {
    name: "mira",
    base_url: "https://api.mira.network/v1",
    api_key_env: "MIRA_API_KEY",
    base_url_env: None,
    request_id_header: None,
    default_max_tokens: None,
    quirks: Quirks {
        emits_complete_single_chunk_tool_calls: false,
        supports_tools: true,
        supports_response_format: true,
        stream_include_usage: true,
        supports_image_tool_results: false,
        string_content_parts: false,
        modern_output_cap_models: &[],
    },
};

pub const PERPLEXITY: Dialect = Dialect {
    name: "perplexity",
    base_url: "https://api.perplexity.ai",
    api_key_env: "PERPLEXITY_API_KEY",
    base_url_env: None,
    request_id_header: None,
    default_max_tokens: None,
    quirks: Quirks {
        emits_complete_single_chunk_tool_calls: false,
        supports_tools: false,
        supports_response_format: true,
        stream_include_usage: true,
        supports_image_tool_results: false,
        string_content_parts: false,
        modern_output_cap_models: &[],
    },
};

pub const TOGETHER: Dialect = Dialect {
    name: "together",
    base_url: "https://api.together.xyz/v1",
    api_key_env: "TOGETHER_API_KEY",
    base_url_env: None,
    request_id_header: None,
    default_max_tokens: None,
    quirks: Quirks {
        emits_complete_single_chunk_tool_calls: false,
        supports_tools: true,
        supports_response_format: true,
        stream_include_usage: true,
        supports_image_tool_results: false,
        string_content_parts: false,
        modern_output_cap_models: &[],
    },
};

pub const HUGGINGFACE: Dialect = Dialect {
    name: "huggingface",
    base_url: "https://api-inference.huggingface.co/v1",
    api_key_env: "HF_TOKEN",
    base_url_env: None,
    request_id_header: None,
    default_max_tokens: None,
    quirks: Quirks {
        emits_complete_single_chunk_tool_calls: false,
        supports_tools: true,
        supports_response_format: true,
        stream_include_usage: true,
        supports_image_tool_results: false,
        string_content_parts: false,
        modern_output_cap_models: &[],
    },
};

pub const LLAMACPP: Dialect = Dialect {
    name: "llamacpp",
    base_url: "http://localhost:8080/v1",
    api_key_env: "LLAMACPP_API_KEY",
    base_url_env: Some("LLAMACPP_BASE_URL"),
    request_id_header: None,
    default_max_tokens: None,
    quirks: Quirks {
        emits_complete_single_chunk_tool_calls: true,
        supports_tools: true,
        supports_response_format: true,
        stream_include_usage: true,
        supports_image_tool_results: true,
        string_content_parts: false,
        modern_output_cap_models: &[],
    },
};

pub const MISTRAL: Dialect = Dialect {
    name: "mistral",
    base_url: "https://api.mistral.ai/v1",
    api_key_env: "MISTRAL_API_KEY",
    base_url_env: None,
    request_id_header: None,
    default_max_tokens: None,
    quirks: Quirks {
        emits_complete_single_chunk_tool_calls: false,
        supports_tools: true,
        supports_response_format: true,
        stream_include_usage: true,
        supports_image_tool_results: false,
        string_content_parts: false,
        modern_output_cap_models: &[],
    },
};

pub const OPENROUTER: Dialect = Dialect {
    name: "openrouter",
    base_url: "https://openrouter.ai/api/v1",
    api_key_env: "OPENROUTER_API_KEY",
    base_url_env: None,
    request_id_header: None,
    default_max_tokens: None,
    quirks: Quirks {
        emits_complete_single_chunk_tool_calls: false,
        supports_tools: true,
        supports_response_format: true,
        stream_include_usage: true,
        supports_image_tool_results: false,
        string_content_parts: false,
        modern_output_cap_models: &[],
    },
};

pub const VENICE: Dialect = Dialect {
    name: "venice",
    base_url: "https://api.venice.ai/api/v1",
    api_key_env: "VENICE_API_KEY",
    base_url_env: None,
    request_id_header: None,
    default_max_tokens: None,
    quirks: Quirks {
        emits_complete_single_chunk_tool_calls: false,
        supports_tools: true,
        supports_response_format: true,
        stream_include_usage: true,
        supports_image_tool_results: false,
        string_content_parts: false,
        modern_output_cap_models: &[],
    },
};

pub const DOUBLEWORD: Dialect = Dialect {
    name: "doubleword",
    base_url: "https://api.doubleword.ai/v1",
    api_key_env: "DOUBLEWORD_API_KEY",
    base_url_env: None,
    request_id_header: None,
    default_max_tokens: None,
    quirks: Quirks {
        emits_complete_single_chunk_tool_calls: false,
        supports_tools: true,
        supports_response_format: true,
        stream_include_usage: true,
        supports_image_tool_results: false,
        string_content_parts: false,
        modern_output_cap_models: &[],
    },
};

pub const MISTRALRS: Dialect = Dialect {
    name: "mistralrs",
    base_url: "http://localhost:1234/v1",
    api_key_env: "MISTRALRS_API_KEY",
    base_url_env: Some("MISTRALRS_BASE_URL"),
    request_id_header: None,
    default_max_tokens: None,
    quirks: Quirks {
        emits_complete_single_chunk_tool_calls: false,
        supports_tools: true,
        supports_response_format: true,
        stream_include_usage: true,
        supports_image_tool_results: false,
        string_content_parts: false,
        modern_output_cap_models: &[],
    },
};

pub const XAI: Dialect = Dialect {
    name: "xai",
    base_url: "https://api.x.ai/v1",
    api_key_env: "XAI_API_KEY",
    base_url_env: None,
    request_id_header: None,
    default_max_tokens: None,
    quirks: Quirks {
        emits_complete_single_chunk_tool_calls: false,
        supports_tools: true,
        supports_response_format: true,
        stream_include_usage: true,
        supports_image_tool_results: false,
        string_content_parts: false,
        modern_output_cap_models: &[],
    },
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpenAI {
    pub api_key: Secret,
    pub base_url: String,
    pub dialect: Dialect,
    #[serde(skip)]
    pub system_instructions_placement: SystemInstructionsPlacement,
}

impl OpenAI {
    pub fn new(api_key: impl Into<Secret>) -> Self {
        Self::from_dialect(OPENAI, api_key)
    }

    pub fn from_dialect(dialect: Dialect, api_key: impl Into<Secret>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: dialect.base_url.to_owned(),
            dialect,
            system_instructions_placement: SystemInstructionsPlacement::default(),
        }
    }

    pub fn from_env() -> Result<Self, crate::client::ProviderClientError> {
        Self::from_env_with(&OPENAI)
    }

    pub fn from_env_with(dialect: &Dialect) -> Result<Self, crate::client::ProviderClientError> {
        let api_key = crate::client::required_env_var(dialect.api_key_env)?;
        let base_url = if let Some(base_url_env) = dialect.base_url_env {
            crate::client::optional_env_var(base_url_env)?
                .unwrap_or_else(|| dialect.base_url.to_owned())
        } else {
            dialect.base_url.to_owned()
        };
        Ok(Self {
            api_key: Secret::new(api_key),
            base_url,
            dialect: dialect.clone(),
            system_instructions_placement: SystemInstructionsPlacement::default(),
        })
    }

    pub fn chat(&self, model: impl Into<String>) -> Chat {
        Chat::new(self.clone(), model)
    }

    pub fn responses(&self, model: impl Into<String>) -> Responses {
        Responses::new(self.clone(), model)
    }

    pub fn embeddings(&self, model: impl Into<String>) -> Embeddings {
        Embeddings::new(self.clone(), model)
    }

    pub fn transcriptions(&self, model: impl Into<String>) -> Transcriptions {
        Transcriptions::new(self.clone(), model)
    }

    #[cfg(feature = "image")]
    pub fn images(&self, model: impl Into<String>) -> Images {
        Images::new(self.clone(), model)
    }

    #[cfg(feature = "audio")]
    pub fn speech(&self, model: impl Into<String>) -> Speech {
        Speech::new(self.clone(), model)
    }
    pub fn models(&self) -> Models {
        Models::new(self.clone())
    }

    pub fn verify(&self) -> Verify {
        Verify::new(self.clone())
    }

    pub fn apply_headers(&self, headers: &mut HeaderMap) {
        if let Ok(val) = HeaderValue::from_str(&format!("Bearer {}", self.api_key.expose_secret()))
        {
            headers.insert(http::header::AUTHORIZATION, val);
        }
    }
}

impl DriverHasCompletion for OpenAI {
    type Wire = Responses;
    fn completion(&self, model: &str) -> Responses {
        self.responses(model)
    }
}

pub type OpenAiCompatible = OpenAI;
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Chat {
    pub provider: OpenAI,
    pub model: String,
    pub strict_tools: bool,
    pub tool_result_array_content: bool,
    pub prompt_caching: bool,
}

impl Chat {
    pub fn new(provider: OpenAI, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            strict_tools: false,
            tool_result_array_content: false,
            prompt_caching: false,
        }
    }
}

impl Wire for Chat {
    type Op = Completion;
    type Decoder = ChatDecoder;

    fn name(&self) -> &str {
        self.provider.dialect.name
    }

    fn capabilities(&self) -> crate::completion::ProviderCapabilities {
        crate::completion::ProviderCapabilities::default().with_native_output_tool_composition(
            self.provider.dialect.quirks.supports_response_format,
        )
    }

    fn encode(
        &self,
        request: crate::completion::CompletionRequest,
    ) -> Result<Encoded, crate::completion::CompletionError> {
        let options = super::completion::CompletionModelOptions {
            strict_tools: self.strict_tools,
            tool_result_array_content: self.tool_result_array_content,
            prompt_caching: self.prompt_caching,
        };
        let req = super::completion::CompletionRequest::try_from(
            super::completion::OpenAIRequestParams {
                model: self.model.clone(),
                request,
                strict_tools: options.strict_tools,
                tool_result_array_content: options.tool_result_array_content,
                supports_response_format: self.provider.dialect.quirks.supports_response_format,
                supports_tools: self.provider.dialect.quirks.supports_tools,
                supports_image_tool_results: self
                    .provider
                    .dialect
                    .quirks
                    .supports_image_tool_results,
            },
        )?;

        let modern_output_cap = self
            .provider
            .dialect
            .quirks
            .requires_modern_output_cap(&req.model);
        let mut request_body = super::completion::request_body(&req, modern_output_cap)?;

        if self.provider.dialect.quirks.string_content_parts
            && let Some(map) = request_body.as_object_mut()
            && let Some(messages) = map
                .get_mut("messages")
                .and_then(serde_json::Value::as_array_mut)
        {
            for message in messages {
                if let Some(message) = message.as_object_mut() {
                    let is_assistant = message.get("role").and_then(serde_json::Value::as_str)
                        == Some("assistant");
                    if let Some(content) = message.get_mut("content") {
                        let separator = if is_assistant { "" } else { "\n" };
                        super::completion::flatten_text_content_parts(content, separator, true);
                    }
                }
            }
        }

        let url = format!(
            "{}/chat/completions",
            self.provider.base_url.trim_end_matches('/')
        );
        let mut http_req = http::Request::builder()
            .method(http::Method::POST)
            .uri(&url)
            .body(Body::Bytes(serde_json::to_vec(&request_body)?))
            .map_err(|e| crate::completion::CompletionError::RequestError(Box::new(e)))?;

        self.provider.apply_headers(http_req.headers_mut());
        http_req.headers_mut().insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );

        let mut encoded = Encoded::new(http_req, Framing::Sse, "/chat/completions");
        if let Some(h) = self.provider.dialect.request_id_header {
            encoded = encoded.with_request_id_header(h);
        }
        Ok(encoded)
    }

    fn decoder(&self) -> Self::Decoder {
        ChatDecoder::new(self.provider.clone(), self.model.clone())
    }
}

pub struct ChatDecoder {
    inner: crate::providers::internal::openai_chat_completions_compatible::CompatAdapter<
        super::completion::streaming::OpenAICompatibleProfile<
            OpenAICompletions,
            super::completion::Usage,
        >,
    >,
}

impl ChatDecoder {
    pub fn new(provider: OpenAI, _model: String) -> Self {
        let profile = super::completion::streaming::OpenAICompatibleProfile::new(
            OpenAICompletions::default(),
        );
        let inner =
            crate::providers::internal::openai_chat_completions_compatible::CompatAdapter::new(
                profile,
                provider.dialect.name.to_string(),
            );
        Self { inner }
    }
}

#[derive(Debug)]
pub struct ChatEvent(
    crate::providers::internal::openai_chat_completions_compatible::CompatEvent<
        super::completion::Usage,
        serde_json::Value,
    >,
);

impl Decoder<Completion> for ChatDecoder {
    type Event = ChatEvent;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        self.inner.classify(frame).map(ChatEvent)
    }

    fn interpret(&mut self, event: Self::Event, out: &mut crate::wire::Output<Completion>) {
        self.inner.interpret(event.0, out);
    }

    fn finish(&mut self, out: &mut crate::wire::Output<Completion>) {
        self.inner.finish(out);
    }

    fn flush_before_terminal_error(&mut self, out: &mut crate::wire::Output<Completion>) {
        self.inner.flush_before_terminal_error(out);
    }

    fn project(&self, payload: &[u8], sink: &mut dyn crate::wire::ObservationSink) {
        self.inner.project(payload, sink);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Responses {
    pub provider: OpenAI,
    pub model: String,
    #[serde(skip)]
    pub system_instructions_placement: SystemInstructionsPlacement,
}

impl Responses {
    pub fn new(provider: OpenAI, model: impl Into<String>) -> Self {
        Self {
            system_instructions_placement: provider.system_instructions_placement,
            provider,
            model: model.into(),
        }
    }
}

impl Wire for Responses {
    type Op = Completion;
    type Decoder = ResponsesDecoder;

    fn name(&self) -> &str {
        self.provider.dialect.name
    }

    fn capabilities(&self) -> crate::completion::ProviderCapabilities {
        crate::completion::ProviderCapabilities::default().with_native_output_tool_composition(true)
    }

    fn encode(
        &self,
        request: crate::completion::CompletionRequest,
    ) -> Result<Encoded, crate::completion::CompletionError> {
        let req = super::responses_api::CompletionRequest::try_from(
            super::responses_api::ResponsesRequestParams {
                model: self.model.clone(),
                request,
                system_instructions_placement: self.system_instructions_placement,
            },
        )?;

        let url = format!("{}/responses", self.provider.base_url.trim_end_matches('/'));
        let mut http_req = http::Request::builder()
            .method(http::Method::POST)
            .uri(&url)
            .body(Body::Bytes(serde_json::to_vec(&req)?))
            .map_err(|e| crate::completion::CompletionError::RequestError(Box::new(e)))?;

        self.provider.apply_headers(http_req.headers_mut());
        http_req.headers_mut().insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );

        let mut encoded = Encoded::new(http_req, Framing::Sse, "/responses");
        if let Some(h) = self.provider.dialect.request_id_header {
            encoded = encoded.with_request_id_header(h);
        }
        Ok(encoded)
    }

    fn decoder(&self) -> Self::Decoder {
        ResponsesDecoder::new(self.provider.dialect.name)
    }
}

pub struct ResponsesDecoder {
    inner: super::responses_api::streaming::ResponsesAdapter,
}

impl ResponsesDecoder {
    pub fn new(provider: &str) -> Self {
        Self {
            inner: super::responses_api::streaming::ResponsesAdapter::live(
                provider,
                super::responses_api::streaming::ResponsesStreamOptions::strict(),
            ),
        }
    }
}

#[derive(Debug)]
pub struct ResponsesEvent(super::responses_api::streaming::ResponsesFrameEvent);

impl Decoder<Completion> for ResponsesDecoder {
    type Event = ResponsesEvent;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        self.inner.classify(frame).map(ResponsesEvent)
    }

    fn interpret(&mut self, event: Self::Event, out: &mut crate::wire::Output<Completion>) {
        self.inner.interpret(event.0, out);
    }

    fn finish(&mut self, out: &mut crate::wire::Output<Completion>) {
        self.inner.finish(out);
    }

    fn flush_before_terminal_error(&mut self, out: &mut crate::wire::Output<Completion>) {
        self.inner.flush_before_terminal_error(out);
    }

    fn project(&self, payload: &[u8], sink: &mut dyn crate::wire::ObservationSink) {
        self.inner.project(payload, sink);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Embeddings {
    pub provider: OpenAI,
    pub model: String,
    pub ndims: Option<usize>,
}

impl Embeddings {
    pub fn new(provider: OpenAI, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            ndims: None,
        }
    }

    pub fn with_ndims(mut self, ndims: usize) -> Self {
        self.ndims = Some(ndims);
        self
    }
}

impl Wire for Embeddings {
    type Op = EmbeddingOp;
    type Decoder = OpenAiEmbeddingDecoder;

    fn name(&self) -> &str {
        self.provider.dialect.name
    }

    fn capabilities(&self) {}

    fn encode(&self, request: Vec<String>) -> Result<Encoded, crate::embeddings::EmbeddingError> {
        let mut body = serde_json::json!({
            "model": self.model,
            "input": request,
        });
        if let Some(ndims) = self.ndims
            && let Some(map) = body.as_object_mut()
        {
            map.insert("dimensions".to_string(), serde_json::json!(ndims));
        }

        let url = format!(
            "{}/embeddings",
            self.provider.base_url.trim_end_matches('/')
        );
        let mut req = http::Request::builder()
            .method(http::Method::POST)
            .uri(&url)
            .body(Body::Bytes(serde_json::to_vec(&body)?))
            .map_err(|e| crate::embeddings::EmbeddingError::HttpError(e.into()))?;

        self.provider.apply_headers(req.headers_mut());
        req.headers_mut().insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );

        let mut encoded = Encoded::new(req, Framing::Whole, "/embeddings");
        if let Some(h) = self.provider.dialect.request_id_header {
            encoded = encoded.with_request_id_header(h);
        }
        Ok(encoded)
    }

    fn decoder(&self) -> Self::Decoder {
        OpenAiEmbeddingDecoder {
            provider: self.provider.dialect.name.to_string(),
        }
    }
}

pub struct OpenAiEmbeddingDecoder {
    provider: String,
}

impl Decoder<EmbeddingOp> for OpenAiEmbeddingDecoder {
    type Event = crate::embeddings::EmbeddingResponse;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        let body = frame.as_str();
        crate::providers::internal::wire::classify_unary_frame::<
            super::embedding::CompatibleEmbeddingResponse,
        >(&body)
        .map(|resp| {
            use crate::embeddings::NormalizeEmbeddingResponse;
            let documents: Vec<String> = resp.data.iter().map(|_| String::new()).collect();
            resp.normalize(&self.provider, documents)
                .unwrap_or_else(|_| crate::embeddings::EmbeddingResponse {
                    embeddings: Vec::new(),
                    usage: crate::completion::Usage::default(),
                    provider: self.provider.clone(),
                    model: None,
                    response_id: None,
                    provider_request_id: None,
                    raw: serde_json::Value::Null,
                })
        })
    }

    fn interpret(&mut self, event: Self::Event, out: &mut crate::wire::Output<EmbeddingOp>) {
        out.emit(event);
    }

    fn finish(&mut self, _out: &mut crate::wire::Output<EmbeddingOp>) {}

    fn flush_before_terminal_error(&mut self, _out: &mut crate::wire::Output<EmbeddingOp>) {}

    fn project(&self, _payload: &[u8], _sink: &mut dyn crate::wire::ObservationSink) {}
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Transcriptions {
    pub provider: OpenAI,
    pub model: String,
}

impl Transcriptions {
    pub fn new(provider: OpenAI, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
        }
    }
}

impl Wire for Transcriptions {
    type Op = TranscriptionOp;
    type Decoder = OpenAiTranscriptionDecoder;

    fn name(&self) -> &str {
        self.provider.dialect.name
    }

    fn capabilities(&self) {}

    fn encode(
        &self,
        request: crate::transcription::TranscriptionRequest,
    ) -> Result<Encoded, crate::transcription::TranscriptionError> {
        let form = crate::http_client::MultipartForm::new()
            .text("model", self.model.clone())
            .file(
                "file",
                request.filename,
                mime::APPLICATION_OCTET_STREAM,
                request.data,
            );

        let url = format!(
            "{}/audio/transcriptions",
            self.provider.base_url.trim_end_matches('/')
        );
        let mut req = http::Request::builder()
            .method(http::Method::POST)
            .uri(&url)
            .body(Body::Multipart(form))
            .map_err(|e| {
                crate::transcription::TranscriptionError::from(crate::http_client::Error::from(e))
            })?;

        self.provider.apply_headers(req.headers_mut());
        let mut encoded = Encoded::new(req, Framing::Whole, "/audio/transcriptions");
        if let Some(h) = self.provider.dialect.request_id_header {
            encoded = encoded.with_request_id_header(h);
        }
        Ok(encoded)
    }

    fn decoder(&self) -> Self::Decoder {
        OpenAiTranscriptionDecoder {
            provider: self.provider.dialect.name.to_string(),
        }
    }
}

pub struct OpenAiTranscriptionDecoder {
    provider: String,
}

impl Decoder<TranscriptionOp> for OpenAiTranscriptionDecoder {
    type Event = crate::transcription::TranscriptionResponse;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        let body = frame.as_str();
        crate::providers::internal::wire::classify_unary_frame::<
            super::transcription::TranscriptionResponse,
        >(&body)
        .map(|resp| {
            use crate::transcription::NormalizeTranscriptionResponse;
            resp.normalize(&self.provider).unwrap_or_else(|_| {
                crate::transcription::TranscriptionResponse {
                    text: String::new(),
                    usage: crate::completion::Usage::default(),
                    provider: self.provider.clone(),
                    model: None,
                    response_id: None,
                    provider_request_id: None,
                    raw: serde_json::Value::Null,
                }
            })
        })
    }

    fn interpret(&mut self, event: Self::Event, out: &mut crate::wire::Output<TranscriptionOp>) {
        out.emit(event);
    }

    fn finish(&mut self, _out: &mut crate::wire::Output<TranscriptionOp>) {}

    fn flush_before_terminal_error(&mut self, _out: &mut crate::wire::Output<TranscriptionOp>) {}

    fn project(&self, _payload: &[u8], _sink: &mut dyn crate::wire::ObservationSink) {}
}

#[cfg(feature = "image")]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Images {
    pub provider: OpenAI,
    pub model: String,
}

#[cfg(feature = "image")]
impl Images {
    pub fn new(provider: OpenAI, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
        }
    }
}

#[cfg(feature = "image")]
impl Wire for Images {
    type Op = ImageGenOp;
    type Decoder = OpenAiImageDecoder;

    fn name(&self) -> &str {
        self.provider.dialect.name
    }

    fn capabilities(&self) {}

    fn encode(
        &self,
        request: crate::image_generation::ImageGenerationRequest,
    ) -> Result<Encoded, crate::image_generation::ImageGenerationError> {
        let body = serde_json::json!({
            "model": self.model,
            "prompt": request.prompt,
        });

        let url = format!(
            "{}/images/generations",
            self.provider.base_url.trim_end_matches('/')
        );
        let mut req = http::Request::builder()
            .method(http::Method::POST)
            .uri(&url)
            .body(Body::Bytes(serde_json::to_vec(&body)?))
            .map_err(|e| {
                crate::image_generation::ImageGenerationError::from(
                    crate::http_client::Error::from(e),
                )
            })?;

        self.provider.apply_headers(req.headers_mut());
        req.headers_mut().insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );

        let mut encoded = Encoded::new(req, Framing::Whole, "/images/generations");
        if let Some(h) = self.provider.dialect.request_id_header {
            encoded = encoded.with_request_id_header(h);
        }
        Ok(encoded)
    }

    fn decoder(&self) -> Self::Decoder {
        OpenAiImageDecoder {
            provider: self.provider.dialect.name.to_string(),
        }
    }
}

#[cfg(feature = "image")]
pub struct OpenAiImageDecoder {
    provider: String,
}

#[cfg(feature = "image")]
impl Decoder<ImageGenOp> for OpenAiImageDecoder {
    type Event = crate::image_generation::ImageGenerationResponse;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        let body = frame.as_str();
        crate::providers::internal::wire::classify_unary_frame::<
            super::image_generation::ImageGenerationResponse,
        >(&body)
        .map(|resp| {
            use crate::image_generation::NormalizeImageGenerationResponse;
            resp.normalize(&self.provider).unwrap_or_else(|_| {
                crate::image_generation::ImageGenerationResponse::new(Vec::new(), &self.provider)
            })
        })
    }

    fn interpret(&mut self, event: Self::Event, out: &mut crate::wire::Output<ImageGenOp>) {
        out.emit(event);
    }

    fn finish(&mut self, _out: &mut crate::wire::Output<ImageGenOp>) {}

    fn flush_before_terminal_error(&mut self, _out: &mut crate::wire::Output<ImageGenOp>) {}

    fn project(&self, _payload: &[u8], _sink: &mut dyn crate::wire::ObservationSink) {}
}

#[cfg(feature = "audio")]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Speech {
    pub provider: OpenAI,
    pub model: String,
}

#[cfg(feature = "audio")]
impl Speech {
    pub fn new(provider: OpenAI, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
        }
    }
}

#[cfg(feature = "audio")]
impl Wire for Speech {
    type Op = AudioGenOp;
    type Decoder = OpenAiSpeechDecoder;

    fn name(&self) -> &str {
        self.provider.dialect.name
    }

    fn capabilities(&self) {}

    fn encode(
        &self,
        request: crate::audio_generation::AudioGenerationRequest,
    ) -> Result<Encoded, crate::audio_generation::AudioGenerationError> {
        let voice = if request.voice.is_empty() {
            "alloy".to_string()
        } else {
            request.voice
        };
        let body = serde_json::json!({
            "model": self.model,
            "input": request.text,
            "voice": voice,
        });

        let url = format!(
            "{}/audio/speech",
            self.provider.base_url.trim_end_matches('/')
        );
        let mut req = http::Request::builder()
            .method(http::Method::POST)
            .uri(&url)
            .body(Body::Bytes(serde_json::to_vec(&body)?))
            .map_err(|e| {
                crate::audio_generation::AudioGenerationError::from(
                    crate::http_client::Error::from(e),
                )
            })?;

        self.provider.apply_headers(req.headers_mut());
        req.headers_mut().insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );

        let mut encoded = Encoded::new(req, Framing::Whole, "/audio/speech");
        if let Some(h) = self.provider.dialect.request_id_header {
            encoded = encoded.with_request_id_header(h);
        }
        Ok(encoded)
    }

    fn decoder(&self) -> Self::Decoder {
        OpenAiSpeechDecoder
    }
}

#[cfg(feature = "audio")]
pub struct OpenAiSpeechDecoder;

#[cfg(feature = "audio")]
impl Decoder<AudioGenOp> for OpenAiSpeechDecoder {
    type Event = crate::audio_generation::AudioGenerationResponse;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        let bytes = match frame {
            WireFrame::Bytes(b) => b,
            WireFrame::Text(s) => s.into_bytes(),
        };
        WireEvent::Known(crate::audio_generation::AudioGenerationResponse::new(
            bytes, "openai",
        ))
    }

    fn interpret(&mut self, event: Self::Event, out: &mut crate::wire::Output<AudioGenOp>) {
        out.emit(event);
    }

    fn finish(&mut self, _out: &mut crate::wire::Output<AudioGenOp>) {}

    fn flush_before_terminal_error(&mut self, _out: &mut crate::wire::Output<AudioGenOp>) {}

    fn project(&self, _payload: &[u8], _sink: &mut dyn crate::wire::ObservationSink) {}
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Models {
    pub provider: OpenAI,
}

impl Models {
    pub fn new(provider: OpenAI) -> Self {
        Self { provider }
    }
}

impl Wire for Models {
    type Op = ModelListingOp;
    type Decoder = OpenAiModelsDecoder;

    fn name(&self) -> &str {
        self.provider.dialect.name
    }

    fn capabilities(&self) {}

    fn encode(&self, _request: ()) -> Result<Encoded, crate::model::ModelListingError> {
        let url = format!("{}/models", self.provider.base_url.trim_end_matches('/'));
        let mut req = http::Request::builder()
            .method(http::Method::GET)
            .uri(&url)
            .body(Body::Bytes(Vec::new()))
            .map_err(|e| crate::model::ModelListingError::request_error(e.to_string()))?;

        self.provider.apply_headers(req.headers_mut());
        req.headers_mut().insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );
        req.headers_mut()
            .insert(http::header::ACCEPT, http::HeaderValue::from_static("*/*"));

        Ok(Encoded::new(req, Framing::Whole, "/models"))
    }

    fn decoder(&self) -> Self::Decoder {
        OpenAiModelsDecoder::new(self.provider.dialect.name)
    }
}

pub struct OpenAiModelsDecoder {
    pub _provider: String,
}

impl OpenAiModelsDecoder {
    pub fn new(provider: impl Into<String>) -> Self {
        Self {
            _provider: provider.into(),
        }
    }
}

impl Decoder<ModelListingOp> for OpenAiModelsDecoder {
    type Event = Vec<crate::model::Model>;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        let body = frame.as_str();
        #[derive(Deserialize)]
        struct ListModelsResponse {
            data: Vec<crate::providers::internal::model_listing::ListModelEntry>,
        }
        crate::providers::internal::wire::classify_unary_frame::<ListModelsResponse>(&body).map(
            |resp| {
                resp.data
                    .into_iter()
                    .map(crate::model::Model::from)
                    .collect()
            },
        )
    }

    fn interpret(&mut self, event: Self::Event, out: &mut crate::wire::Output<ModelListingOp>) {
        out.emit(event);
    }

    fn finish(&mut self, _out: &mut crate::wire::Output<ModelListingOp>) {}

    fn flush_before_terminal_error(&mut self, _out: &mut crate::wire::Output<ModelListingOp>) {}

    fn project(&self, _payload: &[u8], _sink: &mut dyn crate::wire::ObservationSink) {}
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Verify {
    pub provider: OpenAI,
}

impl Verify {
    pub fn new(provider: OpenAI) -> Self {
        Self { provider }
    }
}

pub struct VerifyDecoder;

impl Decoder<VerifyOp> for VerifyDecoder {
    type Event = ();

    fn classify(&self, _frame: WireFrame) -> WireEvent<Self::Event> {
        WireEvent::Known(())
    }

    fn interpret(&mut self, _event: Self::Event, out: &mut crate::wire::Output<VerifyOp>) {
        out.emit(());
    }

    fn finish(&mut self, _out: &mut crate::wire::Output<VerifyOp>) {}

    fn flush_before_terminal_error(&mut self, _out: &mut crate::wire::Output<VerifyOp>) {}

    fn project(&self, _payload: &[u8], _sink: &mut dyn crate::wire::ObservationSink) {}
}

impl Wire for Verify {
    type Op = VerifyOp;
    type Decoder = VerifyDecoder;

    fn name(&self) -> &str {
        self.provider.dialect.name
    }

    fn capabilities(&self) {}

    fn encode(&self, _request: ()) -> Result<Encoded, crate::client::verify::VerifyError> {
        let url = format!("{}/models", self.provider.base_url.trim_end_matches('/'));
        let mut req = http::Request::builder()
            .method(http::Method::GET)
            .uri(&url)
            .body(Body::Bytes(Vec::new()))
            .map_err(|e| {
                crate::client::verify::VerifyError::HttpError(crate::http_client::Error::from(e))
            })?;

        self.provider.apply_headers(req.headers_mut());
        req.headers_mut().insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );
        req.headers_mut()
            .insert(http::header::ACCEPT, http::HeaderValue::from_static("*/*"));

        Ok(Encoded::new(req, Framing::Whole, "/models"))
    }

    fn decoder(&self) -> Self::Decoder {
        VerifyDecoder
    }
}

impl Bound<OpenAI, crate::http_client::BoxedHttpClient> {
    pub fn new_with<H>(
        api_key: impl Into<Secret>,
        http: H,
    ) -> Result<Bound<OpenAI, H>, crate::client::ProviderClientError> {
        Ok(Bound::new(OpenAI::new(api_key), http))
    }
}

impl<H> Bound<OpenAI, H>
where
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    pub fn chat(&self, model: impl Into<String>) -> Bound<Chat, H> {
        Bound::new(self.wire.chat(model), self.http.clone())
    }

    pub fn responses(&self, model: impl Into<String>) -> Bound<Responses, H> {
        Bound::new(self.wire.responses(model), self.http.clone())
    }
    pub fn chat_model(&self, model: impl Into<String>) -> Bound<Chat, H> {
        self.chat(model)
    }

    pub fn completions_api(self) -> Self {
        self
    }

    pub fn responses_api(self) -> Self {
        self
    }

    pub fn embeddings(&self, model: impl Into<String>) -> Bound<Embeddings, H> {
        Bound::new(self.wire.embeddings(model), self.http.clone())
    }

    pub fn embedding_model(
        &self,
        model: impl Into<String>,
        ndims: Option<usize>,
    ) -> Bound<Embeddings, H> {
        let mut w = self.wire.embeddings(model);
        w.ndims = ndims;
        Bound::new(w, self.http.clone())
    }

    pub fn transcriptions(&self, model: impl Into<String>) -> Bound<Transcriptions, H> {
        Bound::new(self.wire.transcriptions(model), self.http.clone())
    }

    pub fn transcription_model(&self, model: impl Into<String>) -> Bound<Transcriptions, H> {
        self.transcriptions(model)
    }

    #[cfg(feature = "image")]
    pub fn images(&self, model: impl Into<String>) -> Bound<Images, H> {
        Bound::new(self.wire.images(model), self.http.clone())
    }

    #[cfg(feature = "image")]
    pub fn image_generation_model(&self, model: impl Into<String>) -> Bound<Images, H> {
        self.images(model)
    }

    #[cfg(feature = "audio")]
    pub fn speech(&self, model: impl Into<String>) -> Bound<Speech, H> {
        Bound::new(self.wire.speech(model), self.http.clone())
    }

    #[cfg(feature = "audio")]
    pub fn audio_generation_model(&self, model: impl Into<String>) -> Bound<Speech, H> {
        self.speech(model)
    }

    pub fn models(&self) -> Bound<Models, H> {
        Bound::new(self.wire.models(), self.http.clone())
    }

    pub fn model_lister(&self) -> Bound<Models, H> {
        self.models()
    }

    pub fn verify(&self) -> Bound<Verify, H> {
        Bound::new(self.wire.verify(), self.http.clone())
    }

    pub fn base_url(&self) -> &str {
        &self.wire.base_url
    }

    pub fn http_client(&self) -> &H {
        &self.http
    }

    pub fn headers(&self) -> http::HeaderMap {
        let mut map = http::HeaderMap::new();
        self.wire.apply_headers(&mut map);
        map
    }
}
impl<H> crate::client::ModelListingClient for Bound<OpenAI, H>
where
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    fn list_models(
        &self,
    ) -> impl std::future::Future<
        Output = Result<crate::model::ModelList, crate::model::ModelListingError>,
    > + WasmCompatSend {
        let lister = self.models();
        async move {
            use crate::client::ModelLister;
            lister.list_all().await
        }
    }
}

impl<H> Bound<Chat, H>
where
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    pub fn with_strict_tools(mut self) -> Self {
        self.wire.strict_tools = true;
        self
    }

    pub fn with_tool_result_array_content(mut self) -> Self {
        self.wire.tool_result_array_content = true;
        self
    }

    pub fn with_prompt_caching(mut self) -> Self {
        self.wire.prompt_caching = true;
        self
    }
}

impl<H> Bound<Responses, H>
where
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    pub fn with_system_instructions_placement(
        mut self,
        placement: SystemInstructionsPlacement,
    ) -> Self {
        self.wire.system_instructions_placement = placement;
        self
    }

    pub fn with_system_instructions_as_messages(self) -> Self {
        self.with_system_instructions_placement(SystemInstructionsPlacement::InputSystemMessages)
    }
}
// ================================================================
// OpenAI Responses API Extension
// ================================================================
#[derive(Debug, Default, Clone, Copy)]
pub struct OpenAIResponses {
    pub(crate) system_instructions_placement: SystemInstructionsPlacement,
}

// ================================================================
// OpenAI Completions API Extension
// ================================================================
#[derive(Debug, Default, Clone, Copy)]
pub struct OpenAICompletions {
    /// Carried through API switches so that a placement configured on a
    /// Responses client survives `completions_api()` → `responses_api()`
    /// round trips. Not used by Chat Completions requests themselves.
    pub(crate) system_instructions_placement: SystemInstructionsPlacement,
}

type OpenAIApiKey = BearerAuth;

// Responses API client (default)
pub type Client<H = crate::http_client::BoxedHttpClient> = client::Client<OpenAIResponses, H>;
pub type ClientBuilder<H = crate::markers::Missing> = client::ClientBuilder<OpenAIResponses, H>;

// Completions API client
pub type CompletionsClient<H = crate::http_client::BoxedHttpClient> =
    client::Client<OpenAICompletions, H>;
pub type CompletionsClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<OpenAICompletions, H>;

impl Provider for OpenAIResponses {
    const NAME: &'static str = "openai";
    const BASE_URL: &'static str = OPENAI_API_BASE_URL;
    const VERIFY_PATH: &'static str = "/models";
    type ApiKey = OpenAIApiKey;
    type Config = ();
    type EnvInput = OpenAIApiKey;

    fn build(_: (), _: &OpenAIApiKey) -> http_client::Result<Self> {
        Ok(OpenAIResponses::default())
    }

    fn from_env<H: HttpClientExt>(http: H) -> ProviderClientResult<Client<H>> {
        Client::from_env_api_key("OPENAI_API_KEY", Some("OPENAI_BASE_URL"), http)
    }

    fn from_val<H: HttpClientExt>(input: OpenAIApiKey, http: H) -> ProviderClientResult<Client<H>> {
        Client::new_with(input, http)
    }
}

impl HasCompletion for OpenAIResponses {
    type Model<H>
        = super::responses_api::ResponsesCompletionModel<H>
    where
        H: ModelTransport;

    fn completion_model<H: ModelTransport>(client: &Client<H>, model: String) -> Self::Model<H> {
        super::responses_api::ResponsesCompletionModel::new(client.clone(), model)
    }
}

impl HasEmbeddings for OpenAIResponses {
    type Model<H>
        = super::EmbeddingModel<H>
    where
        H: ModelTransport;

    fn embedding_model<H: ModelTransport>(
        client: &Client<H>,
        model: String,
        ndims: Option<usize>,
    ) -> Self::Model<H> {
        super::EmbeddingModel::make(client, model, ndims)
    }
}

impl HasTranscription for OpenAIResponses {
    type Model<H>
        = super::TranscriptionModel<H>
    where
        H: ModelTransport;

    fn transcription_model<H: ModelTransport>(client: &Client<H>, model: String) -> Self::Model<H> {
        super::TranscriptionModel::new(client.clone(), model)
    }
}

impl HasModelListing for OpenAIResponses {
    type Lister<H>
        = super::OpenAIModelLister<H>
    where
        H: ModelTransport;

    fn model_lister<H: ModelTransport>(client: &Client<H>) -> Self::Lister<H> {
        super::OpenAIModelLister::new(client.clone())
    }
}

#[cfg(feature = "image")]
impl HasImageGeneration for OpenAIResponses {
    type Model<H>
        = super::ImageGenerationModel<H>
    where
        H: ModelTransport;

    fn image_generation_model<H: ModelTransport>(
        client: &Client<H>,
        model: String,
    ) -> Self::Model<H> {
        super::ImageGenerationModel::new(client.clone(), model)
    }
}

#[cfg(feature = "audio")]
impl HasAudioGeneration for OpenAIResponses {
    type Model<H>
        = super::audio_generation::AudioGenerationModel<H>
    where
        H: ModelTransport;

    fn audio_generation_model<H: ModelTransport>(
        client: &Client<H>,
        model: String,
    ) -> Self::Model<H> {
        super::audio_generation::AudioGenerationModel::new(client.clone(), model)
    }
}

impl ResponsesProviderExt for OpenAIResponses {
    fn system_instructions_placement(&self) -> SystemInstructionsPlacement {
        self.system_instructions_placement
    }
}

impl ConfigurableSystemInstructionsPlacement for OpenAIResponses {}

impl Provider for OpenAICompletions {
    const NAME: &'static str = "openai";
    const BASE_URL: &'static str = OPENAI_API_BASE_URL;
    const VERIFY_PATH: &'static str = "/models";
    type ApiKey = OpenAIApiKey;
    type Config = ();
    type EnvInput = OpenAIApiKey;

    fn build(_: (), _: &OpenAIApiKey) -> http_client::Result<Self> {
        Ok(OpenAICompletions::default())
    }

    fn from_env<H: HttpClientExt>(http: H) -> ProviderClientResult<CompletionsClient<H>> {
        CompletionsClient::from_env_api_key("OPENAI_API_KEY", Some("OPENAI_BASE_URL"), http)
    }

    fn from_val<H: HttpClientExt>(
        input: OpenAIApiKey,
        http: H,
    ) -> ProviderClientResult<CompletionsClient<H>> {
        CompletionsClient::new_with(input, http)
    }
}

impl HasCompletion for OpenAICompletions {
    type Model<H>
        = super::completion::CompletionModel<H>
    where
        H: ModelTransport;

    fn completion_model<H: ModelTransport>(
        client: &CompletionsClient<H>,
        model: String,
    ) -> Self::Model<H> {
        super::completion::CompletionModel::new(client.clone(), model)
    }
}

impl HasEmbeddings for OpenAICompletions {
    type Model<H>
        = super::GenericEmbeddingModel<OpenAICompletions, H>
    where
        H: ModelTransport;

    fn embedding_model<H: ModelTransport>(
        client: &CompletionsClient<H>,
        model: String,
        ndims: Option<usize>,
    ) -> Self::Model<H> {
        super::GenericEmbeddingModel::make(client, model, ndims)
    }
}

impl HasTranscription for OpenAICompletions {
    type Model<H>
        = super::CompletionsTranscriptionModel<H>
    where
        H: ModelTransport;

    fn transcription_model<H: ModelTransport>(
        client: &CompletionsClient<H>,
        model: String,
    ) -> Self::Model<H> {
        super::CompletionsTranscriptionModel::new(client.clone(), model)
    }
}

impl HasModelListing for OpenAICompletions {
    type Lister<H>
        = super::OpenAICompletionsModelLister<H>
    where
        H: ModelTransport;

    fn model_lister<H: ModelTransport>(client: &CompletionsClient<H>) -> Self::Lister<H> {
        super::OpenAICompletionsModelLister::new(client.clone())
    }
}

#[cfg(feature = "image")]
impl HasImageGeneration for OpenAICompletions {
    type Model<H>
        = super::CompletionsImageGenerationModel<H>
    where
        H: ModelTransport;

    fn image_generation_model<H: ModelTransport>(
        client: &CompletionsClient<H>,
        model: String,
    ) -> Self::Model<H> {
        super::CompletionsImageGenerationModel::new(client.clone(), model)
    }
}

#[cfg(feature = "audio")]
impl HasAudioGeneration for OpenAICompletions {
    type Model<H>
        = super::audio_generation::CompletionsAudioGenerationModel<H>
    where
        H: ModelTransport;

    fn audio_generation_model<H: ModelTransport>(
        client: &CompletionsClient<H>,
        model: String,
    ) -> Self::Model<H> {
        super::audio_generation::CompletionsAudioGenerationModel::new(client.clone(), model)
    }
}

impl<H> Client<H>
where
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Sets where Rig system instructions are placed in Responses requests for
    /// every completion model created from this client. Models capture the
    /// placement when they are created, so models built before this call are
    /// unaffected. See [`SystemInstructionsPlacement`] for when each placement applies.
    pub fn with_system_instructions_placement(
        self,
        placement: SystemInstructionsPlacement,
    ) -> Self {
        let mut ext = *self.provider();
        ext.system_instructions_placement = placement;
        self.with_provider(ext)
    }

    /// Sends Rig system instructions as `system` messages in `input` instead of
    /// as top-level Responses API `instructions` for every completion model
    /// created from this client. Models built before this call are unaffected.
    ///
    /// OpenAI's Responses API supports `instructions`, and Rig uses it by
    /// default. Use this compatibility fallback for OpenAI-compatible providers
    /// that reject or ignore top-level `instructions`.
    pub fn with_system_instructions_as_messages(self) -> Self {
        self.with_system_instructions_placement(SystemInstructionsPlacement::InputSystemMessages)
    }

    /// Create a Completions API client from this Responses API client.
    /// Useful for switching to the traditional Chat Completions API.
    pub fn completions_api(self) -> CompletionsClient<H> {
        let system_instructions_placement = self.provider().system_instructions_placement;
        self.with_provider(OpenAICompletions {
            system_instructions_placement,
        })
    }
}

impl<H> CompletionsClient<H>
where
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Create a Responses API client from this Completions API client.
    /// Useful for switching to the newer Responses API. A system-instructions
    /// placement configured before switching to the Completions API is
    /// restored.
    pub fn responses_api(self) -> Client<H> {
        let system_instructions_placement = self.provider().system_instructions_placement;
        self.with_provider(OpenAIResponses {
            system_instructions_placement,
        })
    }
}

/// Error envelope returned by OpenAI-compatible providers alongside 2xx
/// statuses. Providers spell the message field differently (`message`,
/// `error`, nested objects), so anything that isn't a valid success payload
/// is treated as an error envelope and the raw body is preserved for the
/// caller; `message` is only used for logging.
#[derive(Debug)]
pub struct ApiErrorResponse {
    pub(crate) message: String,
}

// Manual impl (not a field-level `alias = "error"`): the alias makes serde
// treat `message` and `error` as one field, so a body carrying both keys
// fails as a duplicate field instead of classifying as this envelope.
impl<'de> Deserialize<'de> for ApiErrorResponse {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Self {
            message: crate::providers::internal::envelope::error_message(deserializer)?,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}

#[cfg(test)]
mod tests;
