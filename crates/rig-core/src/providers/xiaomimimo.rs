//! Xiaomi MiMo API clients and Rig integrations.
//!
//! Xiaomi exposes both OpenAI-compatible and Anthropic-compatible chat APIs
//! under a single global host.
//!
//! # OpenAI-compatible example
//! ```no_run
//! use rig_core::client::CompletionClient;
//! use rig_core::providers::xiaomimimo;
//!
//! let client = xiaomimimo::Client::new("YOUR_API_KEY").expect("Failed to build client");
//! let model = client.completion_model(xiaomimimo::MIMO_V2_5_PRO);
//! ```
//!
//! # Anthropic-compatible example
//! ```no_run
//! use rig_core::client::CompletionClient;
//! use rig_core::providers::xiaomimimo;
//!
//! let client = xiaomimimo::AnthropicClient::new("YOUR_API_KEY").expect("Failed to build client");
//! let model = client.completion_model(xiaomimimo::MIMO_V2_5_PRO);
//! ```

use crate::client::{
    self, BearerAuth, Capabilities, Capable, DebugExt, ModelLister, Nothing, Provider,
    ProviderBuilder, ProviderClient,
};
use crate::http_client::{self, HttpClientExt};
use crate::model::{Model, ModelList, ModelListingError};
use crate::providers::anthropic::client::{
    AnthropicBuilder as AnthropicCompatBuilder, AnthropicKey, finish_anthropic_builder,
};
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};

/// OpenAI-compatible base URL.
pub const API_BASE_URL: &str = "https://api.xiaomimimo.com/v1";
/// Anthropic-compatible base URL.
pub const ANTHROPIC_API_BASE_URL: &str = "https://api.xiaomimimo.com/anthropic/v1";

/// `mimo-v2-flash`
pub const MIMO_V2_FLASH: &str = "mimo-v2-flash";
/// `mimo-v2-omni`
pub const MIMO_V2_OMNI: &str = "mimo-v2-omni";
/// `mimo-v2-pro`
pub const MIMO_V2_PRO: &str = "mimo-v2-pro";
/// `mimo-v2.5`
pub const MIMO_V2_5: &str = "mimo-v2.5";
/// `mimo-v2.5-pro`
pub const MIMO_V2_5_PRO: &str = "mimo-v2.5-pro";

#[derive(Debug, Default, Clone, Copy)]
pub struct XiaomiMimoExt;

#[derive(Debug, Default, Clone, Copy)]
pub struct XiaomiMimoBuilder;

#[derive(Debug, Default, Clone)]
pub struct XiaomiMimoAnthropicBuilder {
    anthropic: AnthropicCompatBuilder,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct XiaomiMimoAnthropicExt;

type XiaomiMimoApiKey = BearerAuth;

pub type Client<H = reqwest::Client> = client::Client<XiaomiMimoExt, H>;
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<XiaomiMimoBuilder, XiaomiMimoApiKey, H>;

pub type AnthropicClient<H = reqwest::Client> = client::Client<XiaomiMimoAnthropicExt, H>;
pub type AnthropicClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<XiaomiMimoAnthropicBuilder, AnthropicKey, H>;

impl Provider for XiaomiMimoExt {
    type Builder = XiaomiMimoBuilder;

    const VERIFY_PATH: &'static str = "/models";
}

impl Provider for XiaomiMimoAnthropicExt {
    type Builder = XiaomiMimoAnthropicBuilder;

    const VERIFY_PATH: &'static str = "/v1/models";
}

impl<H> Capabilities<H> for XiaomiMimoExt {
    type Completion = Capable<super::openai::completion::GenericCompletionModel<XiaomiMimoExt, H>>;
    type Embeddings = Nothing;
    type Transcription = Nothing;
    type ModelListing = Capable<XiaomiMimoModelLister<H>>;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
    type Rerank = Nothing;
}

impl<H> Capabilities<H> for XiaomiMimoAnthropicExt {
    type Completion =
        Capable<super::anthropic::completion::GenericCompletionModel<XiaomiMimoAnthropicExt, H>>;
    type Embeddings = Nothing;
    type Transcription = Nothing;
    type ModelListing = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
    type Rerank = Nothing;
}

impl DebugExt for XiaomiMimoExt {}
impl DebugExt for XiaomiMimoAnthropicExt {}

impl super::openai::completion::OpenAICompatibleProvider for XiaomiMimoExt {
    const PROVIDER_NAME: &'static str = "xiaomimimo";

    type StreamingUsage = super::openai::Usage;

    type Response = super::openai::CompletionResponse;
}

impl ProviderBuilder for XiaomiMimoBuilder {
    type Extension<H>
        = XiaomiMimoExt
    where
        H: HttpClientExt;
    type ApiKey = XiaomiMimoApiKey;

    const BASE_URL: &'static str = API_BASE_URL;

    fn build<H>(
        _builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        Ok(XiaomiMimoExt)
    }
}

impl ProviderBuilder for XiaomiMimoAnthropicBuilder {
    type Extension<H>
        = XiaomiMimoAnthropicExt
    where
        H: HttpClientExt;
    type ApiKey = AnthropicKey;

    const BASE_URL: &'static str = ANTHROPIC_API_BASE_URL;

    fn build<H>(
        _builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        Ok(XiaomiMimoAnthropicExt)
    }

    fn finish<H>(
        &self,
        builder: client::ClientBuilder<Self, AnthropicKey, H>,
    ) -> http_client::Result<client::ClientBuilder<Self, AnthropicKey, H>> {
        finish_anthropic_builder(&self.anthropic, builder)
    }
}

impl super::anthropic::completion::AnthropicCompatibleProvider for XiaomiMimoAnthropicExt {
    const PROVIDER_NAME: &'static str = "xiaomimimo";

    fn default_max_tokens(_model: &str) -> Option<u64> {
        Some(4096)
    }
}

impl ProviderClient for Client {
    type Input = XiaomiMimoApiKey;
    type Error = crate::client::ProviderClientError;

    fn from_env() -> Result<Self, Self::Error> {
        let api_key = crate::client::required_env_var("XIAOMI_MIMO_API_KEY")?;
        let mut builder = Self::builder().api_key(api_key);

        if let Some(base_url) = crate::client::optional_env_var("XIAOMI_MIMO_API_BASE")? {
            builder = builder.base_url(base_url);
        }

        builder.build().map_err(Into::into)
    }

    fn from_val(input: Self::Input) -> Result<Self, Self::Error> {
        Self::new(input).map_err(Into::into)
    }
}

impl ProviderClient for AnthropicClient {
    type Input = String;
    type Error = crate::client::ProviderClientError;

    fn from_env() -> Result<Self, Self::Error> {
        let api_key = crate::client::required_env_var("XIAOMI_MIMO_API_KEY")?;
        let mut builder = Self::builder().api_key(api_key);

        if let Some(base_url) =
            anthropic_base_override("XIAOMI_MIMO_ANTHROPIC_API_BASE", "XIAOMI_MIMO_API_BASE")?
        {
            builder = builder.base_url(base_url);
        }

        builder.build().map_err(Into::into)
    }

    fn from_val(input: Self::Input) -> Result<Self, Self::Error> {
        Self::builder().api_key(input).build().map_err(Into::into)
    }
}

fn anthropic_base_override(
    primary_env: &'static str,
    fallback_env: &'static str,
) -> crate::client::ProviderClientResult<Option<String>> {
    let primary = crate::client::optional_env_var(primary_env)?;
    let fallback = crate::client::optional_env_var(fallback_env)?;

    Ok(resolve_anthropic_base_override(
        primary.as_deref(),
        fallback.as_deref(),
    ))
}

fn resolve_anthropic_base_override(
    primary: Option<&str>,
    fallback: Option<&str>,
) -> Option<String> {
    primary
        .map(str::to_owned)
        .or_else(|| fallback.and_then(normalize_anthropic_base_url))
}

fn normalize_anthropic_base_url(base_url: &str) -> Option<String> {
    if base_url.contains("/anthropic") {
        return Some(base_url.to_owned());
    }

    if base_url.trim_end_matches('/') == API_BASE_URL {
        return Some(ANTHROPIC_API_BASE_URL.to_owned());
    }

    let mut url = url::Url::parse(base_url).ok()?;
    if !matches!(url.path(), "/v1" | "/v1/") {
        return None;
    }
    url.set_path("/anthropic/v1");
    Some(url.to_string())
}

impl<H> AnthropicClientBuilder<H> {
    pub fn anthropic_version(self, anthropic_version: &str) -> Self {
        self.over_ext(|mut ext| {
            ext.anthropic.anthropic_version = anthropic_version.into();
            ext
        })
    }

    pub fn anthropic_betas(self, anthropic_betas: &[&str]) -> Self {
        self.over_ext(|mut ext| {
            ext.anthropic
                .anthropic_betas
                .extend(anthropic_betas.iter().copied().map(String::from));
            ext
        })
    }

    pub fn anthropic_beta(self, anthropic_beta: &str) -> Self {
        self.over_ext(|mut ext| {
            ext.anthropic.anthropic_betas.push(anthropic_beta.into());
            ext
        })
    }
}

#[derive(Debug, serde::Deserialize)]
struct ListModelsResponse {
    data: Vec<ListModelEntry>,
}

#[derive(Debug, serde::Deserialize)]
struct ListModelEntry {
    id: String,
    owned_by: String,
}

impl From<ListModelEntry> for Model {
    fn from(value: ListModelEntry) -> Self {
        let mut model = Model::from_id(value.id);
        model.owned_by = Some(value.owned_by);
        model
    }
}

/// [`ModelLister`] implementation for the Xiaomi MiMo API (`GET /models`).
#[derive(Clone)]
pub struct XiaomiMimoModelLister<H = reqwest::Client> {
    client: Client<H>,
}

impl<H> ModelLister<H> for XiaomiMimoModelLister<H>
where
    H: HttpClientExt + WasmCompatSend + WasmCompatSync + 'static,
{
    type Client = Client<H>;

    fn new(client: Self::Client) -> Self {
        Self { client }
    }

    async fn list_all(&self) -> Result<ModelList, ModelListingError> {
        let path = "/models";
        let req = self.client.get(path)?.body(http_client::NoBody)?;
        let response = self
            .client
            .send::<_, Vec<u8>>(req)
            .await
            .map_err(|error| match error {
                http_client::Error::InvalidStatusCodeWithMessage(status, message) => {
                    ModelListingError::api_error_with_context(
                        "Xiaomi MiMo",
                        path,
                        status.as_u16(),
                        message.as_bytes(),
                    )
                }
                other => ModelListingError::from(other),
            })?;

        let status = response.status();
        let body = response.into_body().await?;
        parse_list_models_response(status, &body)
    }
}

/// Path of the model-listing endpoint, relative to the API base URL.
pub(crate) const LIST_MODELS_PATH: &str = "/models";

/// Parse a `GET /models` response into a [`ModelList`]. Pure.
///
/// Shared by the classic [`XiaomiMimoModelLister`] and
/// [`functions::list_models`].
pub(crate) fn parse_list_models_response(
    status: http::StatusCode,
    body: &[u8],
) -> Result<ModelList, ModelListingError> {
    if !status.is_success() {
        return Err(ModelListingError::api_error_with_context(
            "Xiaomi MiMo",
            LIST_MODELS_PATH,
            status.as_u16(),
            body,
        ));
    }
    let api_resp: ListModelsResponse = serde_json::from_slice(body).map_err(|error| {
        ModelListingError::parse_error_with_context("Xiaomi MiMo", LIST_MODELS_PATH, &error, body)
    })?;
    let models = api_resp.data.into_iter().map(Model::from).collect();
    Ok(ModelList::new(models))
}

#[cfg(test)]
mod tests {
    use super::{
        ANTHROPIC_API_BASE_URL, API_BASE_URL, normalize_anthropic_base_url,
        resolve_anthropic_base_override,
    };

    #[test]
    fn test_client_initialization() {
        let _client =
            crate::providers::xiaomimimo::Client::new("dummy-key").expect("Client::new()");
        let _client_from_builder = crate::providers::xiaomimimo::Client::builder()
            .api_key("dummy-key")
            .build()
            .expect("Client::builder()");
        let _anthropic_client = crate::providers::xiaomimimo::AnthropicClient::new("dummy-key")
            .expect("AnthropicClient::new()");
        let _anthropic_client_from_builder =
            crate::providers::xiaomimimo::AnthropicClient::builder()
                .api_key("dummy-key")
                .build()
                .expect("AnthropicClient::builder()");
    }

    #[test]
    fn normalize_openai_bases_to_anthropic_bases() {
        assert_eq!(
            normalize_anthropic_base_url(API_BASE_URL).as_deref(),
            Some(ANTHROPIC_API_BASE_URL)
        );
        assert_eq!(
            normalize_anthropic_base_url("https://proxy.example.com/v1").as_deref(),
            Some("https://proxy.example.com/anthropic/v1")
        );
    }

    #[test]
    fn normalize_preserves_existing_anthropic_base() {
        assert_eq!(
            normalize_anthropic_base_url(ANTHROPIC_API_BASE_URL).as_deref(),
            Some(ANTHROPIC_API_BASE_URL)
        );
    }

    #[test]
    fn anthropic_primary_override_wins() {
        let override_url = resolve_anthropic_base_override(
            Some("https://primary.example.com/anthropic/v1"),
            Some(API_BASE_URL),
        );

        assert_eq!(
            override_url.as_deref(),
            Some("https://primary.example.com/anthropic/v1")
        );
    }
}

pub mod functions {
    //! Xiaomi MiMo chat completions as config + pure functions.
    //!
    //! The data-oriented face of the Xiaomi MiMo provider, mirroring
    //! [`crate::providers::openai::functions`]: a serde [`Config`], a
    //! [`DESCRIPTOR`] capability sheet, and pure
    //! [`build_request`]/[`parse_response`] free functions plus the async
    //! [`complete`]/[`open_stream`] wrappers over
    //! [`HttpRuntime`](crate::http_runtime::HttpRuntime). The request/parse
    //! mechanics are shared with the other OpenAI-compatible providers via
    //! `openai::functions`; this module instantiates them with
    //! [`XiaomiMimoExt`](super::XiaomiMimoExt) so Xiaomi MiMo's paths, hooks, and
    //! provider name apply.

    use serde::{Deserialize, Serialize};

    use super::XiaomiMimoExt as Ext;
    use crate::completion::{self, CompletionError, CompletionRequest};
    use crate::http_runtime::HttpRuntime;
    use crate::providers::descriptor::{ApiKeyLocation, ProviderDescriptor};
    use crate::providers::openai::functions as openai_functions;

    /// Default Xiaomi MiMo API base URL.
    pub const DEFAULT_BASE_URL: &str = "https://api.xiaomimimo.com/v1";

    /// Xiaomi MiMo's capability sheet.
    pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
        name: "xiaomimimo",
        supports_tools: true,
        supports_response_format: true,
        stream_include_usage: true,
        emits_complete_single_chunk_tool_calls: false,
        composes_native_output_with_tools: true,
        max_embedding_documents: None,
    };

    /// Plain-data Xiaomi MiMo provider configuration.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[non_exhaustive]
    pub struct Config {
        /// API base URL (defaults to [`DEFAULT_BASE_URL`]).
        pub base_url: String,
        /// Credential location.
        pub api_key: ApiKeyLocation,
        /// Model identifier requests are built for.
        pub model: String,
        /// Extra headers attached to every request.
        pub extra_headers: Vec<(String, String)>,
    }

    impl Config {
        /// Config for `model` reading `XIAOMI_MIMO_API_KEY` from the environment.
        pub fn new(model: impl Into<String>) -> Self {
            Self {
                base_url: DEFAULT_BASE_URL.to_string(),
                api_key: ApiKeyLocation::Env("XIAOMI_MIMO_API_KEY".to_string()),
                model: model.into(),
                extra_headers: Vec::new(),
            }
        }

        /// Config for `model` with an explicit API key.
        pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
            self.api_key = ApiKeyLocation::Inline(key.into());
            self
        }

        /// Override the API base URL.
        pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
            self.base_url = base_url.into();
            self
        }
    }

    /// Build the serialized chat-completions request body for `request`. Pure.
    pub fn build_request_body(
        cfg: &Config,
        request: &CompletionRequest,
        stream: bool,
    ) -> Result<Vec<u8>, CompletionError> {
        openai_functions::compatible_request_body(&Ext, &cfg.model, request, stream)
    }

    /// Build the complete HTTP request (URL, headers, body) for `request`.
    ///
    /// Pure except for credential resolution (`ApiKeyLocation::Env` reads the
    /// environment).
    pub fn build_request(
        cfg: &Config,
        request: &CompletionRequest,
        stream: bool,
    ) -> Result<http::Request<Vec<u8>>, CompletionError> {
        openai_functions::compatible_request(
            &Ext,
            &cfg.base_url,
            &cfg.api_key,
            &cfg.extra_headers,
            &cfg.model,
            request,
            stream,
        )
    }

    /// Parse a chat-completions response body into the normalized
    /// [`completion::CompletionResponse`]. Pure.
    pub fn parse_response(
        status: http::StatusCode,
        body: &str,
    ) -> Result<completion::CompletionResponse, CompletionError> {
        openai_functions::compatible_parse_response::<Ext>(status, body)
    }

    /// Open a streaming completion for `request`.
    pub async fn open_stream(
        cfg: &Config,
        rt: &HttpRuntime,
        request: CompletionRequest,
    ) -> Result<crate::streaming::StreamingCompletionResponse, CompletionError> {
        let req = build_request(cfg, &request, true)?;
        openai_functions::compatible_open_stream(Ext, rt, req).await
    }

    /// Send `request` to Xiaomi MiMo and return the normalized response.
    pub async fn complete(
        cfg: &Config,
        rt: &HttpRuntime,
        request: CompletionRequest,
    ) -> Result<completion::CompletionResponse, CompletionError> {
        let req = build_request(cfg, &request, false)?;
        let (status, body) = rt.send(req).await?;
        parse_response(status, &body)
    }

    /// Build the `GET /models` request for [`list_models`].
    ///
    /// Pure except for credential resolution (`ApiKeyLocation::Env` reads
    /// the environment).
    pub fn build_list_models_request(
        cfg: &Config,
    ) -> Result<http::Request<Vec<u8>>, crate::model::ModelListingError> {
        let url = format!(
            "{}{}",
            cfg.base_url.trim_end_matches('/'),
            super::LIST_MODELS_PATH
        );
        openai_functions::bearer_get(url, &cfg.api_key, &cfg.extra_headers)
    }

    /// List the models available to `cfg`'s credentials.
    ///
    /// The classic `ModelListingClient` path parses through the same pure
    /// parser (`super::parse_list_models_response`).
    pub async fn list_models(
        cfg: &Config,
        rt: &HttpRuntime,
    ) -> Result<crate::model::ModelList, crate::model::ModelListingError> {
        let req = build_list_models_request(cfg)?;
        let (status, body) = rt.send_bytes(req).await?;
        super::parse_list_models_response(status, &body)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::OneOrMany;

        fn sample_request() -> CompletionRequest {
            CompletionRequest {
                model: None,
                preamble: None,
                chat_history: OneOrMany::one(crate::message::Message::user("hello")),
                documents: Vec::new(),
                tools: Vec::new(),
                temperature: Some(0.5),
                max_tokens: Some(64),
                tool_choice: None,
                additional_params: None,
                output_schema: None,
                record_telemetry_content: false,
            }
        }

        #[test]
        fn build_request_sets_url_and_model() {
            let cfg = Config::new("test-model").with_api_key("secret");
            let req = build_request(&cfg, &sample_request(), false).expect("build");
            assert_eq!(req.uri(), "https://api.xiaomimimo.com/v1/chat/completions");
            let value: serde_json::Value = serde_json::from_slice(req.body()).expect("json");
            assert_eq!(value["model"], "test-model");
        }

        #[test]
        fn parse_response_normalizes() {
            let body = serde_json::json!({
                "id": "chatcmpl-1",
                "model": "test-model",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "logprobs": null,
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
            })
            .to_string();
            let response = parse_response(http::StatusCode::OK, &body).expect("parse");
            assert_eq!(response.provider, "xiaomimimo");
            assert_eq!(response.usage.input_tokens, 3);
            assert_eq!(response.usage.total_tokens, 5);
        }
    }
}
