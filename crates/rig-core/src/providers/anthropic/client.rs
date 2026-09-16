//! Anthropic client and wire configuration.

use http::HeaderValue;
use serde::{Deserialize, Serialize};

use super::completion::{ANTHROPIC_VERSION_LATEST, Messages};
use super::model_listing::Models;
use crate::driver::{Bound, HasCompletion};
use crate::http_client::{BoxedHttpClient, Framing, HttpClientExt};
use crate::wire::{Body, Decoder, Encoded, Output, Secret, Wire, WireEvent, WireFrame};

/// Dialect configuration for Anthropic and Anthropic-compatible gateways.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Dialect {
    pub name: &'static str,
    pub base_url: &'static str,
    pub api_key_env: &'static str,
    pub base_url_env: Option<&'static str>,
    pub request_id_header: Option<&'static str>,
    pub default_max_tokens: Option<u64>,
    pub strict_tools: bool,
}

impl<'de> serde::Deserialize<'de> for Dialect {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct DialectHelper {
            name: String,
        }
        let helper = DialectHelper::deserialize(deserializer)?;
        match helper.name.as_str() {
            "anthropic" => Ok(ANTHROPIC),
            "minimax" => Ok(compatible(
                "minimax",
                "https://api.minimax.chat/v1",
                "MINIMAX_API_KEY",
                Some("MINIMAX_ANTHROPIC_API_BASE"),
            )),
            "zai" => Ok(compatible(
                "zai",
                "https://api.z.ai/api/v1",
                "ZAI_API_KEY",
                Some("ZAI_ANTHROPIC_API_BASE"),
            )),
            "moonshot" => Ok(compatible(
                "moonshot",
                "https://api.moonshot.cn/v1",
                "MOONSHOT_API_KEY",
                Some("MOONSHOT_ANTHROPIC_API_BASE"),
            )),
            "xiaomimimo" => Ok(compatible(
                "xiaomimimo",
                "https://api.xiaomimimo.com/v1",
                "XIAOMI_MIMO_API_KEY",
                Some("XIAOMI_MIMO_ANTHROPIC_API_BASE"),
            )),
            _ => Ok(ANTHROPIC),
        }
    }
}

pub const ANTHROPIC: Dialect = Dialect {
    name: "anthropic",
    base_url: "https://api.anthropic.com",
    api_key_env: "ANTHROPIC_API_KEY",
    base_url_env: Some("ANTHROPIC_BASE_URL"),
    request_id_header: Some("request-id"),
    default_max_tokens: None,
    strict_tools: true,
};

pub const fn compatible(
    name: &'static str,
    base_url: &'static str,
    api_key_env: &'static str,
    base_url_env: Option<&'static str>,
) -> Dialect {
    Dialect {
        name,
        base_url,
        api_key_env,
        base_url_env,
        request_id_header: Some("request-id"),
        default_max_tokens: Some(4096),
        strict_tools: false,
    }
}
/// Anthropic provider configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Anthropic {
    pub api_key: Secret,
    pub base_url: String,
    pub version: String,
    pub betas: Vec<String>,
    pub dialect: Dialect,
}

impl Anthropic {
    pub fn new(api_key: impl Into<Secret>) -> Self {
        Self::from_dialect(ANTHROPIC, api_key)
    }

    pub fn from_dialect(dialect: Dialect, api_key: impl Into<Secret>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: dialect.base_url.to_owned(),
            version: ANTHROPIC_VERSION_LATEST.to_owned(),
            betas: Vec::new(),
            dialect,
        }
    }

    pub fn from_env() -> Result<Self, crate::client::ProviderClientError> {
        Self::from_env_with(&ANTHROPIC)
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
            base_url: normalize_anthropic_base_url(&base_url),
            version: ANTHROPIC_VERSION_LATEST.to_owned(),
            betas: Vec::new(),
            dialect: dialect.clone(),
        })
    }
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    pub fn with_betas(mut self, betas: &[&str]) -> Self {
        self.betas = betas.iter().map(|s| s.to_string()).collect();
        self
    }

    pub fn with_beta(mut self, beta: impl Into<String>) -> Self {
        self.betas.push(beta.into());
        self
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = normalize_anthropic_base_url(&base_url.into());
        self
    }

    pub fn bind<H: HttpClientExt>(self, http: H) -> Bound<Self, H> {
        Bound::new(self, http)
    }

    pub fn messages(&self, model: impl Into<String>) -> Messages {
        Messages {
            provider: self.clone(),
            model: model.into(),
            default_max_tokens: self.dialect.default_max_tokens,
            prompt_caching: false,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
            strict_tools: false,
        }
    }

    pub fn models(&self) -> Models {
        Models::new(self.clone())
    }

    pub fn verify(&self) -> Verify {
        Verify::new(self.clone())
    }

    pub fn apply_headers(&self, headers: &mut http::HeaderMap) {
        if let Ok(val) = HeaderValue::from_str(self.api_key.expose_secret()) {
            headers.insert(http::HeaderName::from_static("x-api-key"), val);
        }
        if let Ok(val) = HeaderValue::from_str(&self.version) {
            headers.insert(http::HeaderName::from_static("anthropic-version"), val);
        }
        if !self.betas.is_empty()
            && let Ok(val) = HeaderValue::from_str(&self.betas.join(","))
        {
            headers.insert(http::HeaderName::from_static("anthropic-beta"), val);
        }
    }
}

impl HasCompletion for Anthropic {
    type Wire = Messages;
    fn completion(&self, model: &str) -> Messages {
        self.messages(model)
    }
}

pub fn normalize_anthropic_base_url(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    if let Some(stripped) = trimmed.strip_suffix("/v1/messages") {
        stripped.to_string()
    } else if let Some(stripped) = trimmed.strip_suffix("/messages") {
        stripped.to_string()
    } else if let Some(stripped) = trimmed.strip_suffix("/v1") {
        stripped.to_string()
    } else {
        trimmed.to_string()
    }
}

/// The Verify wire for Anthropic (status-only check on /v1/models).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Verify {
    pub provider: Anthropic,
}

impl Verify {
    pub fn new(provider: Anthropic) -> Self {
        Self { provider }
    }
}

impl Wire for Verify {
    type Op = crate::operation::Verify;
    type Decoder = VerifyDecoder;

    fn name(&self) -> &str {
        self.provider.dialect.name
    }

    fn encode(&self, _req: ()) -> Result<Encoded, crate::client::verify::VerifyError> {
        let url = format!("{}/v1/models", self.provider.base_url.trim_end_matches('/'));
        let mut req = http::Request::builder()
            .method(http::Method::GET)
            .uri(url)
            .body(Body::Bytes(Vec::new()))
            .map_err(|e| {
                crate::client::verify::VerifyError::HttpError(crate::http_client::Error::Protocol(
                    e,
                ))
            })?;
        self.provider.apply_headers(req.headers_mut());
        Ok(Encoded::new(req, Framing::Whole, "/v1/models"))
    }

    fn decoder(&self) -> Self::Decoder {
        VerifyDecoder
    }

    fn capabilities(&self) {}
}

pub struct VerifyDecoder;

impl Decoder<crate::operation::Verify> for VerifyDecoder {
    type Event = ();

    fn classify(&self, _frame: WireFrame) -> WireEvent<Self::Event> {
        WireEvent::Known(())
    }

    fn interpret(&mut self, _event: Self::Event, out: &mut Output<crate::operation::Verify>) {
        out.emit(());
    }

    fn finish(&mut self, _out: &mut Output<crate::operation::Verify>) {}

    fn flush_before_terminal_error(&mut self, _out: &mut Output<crate::operation::Verify>) {}

    fn project(&self, _payload: &[u8], _sink: &mut dyn crate::wire::ObservationSink) {}
}

impl Bound<Anthropic, BoxedHttpClient> {
    pub fn new_with<H>(
        api_key: impl Into<Secret>,
        http: H,
    ) -> Result<Bound<Anthropic, H>, crate::client::ProviderClientError> {
        Ok(Bound::new(Anthropic::new(api_key), http))
    }
}

// Convenience methods on Bound<Anthropic, H>
impl<H: Clone> Bound<Anthropic, H> {
    pub fn messages(&self, model: impl Into<String>) -> Bound<Messages, H> {
        Bound::new(self.wire.messages(model), self.http.clone())
    }

    pub fn models(&self) -> Bound<Models, H> {
        Bound::new(self.wire.models(), self.http.clone())
    }

    pub fn verify(&self) -> Bound<Verify, H> {
        Bound::new(self.wire.verify(), self.http.clone())
    }

    pub fn model_lister(&self) -> Bound<Models, H> {
        self.models()
    }

    pub async fn list_models(
        &self,
    ) -> Result<crate::model::ModelList, crate::model::ModelListingError>
    where
        H: HttpClientExt + 'static,
    {
        use crate::client::ModelLister;
        self.models().list_all().await
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
impl<H> crate::client::ModelListingClient for Bound<Anthropic, H>
where
    H: HttpClientExt
        + Clone
        + crate::wasm_compat::WasmCompatSend
        + crate::wasm_compat::WasmCompatSync
        + 'static,
{
    fn list_models(
        &self,
    ) -> impl std::future::Future<
        Output = Result<crate::model::ModelList, crate::model::ModelListingError>,
    > + crate::wasm_compat::WasmCompatSend {
        let lister = self.models();
        async move {
            use crate::client::ModelLister;
            lister.list_all().await
        }
    }
}

// Forwarding methods on Bound<Messages, H>
impl<H: Clone> Bound<Messages, H> {
    pub fn with_prompt_caching(self) -> Self {
        self.map_wire(|w| w.with_prompt_caching())
    }

    pub fn with_automatic_caching(self) -> Self {
        self.map_wire(|w| w.with_automatic_caching())
    }

    pub fn with_automatic_caching_1h(self) -> Self {
        self.map_wire(|w| w.with_automatic_caching_1h())
    }

    pub fn with_static_prefix_cache_ttl(self, ttl: super::completion::CacheTtl) -> Self {
        self.map_wire(|w| w.with_static_prefix_cache_ttl(ttl))
    }

    pub fn with_strict_tools(self) -> Self {
        self.map_wire(|w| w.with_strict_tools())
    }
    pub fn client(&self) -> Bound<Anthropic, H> {
        Bound::new(self.wire.provider.clone(), self.http.clone())
    }

    pub async fn raw_completion(
        &self,
        request: crate::completion::CompletionRequest,
    ) -> Result<super::completion::CompletionResponse, crate::completion::CompletionError>
    where
        H: HttpClientExt + 'static,
    {
        use crate::completion::CompletionModel;
        let resp = self.completion(request).await?;
        serde_json::from_value(resp.raw)
            .map_err(|e| crate::completion::CompletionError::ResponseError(e.to_string()))
    }
}

pub type Client<H = BoxedHttpClient> = Bound<Anthropic, H>;

impl Client<BoxedHttpClient> {
    pub fn builder() -> ClientBuilder<crate::markers::Missing> {
        ClientBuilder::default()
    }
}

#[derive(Default)]
pub struct ClientBuilder<H> {
    api_key: Option<String>,
    base_url: Option<String>,
    version: Option<String>,
    betas: Vec<String>,
    http_client: H,
}

impl ClientBuilder<crate::markers::Missing> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    pub fn anthropic_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    pub fn anthropic_beta(mut self, beta: impl Into<String>) -> Self {
        self.betas.push(beta.into());
        self
    }

    pub fn anthropic_betas(mut self, betas: &[&str]) -> Self {
        self.betas.extend(betas.iter().map(|s| (*s).to_string()));
        self
    }

    pub fn http_client<H2: HttpClientExt>(self, http: H2) -> ClientBuilder<H2> {
        ClientBuilder {
            api_key: self.api_key,
            base_url: self.base_url,
            version: self.version,
            betas: self.betas,
            http_client: http,
        }
    }
}

impl<H: HttpClientExt + Clone> ClientBuilder<H> {
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    pub fn anthropic_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    pub fn anthropic_beta(mut self, beta: impl Into<String>) -> Self {
        self.betas.push(beta.into());
        self
    }

    pub fn anthropic_betas(mut self, betas: &[&str]) -> Self {
        self.betas.extend(betas.iter().map(|s| (*s).to_string()));
        self
    }

    pub fn build(self) -> Result<Bound<Anthropic, H>, crate::client::ProviderClientError> {
        let api_key = self
            .api_key
            .ok_or(crate::client::ProviderClientError::MissingApiKey(
                "anthropic",
            ))?;
        let base_url = self
            .base_url
            .unwrap_or_else(|| ANTHROPIC.base_url.to_owned());
        let version = self
            .version
            .unwrap_or_else(|| ANTHROPIC_VERSION_LATEST.to_owned());
        let anthropic = Anthropic {
            api_key: Secret::new(api_key),
            base_url: normalize_anthropic_base_url(&base_url),
            version,
            betas: self.betas,
            dialect: ANTHROPIC,
        };
        Ok(Bound::new(anthropic, self.http_client))
    }
}
#[derive(Debug, Clone, Default)]
pub struct AnthropicConfig {
    pub anthropic_version: String,
    pub anthropic_betas: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct AnthropicKey(pub String);

impl<S: Into<String>> From<S> for AnthropicKey {
    fn from(s: S) -> Self {
        Self(s.into())
    }
}

impl crate::client::ApiKey for AnthropicKey {
    fn into_header(self) -> Option<crate::http_client::Result<(http::HeaderName, HeaderValue)>> {
        Some(
            HeaderValue::from_str(&self.0)
                .map(|val| (http::HeaderName::from_static("x-api-key"), val))
                .map_err(Into::into),
        )
    }
}

pub fn finish_anthropic_builder<P: crate::client::Provider<Config = AnthropicConfig>, H>(
    mut builder: crate::client::ClientBuilder<P, H>,
) -> crate::http_client::Result<crate::client::ClientBuilder<P, H>> {
    let normalized_base_url = normalize_anthropic_base_url(builder.get_base_url());
    builder = builder.base_url(normalized_base_url);

    let config = builder.config().clone();
    builder.headers_mut().insert(
        "anthropic-version",
        HeaderValue::from_str(&config.anthropic_version)?,
    );

    if !config.anthropic_betas.is_empty() {
        builder.headers_mut().insert(
            "anthropic-beta",
            HeaderValue::from_str(&config.anthropic_betas.join(","))?,
        );
    }

    Ok(builder)
}

#[derive(Debug, Default, Clone, Copy)]
pub struct AnthropicProvider;

impl crate::client::Provider for AnthropicProvider {
    const NAME: &'static str = "anthropic";
    const BASE_URL: &'static str = "https://api.anthropic.com";
    const VERIFY_PATH: &'static str = "/v1/models";
    type ApiKey = AnthropicKey;
    type Config = AnthropicConfig;
    type EnvInput = String;

    fn build(_: Self::Config, _: &Self::ApiKey) -> crate::http_client::Result<Self> {
        Ok(AnthropicProvider)
    }

    fn finish<H>(
        &self,
        builder: crate::client::ClientBuilder<Self, H>,
    ) -> crate::http_client::Result<crate::client::ClientBuilder<Self, H>> {
        finish_anthropic_builder(builder)
    }

    fn from_env<H: HttpClientExt>(
        http: H,
    ) -> crate::client::ProviderClientResult<crate::client::Client<Self, H>> {
        crate::client::Client::from_env_api_key(
            "ANTHROPIC_API_KEY",
            Some("ANTHROPIC_BASE_URL"),
            http,
        )
    }

    fn from_val<H: HttpClientExt>(
        input: String,
        http: H,
    ) -> crate::client::ProviderClientResult<crate::client::Client<Self, H>> {
        crate::client::Client::new_with(input, http)
    }
}

impl super::completion::AnthropicCompatibleProvider for AnthropicProvider {
    const PROVIDER_NAME: &'static str = "anthropic";

    fn default_max_tokens(model: &str) -> Option<u64> {
        super::completion::default_max_tokens_for_model(model)
    }

    fn enable_strict_tool_use(tool: &mut super::completion::ToolDefinition) {
        super::completion::sanitize_strict_tool_schema(&mut tool.input_schema);
        tool.strict = true;
    }
}
