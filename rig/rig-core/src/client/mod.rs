//! This module provides traits for defining and creating provider clients.
//! Clients are used to create models for completion, embeddings, etc.

pub mod audio_generation;
pub mod completion;
pub mod embeddings;
pub mod image_generation;
pub mod model_listing;
pub mod transcription;
pub mod verify;

use bytes::Bytes;
pub use completion::CompletionClient;
pub use embeddings::EmbeddingsClient;
use http::{HeaderMap, HeaderName, HeaderValue};
pub use model_listing::{ModelLister, ModelListingClient};
use std::{env::VarError, fmt::Debug, marker::PhantomData, sync::Arc};
use thiserror::Error;
pub use verify::{VerifyClient, VerifyError};

#[cfg(feature = "image")]
use crate::image_generation::ImageGenerationModel;
#[cfg(feature = "image")]
use image_generation::ImageGenerationClient;

#[cfg(feature = "audio")]
use crate::audio_generation::*;
#[cfg(feature = "audio")]
use audio_generation::*;

use crate::{
    completion::CompletionModel,
    embeddings::EmbeddingModel,
    http_client::{
        self, Builder, HttpClientExt, LazyBody, MultipartForm, Request, Response, make_auth_header,
    },
    markers::Missing,
    prelude::TranscriptionClient,
    transcription::TranscriptionModel,
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ClientBuilderError {
    #[error("reqwest error: {0}")]
    HttpError(
        #[from]
        #[source]
        reqwest::Error,
    ),
    #[error("invalid property: {0}")]
    InvalidProperty(&'static str),
}

/// Errors returned while constructing provider clients from environment variables or explicit input.
///
/// Provider-specific client constructors use this error for configuration problems that can be
/// detected before any model request is sent, such as missing API keys, invalid environment
/// values, or invalid builder configuration.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ProviderClientError {
    /// A required or optional environment variable could not be read as valid Unicode.
    ///
    /// For required variables, this variant is also returned when the variable is not present.
    #[error("environment variable `{name}` is not set or is invalid")]
    EnvironmentVariable {
        /// The environment variable name.
        name: &'static str,
        /// The underlying environment lookup error.
        #[source]
        source: VarError,
    },
    /// The underlying provider client builder failed while constructing HTTP configuration.
    #[error(transparent)]
    Http(#[from] http_client::Error),
    /// The provider received an unsupported or incomplete configuration.
    #[error("{0}")]
    InvalidConfiguration(&'static str),
}

/// Result type returned by provider client construction helpers.
pub type ProviderClientResult<T> = std::result::Result<T, ProviderClientError>;

/// Read a required environment variable for provider client construction.
///
/// Returns [`ProviderClientError::EnvironmentVariable`] when the variable is missing or contains
/// invalid Unicode.
pub fn required_env_var(name: &'static str) -> ProviderClientResult<String> {
    std::env::var(name).map_err(|source| ProviderClientError::EnvironmentVariable { name, source })
}

/// Read an optional environment variable for provider client construction.
///
/// Missing variables return `Ok(None)`. Variables containing invalid Unicode return
/// [`ProviderClientError::EnvironmentVariable`].
pub fn optional_env_var(name: &'static str) -> ProviderClientResult<Option<String>> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(VarError::NotPresent) => Ok(None),
        Err(source) => Err(ProviderClientError::EnvironmentVariable { name, source }),
    }
}

/// Abstracts over the ability to instantiate a client, either via environment variables or some
/// `Self::Input`
pub trait ProviderClient {
    /// Input accepted by [`ProviderClient::from_val`].
    type Input;
    /// Error returned when client construction fails.
    type Error;

    /// Create a client from the process's environment.
    fn from_env() -> Result<Self, Self::Error>
    where
        Self: Sized;

    /// Create a client from an explicit provider-specific input value.
    fn from_val(input: Self::Input) -> Result<Self, Self::Error>
    where
        Self: Sized;
}

/// A trait for API keys. This determines whether the key is inserted into a [Client]'s default
/// headers (in the `Some` case) or handled by a given provider extension (in the `None` case)
pub trait ApiKey: Sized {
    fn into_header(self) -> Option<http_client::Result<(HeaderName, HeaderValue)>> {
        None
    }
}

/// An API key which will be inserted into a `Client`'s default headers as a bearer auth token
pub struct BearerAuth(String);

impl ApiKey for BearerAuth {
    fn into_header(self) -> Option<http_client::Result<(HeaderName, HeaderValue)>> {
        Some(make_auth_header(self.0))
    }
}

impl<S> From<S> for BearerAuth
where
    S: Into<String>,
{
    fn from(value: S) -> Self {
        Self(value.into())
    }
}

/// A type containing nothing at all. For `Option`-like behavior on the type level, i.e. to describe
/// the lack of a capability or field (an API key, for instance)
#[derive(Debug, Default, Clone, Copy)]
pub struct Nothing;

impl ApiKey for Nothing {}

impl TryFrom<String> for Nothing {
    type Error = &'static str;

    fn try_from(_: String) -> Result<Self, Self::Error> {
        Err(
            "Tried to create a Nothing from a string - this should not happen, please file an issue",
        )
    }
}

#[derive(Clone)]
pub struct Client<Ext = Nothing, H = reqwest::Client> {
    base_url: Arc<str>,
    headers: Arc<HeaderMap>,
    http_client: H,
    ext: Ext,
}

pub trait DebugExt: Debug {
    fn fields(&self) -> impl Iterator<Item = (&'static str, &dyn Debug)> {
        std::iter::empty()
    }
}

impl<Ext, H> std::fmt::Debug for Client<Ext, H>
where
    Ext: DebugExt,
    H: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = &mut f.debug_struct("Client");

        d = d
            .field("base_url", &self.base_url)
            .field(
                "headers",
                &self
                    .headers
                    .iter()
                    .filter_map(|(k, v)| {
                        if k == http::header::AUTHORIZATION || k.as_str().contains("api-key") {
                            None
                        } else {
                            Some((k, v))
                        }
                    })
                    .collect::<Vec<(&HeaderName, &HeaderValue)>>(),
            )
            .field("http_client", &self.http_client);

        self.ext
            .fields()
            .fold(d, |d, (name, field)| d.field(name, field))
            .finish()
    }
}

pub enum Transport {
    Http,
    Sse,
    NdJson,
}

/// An API provider extension, this abstracts over extensions which may be used in conjunction with
/// the `Client<Ext, H>` struct to define the behavior of a provider with respect to networking,
/// auth, instantiating models
pub trait Provider: Sized {
    /// The builder type that constructs this provider extension.
    /// This associates extensions with their builders for type inference.
    type Builder: ProviderBuilder;

    const VERIFY_PATH: &'static str;

    fn build_uri(&self, base_url: &str, path: &str, _transport: Transport) -> String {
        // Some providers (like Azure) have a blank base URL to allow users to input their own endpoints.
        let base_url = if base_url.is_empty() {
            base_url.to_string()
        } else {
            base_url.to_string() + "/"
        };

        base_url.to_string() + path.trim_start_matches('/')
    }

    fn with_custom(&self, req: http_client::Builder) -> http_client::Result<http_client::Builder> {
        Ok(req)
    }
}

/// A wrapper type providing runtime checks on a provider's capabilities via the [Capability] trait
pub struct Capable<M>(PhantomData<M>);

pub trait Capability {
    const CAPABLE: bool;
}

impl<M> Capability for Capable<M> {
    const CAPABLE: bool = true;
}

impl Capability for Nothing {
    const CAPABLE: bool = false;
}

/// The capabilities of a given provider, i.e. embeddings, audio transcriptions, text completion
pub trait Capabilities<H = reqwest::Client> {
    type Completion: Capability;
    type Embeddings: Capability;
    type Transcription: Capability;
    type ModelListing: Capability;
    #[cfg(feature = "image")]
    type ImageGeneration: Capability;
    #[cfg(feature = "audio")]
    type AudioGeneration: Capability;
}

/// An API provider extension *builder*, this abstracts over provider-specific builders which are
/// able to configure and produce a given provider's extension type
///
/// See [Provider]
pub trait ProviderBuilder: Sized + Default + Clone {
    type Extension<H>: Provider
    where
        H: HttpClientExt;
    type ApiKey: ApiKey;

    const BASE_URL: &'static str;

    /// Build the provider extension from the client builder configuration.
    fn build<H>(
        builder: &ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt;

    /// This method can be used to customize the fields of `builder` before it is used to create
    /// a client. For example, adding default headers
    fn finish<H>(
        &self,
        builder: ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<ClientBuilder<Self, Self::ApiKey, H>> {
        Ok(builder)
    }
}

impl<Ext> Client<Ext, reqwest::Client>
where
    Ext: Provider,
    Ext::Builder: ProviderBuilder<Extension<reqwest::Client> = Ext> + Default,
{
    pub fn new(
        api_key: impl Into<<Ext::Builder as ProviderBuilder>::ApiKey>,
    ) -> http_client::Result<Self> {
        Self::builder().api_key(api_key).build()
    }
}

impl<Ext, H> Client<Ext, H> {
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn headers(&self) -> &HeaderMap {
        &self.headers
    }

    pub fn ext(&self) -> &Ext {
        &self.ext
    }

    pub fn with_ext<NewExt>(self, new_ext: NewExt) -> Client<NewExt, H> {
        Client {
            base_url: self.base_url,
            headers: self.headers,
            http_client: self.http_client,
            ext: new_ext,
        }
    }
}

impl<Ext, H> HttpClientExt for Client<Ext, H>
where
    H: HttpClientExt + 'static,
    Ext: WasmCompatSend + WasmCompatSync + 'static,
{
    fn send<T, U>(
        &self,
        mut req: Request<T>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes> + WasmCompatSend,
        U: From<Bytes>,
        U: WasmCompatSend + 'static,
    {
        req.headers_mut().insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );

        self.http_client.send(req)
    }

    fn send_multipart<U>(
        &self,
        req: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes>,
        U: WasmCompatSend + 'static,
    {
        self.http_client.send_multipart(req)
    }

    fn send_streaming<T>(
        &self,
        mut req: Request<T>,
    ) -> impl Future<Output = http_client::Result<http_client::StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        req.headers_mut().insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );

        self.http_client.send_streaming(req)
    }
}

impl<Ext> Client<Ext, reqwest::Client>
where
    Ext: Provider,
    Ext::Builder: ProviderBuilder<Extension<reqwest::Client> = Ext> + Default,
{
    pub fn builder() -> ClientBuilder<Ext::Builder, Missing, reqwest::Client> {
        ClientBuilder {
            api_key: Missing,
            headers: Default::default(),
            base_url: <Ext::Builder as ProviderBuilder>::BASE_URL.into(),
            http_client: None,
            ext: Default::default(),
        }
    }
}

impl<Ext, H> Client<Ext, H>
where
    Ext: Provider,
{
    pub fn post<S>(&self, path: S) -> http_client::Result<Builder>
    where
        S: AsRef<str>,
    {
        let uri = self
            .ext
            .build_uri(&self.base_url, path.as_ref(), Transport::Http);

        let mut req = Request::post(uri);

        if let Some(hs) = req.headers_mut() {
            hs.extend(self.headers.iter().map(|(k, v)| (k.clone(), v.clone())));
        }

        self.ext.with_custom(req)
    }

    pub fn post_sse<S>(&self, path: S) -> http_client::Result<Builder>
    where
        S: AsRef<str>,
    {
        let uri = self
            .ext
            .build_uri(&self.base_url, path.as_ref(), Transport::Sse);

        let mut req = Request::post(uri);

        if let Some(hs) = req.headers_mut() {
            hs.extend(self.headers.iter().map(|(k, v)| (k.clone(), v.clone())));
        }

        self.ext.with_custom(req)
    }

    pub fn get_sse<S>(&self, path: S) -> http_client::Result<Builder>
    where
        S: AsRef<str>,
    {
        let uri = self
            .ext
            .build_uri(&self.base_url, path.as_ref(), Transport::Sse);

        let mut req = Request::get(uri);

        if let Some(hs) = req.headers_mut() {
            hs.extend(self.headers.iter().map(|(k, v)| (k.clone(), v.clone())));
        }

        self.ext.with_custom(req)
    }

    pub fn get<S>(&self, path: S) -> http_client::Result<Builder>
    where
        S: AsRef<str>,
    {
        let uri = self
            .ext
            .build_uri(&self.base_url, path.as_ref(), Transport::Http);

        let mut req = Request::get(uri);

        if let Some(hs) = req.headers_mut() {
            hs.extend(self.headers.iter().map(|(k, v)| (k.clone(), v.clone())));
        }

        self.ext.with_custom(req)
    }
}

impl<Ext, H> VerifyClient for Client<Ext, H>
where
    H: HttpClientExt,
    Ext: DebugExt + Provider + WasmCompatSync,
{
    async fn verify(&self) -> Result<(), VerifyError> {
        use http::StatusCode;

        let req = self
            .get(Ext::VERIFY_PATH)?
            .body(http_client::NoBody)
            .map_err(http_client::Error::from)?;

        let response = self.http_client.send(req).await?;

        match response.status() {
            StatusCode::OK => Ok(()),
            StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN => {
                Err(VerifyError::InvalidAuthentication)
            }
            StatusCode::INTERNAL_SERVER_ERROR => {
                let text = http_client::text(response).await?;
                Err(VerifyError::ProviderError(text))
            }
            status if status.as_u16() == 529 => {
                let text = http_client::text(response).await?;
                Err(VerifyError::ProviderError(text))
            }
            _ => {
                let status = response.status();

                if status.is_success() {
                    Ok(())
                } else {
                    let text: String = String::from_utf8_lossy(&response.into_body().await?).into();
                    Err(VerifyError::HttpError(http_client::Error::Instance(
                        format!("Failed with '{status}': {text}").into(),
                    )))
                }
            }
        }
    }
}

// ApiKey is generic because Anthropic uses custom auth header, local models like Ollama use none
#[derive(Clone)]
pub struct ClientBuilder<Ext, ApiKey = Missing, H = reqwest::Client> {
    base_url: String,
    api_key: ApiKey,
    headers: HeaderMap,
    http_client: Option<H>,
    ext: Ext,
}

impl<ExtBuilder, H> Default for ClientBuilder<ExtBuilder, Missing, H>
where
    H: Default,
    ExtBuilder: ProviderBuilder + Default,
{
    fn default() -> Self {
        Self {
            api_key: Missing,
            headers: Default::default(),
            base_url: ExtBuilder::BASE_URL.into(),
            http_client: None,
            ext: Default::default(),
        }
    }
}

impl<Ext, H> ClientBuilder<Ext, Missing, H> {
    /// Set the API key for this client. This *must* be done before the `build` method can be
    /// called
    pub fn api_key<ApiKey>(self, api_key: impl Into<ApiKey>) -> ClientBuilder<Ext, ApiKey, H> {
        ClientBuilder {
            api_key: api_key.into(),
            base_url: self.base_url,
            headers: self.headers,
            http_client: self.http_client,
            ext: self.ext,
        }
    }
}

impl<Ext, ApiKey, H> ClientBuilder<Ext, ApiKey, H>
where
    Ext: Clone,
{
    /// Owned map over the ext field
    pub(crate) fn over_ext<F, NewExt>(self, f: F) -> ClientBuilder<NewExt, ApiKey, H>
    where
        F: FnOnce(Ext) -> NewExt,
    {
        let ClientBuilder {
            base_url,
            api_key,
            headers,
            http_client,
            ext,
        } = self;

        let new_ext = f(ext.clone());

        ClientBuilder {
            base_url,
            api_key,
            headers,
            http_client,
            ext: new_ext,
        }
    }

    /// Set the base URL for this client
    pub fn base_url<S>(self, base_url: S) -> Self
    where
        S: AsRef<str>,
    {
        Self {
            base_url: base_url.as_ref().to_string(),
            ..self
        }
    }

    /// Set the HTTP backend used in this client
    pub fn http_client<U>(self, http_client: U) -> ClientBuilder<Ext, ApiKey, U> {
        ClientBuilder {
            http_client: Some(http_client),
            base_url: self.base_url,
            api_key: self.api_key,
            headers: self.headers,
            ext: self.ext,
        }
    }

    /// Set the HTTP headers used in this client
    pub fn http_headers(self, headers: HeaderMap) -> Self {
        Self { headers, ..self }
    }

    pub(crate) fn headers_mut(&mut self) -> &mut HeaderMap {
        &mut self.headers
    }

    pub(crate) fn ext_mut(&mut self) -> &mut Ext {
        &mut self.ext
    }
}

impl<Ext, ApiKey, H> ClientBuilder<Ext, ApiKey, H> {
    pub(crate) fn get_api_key(&self) -> &ApiKey {
        &self.api_key
    }
}

impl<Ext, Key, H> ClientBuilder<Ext, Key, H> {
    pub fn ext(&self) -> &Ext {
        &self.ext
    }

    pub fn get_base_url(&self) -> &str {
        &self.base_url
    }
}

impl<ExtBuilder, Key, H> ClientBuilder<ExtBuilder, Key, H>
where
    ExtBuilder: ProviderBuilder<ApiKey = Key>,
    Key: ApiKey,
    H: Default + HttpClientExt,
{
    pub fn build(mut self) -> http_client::Result<Client<ExtBuilder::Extension<H>, H>> {
        let ext_builder = self.ext.clone();

        self = ext_builder.finish(self)?;
        let ext = ExtBuilder::build(&self)?;

        let ClientBuilder {
            http_client,
            base_url,
            mut headers,
            api_key,
            ..
        } = self;

        if let Some((k, v)) = api_key.into_header().transpose()?
            && !headers.contains_key(&k)
        {
            headers.insert(k, v);
        }

        let http_client = http_client.unwrap_or_default();

        Ok(Client {
            http_client,
            base_url: Arc::from(base_url.as_str()),
            headers: Arc::new(headers),
            ext,
        })
    }
}

impl<M, Ext, H> CompletionClient for Client<Ext, H>
where
    Ext: Capabilities<H, Completion = Capable<M>>,
    M: CompletionModel<Client = Self>,
{
    type CompletionModel = M;

    fn completion_model(&self, model: impl Into<String>) -> Self::CompletionModel {
        M::make(self, model)
    }
}

impl<M, Ext, H> EmbeddingsClient for Client<Ext, H>
where
    Ext: Capabilities<H, Embeddings = Capable<M>>,
    M: EmbeddingModel<Client = Self>,
{
    type EmbeddingModel = M;

    fn embedding_model(&self, model: impl Into<String>) -> Self::EmbeddingModel {
        M::make(self, model, None)
    }

    fn embedding_model_with_ndims(
        &self,
        model: impl Into<String>,
        ndims: usize,
    ) -> Self::EmbeddingModel {
        M::make(self, model, Some(ndims))
    }
}

impl<M, Ext, H> TranscriptionClient for Client<Ext, H>
where
    Ext: Capabilities<H, Transcription = Capable<M>>,
    M: TranscriptionModel<Client = Self> + WasmCompatSend,
{
    type TranscriptionModel = M;

    fn transcription_model(&self, model: impl Into<String>) -> Self::TranscriptionModel {
        M::make(self, model)
    }
}

#[cfg(feature = "image")]
impl<M, Ext, H> ImageGenerationClient for Client<Ext, H>
where
    Ext: Capabilities<H, ImageGeneration = Capable<M>>,
    M: ImageGenerationModel<Client = Self>,
{
    type ImageGenerationModel = M;

    fn image_generation_model(&self, model: impl Into<String>) -> Self::ImageGenerationModel {
        M::make(self, model)
    }
}

#[cfg(feature = "audio")]
impl<M, Ext, H> AudioGenerationClient for Client<Ext, H>
where
    Ext: Capabilities<H, AudioGeneration = Capable<M>>,
    M: AudioGenerationModel<Client = Self>,
{
    type AudioGenerationModel = M;

    fn audio_generation_model(&self, model: impl Into<String>) -> Self::AudioGenerationModel {
        M::make(self, model)
    }
}

impl<M, Ext, H> ModelListingClient for Client<Ext, H>
where
    Ext: Capabilities<H, ModelListing = Capable<M>> + Clone,
    M: ModelLister<H, Client = Self> + WasmCompatSend + WasmCompatSync + Clone + 'static,
    H: WasmCompatSend + WasmCompatSync + Clone,
{
    fn list_models(
        &self,
    ) -> impl std::future::Future<
        Output = Result<crate::model::ModelList, crate::model::ModelListingError>,
    > + WasmCompatSend {
        let lister = M::new(self.clone());
        async move { lister.list_all().await }
    }
}

#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
mod wasm_model_listing_compile_checks {
    use super::{ModelListingClient, Nothing};
    use crate::{
        http_client::{self, HttpClientExt, LazyBody, MultipartForm, Request, Response},
        providers::{anthropic, deepseek, mistral, ollama, openai, openrouter},
        wasm_compat::WasmCompatSend,
    };
    use bytes::Bytes;
    use std::{
        future::{self, Future},
        marker::PhantomData,
        rc::Rc,
    };

    #[derive(Clone, Default)]
    struct WasmOnlyHttpClient {
        _not_send_sync: PhantomData<Rc<()>>,
    }

    impl HttpClientExt for WasmOnlyHttpClient {
        fn send<T, U>(
            &self,
            _req: Request<T>,
        ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
        where
            T: Into<Bytes> + WasmCompatSend,
            U: From<Bytes> + WasmCompatSend + 'static,
        {
            future::ready(Err(http_client::Error::StreamEnded))
        }

        fn send_multipart<U>(
            &self,
            _req: Request<MultipartForm>,
        ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
        where
            U: From<Bytes> + WasmCompatSend + 'static,
        {
            future::ready(Err(http_client::Error::StreamEnded))
        }

        fn send_streaming<T>(
            &self,
            _req: Request<T>,
        ) -> impl Future<Output = http_client::Result<http_client::StreamingResponse>> + WasmCompatSend
        where
            T: Into<Bytes> + WasmCompatSend,
        {
            future::ready(Err(http_client::Error::StreamEnded))
        }
    }

    fn assert_model_listing_client<C>(client: C)
    where
        C: ModelListingClient,
    {
        let _ = client.list_models();
    }

    fn assert_simple_model_listers_accept_wasm_only_http_clients() {
        let _ = openrouter::Client::builder()
            .api_key("dummy-key")
            .http_client(WasmOnlyHttpClient::default())
            .build()
            .map(assert_model_listing_client);

        let _ = openai::Client::builder()
            .api_key("dummy-key")
            .http_client(WasmOnlyHttpClient::default())
            .build()
            .map(assert_model_listing_client);

        let _ = mistral::Client::builder()
            .api_key("dummy-key")
            .http_client(WasmOnlyHttpClient::default())
            .build()
            .map(assert_model_listing_client);

        let _ = anthropic::Client::builder()
            .api_key("dummy-key")
            .http_client(WasmOnlyHttpClient::default())
            .build()
            .map(assert_model_listing_client);

        let _ = ollama::Client::builder()
            .api_key(Nothing)
            .http_client(WasmOnlyHttpClient::default())
            .build()
            .map(assert_model_listing_client);

        let _ = deepseek::Client::builder()
            .api_key("dummy-key")
            .http_client(WasmOnlyHttpClient::default())
            .build()
            .map(assert_model_listing_client);
    }

    #[allow(dead_code)]
    fn compile_assertions() {
        assert_simple_model_listers_accept_wasm_only_http_clients();
    }
}

#[cfg(test)]
mod tests {
    use crate::providers::anthropic;

    /// Type-level test that `Client::builder()` methods do not require annotation to determine
    /// backig HTTP client
    #[test]
    fn ensures_client_builder_no_annotation() {
        let http_client = reqwest::Client::default();
        let _ = anthropic::Client::builder()
            .http_client(http_client)
            .api_key("Foo")
            .build()
            .unwrap();
    }
}
