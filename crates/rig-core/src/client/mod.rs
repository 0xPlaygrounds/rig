//! Provider clients: one [`Provider`] trait, one `Has*` trait per capability,
//! and the generic [`Client<P, H>`] they attach to.
//!
//! A provider is a value implementing [`Provider`]: its base URL, its API-key
//! type, its builder settings ([`Provider::Config`]), how it assembles request
//! URIs and customises requests, and how it is built from the environment.
//! Each capability the provider offers is one more trait implementation —
//! [`HasCompletion`], [`HasEmbeddings`], [`HasRerank`], [`HasTranscription`],
//! [`HasModelListing`], [`HasImageGeneration`], [`HasAudioGeneration`] — naming
//! the concrete model type and constructing it from a client. The blanket
//! impls in this module turn those into the user-facing
//! [`CompletionClient`], [`EmbeddingsClient`], … traits on `Client<P, H>`, so
//! `client.completion_model(id)` returns the provider's own model type. A
//! capability a provider lacks is a trait it does not implement; the compiler
//! reports the missing method.
//!
//! `H` is the HTTP transport, any [`HttpClientExt`]. rig-core ships no
//! concrete transport: construct with [`Client::new_with`] /
//! [`ClientBuilder::http_client`], or use `rig-reqwest`'s conveniences
//! (re-exported through the `rig` facade prelude), which supply `new(key)`,
//! `from_env()`, and a transport-less `build()`. In type position `H`
//! defaults to the erased [`BoxedHttpClient`].
//!
//! # Writing a provider
//!
//! ```
//! use rig_core::client::{
//!     BearerAuth, Client, ClientBuilder, CompletionClient, HasCompletion, ModelTransport,
//!     Provider, ProviderClientResult,
//! };
//! use rig_core::completion::{CompletionError, CompletionModel, CompletionRequest, CompletionResponse};
//! use rig_core::http_client::HttpClientExt;
//! use rig_core::streaming::StreamingCompletionResponse;
//!
//! #[derive(Debug, Clone, Default)]
//! struct Example;
//!
//! impl Provider for Example {
//!     const NAME: &'static str = "example";
//!     const BASE_URL: &'static str = "https://example.invalid/v1";
//!     const VERIFY_PATH: &'static str = "/models";
//!     type ApiKey = BearerAuth;
//!     type Config = ();
//!     type EnvInput = String;
//!
//!     fn build(_: (), _: &BearerAuth) -> rig_core::http_client::Result<Self> {
//!         Ok(Example)
//!     }
//!     fn from_env<H: HttpClientExt>(http: H) -> ProviderClientResult<Client<Self, H>> {
//!         Client::from_env_api_key("EXAMPLE_API_KEY", None, http)
//!     }
//!     fn from_val<H: HttpClientExt>(key: String, http: H) -> ProviderClientResult<Client<Self, H>> {
//!         Client::new_with(key, http)
//!     }
//! }
//!
//! #[derive(Clone)]
//! struct ExampleModel<H> {
//!     client: Client<Example, H>,
//!     model: String,
//! }
//!
//! impl<H: ModelTransport> CompletionModel for ExampleModel<H> {
//!     async fn completion(&self, _: CompletionRequest) -> Result<CompletionResponse, CompletionError> {
//!         Err(CompletionError::ProviderError(self.model.clone()))
//!     }
//!     async fn stream(&self, _: CompletionRequest) -> Result<StreamingCompletionResponse, CompletionError> {
//!         Err(CompletionError::ProviderError(self.model.clone()))
//!     }
//! }
//!
//! impl HasCompletion for Example {
//!     type Model<H> = ExampleModel<H> where H: ModelTransport;
//!     fn completion_model<H: ModelTransport>(client: &Client<Self, H>, model: String) -> ExampleModel<H> {
//!         ExampleModel { client: client.clone(), model }
//!     }
//! }
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let client = Client::<Example, rig_core::markers::Missing>::builder()
//!     .api_key("k")
//!     .http_client(rig_core::test_utils::RecordingHttpClient::new(""))
//!     .build()?;
//! let model: ExampleModel<_> = client.completion_model("m");
//! assert_eq!(model.model, "m");
//! # Ok(())
//! # }
//! ```

pub mod audio_generation;
pub mod completion;
pub mod embeddings;
pub mod image_generation;
pub mod model_listing;
pub mod rerank;
pub mod transcription;
pub mod verify;

use bytes::Bytes;
pub use completion::{CompletionClient, HasCompletion};
pub use embeddings::{EmbeddingsClient, HasEmbeddings};
use http::{HeaderMap, HeaderName, HeaderValue};
pub use model_listing::{HasModelListing, ModelLister, ModelListingClient};
pub use rerank::{HasRerank, RerankingClient};
use std::{env::VarError, fmt::Debug, sync::Arc};
use thiserror::Error;
pub use transcription::{HasTranscription, TranscriptionClient};
pub use verify::{VerifyClient, VerifyError};

#[cfg(feature = "image")]
pub use image_generation::{HasImageGeneration, ImageGenerationClient};

#[cfg(feature = "audio")]
pub use audio_generation::{AudioGenerationClient, HasAudioGeneration};

use crate::{
    http_client::BoxedHttpClient,
    http_client::{
        self, Builder, HttpClientExt, LazyBody, MultipartForm, Request, Response, make_auth_header,
    },
    markers::Missing,
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

/// Errors returned while constructing provider clients from environment variables or explicit input.
///
/// Provider-specific client constructors use this error for configuration problems that can be
/// detected before any model request is sent, such as missing API keys, invalid environment
/// values, or invalid builder configuration.
#[derive(Debug, Error)]
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
    /// [`ClientBuilder::build`] was called without [`ClientBuilder::api_key`]
    /// for a provider whose key type has no "no credential" value. The payload
    /// is [`Provider::NAME`].
    #[error("{0}: no API key was supplied; call `ClientBuilder::api_key` before `build`")]
    MissingApiKey(&'static str),
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

/// A trait for API key inputs accepted by [`ClientBuilder::api_key`].
///
/// Returning `Some` from [`Self::into_header`] inserts a default header into
/// the generic [`Client`]. Returning `None` leaves credentials to the provider
/// (query-string keys go through [`Provider::build_uri`], per-request headers
/// and token exchange through [`Provider::prepare`]).
pub trait ApiKey: Clone + Sized {
    /// Convert this key into a default request header, if the generic client
    /// should own that authentication header.
    fn into_header(self) -> Option<http_client::Result<(HeaderName, HeaderValue)>> {
        None
    }

    /// The value [`ClientBuilder::build`] uses when [`ClientBuilder::api_key`]
    /// was never called. `None` (the default) makes the key mandatory and
    /// `build` fails with [`ProviderClientError::MissingApiKey`]; a key type
    /// with a genuine "no credential" state ([`Nothing`], an optional local
    /// server key) returns `Some` of it.
    fn absent() -> Option<Self> {
        None
    }
}

/// An API key which will be inserted into a `Client`'s default headers as a bearer auth token
#[derive(Clone)]
pub struct BearerAuth(String);

impl ApiKey for BearerAuth {
    fn into_header(self) -> Option<http_client::Result<(HeaderName, HeaderValue)>> {
        Some(make_auth_header(self.0))
    }
}

impl Debug for BearerAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("BearerAuth(<redacted>)")
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
/// the lack of a field (an API key, for instance)
#[derive(Debug, Default, Clone, Copy)]
pub struct Nothing;

impl ApiKey for Nothing {
    fn absent() -> Option<Self> {
        Some(Nothing)
    }
}

/// A provider: everything rig-core needs to know to build a [`Client`] for it
/// and address its API. Implemented once per provider on a small value type
/// (`openai::OpenAIResponses`, `anthropic::Anthropic`, …); the capability
/// traits ([`HasCompletion`] and friends) are implemented alongside it.
///
/// Provider values are stored inside every [`Client`] and cloned into every
/// model, so they must be cheap to clone. Credentials held here must be
/// redacted by the provider's own [`Debug`] impl.
pub trait Provider: Clone + Debug + WasmCompatSend + WasmCompatSync + 'static {
    /// Provider name used in construction errors and [`Client`]'s debug
    /// output, e.g. `"openai"`.
    const NAME: &'static str;
    /// Default base URL a [`ClientBuilder`] starts from.
    const BASE_URL: &'static str;
    /// Provider endpoint used by [`VerifyClient`] to validate credentials.
    const VERIFY_PATH: &'static str;

    /// What [`ClientBuilder::api_key`] and [`Client::new_with`] accept:
    /// [`BearerAuth`], [`Nothing`], or a provider key type.
    type ApiKey: ApiKey;
    /// Provider-specific builder settings, reachable through
    /// [`ClientBuilder::config`] / [`ClientBuilder::config_mut`]; `()` for
    /// providers that have none.
    type Config: Default + Clone;
    /// What [`Self::from_val`] accepts.
    type EnvInput;

    /// Build the provider value from its settings and the key the builder was
    /// given. Providers that carry the key themselves (query-string auth,
    /// token exchange) copy it out here.
    fn build(config: Self::Config, api_key: &Self::ApiKey) -> http_client::Result<Self>;

    /// Last look at the builder before the client is assembled: default
    /// headers, base-URL normalisation. The default is the identity.
    fn finish<H>(
        &self,
        builder: ClientBuilder<Self, H>,
    ) -> http_client::Result<ClientBuilder<Self, H>> {
        Ok(builder)
    }

    /// Build a client for this provider from the process's environment,
    /// sending through `http`. [`Client::from_env_api_key`] covers the common
    /// "one key variable, optional base-URL variable" case.
    fn from_env<H>(http: H) -> ProviderClientResult<Client<Self, H>>
    where
        H: HttpClientExt;

    /// Build a client for this provider from an explicit input value, sending
    /// through `http`.
    fn from_val<H>(input: Self::EnvInput, http: H) -> ProviderClientResult<Client<Self, H>>
    where
        H: HttpClientExt;

    /// Build a complete request URI for the given base URL and provider path.
    /// The default joins them with a single `/`; a provider that authenticates
    /// through the query string appends its key here.
    fn build_uri(&self, base_url: &str, path: &str) -> String {
        // Some providers (like Azure) have a blank base URL to allow users to input their own endpoints.
        let base_url = if base_url.is_empty() || base_url.ends_with('/') {
            base_url.to_string()
        } else {
            // Only add a slash to the base_url when it doesn't already end with a slash
            base_url.to_string() + "/"
        };

        base_url + path.trim_start_matches('/')
    }

    /// Per-request customisation applied after the client's default headers:
    /// per-request auth headers, `Accept` overrides. The default is the
    /// identity.
    fn prepare(&self, req: http_client::Builder) -> http_client::Result<http_client::Builder> {
        Ok(req)
    }
}

/// The one bound set a transport must satisfy for a provider model to be
/// built over it: every `Has*` trait's `Model<H>` is declared against this,
/// and nothing else names the set. Implemented automatically.
pub trait ModelTransport:
    HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static
{
}

impl<H> ModelTransport for H where
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static
{
}

/// Generic provider client shared by Rig provider integrations.
///
/// `P` is the [`Provider`]: URL construction, request customisation, and (via
/// its `Has*` impls) which models this client can build. `H` is the HTTP
/// backend — any [`HttpClientExt`] implementation. rig-core has no default
/// *concrete* transport: construct with [`Client::new_with`] /
/// [`ClientBuilder::http_client`], or use the bundled `reqwest` transport's
/// conveniences (`rig-reqwest`, re-exported by the `rig` facade) which pin
/// `H` for you. In type position `H` defaults to the erased
/// [`BoxedHttpClient`], so `Client<P>` means "any transport" — the shape a
/// host that owns one transport for many providers holds (see
/// [`Client::boxed`]). The default does not apply in expression position, so
/// `Client::new_with(..)` still infers `H` from its argument.
#[derive(Clone)]
pub struct Client<P, H = BoxedHttpClient> {
    base_url: Arc<str>,
    headers: Arc<HeaderMap>,
    http_client: H,
    provider: P,
}

impl<P, H> Debug for Client<P, H>
where
    P: Provider,
    H: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("provider", &self.provider)
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
            .field("http_client", &self.http_client)
            .finish()
    }
}

/// Construction with an explicit transport. rig-core never chooses a transport
/// for you; the bundled `reqwest` one lives in `rig-reqwest`, whose
/// `DefaultTransportClient` (re-exported via the `rig` facade prelude) supplies
/// the one-argument `new(api_key)` on top of this.
impl<P, H> Client<P, H>
where
    P: Provider,
    H: HttpClientExt,
{
    /// Construct a provider client that sends through `http`.
    pub fn new_with(api_key: impl Into<P::ApiKey>, http: H) -> ProviderClientResult<Self> {
        Client::<P, Missing>::builder()
            .api_key(api_key)
            .http_client(http)
            .build()
    }

    /// Construct a provider client from the environment for the common
    /// provider shape: a required `api_key_env` variable holding the key and,
    /// when `base_url_env` is given, an optional variable overriding
    /// [`Provider::BASE_URL`].
    pub fn from_env_api_key(
        api_key_env: &'static str,
        base_url_env: Option<&'static str>,
        http: H,
    ) -> ProviderClientResult<Self>
    where
        P::ApiKey: From<String>,
    {
        let api_key = required_env_var(api_key_env)?;
        let mut builder = Client::<P, Missing>::builder().api_key(api_key);
        if let Some(base_url) = base_url_env.map(optional_env_var).transpose()?.flatten() {
            builder = builder.base_url(base_url);
        }
        builder.http_client(http).build()
    }
}

impl<P, H> Client<P, H>
where
    P: Provider,
    H: HttpClientExt + 'static,
{
    /// Erase this client's transport behind [`BoxedHttpClient`].
    ///
    /// The result is the same client — base URL, headers, provider — sending
    /// through the same transport, but its type no longer names `H`. A host
    /// that builds clients for several providers over one transport uses this
    /// so every client it holds is a `Client<P>`. Boxing an already boxed
    /// client is a no-op clone of the transport handle.
    pub fn boxed(self) -> Client<P, BoxedHttpClient> {
        Client {
            base_url: self.base_url,
            headers: self.headers,
            http_client: BoxedHttpClient::new(self.http_client),
            provider: self.provider,
        }
    }
}

impl<P, H> Client<P, H> {
    /// Returns the configured provider base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Returns default headers applied to outgoing provider requests.
    pub fn headers(&self) -> &HeaderMap {
        &self.headers
    }

    /// Returns the provider value.
    pub fn provider(&self) -> &P {
        &self.provider
    }

    /// The HTTP transport this client sends through, for callers that must
    /// talk to an absolute URL outside the provider's API base (OAuth/device
    /// flows) with the same transport.
    pub fn http_client(&self) -> &H {
        &self.http_client
    }

    /// Reuse this client's base URL, headers, and HTTP backend with a different provider.
    pub fn with_provider<Q>(self, provider: Q) -> Client<Q, H> {
        Client {
            base_url: self.base_url,
            headers: self.headers,
            http_client: self.http_client,
            provider,
        }
    }
}

impl<P, H> Client<P, H>
where
    P: Provider,
{
    fn request(&self, method: http::Method, path: &str) -> http_client::Result<Builder> {
        let uri = self.provider.build_uri(&self.base_url, path);

        let mut req = Request::builder().method(method).uri(uri);

        if let Some(hs) = req.headers_mut() {
            hs.extend(self.headers.iter().map(|(k, v)| (k.clone(), v.clone())));
        }

        self.provider.prepare(req)
    }

    /// Build a provider-customized POST request.
    pub fn post<S>(&self, path: S) -> http_client::Result<Builder>
    where
        S: AsRef<str>,
    {
        self.request(http::Method::POST, path.as_ref())
    }

    /// Build a provider-customized GET request.
    pub fn get<S>(&self, path: S) -> http_client::Result<Builder>
    where
        S: AsRef<str>,
    {
        self.request(http::Method::GET, path.as_ref())
    }

    /// Build a provider-customized PATCH request.
    ///
    /// REST resources that support partial update need this: Gemini's
    /// `cachedContents` only allows the expiry to be changed, and does it with
    /// `PATCH ?updateMask=ttl`.
    pub fn patch<S>(&self, path: S) -> http_client::Result<Builder>
    where
        S: AsRef<str>,
    {
        self.request(http::Method::PATCH, path.as_ref())
    }

    /// Build a provider-customized DELETE request.
    ///
    /// Needed by any provider resource with a lifecycle rather than a single
    /// call — a cached-content handle bills for storage until it is deleted, so
    /// deleting one is a first-class operation, not a convenience.
    pub fn delete<S>(&self, path: S) -> http_client::Result<Builder>
    where
        S: AsRef<str>,
    {
        self.request(http::Method::DELETE, path.as_ref())
    }
}

impl<P, H> HttpClientExt for Client<P, H>
where
    P: Provider,
    H: HttpClientExt + 'static,
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

/// `builder()` lives on `Client<P, Missing>` — the "no transport chosen yet"
/// state — so `provider::Client::builder()` resolves without an `H` annotation
/// (it is the only `builder` inherent fn, so `H` infers to `Missing`). The
/// returned builder's `H` slot is `Missing` too; [`ClientBuilder::http_client`]
/// must be called before [`ClientBuilder::build`] (or a transport crate's
/// default-substituting `build`, such as `rig-reqwest`'s
/// `DefaultTransportBuilder`).
impl<P> Client<P, Missing>
where
    P: Provider,
{
    /// Start constructing a provider client.
    pub fn builder() -> ClientBuilder<P, Missing> {
        ClientBuilder::default()
    }
}

impl<P, H> VerifyClient for Client<P, H>
where
    P: Provider,
    H: HttpClientExt,
{
    async fn verify(&self) -> Result<(), VerifyError> {
        use http::StatusCode;

        let req = self
            .get(P::VERIFY_PATH)?
            .body(http_client::NoBody)
            .map_err(http_client::Error::from)?;

        // The reqwest transport reports non-success as an error before this
        // status match can run (found live on rig#2315's error matrix: the
        // 401/403 arms below were dead and every bogus key surfaced as a raw
        // HttpError). Recover the status from the transport error so the
        // documented VerifyError classification actually fires.
        let response = match self.http_client.send(req).await {
            Ok(response) => response,
            Err(error) => {
                return Err(match error.non_success_status() {
                    Some(StatusCode::UNAUTHORIZED) | Some(StatusCode::FORBIDDEN) => {
                        VerifyError::InvalidAuthentication
                    }
                    _ => VerifyError::HttpError(error),
                });
            }
        };

        match response.status() {
            StatusCode::OK => Ok(()),
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
                Err(VerifyError::InvalidAuthentication)
            }
            // The failed response's headers are preserved on every branch, so
            // a caller can read rate-limit metadata such as `Retry-After` off
            // a rejected verification (rig#2210).
            StatusCode::INTERNAL_SERVER_ERROR => {
                let headers = Box::new(response.headers().clone());
                let body = http_client::text(response).await?;
                Err(VerifyError::HttpError(
                    http_client::Error::InvalidStatusCodeWithDetails {
                        status: StatusCode::INTERNAL_SERVER_ERROR,
                        body,
                        headers,
                    },
                ))
            }
            status if status.as_u16() == 529 => {
                let headers = Box::new(response.headers().clone());
                let body = http_client::text(response).await?;
                Err(VerifyError::HttpError(
                    http_client::Error::InvalidStatusCodeWithDetails {
                        status,
                        body,
                        headers,
                    },
                ))
            }
            _ => {
                let status = response.status();

                if status.is_success() {
                    Ok(())
                } else {
                    let headers = Box::new(response.headers().clone());
                    let body: String = String::from_utf8_lossy(&response.into_body().await?).into();
                    Err(VerifyError::HttpError(
                        http_client::Error::InvalidStatusCodeWithDetails {
                            status,
                            body,
                            headers,
                        },
                    ))
                }
            }
        }
    }
}

/// Builder for [`Client`].
///
/// `H = Missing` means the caller has not yet called [`Self::http_client`];
/// rig-core's own `build()` is only reachable once a concrete
/// [`HttpClientExt`] backend has been supplied. A transport crate may add a
/// default-substituting `build` for the `Missing` state (the bundled one is
/// `rig-reqwest`'s `DefaultTransportBuilder`, in the `rig` facade prelude).
///
/// The API key is an `Option`: [`Self::build`] fails with
/// [`ProviderClientError::MissingApiKey`] when the provider's key type has no
/// [`ApiKey::absent`] value and [`Self::api_key`] was never called.
#[derive(Clone)]
pub struct ClientBuilder<P: Provider, H = Missing> {
    base_url: String,
    api_key: Option<P::ApiKey>,
    headers: HeaderMap,
    http_client: H,
    config: P::Config,
}

impl<P> Default for ClientBuilder<P, Missing>
where
    P: Provider,
{
    fn default() -> Self {
        Self {
            api_key: None,
            headers: Default::default(),
            base_url: P::BASE_URL.into(),
            http_client: Missing,
            config: Default::default(),
        }
    }
}

impl<P, H> ClientBuilder<P, H>
where
    P: Provider,
{
    /// Set the API key for this client.
    pub fn api_key(self, api_key: impl Into<P::ApiKey>) -> Self {
        Self {
            api_key: Some(api_key.into()),
            ..self
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

    /// Set the HTTP backend used in this client.
    ///
    /// Calling this advances the builder's `H` slot from whatever it was (typically `Missing`)
    /// to the supplied client's type, which selects the H-generic [`Self::build`] impl below.
    pub fn http_client<U>(self, http_client: U) -> ClientBuilder<P, U> {
        ClientBuilder {
            http_client,
            base_url: self.base_url,
            api_key: self.api_key,
            headers: self.headers,
            config: self.config,
        }
    }

    /// Set the HTTP headers used in this client
    pub fn http_headers(self, headers: HeaderMap) -> Self {
        Self { headers, ..self }
    }

    /// Default headers accumulated so far; [`Provider::finish`] adds to them.
    pub fn headers_mut(&mut self) -> &mut HeaderMap {
        &mut self.headers
    }

    /// The provider-specific settings.
    pub fn config(&self) -> &P::Config {
        &self.config
    }

    /// The provider-specific settings, for a provider module's builder setters.
    pub fn config_mut(&mut self) -> &mut P::Config {
        &mut self.config
    }

    /// Owned map over the provider-specific settings.
    pub fn map_config<F>(mut self, f: F) -> Self
    where
        F: FnOnce(P::Config) -> P::Config,
    {
        let config = std::mem::take(&mut self.config);
        self.config = f(config);
        self
    }

    /// Returns the configured base URL.
    pub fn get_base_url(&self) -> &str {
        &self.base_url
    }
}

/// `build`: the caller supplied an HTTP client via [`ClientBuilder::http_client`], so `H` is a
/// real `HttpClientExt` type and we use it directly.
impl<P, H> ClientBuilder<P, H>
where
    P: Provider,
    H: HttpClientExt,
{
    /// Build a client using the HTTP backend supplied with [`ClientBuilder::http_client`].
    pub fn build(mut self) -> ProviderClientResult<Client<P, H>> {
        let api_key = match self.api_key.take() {
            Some(key) => key,
            None => P::ApiKey::absent().ok_or(ProviderClientError::MissingApiKey(P::NAME))?,
        };
        let provider = P::build(self.config.clone(), &api_key)?;
        self = provider.finish(self)?;

        let ClientBuilder {
            http_client,
            base_url,
            mut headers,
            ..
        } = self;

        if let Some((k, v)) = api_key.into_header().transpose()?
            && !headers.contains_key(&k)
        {
            headers.insert(k, v);
        }

        Ok(Client {
            http_client,
            base_url: Arc::from(base_url.as_str()),
            headers: Arc::new(headers),
            provider,
        })
    }
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
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

    // Only referenced on the wasm target's compile pass; native builds see it
    // as dead code.
    #[allow(dead_code)]
    fn compile_assertions() {
        assert_simple_model_listers_accept_wasm_only_http_clients();
    }
}

#[cfg(test)]
mod tests;
