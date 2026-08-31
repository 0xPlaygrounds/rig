//! This module provides traits for defining and creating provider clients.
//! Clients are used to create models for completion, embeddings, etc.

pub mod audio_generation;
pub mod completion;
pub mod embeddings;
pub mod image_generation;
pub mod model_listing;
pub mod rerank;
pub mod transcription;
pub mod verify;

#[cfg(feature = "reqwest")]
mod default_transport;

use bytes::Bytes;
pub use completion::{CompletionClient, ConstructCompletionModel};
pub use embeddings::{ConstructEmbeddingModel, EmbeddingsClient};
use http::{HeaderMap, HeaderName, HeaderValue};
pub use model_listing::{ConstructModelLister, ModelLister, ModelListingClient};
pub use rerank::{ConstructRerankModel, RerankingClient};
use std::{env::VarError, fmt::Debug, marker::PhantomData, sync::Arc};
use thiserror::Error;
pub use transcription::ConstructTranscriptionModel;
pub use verify::{VerifyClient, VerifyError};

#[cfg(feature = "image")]
use crate::image_generation::ImageGenerationModel;
#[cfg(feature = "image")]
pub use image_generation::{ConstructImageGenerationModel, ImageGenerationClient};

#[cfg(feature = "audio")]
use crate::audio_generation::*;
#[cfg(feature = "audio")]
pub use audio_generation::{AudioGenerationClient, ConstructAudioGenerationModel};

use crate::{
    completion::CompletionModel,
    embeddings::EmbeddingModel,
    http_client::BoxedHttpClient,
    http_client::{
        self, Builder, HttpClientExt, LazyBody, MultipartForm, Request, Response, make_auth_header,
    },
    markers::Missing,
    prelude::TranscriptionClient,
    rerank::RerankModel,
    transcription::TranscriptionModel,
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

/// Provider-specific environment configuration, generic over the transport.
///
/// Implemented by each provider *extension* type. This is where rig-core keeps
/// the knowledge of which environment variables configure a provider; it
/// never picks a transport itself. A transport crate supplies the ergonomic
/// `from_env()` / `from_val()` by calling these with its client — see
/// the inherent default-transport constructors, available through the `rig`
/// facade prelude.
pub trait ProviderFromEnv: Provider {
    /// Provider-specific input accepted by [`Self::from_val_with`].
    type Input;

    /// Build a client for this provider from the process's environment,
    /// sending through `http`.
    fn from_env_with<H>(http: H) -> Result<Client<Self, H>, ProviderClientError>
    where
        H: HttpClientExt,
        Self::Builder: ProviderBuilder<Extension<H> = Self>;

    /// Build a client for this provider from an explicit input value,
    /// sending through `http`.
    fn from_val_with<H>(
        input: Self::Input,
        http: H,
    ) -> Result<Client<Self, H>, ProviderClientError>
    where
        H: HttpClientExt,
        Self::Builder: ProviderBuilder<Extension<H> = Self>;

    /// [`Self::from_env_with`], with the transport erased behind
    /// [`BoxedHttpClient`] so the returned client names no concrete `H`.
    fn from_env_boxed<H>(http: H) -> Result<Client<Self, BoxedHttpClient>, ProviderClientError>
    where
        H: HttpClientExt + 'static,
        Self::Builder: ProviderBuilder<Extension<BoxedHttpClient> = Self>,
    {
        Self::from_env_with(BoxedHttpClient::new(http))
    }

    /// [`Self::from_val_with`], with the transport erased behind
    /// [`BoxedHttpClient`] so the returned client names no concrete `H`.
    fn from_val_boxed<H>(
        input: Self::Input,
        http: H,
    ) -> Result<Client<Self, BoxedHttpClient>, ProviderClientError>
    where
        H: HttpClientExt + 'static,
        Self::Builder: ProviderBuilder<Extension<BoxedHttpClient> = Self>,
    {
        Self::from_val_with(input, BoxedHttpClient::new(http))
    }
}

/// A trait for API key inputs accepted by [`ClientBuilder::api_key`].
///
/// Returning `Some` inserts a header into the generic [`Client`]. Returning `None`
/// lets the provider extension handle credentials itself.
pub trait ApiKey: Sized {
    /// Convert this key into a default request header, if the generic client
    /// should own that authentication header.
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

#[derive(Clone)]
/// Generic provider client shared by Rig provider integrations.
///
/// `Ext` stores provider-specific behavior such as URL construction, request
/// customization, and capabilities. `H` is the HTTP backend — any
/// [`crate::http_client::HttpClientExt`] implementation. rig-core has no
/// default *concrete* transport: construct with [`Client::new_with`] /
/// [`ClientBuilder::http_client`], or use the bundled `reqwest` transport's
/// conveniences (the `reqwest` feature, on by default in the `rig` facade) which pin
/// `H` for you. In type position `H` defaults to the erased
/// [`BoxedHttpClient`], so `Client<Ext>` means "any transport" — the shape a
/// host that owns one transport for many providers holds (see
/// [`Client::boxed`] and [`ProviderFromEnv::from_env_boxed`]). The default
/// does not apply in expression position, so `Client::new_with(..)` still
/// infers `H` from its argument.
pub struct Client<Ext = Nothing, H = BoxedHttpClient> {
    base_url: Arc<str>,
    headers: Arc<HeaderMap>,
    http_client: H,
    ext: Ext,
}

/// Provider extension hook for redacted [`Debug`] output.
pub trait DebugExt: Debug {
    /// Additional provider-specific fields to include in `Client` debug output.
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
    /// Regular request/response HTTP transport.
    Http,
    /// Server-sent events streaming transport.
    Sse,
}

/// An API provider extension, this abstracts over extensions which may be used in conjunction with
/// the `Client<Ext, H>` struct to define the behavior of a provider with respect to networking,
/// auth, instantiating models
pub trait Provider: Sized {
    /// The builder type that constructs this provider extension.
    /// This associates extensions with their builders for type inference.
    type Builder: ProviderBuilder;

    /// Provider endpoint used by [`VerifyClient`] to validate credentials.
    const VERIFY_PATH: &'static str;

    /// Build a complete request URI for the given base URL, provider path, and transport.
    fn build_uri(&self, base_url: &str, path: &str, _transport: Transport) -> String {
        // Some providers (like Azure) have a blank base URL to allow users to input their own endpoints.
        let base_url = if base_url.is_empty() || base_url.ends_with('/') {
            base_url.to_string()
        } else {
            // Only add a slash to the base_url when it doesn't already end with a slash
            base_url.to_string() + "/"
        };

        base_url + path.trim_start_matches('/')
    }

    /// Apply provider-specific request customization before sending.
    fn with_custom(&self, req: http_client::Builder) -> http_client::Result<http_client::Builder> {
        Ok(req)
    }
}

/// A wrapper type providing runtime checks on a provider's capabilities via the [Capability] trait
pub struct Capable<M>(PhantomData<M>);

/// Type-level marker for whether a provider supports a capability.
pub trait Capability {
    /// Whether this marker represents a supported capability.
    const CAPABLE: bool;
}

impl<M> Capability for Capable<M> {
    const CAPABLE: bool = true;
}

impl Capability for Nothing {
    const CAPABLE: bool = false;
}

/// The capabilities of a given provider, i.e. embeddings, audio transcriptions, text completion
pub trait Capabilities<H> {
    /// Completion model capability marker.
    type Completion: Capability;
    /// Embedding model capability marker.
    type Embeddings: Capability;
    /// Rerank model capability marker.
    type Rerank: Capability;
    /// Audio transcription model capability marker.
    type Transcription: Capability;
    /// Model listing capability marker.
    type ModelListing: Capability;
    #[cfg(feature = "image")]
    /// Image generation model capability marker.
    type ImageGeneration: Capability;
    #[cfg(feature = "audio")]
    /// Audio generation model capability marker.
    type AudioGeneration: Capability;
}

/// An API provider extension *builder*, this abstracts over provider-specific builders which are
/// able to configure and produce a given provider's extension type
///
/// See [Provider]
pub trait ProviderBuilder: Sized + Default + Clone {
    /// Provider extension type built for a concrete HTTP backend.
    type Extension<H>: Provider
    where
        H: HttpClientExt;
    /// API key input type accepted by the provider's client builder.
    type ApiKey: ApiKey;

    /// Default base URL for the provider.
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

// These implementations are declarations of associated types and constants,
// so ordinary helper functions cannot express the repeated structure. Keeping
// the variation points in one invocation makes each provider's configuration
// visible without duplicating the generic builder plumbing.
macro_rules! impl_default_provider_builder {
    (
        $builder:ty => $extension:ty,
        api_key = $api_key:ty,
        base_url = $base_url:expr
        $(, finish = $finish:path, state = $state:ident)? $(,)?
    ) => {
        impl $crate::client::ProviderBuilder for $builder {
            type Extension<H>
                = $extension
            where
                H: $crate::http_client::HttpClientExt;
            type ApiKey = $api_key;

            const BASE_URL: &'static str = $base_url;

            fn build<H>(
                _builder: &$crate::client::ClientBuilder<Self, Self::ApiKey, H>,
            ) -> $crate::http_client::Result<Self::Extension<H>>
            where
                H: $crate::http_client::HttpClientExt,
            {
                Ok(<$extension>::default())
            }

            $(
                fn finish<H>(
                    &self,
                    builder: $crate::client::ClientBuilder<Self, Self::ApiKey, H>,
                ) -> $crate::http_client::Result<
                    $crate::client::ClientBuilder<Self, Self::ApiKey, H>,
                > {
                    $finish(&self.$state, builder)
                }
            )?
        }
    };
}
pub(crate) use impl_default_provider_builder;

// A provider's Capabilities impl is a pure associated-type table where every
// slot a provider does not support is `Nothing`. The named optional slots
// keep each provider's invocation down to what it actually supports, and the
// macro owns the feature gating on the image/audio slots.
macro_rules! impl_capabilities {
    (
        $ext:ty
        $(, completion = $completion:ty)?
        $(, embeddings = $embeddings:ty)?
        $(, transcription = $transcription:ty)?
        $(, model_listing = $model_listing:ty)?
        $(, image_generation = $image_generation:ty)?
        $(, audio_generation = $audio_generation:ty)?
        $(, rerank = $rerank:ty)?
        $(,)?
    ) => {
        impl<H> $crate::client::Capabilities<H> for $ext {
            type Completion = $crate::client::impl_capabilities!(@slot $($completion)?);
            type Embeddings = $crate::client::impl_capabilities!(@slot $($embeddings)?);
            type Transcription = $crate::client::impl_capabilities!(@slot $($transcription)?);
            type ModelListing = $crate::client::impl_capabilities!(@slot $($model_listing)?);
            #[cfg(feature = "image")]
            type ImageGeneration = $crate::client::impl_capabilities!(@slot $($image_generation)?);
            #[cfg(feature = "audio")]
            type AudioGeneration = $crate::client::impl_capabilities!(@slot $($audio_generation)?);
            type Rerank = $crate::client::impl_capabilities!(@slot $($rerank)?);
        }
    };
    (@slot $model:ty) => { $crate::client::Capable<$model> };
    (@slot) => { $crate::client::Nothing };
}
pub(crate) use impl_capabilities;

// `ProviderFromEnv` is implemented per provider *extension* type, generic over
// the transport. The optional base-URL forms capture the only common
// construction variation without hiding provider-specific auth.
macro_rules! impl_provider_from_env {
    (
        $ext:ty,
        input = $input:ty,
        api_key_env = $api_key_env:literal,
        base_url_env_first = $base_url_env:literal $(,)?
    ) => {
        $crate::client::impl_provider_from_env!(@with_base
            $ext,
            input = $input,
            api_key_env = $api_key_env,
            configuration = {
                let base_url = $crate::client::optional_env_var($base_url_env)?;
                let api_key = $crate::client::required_env_var($api_key_env)?;
                (api_key, base_url)
            }
        );
    };
    (
        $ext:ty,
        input = $input:ty,
        api_key_env = $api_key_env:literal,
        base_url_env = $base_url_env:literal $(,)?
    ) => {
        $crate::client::impl_provider_from_env!(@with_base
            $ext,
            input = $input,
            api_key_env = $api_key_env,
            configuration = {
                let api_key = $crate::client::required_env_var($api_key_env)?;
                let base_url = $crate::client::optional_env_var($base_url_env)?;
                (api_key, base_url)
            }
        );
    };
    (
        $ext:ty,
        input = $input:ty,
        api_key_env = $api_key_env:literal,
        base_url = $base_url:expr $(,)?
    ) => {
        $crate::client::impl_provider_from_env!(@with_base
            $ext,
            input = $input,
            api_key_env = $api_key_env,
            configuration = {
                let api_key = $crate::client::required_env_var($api_key_env)?;
                (api_key, $base_url)
            }
        );
    };
    (@with_base
        $ext:ty,
        input = $input:ty,
        api_key_env = $api_key_env:literal,
        configuration = $configuration:block
    ) => {
        impl $crate::client::ProviderFromEnv for $ext {
            type Input = $input;

            #[doc = concat!("Configure this provider from the `", $api_key_env, "` environment variable.")]
            fn from_env_with<H>(
                http: H,
            ) -> Result<$crate::client::Client<Self, H>, $crate::client::ProviderClientError>
            where
                H: $crate::http_client::HttpClientExt,
                Self::Builder: $crate::client::ProviderBuilder<Extension<H> = Self>,
            {
                let (api_key, base_url) = $configuration;
                let mut builder = $crate::client::Client::<Self, $crate::markers::Missing>::builder()
                    .api_key(api_key);
                if let Some(base_url) = base_url {
                    builder = builder.base_url(base_url);
                }
                builder.http_client(http).build().map_err(Into::into)
            }

            fn from_val_with<H>(
                input: Self::Input,
                http: H,
            ) -> Result<$crate::client::Client<Self, H>, $crate::client::ProviderClientError>
            where
                H: $crate::http_client::HttpClientExt,
                Self::Builder: $crate::client::ProviderBuilder<Extension<H> = Self>,
            {
                $crate::client::Client::new_with(input, http).map_err(Into::into)
            }
        }
    };
    (
        $ext:ty,
        input = $input:ty,
        api_key_env = $api_key_env:literal $(,)?
    ) => {
        impl $crate::client::ProviderFromEnv for $ext {
            type Input = $input;

            #[doc = concat!("Configure this provider from the `", $api_key_env, "` environment variable.")]
            fn from_env_with<H>(
                http: H,
            ) -> Result<$crate::client::Client<Self, H>, $crate::client::ProviderClientError>
            where
                H: $crate::http_client::HttpClientExt,
                Self::Builder: $crate::client::ProviderBuilder<Extension<H> = Self>,
            {
                let api_key = $crate::client::required_env_var($api_key_env)?;
                $crate::client::Client::new_with(api_key, http).map_err(Into::into)
            }

            fn from_val_with<H>(
                input: Self::Input,
                http: H,
            ) -> Result<$crate::client::Client<Self, H>, $crate::client::ProviderClientError>
            where
                H: $crate::http_client::HttpClientExt,
                Self::Builder: $crate::client::ProviderBuilder<Extension<H> = Self>,
            {
                $crate::client::Client::new_with(input, http).map_err(Into::into)
            }
        }
    };
}
pub(crate) use impl_provider_from_env;

/// Construction with an explicit transport. rig-core never chooses a transport
/// for you; the bundled `reqwest` one is behind the `reqwest` feature, whose
/// The `reqwest` feature adds inherent constructors that supply
/// the one-argument `new(api_key)` on top of this.
impl<Ext, H> Client<Ext, H>
where
    Ext: Provider,
    Ext::Builder: ProviderBuilder<Extension<H> = Ext> + Default,
    H: HttpClientExt,
{
    /// Construct a provider client that sends through `http`.
    pub fn new_with(
        api_key: impl Into<<Ext::Builder as ProviderBuilder>::ApiKey>,
        http: H,
    ) -> http_client::Result<Self> {
        Client::<Ext, Missing>::builder()
            .api_key(api_key)
            .http_client(http)
            .build()
    }
}

impl<Ext, H> Client<Ext, H>
where
    H: HttpClientExt + 'static,
{
    /// Erase this client's transport behind [`BoxedHttpClient`].
    ///
    /// The result is the same client — base URL, headers, extension — sending
    /// through the same transport, but its type no longer names `H`. A host
    /// that builds clients for several providers over one transport uses this
    /// (or [`ProviderFromEnv::from_env_boxed`]) so every client it holds is a
    /// `Client<Ext>`. Boxing an already boxed client is a no-op clone of the
    /// transport handle.
    pub fn boxed(self) -> Client<Ext, BoxedHttpClient> {
        Client {
            base_url: self.base_url,
            headers: self.headers,
            http_client: BoxedHttpClient::new(self.http_client),
            ext: self.ext,
        }
    }
}

impl<Ext, H> Client<Ext, H> {
    /// Returns the configured provider base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Returns default headers applied to outgoing provider requests.
    pub fn headers(&self) -> &HeaderMap {
        &self.headers
    }

    /// Returns the provider extension.
    pub fn ext(&self) -> &Ext {
        &self.ext
    }

    /// The HTTP transport this client sends through, for callers that must
    /// talk to an absolute URL outside the provider's API base (OAuth/device
    /// flows) with the same transport.
    pub fn http_client(&self) -> &H {
        &self.http_client
    }

    /// Reuse this client's base URL, headers, and HTTP backend with a different extension.
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

/// `builder()` lives on `Client<Ext, Missing>` — the "no transport chosen yet"
/// state — so `provider::Client::builder()` resolves without an `H` annotation
/// (it is the only `builder` inherent fn, so `H` infers to `Missing`). The
/// returned builder's `H` slot is `Missing` too; [`ClientBuilder::http_client`]
/// must be called before [`ClientBuilder::build`] (or a transport crate's
/// default-substituting `build`, such as the `reqwest` feature's
/// the `reqwest` feature's inherent `build`).
impl<Ext> Client<Ext, Missing>
where
    Ext: Provider,
    Ext::Builder: ProviderBuilder + Default,
{
    /// Start constructing a provider client.
    pub fn builder() -> ClientBuilder<Ext::Builder, Missing, Missing> {
        ClientBuilder::default()
    }
}

impl<Ext, H> Client<Ext, H>
where
    Ext: Provider,
{
    fn request(
        &self,
        method: http::Method,
        path: &str,
        transport: Transport,
    ) -> http_client::Result<Builder> {
        let uri = self.ext.build_uri(&self.base_url, path, transport);

        let mut req = Request::builder().method(method).uri(uri);

        if let Some(hs) = req.headers_mut() {
            hs.extend(self.headers.iter().map(|(k, v)| (k.clone(), v.clone())));
        }

        self.ext.with_custom(req)
    }

    /// Build a provider-customized POST request for a regular HTTP endpoint.
    pub fn post<S>(&self, path: S) -> http_client::Result<Builder>
    where
        S: AsRef<str>,
    {
        self.request(http::Method::POST, path.as_ref(), Transport::Http)
    }

    /// Build a provider-customized POST request for an SSE endpoint.
    pub fn post_sse<S>(&self, path: S) -> http_client::Result<Builder>
    where
        S: AsRef<str>,
    {
        self.request(http::Method::POST, path.as_ref(), Transport::Sse)
    }

    /// Build a provider-customized GET request for an SSE endpoint.
    pub fn get_sse<S>(&self, path: S) -> http_client::Result<Builder>
    where
        S: AsRef<str>,
    {
        self.request(http::Method::GET, path.as_ref(), Transport::Sse)
    }

    /// Build a provider-customized GET request for a regular HTTP endpoint.
    pub fn get<S>(&self, path: S) -> http_client::Result<Builder>
    where
        S: AsRef<str>,
    {
        self.request(http::Method::GET, path.as_ref(), Transport::Http)
    }

    /// Build a provider-customized PATCH request for a regular HTTP endpoint.
    ///
    /// REST resources that support partial update need this: Gemini's
    /// `cachedContents` only allows the expiry to be changed, and does it with
    /// `PATCH ?updateMask=ttl`.
    pub fn patch<S>(&self, path: S) -> http_client::Result<Builder>
    where
        S: AsRef<str>,
    {
        self.request(http::Method::PATCH, path.as_ref(), Transport::Http)
    }

    /// Build a provider-customized DELETE request for a regular HTTP endpoint.
    ///
    /// Needed by any provider resource with a lifecycle rather than a single
    /// call — a cached-content handle bills for storage until it is deleted, so
    /// deleting one is a first-class operation, not a convenience.
    pub fn delete<S>(&self, path: S) -> http_client::Result<Builder>
    where
        S: AsRef<str>,
    {
        self.request(http::Method::DELETE, path.as_ref(), Transport::Http)
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

/// Type-state builder for [`Client`].
///
/// Each generic slot encodes a separate "has the user supplied this yet?" question:
///
/// - `ApiKey = Missing` means the caller has not yet called [`Self::api_key`]; transitioning to a
///   concrete `ApiKey` type is required before [`Self::build`] is reachable.
/// - `H = Missing` means the caller has not yet called [`Self::http_client`]; rig-core's own
///   `build()` is only reachable once a concrete `HttpClientExt` backend has been supplied. A
///   transport crate may add a default-substituting `build` for the `Missing` state (the
///   bundled one is the inherent `build` the `reqwest` feature adds).
///
/// Keeping `Missing` as the *type-level* placeholder means the builder's generics describe what
/// the caller has actually provided, instead of pretending a default value is already present.
/// It also avoids carrying an `Option<H>` whose `None` branch existed only to model the same
/// "user hasn't picked a backend" state.
#[derive(Clone)]
pub struct ClientBuilder<Ext, ApiKey = Missing, H = Missing> {
    base_url: String,
    api_key: ApiKey,
    headers: HeaderMap,
    http_client: H,
    ext: Ext,
}

impl<ExtBuilder> Default for ClientBuilder<ExtBuilder, Missing, Missing>
where
    ExtBuilder: ProviderBuilder + Default,
{
    fn default() -> Self {
        Self {
            api_key: Missing,
            headers: Default::default(),
            base_url: ExtBuilder::BASE_URL.into(),
            http_client: Missing,
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

        let new_ext = f(ext);

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

    /// Set the HTTP backend used in this client.
    ///
    /// Calling this advances the builder's `H` slot from whatever it was (typically `Missing`)
    /// to the supplied client's type, which selects the H-generic [`Self::build`] impl below.
    pub fn http_client<U>(self, http_client: U) -> ClientBuilder<Ext, ApiKey, U> {
        ClientBuilder {
            http_client,
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
    /// Returns the provider extension builder state.
    pub fn ext(&self) -> &Ext {
        &self.ext
    }

    /// Returns the configured base URL.
    pub fn get_base_url(&self) -> &str {
        &self.base_url
    }
}

/// `build`: the caller supplied an HTTP client via [`ClientBuilder::http_client`], so `H` is a
/// real `HttpClientExt` type and we use it directly.
impl<ExtBuilder, Key, H> ClientBuilder<ExtBuilder, Key, H>
where
    ExtBuilder: ProviderBuilder<ApiKey = Key>,
    Key: ApiKey,
    H: HttpClientExt,
{
    /// Build a client using the HTTP backend supplied with [`ClientBuilder::http_client`].
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

        Ok(Client {
            http_client,
            base_url: Arc::from(base_url.as_str()),
            headers: Arc::new(headers),
            ext,
        })
    }
}

// Every single-model capability client impl on `Client<Ext, H>` shares the
// same shape: gate on the matching `Capabilities` slot, name the model type,
// and construct it through the capability's public `Construct*Model` hook.
// The macro keeps the per-capability variation (trait, slot, associated type,
// method, hook, and feature gate) in one invocation each. `EmbeddingsClient`
// (extra `_with_ndims` method and a `dims` argument on its hook) stays
// hand-written below.
macro_rules! impl_capability_client {
    (
        $(#[cfg(feature = $feature:literal)])?
        $client_trait:ident { $slot:ident, $assoc:ident, $method:ident, $model_trait:ident, $construct:ident }
    ) => {
        $(#[cfg(feature = $feature)])?
        impl<M, Ext, H> $client_trait for Client<Ext, H>
        where
            Ext: Capabilities<H, $slot = Capable<M>>,
            M: $model_trait + $construct<Self>,
        {
            type $assoc = M;

            fn $method(&self, model: impl Into<String>) -> Self::$assoc {
                M::construct(self, model.into())
            }
        }
    };
}

impl_capability_client!(CompletionClient {
    Completion,
    CompletionModel,
    completion_model,
    CompletionModel,
    ConstructCompletionModel
});

impl<M, Ext, H> EmbeddingsClient for Client<Ext, H>
where
    Ext: Capabilities<H, Embeddings = Capable<M>>,
    M: EmbeddingModel + ConstructEmbeddingModel<Self>,
{
    type EmbeddingModel = M;

    fn embedding_model(&self, model: impl Into<String>) -> Self::EmbeddingModel {
        M::construct(self, model.into(), None)
    }

    fn embedding_model_with_ndims(
        &self,
        model: impl Into<String>,
        ndims: usize,
    ) -> Self::EmbeddingModel {
        M::construct(self, model.into(), Some(ndims))
    }
}

impl_capability_client!(RerankingClient {
    Rerank,
    RerankModel,
    rerank_model,
    RerankModel,
    ConstructRerankModel
});

impl_capability_client!(TranscriptionClient {
    Transcription,
    TranscriptionModel,
    transcription_model,
    TranscriptionModel,
    ConstructTranscriptionModel
});

impl_capability_client!(
    #[cfg(feature = "image")]
    ImageGenerationClient {
        ImageGeneration,
        ImageGenerationModel,
        image_generation_model,
        ImageGenerationModel,
        ConstructImageGenerationModel
    }
);

impl_capability_client!(
    #[cfg(feature = "audio")]
    AudioGenerationClient {
        AudioGeneration,
        AudioGenerationModel,
        audio_generation_model,
        AudioGenerationModel,
        ConstructAudioGenerationModel
    }
);

impl<M, Ext, H> ModelListingClient for Client<Ext, H>
where
    Ext: Capabilities<H, ModelListing = Capable<M>>,
    M: ModelLister<H> + ConstructModelLister<Self> + 'static,
    H: WasmCompatSend + WasmCompatSync,
{
    fn list_models(
        &self,
    ) -> impl std::future::Future<
        Output = Result<crate::model::ModelList, crate::model::ModelListingError>,
    > + WasmCompatSend {
        let lister = M::construct(self);
        async move { lister.list_all().await }
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
mod tests {
    use crate::providers::anthropic;

    /// Type-level test that `Client::builder()` methods do not require annotation to determine
    /// backig HTTP client
    #[test]
    fn ensures_client_builder_no_annotation() {
        let http_client = crate::test_utils::RecordingHttpClient::new("");
        let _ = anthropic::Client::builder()
            .http_client(http_client)
            .api_key("Foo")
            .build()
            .unwrap();
    }
}

/// Compile coverage for an out-of-tree provider extension built on the generic
/// [`Client`] that offers every non-completion modality: implementing the
/// public `Construct*Model` hooks is all it takes for the blanket capability
/// client impls to apply. Everything here uses only public API, mirroring what
/// a downstream crate can write — the same probe [`completion`] ships for
/// [`ConstructCompletionModel`].
#[cfg(test)]
mod external_modality_extension_probe {
    use super::*;
    use crate::embeddings::{EmbeddingError, EmbeddingModel, EmbeddingResponse};
    use crate::rerank::{RerankError, RerankModel, RerankResponse};
    use crate::transcription::{
        TranscriptionError, TranscriptionModel, TranscriptionRequest, TranscriptionResponse,
    };

    #[derive(Debug, Default, Clone, Copy)]
    struct ExternalExt;
    #[derive(Debug, Default, Clone, Copy)]
    struct ExternalExtBuilder;

    impl Provider for ExternalExt {
        type Builder = ExternalExtBuilder;
        const VERIFY_PATH: &'static str = "/";
    }

    impl ProviderBuilder for ExternalExtBuilder {
        type Extension<H>
            = ExternalExt
        where
            H: HttpClientExt;
        type ApiKey = BearerAuth;

        const BASE_URL: &'static str = "https://external.invalid";

        fn build<H>(
            _builder: &ClientBuilder<Self, Self::ApiKey, H>,
        ) -> http_client::Result<Self::Extension<H>>
        where
            H: HttpClientExt,
        {
            Ok(ExternalExt)
        }
    }

    impl<H> Capabilities<H> for ExternalExt {
        type Completion = Nothing;
        type Embeddings = Capable<ExternalModel<H>>;
        type Transcription = Capable<ExternalModel<H>>;
        type ModelListing = Capable<ExternalModel<H>>;
        #[cfg(feature = "image")]
        type ImageGeneration = Capable<ExternalModel<H>>;
        #[cfg(feature = "audio")]
        type AudioGeneration = Capable<ExternalModel<H>>;
        type Rerank = Capable<ExternalModel<H>>;
    }

    impl DebugExt for ExternalExt {}

    /// One model type standing in for every modality; deliberately not
    /// `Clone`, which the relaxed supertraits no longer require.
    struct ExternalModel<H> {
        _client: Client<ExternalExt, H>,
        model: String,
        ndims: Option<usize>,
    }

    impl<H> TranscriptionModel for ExternalModel<H>
    where
        H: Send + Sync + 'static,
    {
        async fn transcription(
            &self,
            _request: TranscriptionRequest,
        ) -> Result<TranscriptionResponse, TranscriptionError> {
            Err(TranscriptionError::ResponseError(self.model.clone()))
        }
    }

    impl<H> ConstructTranscriptionModel<Client<ExternalExt, H>> for ExternalModel<H>
    where
        H: Clone,
    {
        fn construct(client: &Client<ExternalExt, H>, model: String) -> Self {
            Self {
                _client: client.clone(),
                model,
                ndims: None,
            }
        }
    }

    impl<H> EmbeddingModel for ExternalModel<H>
    where
        H: Send + Sync + 'static,
    {
        fn max_documents(&self) -> usize {
            1
        }

        fn ndims(&self) -> usize {
            self.ndims.unwrap_or(3)
        }

        async fn embed_texts_response(
            &self,
            _texts: impl IntoIterator<Item = String> + Send,
        ) -> Result<EmbeddingResponse, EmbeddingError> {
            Err(EmbeddingError::ResponseError(self.model.clone()))
        }
    }

    impl<H> ConstructEmbeddingModel<Client<ExternalExt, H>> for ExternalModel<H>
    where
        H: Clone,
    {
        fn construct(client: &Client<ExternalExt, H>, model: String, ndims: Option<usize>) -> Self {
            Self {
                _client: client.clone(),
                model,
                ndims,
            }
        }
    }

    impl<H> RerankModel for ExternalModel<H>
    where
        H: Send + Sync + 'static,
    {
        fn max_documents(&self) -> usize {
            1
        }

        async fn rerank(
            &self,
            _query: &str,
            _documents: Vec<String>,
        ) -> Result<RerankResponse, RerankError> {
            Err(RerankError::ResponseError(self.model.clone()))
        }
    }

    impl<H> ConstructRerankModel<Client<ExternalExt, H>> for ExternalModel<H>
    where
        H: Clone,
    {
        fn construct(client: &Client<ExternalExt, H>, model: String) -> Self {
            Self {
                _client: client.clone(),
                model,
                ndims: None,
            }
        }
    }

    #[cfg(feature = "image")]
    impl<H> ImageGenerationModel for ExternalModel<H>
    where
        H: Send + Sync + 'static,
    {
        async fn image_generation(
            &self,
            _request: crate::image_generation::ImageGenerationRequest,
        ) -> Result<
            crate::image_generation::ImageGenerationResponse,
            crate::image_generation::ImageGenerationError,
        > {
            Err(crate::image_generation::ImageGenerationError::ResponseError(self.model.clone()))
        }
    }

    #[cfg(feature = "image")]
    impl<H> ConstructImageGenerationModel<Client<ExternalExt, H>> for ExternalModel<H>
    where
        H: Clone,
    {
        fn construct(client: &Client<ExternalExt, H>, model: String) -> Self {
            Self {
                _client: client.clone(),
                model,
                ndims: None,
            }
        }
    }

    #[cfg(feature = "audio")]
    impl<H> AudioGenerationModel for ExternalModel<H>
    where
        H: Send + Sync + 'static,
    {
        async fn audio_generation(
            &self,
            _request: AudioGenerationRequest,
        ) -> Result<AudioGenerationResponse, AudioGenerationError> {
            Err(AudioGenerationError::ResponseError(self.model.clone()))
        }
    }

    #[cfg(feature = "audio")]
    impl<H> ConstructAudioGenerationModel<Client<ExternalExt, H>> for ExternalModel<H>
    where
        H: Clone,
    {
        fn construct(client: &Client<ExternalExt, H>, model: String) -> Self {
            Self {
                _client: client.clone(),
                model,
                ndims: None,
            }
        }
    }

    impl<H> ModelLister<H> for ExternalModel<H>
    where
        H: Send + Sync + 'static,
    {
        async fn list_all(
            &self,
        ) -> Result<crate::model::ModelList, crate::model::ModelListingError> {
            Err(crate::model::ModelListingError::ParseError {
                message: self.model.clone(),
            })
        }
    }

    impl<H> ConstructModelLister<Client<ExternalExt, H>> for ExternalModel<H>
    where
        H: Clone,
    {
        fn construct(client: &Client<ExternalExt, H>) -> Self {
            Self {
                _client: client.clone(),
                model: "lister".to_owned(),
                ndims: None,
            }
        }
    }

    #[test]
    fn external_extension_reaches_every_blanket_capability_client_impl() {
        fn assert_transcription<C: TranscriptionClient>() {}
        fn assert_embeddings<C: EmbeddingsClient>() {}
        fn assert_rerank<C: RerankingClient>() {}
        fn assert_listing<C: ModelListingClient>() {}
        #[cfg(feature = "image")]
        fn assert_image<C: ImageGenerationClient>() {}
        #[cfg(feature = "audio")]
        fn assert_audio<C: AudioGenerationClient>() {}

        type ExternalClient = Client<ExternalExt, crate::test_utils::RecordingHttpClient>;
        assert_transcription::<ExternalClient>();
        assert_embeddings::<ExternalClient>();
        assert_rerank::<ExternalClient>();
        assert_listing::<ExternalClient>();
        #[cfg(feature = "image")]
        assert_image::<ExternalClient>();
        #[cfg(feature = "audio")]
        assert_audio::<ExternalClient>();
    }

    #[test]
    fn embedding_hook_receives_the_requested_dims() {
        let client: Client<ExternalExt, crate::test_utils::RecordingHttpClient> =
            Client::<ExternalExt, Missing>::builder()
                .api_key("key")
                .http_client(crate::test_utils::RecordingHttpClient::new(""))
                .build()
                .expect("client should build");
        assert_eq!(client.embedding_model("m").ndims(), 3);
        assert_eq!(client.embedding_model_with_ndims("m", 7).ndims(), 7);
    }

    /// `Arc<M>` is a model: the relaxed supertraits make "wrap it in an Arc"
    /// real through the generic APIs, as for `CompletionModel`.
    #[test]
    fn arc_wrapped_models_satisfy_the_modality_traits() {
        fn assert_transcription_model<M: TranscriptionModel>() {}
        #[cfg(feature = "image")]
        fn assert_image_model<M: ImageGenerationModel>() {}
        #[cfg(feature = "audio")]
        fn assert_audio_model<M: AudioGenerationModel>() {}

        assert_transcription_model::<Arc<ExternalModel<crate::test_utils::RecordingHttpClient>>>();
        #[cfg(feature = "image")]
        assert_image_model::<Arc<ExternalModel<crate::test_utils::RecordingHttpClient>>>();
        #[cfg(feature = "audio")]
        assert_audio_model::<Arc<ExternalModel<crate::test_utils::RecordingHttpClient>>>();
    }
}
