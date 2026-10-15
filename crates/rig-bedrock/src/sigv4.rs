//! AWS Signature Version 4 signing as an HTTP transport. [`SigV4Http`] wraps
//! another transport and signs each request with the current AWS credentials
//! just before it is sent, so a provider wire that speaks plain HTTP
//! (Anthropic Messages, OpenAI Chat Completions or Responses) can reach an
//! AWS endpoint that authenticates with IAM.
//!
//! ```no_run
//! use rig_bedrock::sigv4::SigV4Http;
//!
//! # async fn run(http: rig_core::http_client::DynHttpClient) -> Result<(), rig_bedrock::sigv4::AwsError> {
//! let aws = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
//! let signed = SigV4Http::from_sdk_config(&aws, "bedrock-mantle", http)?;
//! # let _ = signed;
//! # Ok(())
//! # }
//! ```

use std::borrow::Cow;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use aws_config::SdkConfig;
use aws_credential_types::Credentials;
use aws_credential_types::provider::error::CredentialsError;
use aws_credential_types::provider::{ProvideCredentials, SharedCredentialsProvider};
use aws_sigv4::http_request::{SignableBody, SignableRequest, SigningSettings, sign};
use aws_sigv4::sign::v4;
use rig_core::client::env::EnvError;
use rig_core::http_client::{
    self, Bytes, DynHttpClient, HeaderValue, HttpClientExt, LazyBody, MultipartForm, Request,
    Response, StreamingResponse, Uri,
};
use rig_core::wasm_compat::WasmCompatSend;
use tokio::sync::Mutex;

/// Why an AWS-authenticated client could not be built or a request could
/// not be signed.
#[derive(Debug, thiserror::Error)]
pub enum AwsError {
    /// The AWS configuration names no region.
    #[error("the AWS configuration names no region")]
    NoRegion,
    /// The AWS configuration has no credentials provider.
    #[error("the AWS configuration has no credentials provider")]
    NoCredentialsProvider,
    /// An environment variable could not be read.
    #[error(transparent)]
    Env(#[from] EnvError),
    /// The credentials provider failed.
    #[error("could not resolve AWS credentials")]
    Credentials(#[source] CredentialsError),
    /// The signer rejected the request.
    #[error("could not sign the request")]
    Signing(#[source] aws_sigv4::http_request::SigningError),
    /// The signing parameters were incomplete.
    #[error("invalid SigV4 signing parameters")]
    Params(#[source] v4::signing_params::BuildError),
    /// The request URI has no host, which the signature covers.
    #[error("request URI `{0}` has no host to sign")]
    NoHost(Uri),
    /// The request URI could not be put in the form it is sent in.
    #[error("request URI `{0}` cannot be signed")]
    InvalidUri(Uri),
    /// A request or signature header value is not visible ASCII.
    #[error("header `{0}` cannot be signed")]
    InvalidHeader(String),
    /// Multipart bodies are assembled by the transport after signing.
    #[error("a multipart request cannot be SigV4-signed")]
    Multipart,
}

/// Headers a hop between client and AWS may change, left out of the
/// signature in addition to the signer's own exclusions.
const UNSIGNED_HEADERS: &[&str] = &["connection"];

/// Credential headers a provider wire sets that the signature replaces.
const CREDENTIAL_HEADERS: &[&str] = &["authorization", "x-api-key"];

/// How long before they expire cached credentials are refreshed.
const REFRESH_MARGIN: Duration = Duration::from_secs(300);

/// A transport that SigV4-signs every request for one AWS region and
/// service, then sends it through the transport it wraps. Credentials are
/// cached and fetched again from the provider shortly before they expire.
/// Any `authorization` or `x-api-key` header the wire set is dropped, since
/// the signature is the credential. Clones share the signer and its cache.
#[derive(Clone)]
pub struct SigV4Http {
    inner: DynHttpClient,
    signer: Arc<Signer>,
}

struct Signer {
    credentials: SharedCredentialsProvider,
    cached: Mutex<Option<Credentials>>,
    region: String,
    service: String,
}

impl fmt::Debug for SigV4Http {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SigV4Http")
            .field("region", &self.signer.region)
            .field("service", &self.signer.service)
            .finish_non_exhaustive()
    }
}

impl SigV4Http {
    /// Sign for `service` in `region` with `credentials`, sending through
    /// `http`. `service` is the endpoint's signing name, such as `bedrock`
    /// or `bedrock-mantle`. Middleware attached to `http` runs after
    /// signing, so it must not change the body or a signed header.
    pub fn new(
        http: impl HttpClientExt + 'static,
        credentials: SharedCredentialsProvider,
        region: impl Into<String>,
        service: impl Into<String>,
    ) -> Self {
        Self {
            inner: DynHttpClient::new(http),
            signer: Arc::new(Signer {
                credentials,
                cached: Mutex::new(None),
                region: region.into(),
                service: service.into(),
            }),
        }
    }

    /// Sign for `service` with the region and credentials provider of
    /// `config`.
    pub fn from_sdk_config(
        config: &SdkConfig,
        service: impl Into<String>,
        http: impl HttpClientExt + 'static,
    ) -> Result<Self, AwsError> {
        let region = config.region().ok_or(AwsError::NoRegion)?;
        let credentials = config
            .credentials_provider()
            .ok_or(AwsError::NoCredentialsProvider)?;
        Ok(Self::new(http, credentials, region.as_ref(), service))
    }

    /// The region requests are signed for.
    pub fn region(&self) -> &str {
        &self.signer.region
    }

    /// The signing name requests are signed for.
    pub fn service(&self) -> &str {
        &self.signer.service
    }
}

impl Signer {
    /// The cached credentials, fetched again when they expire within
    /// [`REFRESH_MARGIN`] of `time`. Credentials without an expiry are kept.
    async fn credentials(&self, time: SystemTime) -> Result<Credentials, AwsError> {
        let mut cached = self.cached.lock().await;
        let fresh = |credentials: &&Credentials| {
            credentials
                .expiry()
                .is_none_or(|expiry| expiry > time + REFRESH_MARGIN)
        };
        if let Some(credentials) = cached.as_ref().filter(fresh) {
            return Ok(credentials.clone());
        }
        let credentials = self
            .credentials
            .provide_credentials()
            .await
            .map_err(AwsError::Credentials)?;
        *cached = Some(credentials.clone());
        Ok(credentials)
    }

    /// `request` with its credential headers replaced by a signature taken
    /// at `time`. The host is lowercased first, as HTTP clients send it, so
    /// the signed host is the one on the wire.
    async fn sign(
        &self,
        mut request: Request<Bytes>,
        time: SystemTime,
    ) -> Result<Request<Bytes>, AwsError> {
        let Some(authority) = request.uri().authority() else {
            return Err(AwsError::NoHost(request.uri().clone()));
        };
        if authority
            .as_str()
            .bytes()
            .any(|byte| byte.is_ascii_uppercase())
        {
            let uri = request.uri().clone();
            let mut parts = uri.clone().into_parts();
            parts.authority = Some(
                authority
                    .as_str()
                    .to_ascii_lowercase()
                    .parse()
                    .map_err(|_| AwsError::InvalidUri(uri.clone()))?,
            );
            *request.uri_mut() = Uri::from_parts(parts).map_err(|_| AwsError::InvalidUri(uri))?;
        }
        for name in CREDENTIAL_HEADERS {
            request.headers_mut().remove(*name);
        }
        let identity = self.credentials(time).await?.into();
        let mut settings = SigningSettings::default();
        settings
            .excluded_headers
            .get_or_insert_with(Vec::new)
            .extend(UNSIGNED_HEADERS.iter().map(|name| Cow::Borrowed(*name)));
        let params = v4::SigningParams::builder()
            .identity(&identity)
            .region(&self.region)
            .name(&self.service)
            .time(time)
            .settings(settings)
            .build()
            .map_err(AwsError::Params)?
            .into();

        let headers = request
            .headers()
            .iter()
            .map(|(name, value)| {
                value
                    .to_str()
                    .map(|value| (name.as_str(), value))
                    .map_err(|_| AwsError::InvalidHeader(name.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let uri = request.uri().to_string();
        let signable = SignableRequest::new(
            request.method().as_str(),
            uri,
            headers.into_iter(),
            SignableBody::Bytes(request.body()),
        )
        .map_err(AwsError::Signing)?;
        let (instructions, _signature) = sign(signable, &params)
            .map_err(AwsError::Signing)?
            .into_parts();

        let (signature_headers, _query) = instructions.into_parts();
        for header in signature_headers {
            let mut value = HeaderValue::from_str(header.value())
                .map_err(|_| AwsError::InvalidHeader(header.name().to_owned()))?;
            value.set_sensitive(header.sensitive());
            request.headers_mut().insert(header.name(), value);
        }
        Ok(request)
    }
}

impl HttpClientExt for SigV4Http {
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes> + WasmCompatSend,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        let this = self.clone();
        let req = req.map(Into::into);
        async move {
            let req = this
                .signer
                .sign(req, SystemTime::now())
                .await
                .map_err(http_client::Error::instance)?;
            this.inner.send::<Bytes, U>(req).await
        }
    }

    fn send_multipart<U>(
        &self,
        _req: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        std::future::ready(Err(http_client::Error::instance(AwsError::Multipart)))
    }

    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        let req = req.map(Into::into);
        async move {
            let req = self
                .signer
                .sign(req, SystemTime::now())
                .await
                .map_err(http_client::Error::instance)?;
            self.inner.send_streaming(req).await
        }
    }
}

#[cfg(test)]
mod tests;
