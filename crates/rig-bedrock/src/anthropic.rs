//! The Anthropic Messages API on an AWS-fronted endpoint.
//!
//! This is not the Bedrock Runtime `Converse`/`InvokeModel` API that the rest of this crate speaks.
//! It is the other shape: an endpoint that accepts Anthropic's own `/v1/messages` request body, and
//! therefore carries Anthropic-shaped parameters, but authenticates with AWS SigV4 instead of an
//! `x-api-key` header. Different wire protocols against different endpoints, so neither substitutes
//! for the other, and this module reuses rig-core's Anthropic request and response mapping rather
//! than duplicating it.
//!
//! It lives in `rig-bedrock` because the credential side is pure AWS. Signing needs `aws-config`,
//! `aws-sigv4`, `aws-credential-types` and tokio; all four already belong to this crate, and none
//! of them belongs in rig-core. What rig-core contributes is one defaulted, dependency-free hook —
//! [`AnthropicCompatibleProvider::signed_headers`] — which it calls immediately before the request
//! body is attached, in both the unary and the streaming path.
//!
//! # Example
//!
//! ```no_run
//! use rig_bedrock::anthropic::{AnthropicKey, Client};
//!
//! # fn run<H: rig_core::http_client::HttpClientExt>(http: H) -> Result<(), Box<dyn std::error::Error>> {
//! let client = Client::builder()
//!     .api_key(AnthropicKey::sigv4("us-east-1"))
//!     .base_url("https://your-endpoint.example/anthropic")
//!     .http_client(http)
//!     .build()?;
//! # Ok(())
//! # }
//! ```
//!
//! The endpoint is not defaulted. There is no regional URL baked in, because signing a request to
//! `api.anthropic.com` is never what the caller meant and a wrong default would produce a 403 that
//! reads like a credential problem. Supply it with [`ClientBuilder::base_url`] or the
//! `ANTHROPIC_BASE_URL` environment variable; building without one fails immediately.
//!
//! # Known limitation: `verify()`
//!
//! [`VerifyClient::verify`](rig_core::client::VerifyClient::verify) does not work against a signed
//! endpoint. It is a blanket implementation over every [`rig_core::client::Client`] and sends a
//! plain GET to [`Provider::VERIFY_PATH`] carrying only the client's default headers — and a SigV4
//! client has none, by design, since the signature covers the body and the clock and so cannot be
//! precomputed at construction. Signing is applied at the two Anthropic request builders, which
//! `verify` does not go through, so it returns 403 rather than a credential verdict. Send a small
//! completion instead. Closing this would mean a second signing hook in rig-core covering arbitrary
//! client requests, which is more surface there than one verification convenience is worth.

mod sigv4;

use http::{HeaderName, HeaderValue};
use rig_core::client::{
    self, ApiKey, HasCompletion, ModelTransport, Provider, ProviderClientError,
    ProviderClientResult,
};
use rig_core::completion::CompletionError;
use rig_core::http_client::{self, HttpClientExt};
use rig_core::markers::Missing;
use rig_core::providers::anthropic::client::{AnthropicConfig, finish_anthropic_builder};
use rig_core::providers::anthropic::completion::{
    AnthropicCompatibleProvider, GenericCompletionModel, default_max_tokens_for_model,
};

/// Environment variable naming the endpoint, shared with rig-core's Anthropic provider: a caller
/// who has already pointed `ANTHROPIC_BASE_URL` at the AWS-fronted endpoint is exactly the caller
/// this provider is for.
const BASE_URL_ENV: &str = "ANTHROPIC_BASE_URL";

/// How to authenticate against the AWS-fronted Anthropic endpoint, which accepts either a Bedrock
/// API key or SigV4 credentials.
#[derive(Debug, Clone)]
pub enum AnthropicKey {
    /// A key sent as Anthropic's own `x-api-key` header, the same way rig-core's Anthropic provider
    /// sends one. What `From<S: Into<String>>` builds.
    ApiKey(String),
    /// AWS SigV4 request signing for `region`.
    ///
    /// No static header is produced; signing is per-request because the signature covers the body
    /// and the current time. Credentials come from the standard AWS provider chain.
    SigV4 { region: String },
}

impl AnthropicKey {
    /// Sign requests with AWS SigV4 for `region` instead of sending an API key.
    pub fn sigv4(region: impl Into<String>) -> Self {
        Self::SigV4 {
            region: region.into(),
        }
    }
}

impl<S> From<S> for AnthropicKey
where
    S: Into<String>,
{
    fn from(value: S) -> Self {
        Self::ApiKey(value.into())
    }
}

impl ApiKey for AnthropicKey {
    fn into_header(self) -> Option<http_client::Result<(HeaderName, HeaderValue)>> {
        match self {
            Self::ApiKey(key) => Some(
                HeaderValue::from_str(&key)
                    .map(|val| (HeaderName::from_static("x-api-key"), val))
                    .map_err(Into::into),
            ),
            // Deliberately no header: a SigV4 request must not also carry x-api-key, and the
            // signature cannot be computed here because the body does not exist yet.
            Self::SigV4 { .. } => None,
        }
    }
}

/// Provider for the Anthropic Messages API served over an AWS-fronted endpoint.
#[derive(Debug, Default, Clone)]
pub struct AnthropicOnAws {
    /// Set when the client was built with [`AnthropicKey::sigv4`]. Carries the AWS region to sign
    /// for. `None` means header-based auth (`x-api-key`), and then nothing is signed.
    ///
    /// This lives on the provider value rather than being inferred from the base URL so that the
    /// choice is explicit: a blank API key against an AWS host would otherwise silently become a
    /// signed request, and the resulting 403 would look nothing like a missing credential.
    signing_region: Option<String>,
}

pub type Client<H = rig_core::http_client::BoxedHttpClient> = client::Client<AnthropicOnAws, H>;
pub type ClientBuilder<H = Missing> = client::ClientBuilder<AnthropicOnAws, H>;

impl Provider for AnthropicOnAws {
    // Construction-error and `Debug` name. Deliberately distinct from `PROVIDER_NAME` below, which
    // is the telemetry system name and matches the rest of this crate.
    const NAME: &'static str = "anthropic-sigv4";
    // No default. See the module docs: a wrong endpoint here would sign a request to the wrong host.
    const BASE_URL: &'static str = "";
    const VERIFY_PATH: &'static str = "/v1/models";
    type ApiKey = AnthropicKey;
    type Config = AnthropicConfig;
    /// The AWS region to sign for.
    type EnvInput = String;

    fn build(_: AnthropicConfig, api_key: &AnthropicKey) -> http_client::Result<Self> {
        // Carry the signing region from the key onto the provider value, which is the only part of
        // the builder the request path can still see. `into_header()` discards the key itself.
        let signing_region = match api_key {
            AnthropicKey::SigV4 { region } => Some(region.clone()),
            AnthropicKey::ApiKey(_) => None,
        };
        Ok(AnthropicOnAws { signing_region })
    }

    fn finish<H>(
        &self,
        builder: client::ClientBuilder<Self, H>,
    ) -> http_client::Result<client::ClientBuilder<Self, H>> {
        if builder.get_base_url().is_empty() {
            return Err(http_client::Error::Instance(
                "no endpoint configured for the AWS-fronted Anthropic API: set one with \
                 ClientBuilder::base_url, or the ANTHROPIC_BASE_URL environment variable"
                    .into(),
            ));
        }

        finish_anthropic_builder(builder)
    }

    fn from_env<H: HttpClientExt>(http: H) -> ProviderClientResult<Client<H>> {
        // The region the AWS SDK would resolve anyway, read here because it goes into the signature
        // rather than into the SDK config.
        let region = match client::optional_env_var("AWS_REGION")? {
            Some(region) => region,
            None => client::optional_env_var("AWS_DEFAULT_REGION")?.ok_or(
                ProviderClientError::InvalidConfiguration(
                    "no AWS region for SigV4 signing: set AWS_REGION or AWS_DEFAULT_REGION",
                ),
            )?,
        };

        Self::from_val(region, http)
    }

    fn from_val<H: HttpClientExt>(region: String, http: H) -> ProviderClientResult<Client<H>> {
        let mut builder = Client::<Missing>::builder().api_key(AnthropicKey::sigv4(region));
        if let Some(base_url) = client::optional_env_var(BASE_URL_ENV)? {
            builder = builder.base_url(base_url);
        }

        builder.http_client(http).build()
    }
}

impl HasCompletion for AnthropicOnAws {
    type Model<H>
        = GenericCompletionModel<AnthropicOnAws, H>
    where
        H: ModelTransport;

    fn completion_model<H: ModelTransport>(client: &Client<H>, model: String) -> Self::Model<H> {
        GenericCompletionModel::new(client.clone(), model)
    }
}

impl AnthropicCompatibleProvider for AnthropicOnAws {
    // Telemetry system name. `aws_bedrock`, matching this crate's Bedrock Runtime models, because
    // the call is billed and logged by AWS even though the wire dialect is Anthropic's.
    const PROVIDER_NAME: &'static str = "aws_bedrock";

    // The endpoint serves Anthropic's own models, so Anthropic's published output limits apply.
    // Reusing rig-core's table rather than a flat number keeps `max_tokens` optional here exactly as
    // it is for `anthropic::Client`.
    fn default_max_tokens(model: &str) -> Option<u64> {
        default_max_tokens_for_model(model)
    }

    async fn signed_headers(
        &self,
        method: &str,
        uri: &str,
        body: &[u8],
    ) -> Result<Vec<(String, String)>, CompletionError> {
        match &self.signing_region {
            Some(region) => sigv4::signed_headers(method, uri, body, region).await,
            None => Ok(Vec::new()),
        }
    }
}

/// Install the AWS example credentials this module's SigV4 tests sign with.
///
/// One writer, run exactly once, because [`sigv4`] resolves the credential chain once per process:
/// whichever test reaches it first fixes what every later test signs with. Every signing test calls
/// this before its first `await` and `call_once` blocks until the variables are set, so resolution
/// order stops mattering. Two tests each setting up their own environment would instead make the
/// outcome depend on test order, and on a machine with no AWS configuration the loser would see
/// "no AWS credentials could be resolved".
///
/// A `Mutex` held across the awaits would have served too, but `clippy::await_holding_lock` is
/// denied workspace-wide and is right to be: `Once` needs no guard to outlive anything.
#[cfg(test)]
fn install_static_test_credentials() {
    static ONCE: std::sync::Once = std::sync::Once::new();

    ONCE.call_once(|| {
        // SAFETY: this is the only writer of these variables in the crate, and `call_once` runs it
        // exactly once, before any test here resolves credentials.
        unsafe {
            std::env::set_var("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE");
            std::env::set_var(
                "AWS_SECRET_ACCESS_KEY",
                "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            );
            std::env::set_var("AWS_REGION", "us-east-1");
        }
    });
}

#[cfg(test)]
mod tests;
