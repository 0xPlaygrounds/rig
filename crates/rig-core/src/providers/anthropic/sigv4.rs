//! AWS SigV4 request signing for Anthropic-compatible endpoints fronted by AWS.
//!
//! Only compiled with the `sigv4` feature. Selected per-client via [`super::AnthropicKey::sigv4`];
//! never inferred from the URL, so an unsigned request can never silently become a signed one.
//!
//! Signing is per-request rather than a static header because the signature covers the request body
//! and the current time. That imposes an ordering constraint on callers: **sign last**. The payload
//! hash is taken over the exact bytes sent, so any body shaping must already have happened.

use std::sync::OnceLock;
use std::time::SystemTime;

use aws_config::BehaviorVersion;
use aws_credential_types::provider::ProvideCredentials;
use aws_sigv4::http_request::{SignableBody, SignableRequest, SigningSettings, sign};
use aws_sigv4::sign::v4;
use tokio::sync::OnceCell;

use crate::completion::CompletionError;

/// The AWS service name that goes in the SigV4 credential scope.
///
/// `bedrock-mantle`, NOT `bedrock`. Taken from the endpoint's own CloudTrail integration, whose
/// `eventSource` is `bedrock-mantle.amazonaws.com` and which logs inference as `CreateInference`,
/// matching the IAM prefix `bedrock-mantle:CreateInference`. The `bedrock-runtime` endpoint signs
/// with `bedrock` by the same rule. Getting this wrong yields a 403 whose message does not mention
/// the credential scope, so it is worth stating where the value came from.
const SIGNING_SERVICE: &str = "bedrock-mantle";

/// Resolved once per process. The credential chain performs I/O -- profile files, SSO cache, IMDS --
/// and re-running it on every request would add that cost to each model call.
static SDK_CONFIG: OnceCell<aws_config::SdkConfig> = OnceCell::const_new();

/// Cached so the "feature is on but no credentials" message is produced once.
static WARNED: OnceLock<()> = OnceLock::new();

async fn sdk_config() -> &'static aws_config::SdkConfig {
    SDK_CONFIG
        .get_or_init(|| async { aws_config::load_defaults(BehaviorVersion::latest()).await })
        .await
}

/// Compute the SigV4 headers for one request.
///
/// Returns the headers to add. Applying them is the caller's job, because the two Anthropic request
/// paths build their requests differently and neither owns an `http::Request` at this point.
///
/// `host` is derived from `uri` and signed. It is required in the canonical request, and it is not
/// present on the builder at this stage -- the HTTP client adds it later, so relying on the
/// builder's headers alone would sign a canonical request the server cannot reproduce.
pub(crate) async fn signed_headers(
    method: &str,
    uri: &str,
    body: &[u8],
    region: &str,
) -> Result<Vec<(String, String)>, CompletionError> {
    let host = uri
        .split("://")
        .nth(1)
        .and_then(|rest| rest.split('/').next())
        .ok_or_else(|| {
            CompletionError::RequestError(format!("could not derive a host to sign from {uri:?}").into())
        })?
        .to_owned();

    let config = sdk_config().await;
    let provider = config.credentials_provider().ok_or_else(|| {
        let _ = WARNED.set(());
        CompletionError::RequestError(
            "SigV4 auth was requested but no AWS credentials could be resolved. The standard chain \
             was consulted (environment, shared config/credentials, SSO cache, IMDS)."
                .into(),
        )
    })?;
    let credentials = provider
        .provide_credentials()
        .await
        .map_err(|e| CompletionError::RequestError(format!("AWS credential resolution failed: {e}").into()))?;
    let identity = credentials.into();

    let params = v4::SigningParams::builder()
        .identity(&identity)
        .region(region)
        .name(SIGNING_SERVICE)
        .time(SystemTime::now())
        .settings(SigningSettings::default())
        .build()
        .map_err(|e| CompletionError::RequestError(format!("SigV4 params: {e}").into()))?;

    // Only `host` is declared as signed. SignedHeaders must list exactly what was included, so
    // adding headers here that the client may alter in flight would break the signature.
    let signable = SignableRequest::new(
        method,
        uri,
        std::iter::once(("host", host.as_str())),
        SignableBody::Bytes(body),
    )
    .map_err(|e| CompletionError::RequestError(format!("SigV4 signable request: {e}").into()))?;

    let out = sign(signable, &params.into())
        .map_err(|e| CompletionError::RequestError(format!("SigV4 signing failed: {e}").into()))?;

    let (headers, _query) = out.into_parts().0.into_parts();
    // Return ONLY the signer's own headers (authorization, x-amz-date, x-amz-security-token).
    //
    // Do NOT add `host`. It is signed above because the canonical request requires it, but the HTTP
    // client sets its own `Host` from the URI, so adding one here puts the header on the wire TWICE.
    // The server then canonicalises it as a comma-joined pair --
    // `host:bedrock-mantle.us-west-2.api.aws,bedrock-mantle.us-west-2.api.aws` -- which cannot match
    // a signature computed over the single value, and every request fails 401 with
    // "the request signature we calculated does not match".
    //
    // This survived the hermetic tests because they assert the Authorization header's SHAPE and
    // credential scope; nothing offline verifies a signature. It took one live call to find.
    Ok(headers
        .into_iter()
        .map(|h| (h.name().to_owned(), h.value().to_owned()))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The returned headers must NOT include `host`.
    ///
    /// `host` is signed -- the canonical request requires it -- but must not be re-sent, because the
    /// HTTP client supplies it. Sending it too produced a live 401 whose message showed the service
    /// canonicalising `host` as a comma-joined PAIR of the same value.
    ///
    /// Asserted here, on the returned header list, rather than on the wire. Over HTTP/1.1 hyper
    /// collapses a user-supplied Host into its own, so a capture server on localhost sees exactly
    /// one either way and a wire-level assertion cannot fail. The duplication is only observable
    /// against the real HTTPS endpoint, where the connection is HTTP/2 and carries `:authority`
    /// instead. This test guards the invariant at the layer where it is deterministic.
    #[tokio::test]
    async fn signed_headers_never_include_host() {
        // SAFETY: set before the first credential resolution in this process.
        unsafe {
            std::env::set_var("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE");
            std::env::set_var(
                "AWS_SECRET_ACCESS_KEY",
                "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            );
            std::env::set_var("AWS_REGION", "us-east-1");
        }
        let headers = signed_headers(
            "POST",
            "https://bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages",
            b"{}",
            "us-east-1",
        )
        .await
        .expect("signing should succeed with static credentials");

        let names: Vec<String> = headers.iter().map(|(n, _)| n.to_lowercase()).collect();
        assert!(
            !names.iter().any(|n| n == "host"),
            "signed_headers returned a host header; the client sets its own, and sending both \
             breaks the signature. names={names:?}"
        );
        // Anti-vacuity: if this returned nothing at all the assertion above would pass trivially.
        assert!(
            names.iter().any(|n| n == "authorization"),
            "no authorization header produced, so the absence of host proves nothing. names={names:?}"
        );
    }
}
