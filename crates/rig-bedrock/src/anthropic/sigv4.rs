//! AWS SigV4 request signing for the Anthropic-compatible endpoint.
//!
//! Selected per-client via [`super::AnthropicKey::sigv4`]; never inferred from the URL, so an
//! unsigned request can never silently become a signed one.
//!
//! Signing is per-request rather than a static header because the signature covers the request body
//! and the current time. That imposes an ordering constraint on callers: **sign last**. The payload
//! hash is taken over the exact bytes sent, so any body shaping must already have happened.
//! rig-core's Anthropic request builders call
//! [`AnthropicCompatibleProvider::signed_headers`](rig_core::providers::anthropic::completion::AnthropicCompatibleProvider::signed_headers)
//! at exactly that point.

use std::sync::Arc;
use std::time::SystemTime;

use aws_config::BehaviorVersion;
use aws_credential_types::provider::ProvideCredentials;
use aws_sigv4::http_request::{SignableBody, SignableRequest, SigningSettings, sign};
use aws_sigv4::sign::v4;
use rig_core::completion::CompletionError;
use tokio::sync::OnceCell;

/// The AWS service name that goes in the SigV4 credential scope.
///
/// `bedrock-mantle`, NOT `bedrock`. Taken from the endpoint's own CloudTrail integration, whose
/// `eventSource` is `bedrock-mantle.amazonaws.com` and which logs inference as `CreateInference`,
/// matching the IAM prefix `bedrock-mantle:CreateInference`. The `bedrock-runtime` endpoint signs
/// with `bedrock` by the same rule. Getting this wrong yields a 403 whose message does not mention
/// the credential scope, so it is worth stating where the value came from.
const SIGNING_SERVICE: &str = "bedrock-mantle";

/// One client's resolved AWS SDK configuration, shared with that client's clones.
///
/// Per client, deliberately not per process. The credential chain performs I/O -- profile files,
/// SSO cache, IMDS -- so it is resolved once and reused rather than re-run on every model call.
/// Caching it in a `static`, which is what this used to do, made that "once per process": the
/// FIRST client built would fix the identity every later client signed with. A process that builds
/// one client under one profile or assumed role and a second expecting another would sign both as
/// whichever chain resolved first -- at best a 403 naming an unexpected principal, at worst a call
/// authorized and billed against the wrong account, with nothing in the request to show why.
///
/// `Arc<OnceCell<_>>` rather than an owned `SdkConfig` field because [`Provider::build`] is
/// synchronous and cannot await `load_defaults`. The cell defers resolution to the client's first
/// signed request while keeping "resolve once" intact, and the `Arc` lets a cloned client -- same
/// credentials by construction -- share that one resolution instead of repeating it.
///
/// Residual limitation: resolution happens at first request, not at construction. Two clients
/// built back to back each get their own cell and so their own chain, but each reads the ambient
/// environment when it first signs. A caller who mutates `AWS_PROFILE` between constructing two
/// clients and only then sends through either does not get the profile that was set at
/// construction time. Binding at construction would require an `async` `Provider::build`, which
/// the trait does not offer.
///
/// `tokio::sync::OnceCell`, not `std::sync::OnceLock`: initialisation is `async` because it awaits
/// `load_defaults`, and `OnceLock` has no async initialiser. tokio is a first-class dependency of
/// this crate, which is one of the reasons the signing belongs here rather than in rig-core.
///
/// [`Provider::build`]: rig_core::client::Provider::build
pub(crate) type SharedSdkConfig = Arc<OnceCell<aws_config::SdkConfig>>;

async fn sdk_config(cell: &OnceCell<aws_config::SdkConfig>) -> &aws_config::SdkConfig {
    cell.get_or_init(|| async { aws_config::load_defaults(BehaviorVersion::latest()).await })
        .await
}

/// The `host` value to put in the canonical request: exactly what the transport will send.
///
/// Read off the parsed authority rather than recovered from the URI's string form. Two things make
/// the string form wrong, and each yields a canonical request the service cannot reproduce -- so
/// every call fails 401 "the request signature we calculated does not match", the same opaque
/// failure as the duplicated host header described below:
///
/// - **A default port.** `https://host:443/...` goes on the wire as `Host: host`, with no port.
///   Two independent reasons, both in the transport: `hyper_util`'s client builds `Host` from
///   `uri.host()` plus `get_non_default_port`, which maps 443/https and 80/http to `None`; and
///   `rig-reqwest` passes `uri.to_string()` to reqwest, which re-parses it with the `url` crate,
///   whose `default_port` table drops the port during parsing. Signing `host:443` signs a host
///   nothing sends.
/// - **Userinfo.** `https://user@host/...` has host `host`. HTTP/2 forbids userinfo in
///   `:authority` outright, and no HTTP/1.1 client puts it in `Host`.
///
/// The default-port table below is the intersection of those two -- they agree: 443 for `https`
/// and `wss`, 80 for `http` and `ws`. A scheme outside that set has no known default, so any port
/// it carries is significant and is signed.
fn host_to_sign(uri: &http::Uri) -> Result<String, CompletionError> {
    let host = uri.host().ok_or_else(|| {
        CompletionError::RequestError(
            format!("could not derive a host to sign from {uri:?}").into(),
        )
    })?;

    let default_port = match uri.scheme_str() {
        Some("https" | "wss") => Some(443),
        Some("http" | "ws") => Some(80),
        _ => None,
    };

    match uri.port_u16() {
        Some(port) if Some(port) != default_port => Ok(format!("{host}:{port}")),
        _ => Ok(host.to_owned()),
    }
}

/// Compute the SigV4 headers for one request.
///
/// Returns the headers to add. Applying them is the caller's job, because the two Anthropic request
/// paths build their requests differently and neither owns an `http::Request` at this point.
///
/// `host` is derived from `uri` by [`host_to_sign`] and signed. It is required in the canonical
/// request, and it is not present on the builder at this stage -- the HTTP client adds it later, so
/// relying on the builder's headers alone would sign a canonical request the server cannot
/// reproduce.
pub(crate) async fn signed_headers(
    method: &str,
    uri: &http::Uri,
    body: &[u8],
    region: &str,
    config_cell: &OnceCell<aws_config::SdkConfig>,
) -> Result<Vec<(String, String)>, CompletionError> {
    signed_headers_at(method, uri, body, region, config_cell, SystemTime::now()).await
}

/// [`signed_headers`] with the signing instant supplied.
///
/// Split out only so tests can hold the clock still. A signature covers the time, so two
/// signatures taken a second apart differ for that reason alone -- which would make it impossible
/// to assert that some *other* input, such as a default port in the URI, left the signature
/// untouched.
async fn signed_headers_at(
    method: &str,
    uri: &http::Uri,
    body: &[u8],
    region: &str,
    config_cell: &OnceCell<aws_config::SdkConfig>,
    time: SystemTime,
) -> Result<Vec<(String, String)>, CompletionError> {
    let host = host_to_sign(uri)?;
    let uri = uri.to_string();

    let config = sdk_config(config_cell).await;
    let provider = config.credentials_provider().ok_or_else(|| {
        CompletionError::RequestError(
            "SigV4 auth was requested but no AWS credentials could be resolved. The standard chain \
             was consulted (environment, shared config/credentials, SSO cache, IMDS)."
                .into(),
        )
    })?;
    let credentials = provider.provide_credentials().await.map_err(|e| {
        CompletionError::RequestError(format!("AWS credential resolution failed: {e}").into())
    })?;
    let identity = credentials.into();

    let params = v4::SigningParams::builder()
        .identity(&identity)
        .region(region)
        .name(SIGNING_SERVICE)
        .time(time)
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

    fn uri(s: &str) -> http::Uri {
        s.parse().expect("test URI is valid")
    }

    /// A fixed instant, so a signature depends only on the inputs under test.
    fn fixed_time() -> SystemTime {
        SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000)
    }

    fn authorization(headers: &[(String, String)]) -> String {
        headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("authorization"))
            .map(|(_, value)| value.clone())
            .expect("signing produced no authorization header")
    }

    /// The host that goes into the canonical request must be the one the transport puts on the
    /// wire, and the transport drops a scheme's default port.
    ///
    /// This is the whole correctness condition for the host. Both sides are pinned to a rule
    /// verified in the transport's own source rather than assumed:
    ///
    /// - `hyper_util::client::legacy::Client` builds `Host` from `uri.host()` plus
    ///   `get_non_default_port(uri)`, which maps 443/https and 80/http to `None`. So a URI
    ///   carrying an explicit `:443` is sent as `Host: <host>`, with no port.
    /// - `rig-reqwest` hands `parts.uri.to_string()` to `reqwest::Client::request`, so the URI is
    ///   re-parsed by the `url` crate, whose `default_port` table (443 for https/wss, 80 for
    ///   http/ws) drops the port during parsing -- before hyper ever sees it.
    ///
    /// Both paths therefore sign and send a bare host. Signing `<host>:443` against either
    /// produces a canonical request the service cannot reproduce, and every call fails 401 with
    /// "the request signature we calculated does not match" -- the same failure class as the
    /// duplicated host header documented above, and just as invisible to a shape assertion.
    #[test]
    fn the_host_to_sign_is_what_the_transport_will_send() {
        let mut wrong = Vec::new();

        for (input, expected, why) in [
            (
                "https://bedrock-mantle.us-east-1.api.aws:443/anthropic/v1/messages",
                "bedrock-mantle.us-east-1.api.aws",
                "443 is the default for https, so the transport omits it",
            ),
            (
                "http://localhost:80/anthropic/v1/messages",
                "localhost",
                "80 is the default for http, so the transport omits it",
            ),
            (
                "https://localhost:8443/anthropic/v1/messages",
                "localhost:8443",
                "a non-default port is part of the host the transport sends",
            ),
            (
                "https://bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages",
                "bedrock-mantle.us-east-1.api.aws",
                "no port in, no port out",
            ),
            (
                "https://user@bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages",
                "bedrock-mantle.us-east-1.api.aws",
                "userinfo is not part of the host, and HTTP/2 forbids it in :authority",
            ),
            (
                "https://[2001:db8::1]:8443/anthropic/v1/messages",
                "[2001:db8::1]:8443",
                "an IPv6 literal keeps its brackets and its non-default port",
            ),
        ] {
            // Collected rather than asserted case by case: one run should report every host it
            // gets wrong, not just the first.
            let actual = host_to_sign(&uri(input)).expect("a URI with an authority yields a host");
            if actual != expected {
                wrong.push(format!(
                    "{input}\n  signed {actual:?}, transport sends {expected:?} ({why})"
                ));
            }
        }

        assert!(
            wrong.is_empty(),
            "{} of the cases signed a host the transport will not send:\n{}",
            wrong.len(),
            wrong.join("\n")
        );
    }

    /// A URI with no authority cannot be signed, and must say so rather than sign something else.
    #[test]
    fn a_uri_without_a_host_is_an_error() {
        let error = host_to_sign(&uri("/anthropic/v1/messages"))
            .expect_err("a path-only URI has no host to sign");
        assert!(
            error.to_string().contains("host"),
            "the error should say a host could not be derived: {error}"
        );
    }

    /// End-to-end at the signing call: an explicit default port must not reach the signature.
    ///
    /// The value-level assertion above pins `host_to_sign`; this pins that `signed_headers`
    /// actually routes through it, which is the wiring a refactor could quietly drop.
    #[tokio::test]
    async fn a_default_port_does_not_change_the_signature() {
        super::super::install_static_test_credentials();
        // One cell for both calls, so the two signatures are made with the same credentials and
        // the only difference left between them is the port.
        let credentials = OnceCell::new();

        let with_port = signed_headers_at(
            "POST",
            &uri("https://bedrock-mantle.us-east-1.api.aws:443/anthropic/v1/messages"),
            b"{}",
            "us-east-1",
            &credentials,
            fixed_time(),
        )
        .await
        .expect("signing should succeed with static credentials");

        let without_port = signed_headers_at(
            "POST",
            &uri("https://bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages"),
            b"{}",
            "us-east-1",
            &credentials,
            fixed_time(),
        )
        .await
        .expect("signing should succeed with static credentials");

        assert_eq!(
            authorization(&with_port),
            authorization(&without_port),
            "an explicit :443 changed the signature, so the signed host carried the port the \
             transport will not send"
        );
    }

    /// Control for the test above: the port genuinely reaches the signature when it is not the
    /// default. Without this, signing a constant host -- or no host at all -- would satisfy that
    /// test while breaking every request to a non-standard port.
    #[tokio::test]
    async fn a_non_default_port_does_change_the_signature() {
        super::super::install_static_test_credentials();
        let credentials = OnceCell::new();

        let with_port = signed_headers_at(
            "POST",
            &uri("https://bedrock-mantle.us-east-1.api.aws:8443/anthropic/v1/messages"),
            b"{}",
            "us-east-1",
            &credentials,
            fixed_time(),
        )
        .await
        .expect("signing should succeed with static credentials");

        let without_port = signed_headers_at(
            "POST",
            &uri("https://bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages"),
            b"{}",
            "us-east-1",
            &credentials,
            fixed_time(),
        )
        .await
        .expect("signing should succeed with static credentials");

        assert_ne!(
            authorization(&with_port),
            authorization(&without_port),
            ":8443 must be part of the signed host, so the two signatures must differ"
        );
    }

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
        super::super::install_static_test_credentials();

        let headers = signed_headers(
            "POST",
            &uri("https://bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages"),
            b"{}",
            "us-east-1",
            &OnceCell::new(),
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
