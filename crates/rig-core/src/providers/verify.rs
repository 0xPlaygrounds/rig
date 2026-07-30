//! Credential verification as data plus free functions.
//!
//! The replacement for the deleted `Provider::VERIFY_PATH` const and the
//! `VerifyClient` trait: the endpoint is a
//! [`ProviderDescriptor::verify_path`] field, and the round trip is
//! [`send_verify`] — a free function over
//! [`HttpRuntime`] that every provider's
//! `functions::verify` calls.
//!
//! The status mapping is the one the deleted blanket
//! `impl VerifyClient for Client<Ext, H>` used, so the error semantics a
//! caller could match on are unchanged:
//!
//! - `200` (and any other success status) → `Ok(())`;
//! - `401` / `403` → [`VerifyError::InvalidAuthentication`];
//! - `500` and `529` → [`VerifyError::HttpError`] with the body preserved;
//! - any other non-success status → [`VerifyError::HttpError`] with the body
//!   preserved;
//! - transport failures → [`VerifyError::HttpError`].
//!
//! The one addition is [`VerifyError::Unsupported`], for providers whose
//! descriptor carries no `verify_path`. Those providers declared
//! `const VERIFY_PATH: &'static str = ""` before the deletion, which made the
//! classic `verify()` issue a bare `GET` of the base URL — a request that
//! validated no credential and, for the endpoints in question, did not even
//! return a success status. Reporting it as unsupported states that fact
//! instead of dressing it up as a check.

use http::StatusCode;

use crate::http_client;
use crate::http_runtime::HttpRuntime;
use crate::provider_response;
use crate::providers::descriptor::{ApiKeyLocation, ProviderDescriptor};

/// Errors from provider credential verification.
///
/// Inspect provider failures with [`Self::provider_response_body`],
/// [`Self::provider_response_json`], and [`Self::provider_response_status`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum VerifyError {
    /// The provider rejected the credential (`401`/`403`).
    #[error("invalid authentication")]
    InvalidAuthentication,
    /// A provider-side diagnostic that is not a preserved response body.
    #[error("provider error: {0}")]
    ProviderError(String),
    /// Raw error response preserved from the provider.
    #[error("provider response error: {0}")]
    ProviderResponse(provider_response::ProviderResponseError),
    /// Transport failure, or a non-success status with its body preserved.
    #[error("http error: {0}")]
    HttpError(
        #[from]
        #[source]
        http_client::Error,
    ),
    /// The provider exposes no credential-verification endpoint.
    #[error("provider `{provider}` does not support credential verification")]
    Unsupported {
        /// The provider whose descriptor carries no `verify_path`.
        provider: &'static str,
    },
}

crate::provider_response::impl_provider_response_helpers!(VerifyError);

/// The absolute verification URL for `descriptor` against `base_url`.
///
/// Joins exactly as the deleted `Provider::build_uri` default did: one
/// separating slash, no matter how either side is punctuated.
///
/// # Errors
/// [`VerifyError::Unsupported`] when the descriptor has no `verify_path`.
pub fn verify_url(descriptor: &ProviderDescriptor, base_url: &str) -> Result<String, VerifyError> {
    let path = descriptor.verify_path.ok_or(VerifyError::Unsupported {
        provider: descriptor.name,
    })?;
    Ok(format!(
        "{}/{}",
        base_url.trim_end_matches('/'),
        path.trim_start_matches('/')
    ))
}

/// Build the `GET` verification request for a Bearer-authenticated provider.
///
/// The overwhelmingly common shape: `Authorization: Bearer <key>` plus the
/// config's extra headers. A resolved-to-`None` credential
/// ([`ApiKeyLocation::None`], e.g. a local Ollama) sends no auth header, as
/// the classic client did.
///
/// # Errors
/// [`VerifyError::Unsupported`] when the descriptor has no `verify_path`;
/// [`VerifyError::ProviderError`] when the credential cannot be resolved.
pub fn build_bearer_verify_request(
    descriptor: &ProviderDescriptor,
    base_url: &str,
    api_key: &ApiKeyLocation,
    extra_headers: &[(String, String)],
) -> Result<http::Request<Vec<u8>>, VerifyError> {
    let url = verify_url(descriptor, base_url)?;
    let mut builder = http::Request::get(url);
    if let Some(key) = api_key
        .resolve()
        .map_err(|e| VerifyError::ProviderError(e.to_string()))?
    {
        builder = builder.header(http::header::AUTHORIZATION, format!("Bearer {key}"));
    }
    for (name, value) in extra_headers {
        builder = builder.header(name.as_str(), value.as_str());
    }
    builder
        .body(Vec::new())
        .map_err(|e| VerifyError::ProviderError(e.to_string()))
}

/// Send a built verification request and map its status to the classic
/// `VerifyClient::verify` semantics.
///
/// # Errors
/// See the [module documentation](self) for the full status mapping.
pub async fn send_verify(
    rt: &HttpRuntime,
    request: http::Request<Vec<u8>>,
) -> Result<(), VerifyError> {
    let (status, body) = rt.send(request).await.map_err(|error| match error {
        crate::completion::CompletionError::HttpError(error) => VerifyError::HttpError(error),
        other => VerifyError::ProviderError(other.to_string()),
    })?;
    map_verify_status(status, body)
}

/// The pure half of [`send_verify`]: status + body → verification verdict.
///
/// # Errors
/// See the [module documentation](self).
pub fn map_verify_status(status: StatusCode, body: String) -> Result<(), VerifyError> {
    match status {
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => Err(VerifyError::InvalidAuthentication),
        status if status.is_success() => Ok(()),
        status => Err(VerifyError::HttpError(
            http_client::Error::InvalidStatusCodeWithMessage(status, body),
        )),
    }
}

/// Verify a Bearer-authenticated provider's credential end to end.
///
/// The one-call shape every Bearer provider's `functions::verify` delegates
/// to.
///
/// # Errors
/// See the [module documentation](self).
pub async fn verify_bearer(
    descriptor: &ProviderDescriptor,
    base_url: &str,
    api_key: &ApiKeyLocation,
    extra_headers: &[(String, String)],
    rt: &HttpRuntime,
) -> Result<(), VerifyError> {
    let request = build_bearer_verify_request(descriptor, base_url, api_key, extra_headers)?;
    send_verify(rt, request).await
}

#[cfg(test)]
mod tests {
    use super::*;

    const WITH_PATH: ProviderDescriptor =
        ProviderDescriptor::named("test-provider").with_verify_path("/models");
    const WITHOUT_PATH: ProviderDescriptor = ProviderDescriptor::named("pathless-provider");

    /// Every provider's `verify_path` against the `Provider::VERIFY_PATH`
    /// const it replaces, as of the pre-deletion tree (`8afa8b752`). An empty
    /// classic path becomes `None`; see the module docs.
    #[test]
    fn every_descriptor_carries_its_classic_verify_path() {
        use crate::providers::*;

        let expected: &[(ProviderDescriptor, Option<&'static str>)] = &[
            (anthropic::functions::DESCRIPTOR, Some("/v1/models")),
            (azure::functions::DESCRIPTOR, None),
            (chatgpt::functions::DESCRIPTOR, None),
            (cohere::functions::DESCRIPTOR, Some("/models")),
            (copilot::functions::DESCRIPTOR, None),
            (deepseek::functions::DESCRIPTOR, Some("/user/balance")),
            (doubleword::functions::DESCRIPTOR, Some("/models")),
            (gemini::functions::DESCRIPTOR, Some("/v1beta/models")),
            (
                gemini::interactions_api::functions::DESCRIPTOR,
                Some("/v1beta/models"),
            ),
            (groq::functions::DESCRIPTOR, Some("/models")),
            (huggingface::functions::DESCRIPTOR, Some("/api/whoami-v2")),
            (hyperbolic::functions::DESCRIPTOR, Some("/models")),
            (llamafile::functions::DESCRIPTOR, Some("/models")),
            (minimax::functions::DESCRIPTOR, Some("/models")),
            (mira::functions::DESCRIPTOR, Some("/user-credits")),
            (mistral::functions::DESCRIPTOR, Some("/models")),
            (moonshot::functions::DESCRIPTOR, Some("/models")),
            (ollama::functions::DESCRIPTOR, Some("api/tags")),
            (openai::functions::DESCRIPTOR, Some("/models")),
            (
                openai::responses_api::functions::DESCRIPTOR,
                Some("/models"),
            ),
            (openrouter::functions::DESCRIPTOR, Some("/key")),
            (perplexity::functions::DESCRIPTOR, None),
            (together::functions::DESCRIPTOR, Some("/models")),
            (voyageai::functions::DESCRIPTOR, None),
            (xai::functions::DESCRIPTOR, Some("/v1/api-key")),
            (xiaomimimo::functions::DESCRIPTOR, Some("/models")),
            (zai::functions::DESCRIPTOR, Some("/models")),
        ];

        for (descriptor, path) in expected {
            assert_eq!(
                descriptor.verify_path, *path,
                "verify_path drifted for `{}`",
                descriptor.name
            );
        }
    }

    #[tokio::test]
    async fn verify_round_trips_and_maps_provider_failures() {
        use crate::providers::openai;
        use crate::test_utils::RecordingHttpClient;

        // Success: the credential is accepted.
        let rt = HttpRuntime::recording(RecordingHttpClient::new(r#"{"data":[]}"#));
        let cfg = openai::functions::Config::new("gpt-4o").with_api_key("secret");
        openai::functions::verify(&cfg, &rt)
            .await
            .expect("a 200 verifies");

        // 401: invalid authentication, not a generic HTTP error.
        let rt = HttpRuntime::recording(RecordingHttpClient::with_error_response(
            StatusCode::UNAUTHORIZED,
            r#"{"error":{"message":"bad key"}}"#,
        ));
        let error = openai::functions::verify(&cfg, &rt)
            .await
            .expect_err("a 401 must fail");
        assert!(matches!(error, VerifyError::InvalidAuthentication));

        // 500: the provider's response body survives for inspection, exactly
        // as the deleted `VerifyClient::verify` preserved it.
        let body = r#"{"error":{"message":"server exploded","type":"server_error"}}"#;
        let rt = HttpRuntime::recording(RecordingHttpClient::with_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            body,
        ));
        let error = openai::functions::verify(&cfg, &rt)
            .await
            .expect_err("a 500 must fail");
        assert_eq!(
            error.provider_response_status(),
            Some(StatusCode::INTERNAL_SERVER_ERROR)
        );
        assert_eq!(error.provider_response_body(), Some(body));
        let json = error
            .provider_response_json()
            .expect("valid JSON")
            .expect("a body");
        assert_eq!(json["error"]["type"], "server_error");
    }

    #[tokio::test]
    async fn pathless_providers_report_unsupported_without_sending_a_request() {
        use crate::providers::perplexity;
        use crate::test_utils::RecordingHttpClient;

        let http_client = RecordingHttpClient::new("{}");
        let rt = HttpRuntime::recording(http_client.clone());
        let cfg = perplexity::functions::Config::new("sonar").with_api_key("secret");

        let error = perplexity::functions::verify(&cfg, &rt)
            .await
            .expect_err("perplexity has no verify endpoint");
        assert!(matches!(
            error,
            VerifyError::Unsupported {
                provider: "perplexity"
            }
        ));
        assert!(
            http_client.requests().is_empty(),
            "no request should reach the transport"
        );
    }

    #[test]
    fn verify_url_joins_with_exactly_one_slash() {
        assert_eq!(
            verify_url(&WITH_PATH, "https://api.example.com").expect("url"),
            "https://api.example.com/models"
        );
        assert_eq!(
            verify_url(&WITH_PATH, "https://api.example.com/").expect("url"),
            "https://api.example.com/models"
        );
    }

    #[test]
    fn verify_url_reports_unsupported_without_a_path() {
        let error = verify_url(&WITHOUT_PATH, "https://api.example.com")
            .expect_err("a pathless descriptor cannot verify");
        assert!(matches!(
            error,
            VerifyError::Unsupported {
                provider: "pathless-provider"
            }
        ));
    }

    #[test]
    fn bearer_request_carries_the_credential_and_extra_headers() {
        let request = build_bearer_verify_request(
            &WITH_PATH,
            "https://api.example.com",
            &ApiKeyLocation::Inline("secret".to_string()),
            &[("x-trace".to_string(), "on".to_string())],
        )
        .expect("build");
        assert_eq!(request.method(), http::Method::GET);
        assert_eq!(request.uri(), "https://api.example.com/models");
        assert_eq!(
            request
                .headers()
                .get(http::header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok()),
            Some("Bearer secret")
        );
        assert_eq!(
            request
                .headers()
                .get("x-trace")
                .and_then(|v| v.to_str().ok()),
            Some("on")
        );
    }

    #[test]
    fn bearer_request_omits_auth_when_there_is_no_credential() {
        let request = build_bearer_verify_request(
            &WITH_PATH,
            "http://localhost:11434",
            &ApiKeyLocation::None,
            &[],
        )
        .expect("build");
        assert!(request.headers().get(http::header::AUTHORIZATION).is_none());
    }

    #[test]
    fn status_mapping_matches_the_classic_verify_client() {
        assert!(map_verify_status(StatusCode::OK, String::new()).is_ok());
        // Any success status verifies, not just 200.
        assert!(map_verify_status(StatusCode::NO_CONTENT, String::new()).is_ok());

        for status in [StatusCode::UNAUTHORIZED, StatusCode::FORBIDDEN] {
            assert!(matches!(
                map_verify_status(status, "nope".to_string()),
                Err(VerifyError::InvalidAuthentication)
            ));
        }

        let error = map_verify_status(
            StatusCode::INTERNAL_SERVER_ERROR,
            r#"{"error":"boom"}"#.to_string(),
        )
        .expect_err("500 must error");
        assert_eq!(
            error.provider_response_status(),
            Some(StatusCode::INTERNAL_SERVER_ERROR)
        );
        assert_eq!(error.provider_response_body(), Some(r#"{"error":"boom"}"#));

        let overloaded = map_verify_status(
            StatusCode::from_u16(529).expect("529"),
            "overloaded".to_string(),
        )
        .expect_err("529 must error");
        assert_eq!(overloaded.provider_response_body(), Some("overloaded"));

        let not_found = map_verify_status(StatusCode::NOT_FOUND, "missing".to_string())
            .expect_err("404 must error");
        assert_eq!(
            not_found.provider_response_status(),
            Some(StatusCode::NOT_FOUND)
        );
    }
}
