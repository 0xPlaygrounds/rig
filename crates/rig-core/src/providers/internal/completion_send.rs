//! Shared request driver for unary completion endpoints.
//!
//! Every provider builds its own request body and path — those are the real
//! wire differences — but the tail is identical: send the request, split
//! status and body, decode through the provider's success-or-error envelope,
//! record telemetry, trace-log the payload, and preserve raw error bodies via
//! [`CompletionError::from_http_response`]. This driver owns that tail.

use bytes::Bytes;
use serde::de::DeserializeOwned;

use super::envelope::ProviderEnvelope;
use crate::completion::CompletionError;
use crate::http_client::HttpClientExt;

/// Sends a unary completion request and decodes the provider's
/// success-or-error envelope.
///
/// `request` is the provider's fully built POST request; `A` is the
/// provider's own response envelope (use
/// [`DirectPayload`](super::envelope::DirectPayload) when the 2xx body IS the
/// payload); `record_telemetry` records response metadata and token usage on
/// the current span; `label` names the provider in trace/error logs (e.g.
/// `"Gemini completion"`).
///
/// `request_id_header` names the provider's transport request-id response
/// header (e.g. Anthropic `request-id`, OpenAI `x-request-id`); when present
/// and the response carries it, its value is returned alongside the payload.
/// `None` — as the parameter or in the returned pair — means "this provider
/// does not report one", never an error.
///
/// Error paths, preserved exactly:
/// - non-success status → `from_http_response(status, raw_body)` with the
///   provider's request id and the failed response's headers attached, so
///   support's id and rate-limit metadata such as `Retry-After` stay
///   readable (rig#2314, rig#2210);
/// - 2xx error envelope → warn-log the provider message, preserve raw body;
/// - undecodable 2xx body → error-log the body, surface the JSON error.
pub(crate) async fn send_completion<C, A, F>(
    client: &C,
    request: crate::http_client::Request<Vec<u8>>,
    label: &str,
    request_id_header: Option<&str>,
    record_telemetry: F,
) -> Result<(A::Payload, Option<String>), CompletionError>
where
    C: HttpClientExt,
    A: DeserializeOwned + ProviderEnvelope,
    A::Payload: serde::Serialize,
    F: FnOnce(&A::Payload),
{
    send_completion_with::<C, A, F, _, _>(
        client,
        request,
        label,
        request_id_header,
        record_telemetry,
        Ok,
    )
    .await
}

/// Keep the attempt alive through provider normalization, including rejection
/// of a decoded empty/blocked response. The raw API uses an identity mapping.
pub(crate) async fn send_completion_with<C, A, F, N, R>(
    client: &C,
    request: crate::http_client::Request<Vec<u8>>,
    label: &str,
    request_id_header: Option<&str>,
    record_telemetry: F,
    normalize: N,
) -> Result<(R, Option<String>), CompletionError>
where
    C: HttpClientExt,
    A: DeserializeOwned + ProviderEnvelope,
    A::Payload: serde::Serialize,
    F: FnOnce(&A::Payload),
    N: FnOnce(A::Payload) -> Result<R, CompletionError>,
{
    let mut attempt = crate::observe::AdapterContext::from_request(&request);
    let result = async {
        let response = match client.send::<_, Bytes>(request).await.inspect_err(|error| {
            if let Some(attempt) = &mut attempt {
                if let Some(status) = error.non_success_status() {
                    attempt.response_with_headers(status, error.non_success_headers());
                }
                if let Some(body) = error.non_success_body() {
                    attempt.payload(body.as_bytes());
                }
            }
        }) {
            Ok(response) => response,
            // A transport that reports the non-success reply as an error: the
            // reply is the provider's, so it funnels to ProviderResponse with
            // the id read off its headers (rig#2314) and the headers themselves
            // (rig#2210); a response-less failure stays a transport error.
            Err(error) => {
                let provider_request_id = error
                    .non_success_headers()
                    .and_then(|headers| super::request_id_from_headers(headers, request_id_header));
                return Err(CompletionError::from_transport_error(error)
                    .with_provider_request_id(provider_request_id));
            }
        };

        // Take the response apart before awaiting the body: that hands over the
        // headers already owned, so preserving them onto an error (rig#2210) costs
        // no clone and every error path below can afford them — including the 2xx
        // error envelope, which is a failure the caller may need to back off from
        // even though its status says success.
        let (parts, body) = response.into_parts();
        let status = parts.status;
        if let Some(attempt) = &mut attempt {
            attempt.response_with_headers(status, Some(&parts.headers));
        }
        let provider_request_id = request_id_header.and_then(|header| {
            parts
                .headers
                .get(header)
                .and_then(|value| value.to_str().ok())
                .filter(|value| !value.is_empty())
                .map(str::to_string)
        });
        let response_headers = Some(Box::new(parts.headers));
        let body = body.await.map_err(CompletionError::HttpError)?;
        if let Some(attempt) = &mut attempt {
            attempt.payload(&body);
        }

        if !status.is_success() {
            return Err(CompletionError::from_http_response(
                status,
                String::from_utf8_lossy(&body),
            )
            .with_provider_request_id(provider_request_id)
            .with_response_headers(response_headers));
        }

        let envelope: A = serde_json::from_slice(&body).map_err(|err| {
            tracing::error!(
                error = %err,
                body = %String::from_utf8_lossy(&body),
                "failed to deserialize {label} response"
            );
            CompletionError::JsonError(err)
        })?;

        match envelope.into_payload() {
            Ok(payload) => {
                record_telemetry(&payload);
                super::trace_json(
                    crate::providers::internal::LogTarget::Completions,
                    &format!("{label} response"),
                    &payload,
                );
                normalize(payload).map(|payload| (payload, provider_request_id))
            }
            Err(message) => {
                tracing::warn!(message = %message, "provider returned an error response");
                // A 2xx error envelope's headers matter as much as a
                // non-success response's: gateways report rate limits this
                // way, with `Retry-After` alongside a 200 (rig#2210).
                Err(
                    CompletionError::from_http_response(status, String::from_utf8_lossy(&body))
                        .with_provider_request_id(provider_request_id)
                        .with_response_headers(response_headers),
                )
            }
        }
    }
    .await;
    if let Some(attempt) = &mut attempt {
        let ending = match &result {
            Ok(_) => crate::observe::AdapterEnding::Decoded,
            Err(error) => {
                if let Some(status) = error.provider_response_status() {
                    attempt.response(status);
                }
                let report = crate::error::ErrorReport::from(error);
                crate::observe::AdapterEnding::Error {
                    boundary: crate::observe::AdapterErrorBoundary::from_completion(error),
                    kind: report.kind.code().to_owned(),
                    status: report.http_status,
                    retryable: report.is_retryable(),
                }
            }
        };
        attempt.finish(ending);
    }
    result
}

/// rig#2210: the failed response's headers must survive the driver, so a
/// caller can read `Retry-After` off a 429 and back off correctly.
///
/// The driver sees two transport shapes (the bundled reqwest client reports
/// non-success as an *error* carrying the response's headers; a custom
/// `HttpClientExt` may hand the non-success *response* back) and classifies by
/// two contracts (a provider with a request-id header vs one without). All
/// four cells must preserve the headers.
#[cfg(test)]
mod header_preservation_tests;
