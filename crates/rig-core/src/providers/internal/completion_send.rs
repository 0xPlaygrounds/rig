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
/// - non-success status → `from_http_response(status, raw_body)`, with the
///   failed response's headers attached so rate-limit metadata such as
///   `Retry-After` stays readable (rig#2210);
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
    let response = match client.send::<_, Bytes>(request).await {
        Ok(response) => response,
        // The reqwest transport reports a non-success status as an error with
        // the failed response's headers preserved. A provider with a
        // request-id contract reads its header off them so the failed call's
        // transport id — the one support asks for — survives onto the error
        // (rig#2314); classification then follows the contract, so a given
        // provider's errors stay one shape. Either way the whole header map
        // rides along, so a caller can still read `Retry-After` off a 429
        // (rig#2210).
        Err(crate::http_client::Error::InvalidStatusCodeWithDetails {
            status,
            body,
            headers,
        }) => {
            return Err(match request_id_header {
                Some(header) => {
                    let provider_request_id = headers
                        .get(header)
                        .and_then(|value| value.to_str().ok())
                        .filter(|value| !value.is_empty())
                        .map(str::to_string);
                    CompletionError::from_http_response_with_request_id(
                        status,
                        body,
                        provider_request_id,
                    )
                    .with_response_headers(Some(headers))
                }
                // Contract-less providers keep the pre-#2314 transport shape;
                // the details variant is that shape plus the headers, and
                // displays identically.
                None => CompletionError::HttpError(
                    crate::http_client::Error::InvalidStatusCodeWithDetails {
                        status,
                        body,
                        headers,
                    },
                ),
            });
        }
        // A transport that reports non-success without preserved headers (a
        // custom `HttpClientExt`): a contract provider still classifies as
        // ProviderResponse — the shape follows the contract on every
        // transport — with no id to read.
        Err(crate::http_client::Error::InvalidStatusCodeWithMessage(status, body))
            if request_id_header.is_some() =>
        {
            return Err(CompletionError::from_http_response_with_request_id(
                status, body, None,
            ));
        }
        Err(other) => return Err(other.into()),
    };

    // Take the response apart before awaiting the body: that hands over the
    // headers already owned, so preserving them onto an error (rig#2210) costs
    // no clone and every error path below can afford them — including the 2xx
    // error envelope, which is a failure the caller may need to back off from
    // even though its status says success.
    let (parts, body) = response.into_parts();
    let status = parts.status;
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

    if !status.is_success() {
        // A provider with a request-id contract routes through the
        // metadata-aware funnel so the failed call's transport id — the one
        // support asks for — survives onto the error (rig#2314).
        // Classification follows the contract, not the header's presence on
        // a particular response, so a given provider's errors stay one shape.
        return Err(match request_id_header {
            Some(_) => CompletionError::from_http_response_with_request_id(
                status,
                String::from_utf8_lossy(&body),
                provider_request_id,
            ),
            None => CompletionError::from_http_response(status, String::from_utf8_lossy(&body)),
        }
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
            Ok((payload, provider_request_id))
        }
        Err(message) => {
            tracing::warn!(message = %message, "provider returned an error response");
            // A 2xx error envelope preserves as ProviderResponse either way;
            // the metadata-aware funnel just adds the captured id. Its headers
            // matter as much as a non-success response's: gateways report rate
            // limits this way, with `Retry-After` alongside a 200 (rig#2210).
            Err(match request_id_header {
                Some(_) => CompletionError::from_http_response_with_request_id(
                    status,
                    String::from_utf8_lossy(&body),
                    provider_request_id,
                ),
                None => CompletionError::from_http_response(status, String::from_utf8_lossy(&body)),
            }
            .with_response_headers(response_headers))
        }
    }
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
