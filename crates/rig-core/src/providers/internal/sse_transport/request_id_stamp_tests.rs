use super::*;
use crate::completion::CompletionError;
use crate::provider_response::ProviderResponseError;
use crate::streaming::RawStreamingChoice;

/// rig#2314: an in-band provider error envelope yielded as a stream error
/// item is stamped with the delivering connection's request id; other
/// error variants and pass-through items are untouched.
#[tokio::test]
async fn stamps_provider_response_stream_errors_from_the_slot() {
    let slot = crate::http_client::sse::RequestIdSlot::default();
    *slot.lock().expect("slot") = Some("req_conn_1".to_string());

    let stream: crate::streaming::RawStreamingResult<String> =
        Box::pin(futures::stream::iter(vec![
            Ok(RawStreamingChoice::Message("hi".to_string())),
            Err(CompletionError::ProviderResponse(
                ProviderResponseError::new(http::StatusCode::OK, r#"{"type":"error"}"#),
            )),
            Err(CompletionError::ResponseError("unrelated".to_string())),
        ]));

    let stamped = stamp_terminal_request_id(stream, Some(slot), None, |_, _| {});
    let items: Vec<_> = stamped.collect().await;

    assert!(matches!(
        &items[0],
        Ok(RawStreamingChoice::Message(text)) if text == "hi"
    ));
    match &items[1] {
        Err(error) => {
            assert_eq!(error.provider_request_id(), Some("req_conn_1"));
        }
        other => panic!("expected the stamped provider error, got {other:?}"),
    }
    match &items[2] {
        Err(CompletionError::ResponseError(_)) => {}
        other => panic!("non-provider errors pass through untouched, got {other:?}"),
    }
}

/// rig#2315 follow-up: a failed SSE handshake (details-preserving
/// transport error) classifies like the unary driver for contract
/// providers — ProviderResponse carrying the failed response's own id.
#[tokio::test]
async fn handshake_details_error_classifies_with_contract() {
    let mut headers = http::HeaderMap::new();
    headers.insert("x-request-id", "req_handshake".parse().expect("value"));
    let stream: crate::streaming::RawStreamingResult<String> =
        Box::pin(futures::stream::iter(vec![Err(
            CompletionError::HttpError(crate::http_client::Error::InvalidStatusCodeWithDetails {
                status: http::StatusCode::NOT_FOUND,
                body: r#"{"error":"no model"}"#.to_string(),
                headers: Box::new(headers),
            }),
        )]));

    let stamped = stamp_terminal_request_id(
        stream,
        Some(crate::http_client::sse::RequestIdSlot::default()),
        Some("x-request-id"),
        |_, _| {},
    );
    let items: Vec<_> = stamped.collect().await;
    match &items[0] {
        Err(error) => {
            assert!(matches!(error, CompletionError::ProviderResponse(_)));
            assert_eq!(error.provider_request_id(), Some("req_handshake"));
            assert_eq!(
                error.provider_response_status(),
                Some(http::StatusCode::NOT_FOUND)
            );
        }
        other => panic!("expected the classified handshake error, got {other:?}"),
    }
}

/// rig#2210: a rate-limited *streaming* handshake must expose the same
/// `Retry-After` its blocking twin does — the classification hands the
/// error to a different constructor, which is exactly where headers get
/// lost if the conversion forgets them.
#[tokio::test]
async fn handshake_details_error_preserves_rate_limit_headers() {
    let mut headers = http::HeaderMap::new();
    headers.insert("x-request-id", "req_handshake".parse().expect("value"));
    headers.insert("retry-after", "20".parse().expect("value"));
    headers.insert("x-ratelimit-remaining", "0".parse().expect("value"));
    let stream: crate::streaming::RawStreamingResult<String> =
        Box::pin(futures::stream::iter(vec![Err(
            CompletionError::HttpError(crate::http_client::Error::InvalidStatusCodeWithDetails {
                status: http::StatusCode::TOO_MANY_REQUESTS,
                body: r#"{"error":"slow down"}"#.to_string(),
                headers: Box::new(headers),
            }),
        )]));

    let stamped = stamp_terminal_request_id(
        stream,
        Some(crate::http_client::sse::RequestIdSlot::default()),
        Some("x-request-id"),
        |_, _| {},
    );
    let items: Vec<_> = stamped.collect().await;
    match &items[0] {
        Err(error) => {
            let headers = error
                .provider_response_headers()
                .expect("handshake headers preserved onto the classified error");
            assert_eq!(
                headers
                    .get(http::header::RETRY_AFTER)
                    .and_then(|value| value.to_str().ok()),
                Some("20"),
            );
            assert_eq!(
                headers
                    .get("x-ratelimit-remaining")
                    .and_then(|value| value.to_str().ok()),
                Some("0"),
            );
            // The #2314 metadata is untouched by the header capture.
            assert_eq!(error.provider_request_id(), Some("req_handshake"));
            assert_eq!(
                error.provider_response_status(),
                Some(http::StatusCode::TOO_MANY_REQUESTS)
            );
            assert_eq!(
                error.provider_response_body(),
                Some(r#"{"error":"slow down"}"#)
            );
        }
        other => panic!("expected the classified handshake error, got {other:?}"),
    }
}
