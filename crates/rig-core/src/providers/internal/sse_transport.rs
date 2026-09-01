//! The shared SSE transport preamble: `GenericEventSource` events →
//! [`WireFrame`]s.
//!
//! Every SSE-backed streaming wire opens with the same loop — log the `Open`
//! event, skip payload-less heartbeat frames, yield each `data:` payload as a
//! text frame, end on `StreamEnded`, and surface transport errors in-band —
//! so the loop is stated once here. Byte splitting and framing only:
//! classification and policy live downstream in the wire adapters and
//! [`run_wire_stream`](super::adapter::run_wire_stream).
//!
//! The per-provider deltas are the variation points: the `Open` log level
//! ([`OpenLog`]), whether `StreamEnded` is a normal end or an error
//! (Anthropic's historical behavior), whether transport errors are logged,
//! and a per-frame triage closure that owns heartbeat/`[DONE]` filtering and
//! any in-band provider-error pre-filter.

use async_stream::stream;
use futures::{Stream, StreamExt};
use tracing_futures::Instrument;

use super::adapter::{WireAdapter, WireFrame, run_wire_stream};
use crate::completion::CompletionError;
use crate::http_client::HttpClientExt;
use crate::http_client::sse::{Event, GenericEventSource};
use crate::wasm_compat::WasmCompatSend;

/// How the transport logs the SSE `Open` event (a per-provider historical
/// delta, preserved exactly).
#[derive(Clone, Copy)]
pub(crate) enum OpenLog {
    Silent,
    Trace,
    Debug,
}

/// One frame's disposition, decided by the caller's triage closure.
pub(crate) enum FrameDisposition {
    /// Not a wire frame (heartbeat, `[DONE]` on wires that drop it): skip.
    Skip,
    /// A payload frame: yield it as [`WireFrame::Text`].
    Frame(String),
    /// An in-band terminal provider error (the wire's error envelope,
    /// detected pre-classification exactly as an HTTP failure would be):
    /// yield the error and end the transport.
    Fail(CompletionError),
}

/// Skip-blank triage shared by wires with no `[DONE]` sentinel and no
/// in-band error envelope: heartbeats carry no payload and are not wire
/// frames; everything else passes through untrimmed.
pub(crate) fn skip_blank_frames(data: String) -> FrameDisposition {
    if data.trim().is_empty() {
        FrameDisposition::Skip
    } else {
        FrameDisposition::Frame(data)
    }
}

/// Triage shared by wires whose heartbeats and `[DONE]` sentinel are both
/// dropped at the transport: trim the payload, skip blanks and `[DONE]`,
/// yield everything else trimmed.
pub(crate) fn skip_blank_and_done(data: &str) -> FrameDisposition {
    let data = data.trim();
    if data.is_empty() || data == "[DONE]" {
        FrameDisposition::Skip
    } else {
        FrameDisposition::Frame(data.to_owned())
    }
}

/// The per-wire transport deltas.
#[derive(Clone, Copy)]
pub(crate) struct SseTransportOptions {
    pub open_log: OpenLog,
    /// `false`: `StreamEnded` is the normal end of the stream (break).
    /// `true` (Anthropic): `StreamEnded` maps through
    /// [`CompletionError::from_stream_transport`] like any other transport
    /// error — its historical loop had no separate `StreamEnded` arm.
    pub stream_ended_is_error: bool,
    /// Whether transport errors are logged (`error!(?error, "SSE error")`)
    /// before being yielded in-band. Anthropic historically did not log.
    pub log_transport_errors: bool,
}

/// Run the SSE transport preamble: drain `event_source` into a stream of
/// [`WireFrame`]s for [`run_wire_stream`](super::adapter::run_wire_stream),
/// closing the event source when the loop ends.
pub(crate) fn sse_frames<HttpClient, RequestBody, F>(
    event_source: GenericEventSource<HttpClient, RequestBody>,
    options: SseTransportOptions,
    mut triage: F,
) -> impl Stream<Item = Result<WireFrame, CompletionError>>
where
    HttpClient: HttpClientExt + Clone + 'static,
    RequestBody: Into<bytes::Bytes> + Clone + WasmCompatSend + 'static,
    F: FnMut(String) -> FrameDisposition + WasmCompatSend + 'static,
{
    stream! {
        let mut event_source = Box::pin(event_source);
        while let Some(event_result) = event_source.next().await {
            match event_result {
                Ok(Event::Open) => match options.open_log {
                    OpenLog::Silent => {}
                    OpenLog::Trace => tracing::trace!("SSE connection opened"),
                    OpenLog::Debug => tracing::debug!("SSE connection opened"),
                },
                Ok(Event::Message(message)) => match triage(message.data) {
                    FrameDisposition::Skip => {}
                    FrameDisposition::Frame(data) => yield Ok(WireFrame::Text(data)),
                    FrameDisposition::Fail(error) => {
                        yield Err(error);
                        break;
                    }
                },
                Err(crate::http_client::Error::StreamEnded)
                    if !options.stream_ended_is_error =>
                {
                    break;
                }
                Err(error) => {
                    if options.log_transport_errors {
                        tracing::error!(?error, "SSE error");
                    }
                    yield Err(CompletionError::from_stream_transport(error));
                    break;
                }
            }
        }
        // Ensure event source is closed when stream ends
        event_source.close();
    }
}

/// Stamp the transport request id captured off the SSE connection onto the
/// stream's terminal record. `slot` is filled at each successful (re)connect
/// ([`crate::http_client::sse::GenericEventSource::capture_request_id`]), so
/// by the time a terminal flows through here it holds the id of the
/// connection that delivered it. With no slot (provider reports no request-id
/// header), the stream passes through untouched and the terminal's id stays
/// `None`.
pub(crate) fn stamp_terminal_request_id<R>(
    stream: crate::streaming::RawStreamingResult<R>,
    slot: Option<crate::http_client::sse::RequestIdSlot>,
    request_id_header: Option<&'static str>,
    stamp: impl Fn(&mut R, String) + WasmCompatSend + 'static,
) -> crate::streaming::RawStreamingResult<R>
where
    R: 'static,
{
    let Some(slot) = slot else {
        return stream;
    };
    Box::pin(stream.map(move |item| {
        let request_id = slot.lock().ok().and_then(|guard| guard.clone());
        match item {
            Ok(crate::streaming::RawStreamingChoice::FinalResponse(mut response)) => {
                if let Some(id) = request_id {
                    stamp(&mut response, id);
                }
                Ok(crate::streaming::RawStreamingChoice::FinalResponse(
                    response,
                ))
            }
            // A mid-stream in-band provider error envelope (yielded as an
            // error item) also came over this connection: attach the same
            // connection's transport id so a failed stream reports the id
            // support asks for (rig#2314). Only the ProviderResponse variant
            // has a slot for it; transport-level failures stay untouched.
            // A failed SSE handshake (connect-time non-success) surfaces as
            // a details-preserving transport error; this helper is installed
            // exactly by providers with a request-id contract, so classify it
            // like the unary driver would — ProviderResponse with the failed
            // response's own id (rig#2314 follow-up: the streaming 4xx now
            // matches its blocking twin instead of losing body and id).
            Err(crate::completion::CompletionError::HttpError(
                crate::http_client::Error::InvalidStatusCodeWithDetails {
                    status,
                    body,
                    headers,
                },
            )) if request_id_header.is_some() => {
                let provider_request_id = request_id_header
                    .and_then(|header| headers.get(header))
                    .and_then(|value| value.to_str().ok())
                    .filter(|value| !value.is_empty())
                    .map(str::to_string);
                Err(
                    crate::completion::CompletionError::from_http_response_with_request_id(
                        status,
                        body,
                        provider_request_id,
                    )
                    // The handshake's headers ride along too, so a streamed
                    // 429 exposes `Retry-After` exactly like its blocking
                    // twin (rig#2210).
                    .with_response_headers(Some(headers)),
                )
            }
            Err(crate::completion::CompletionError::ProviderResponse(response)) => {
                // Never clear an id an upstream constructor already attached;
                // the slot only fills the gap.
                let stamped = if response.provider_request_id.is_none() {
                    response.with_provider_request_id(request_id)
                } else {
                    response
                };
                Err(crate::completion::CompletionError::ProviderResponse(
                    stamped,
                ))
            }
            other => other,
        }
    }))
}

/// Open an SSE-backed wire stream: build the event source, run the transport
/// preamble ([`sse_frames`]) with the wire's options and triage, and drive the
/// frames through the shared adapter driver
/// ([`run_wire_stream`](super::adapter::run_wire_stream)) under `span`.
pub(crate) fn open_wire_stream<HttpClient, RequestBody, A, F>(
    event_source: GenericEventSource<HttpClient, RequestBody>,
    options: SseTransportOptions,
    triage: F,
    adapter: A,
    span: tracing::Span,
) -> crate::streaming::RawStreamingResult<A::Response>
where
    HttpClient: HttpClientExt + Clone + 'static,
    RequestBody: Into<bytes::Bytes> + Clone + WasmCompatSend + 'static,
    A: WireAdapter<Frame = WireFrame> + WasmCompatSend + 'static,
    A::Event: WasmCompatSend,
    A::Response: WasmCompatSend + 'static,
    F: FnMut(String) -> FrameDisposition + WasmCompatSend + 'static,
{
    // Transport layer: SSE events → `WireFrame`s. Byte splitting, framing,
    // and any in-band provider-error pre-filter carried by `triage` —
    // classification and policy live downstream.
    let transport = sse_frames(event_source, options, triage);

    Box::pin(run_wire_stream(transport, adapter).instrument(span))
}

#[cfg(test)]
mod tests;

#[cfg(all(test, not(all(target_arch = "wasm32", target_os = "unknown"))))]
mod request_id_stamp_tests {
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
        let stream: crate::streaming::RawStreamingResult<String> = Box::pin(futures::stream::iter(
            vec![Err(CompletionError::HttpError(
                crate::http_client::Error::InvalidStatusCodeWithDetails {
                    status: http::StatusCode::NOT_FOUND,
                    body: r#"{"error":"no model"}"#.to_string(),
                    headers: Box::new(headers),
                },
            ))],
        ));

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
        let stream: crate::streaming::RawStreamingResult<String> = Box::pin(futures::stream::iter(
            vec![Err(CompletionError::HttpError(
                crate::http_client::Error::InvalidStatusCodeWithDetails {
                    status: http::StatusCode::TOO_MANY_REQUESTS,
                    body: r#"{"error":"slow down"}"#.to_string(),
                    headers: Box::new(headers),
                },
            ))],
        ));

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
}
