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

use super::adapter::{WireAdapter, WireFrame, run_wire_stream_observed};
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
        let observation = event_source.observation();
        let mut event_source = Box::pin(event_source);
        while let Some(event_result) = event_source.next().await {
            match event_result {
                Ok(Event::Open) => match options.open_log {
                    OpenLog::Silent => {}
                    OpenLog::Trace => tracing::trace!("SSE connection opened"),
                    OpenLog::Debug => tracing::debug!("SSE connection opened"),
                },
                Ok(Event::Message(message)) => {
                    if let Some(observation) = &observation {
                        observation.payload(message.data.as_bytes());
                    }
                    match triage(message.data) {
                    FrameDisposition::Skip => {}
                    FrameDisposition::Frame(data) => yield Ok(WireFrame::Text(data)),
                    FrameDisposition::Fail(error) => {
                        yield Err(error);
                        break;
                    }
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
pub(crate) fn stamp_terminal_request_id(
    stream: crate::streaming::StreamingResult,
    slot: Option<crate::http_client::sse::RequestIdSlot>,
    request_id_header: Option<&'static str>,
) -> crate::streaming::StreamingResult {
    let Some(slot) = slot else {
        return stream;
    };
    Box::pin(stream.map(move |item| {
        let request_id = slot.lock().ok().and_then(|guard| guard.clone());
        match item {
            Ok(crate::streaming::StreamEvent::Final(response)) => {
                Ok(crate::streaming::StreamEvent::Final(match request_id {
                    Some(id) if response.provider_request_id.is_none() => {
                        response.with_provider_request_id(id)
                    }
                    _ => response,
                }))
            }
            // A failed SSE handshake (connect-time non-success) is the
            // provider's reply: `from_transport_error` already classified it
            // as ProviderResponse with the handshake's headers (rig#2210);
            // stamp the id read off those headers, like the unary driver
            // (rig#2314). An in-band envelope yielded as an error item came
            // over the same connection and takes the slot's id below.
            Err(crate::completion::CompletionError::ProviderResponse(response)) => {
                // Never clear an id an upstream constructor already attached;
                // the slot only fills the gap, and a handshake rejection's own
                // headers fill it before the slot does.
                let from_headers = response
                    .headers
                    .as_deref()
                    .and_then(|headers| super::request_id_from_headers(headers, request_id_header));
                let stamped = if response.provider_request_id.is_none() {
                    response.with_provider_request_id(from_headers.or(request_id))
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
) -> crate::streaming::StreamingResult
where
    HttpClient: HttpClientExt + Clone + 'static,
    RequestBody: Into<bytes::Bytes> + Clone + WasmCompatSend + 'static,
    A: WireAdapter<Frame = WireFrame> + WasmCompatSend + 'static,
    A::Event: WasmCompatSend,
    F: FnMut(String) -> FrameDisposition + WasmCompatSend + 'static,
{
    // Transport layer: SSE events → `WireFrame`s. Byte splitting, framing,
    // and any in-band provider-error pre-filter carried by `triage` —
    // classification and policy live downstream.
    let observation = event_source.observation();
    let transport = sse_frames(event_source, options, triage);

    Box::pin(run_wire_stream_observed(transport, adapter, observation).instrument(span))
}

#[cfg(test)]
mod tests;

#[cfg(all(test, not(all(target_arch = "wasm32", target_os = "unknown"))))]
mod request_id_stamp_tests;
