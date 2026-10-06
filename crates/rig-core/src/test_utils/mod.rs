//! Test utilities for deterministic completion-model tests.

mod completion;
mod embeddings;
pub mod history;
pub mod history_conformance;
mod memory;
pub mod observations;
#[cfg(test)]
pub(crate) mod provider_extensions;
mod relay;
mod streaming;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub mod streaming_conformance;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
mod streaming_conformance_suite;
mod trace_capture;
mod tracing_isolation;

pub use completion::{
    MOCK_API, MOCK_MODEL, MockCompletionModel, MockError, MockRuntime, MockScript, MockTurn,
    refuse_options,
};
pub use embeddings::{MockEmbeddingModel, MockEmbeddings, MockMultiTextDocument, MockTextDocument};
pub use memory::{AppendFailingMemory, CountingMemory, FailingMemory};
pub use relay::MockRelay;
pub use rig_http::test_utils::{
    CapturedHttpRequest, HttpErrorStreamingClient, MockHttpResponse, MockStreamingClient,
    NonSuccessStreamingClient, RecordingHttpClient, SequencedHttpClient,
    SequencedStreamingHttpClient,
};
pub use streaming::{
    MOCK_PROVIDER, MockDecoder, MockFrame, MockStreamEvent, mock_final,
    mock_final_with_total_tokens,
};
pub use trace_capture::{CapturedEvent, CapturedSpan, TraceCapture};
pub use tracing_isolation::{
    scoped_tracing_subscriber_guard, scoped_tracing_subscriber_guard_blocking,
};

/// The JSON document an encoded request sends.
///
/// # Panics
///
/// When the body is multipart or is not JSON.
#[cfg(test)]
pub(crate) fn json_body(request: &http::Request<crate::wire::Body>) -> serde_json::Value {
    let crate::wire::Body::Bytes(bytes) = request.body() else {
        panic!("the request body is multipart, not JSON");
    };
    serde_json::from_slice(bytes).expect("the request body is JSON")
}

/// Decode one reply of `wire` to `request` from frames already in hand,
/// folded as `mode` folds it: the one decoder and fold a live call runs.
#[cfg(test)]
pub(crate) fn decode_reply<W: crate::wire::Wire>(
    wire: &W,
    request: &crate::wire::Request<W>,
    mode: crate::wire::Mode,
    frames: impl IntoIterator<Item = W::Frame>,
    raw: serde_json::Value,
) -> Result<crate::wire::Response<W>, crate::error::ProviderError> {
    let shared = std::sync::Mutex::new(crate::wire::Shared::new(fold_for(request, wire, mode)));
    let fed = crate::driver::feed(&mut wire.decoder(), &shared, frames);
    crate::driver::settle(
        shared,
        fed,
        crate::wire::Reply {
            provider: wire.describe().name.to_owned(),
            raw,
            provider_request_id: None,
        },
    )
    .outcome
}

/// The fold a call to `wire` in `mode` opens for `request`, for tests that
/// drive a decoder by hand.
#[cfg(test)]
pub(crate) fn fold_for<W: crate::wire::Wire>(
    request: &crate::wire::Request<W>,
    wire: &W,
    mode: crate::wire::Mode,
) -> <W::Op as crate::wire::Operation>::Fold {
    <W::Op as crate::wire::Operation>::fold(
        request,
        &mut crate::wire::Call::new(&wire.describe(), mode),
    )
}

/// The documents a prepared request sent, as `(id, text)`: `prepare` folds
/// a request's documents into its first user message as `<file id: ...>`
/// text documents.
pub fn sent_documents(request: &crate::completion::CompletionRequest) -> Vec<(String, String)> {
    request
        .chat_history
        .iter()
        .flat_map(|message| match message {
            crate::message::Message::User { content } => content.iter().collect::<Vec<_>>(),
            crate::message::Message::System { .. } | crate::message::Message::Assistant(_) => {
                Vec::new()
            }
        })
        .filter_map(|part| match part {
            crate::message::UserContent::Document(document) => match &document.data {
                crate::message::DocumentSourceKind::String(text) => {
                    let body = text.strip_prefix("<file id: ")?;
                    let (id, rest) = body.split_once(">\n")?;
                    let text = rest.strip_suffix("\n</file>\n")?;
                    Some((id.to_owned(), text.to_owned()))
                }
                _ => None,
            },
            _ => None,
        })
        .collect()
}
