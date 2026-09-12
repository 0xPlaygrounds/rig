//! Gemini stream faults through the runner: the recorded 404 on stream
//! setup, and scripted faults — a blocked prompt, a stream cut before its
//! terminal, an in-band error after content — served to the real
//! `streamGenerateContent` adapter. Native counterparts:
//! `ecs_stream_faults.rs`. Every cell's hypothesis is that the fault
//! surfaces as its own kind, with nothing recorded, committed or executed
//! after it.

use bytes::Bytes;
use rig::error::ErrorKind;
use rig::prelude::*;
use rig::providers::gemini::{self, completion::GEMINI_2_5_FLASH};
use rig::test_utils::SequencedStreamingHttpClient;

use super::super::support::with_gemini_cassette;
use crate::stream_faults::{
    assert_setup_failure, drain, recorded_stream_errors, scripted, sole_failed_completion,
};

/// The recorded stream-setup failure, `error_envelope/nonexistent_model_streaming_error_preserves_status_and_body`,
/// carries this model, prompt and budget.
pub(super) const MISSING_MODEL: &str = "gemini-nonexistent-rig-test";
pub(super) const SETUP_PROMPT: &str = "Say hi.";
pub(super) const SETUP_MAX_TOKENS: u64 = 16;

/// A real `streamGenerateContent` content frame without a finish reason,
/// as `gemini-3.8-flash` streamed it on 2026-09-08 (captured in the
/// downstream rigcoder Gemini matrix; ids scrubbed by this engine). The
/// recordings in this tree stream whole answers in one terminal frame, so a
/// prefix-then-fault stream is assembled from this frame. Synthetic order,
/// real frames.
pub(super) const CONTENT_FRAME: &str = r#"{"candidates":[{"content":{"parts":[{"text":"pong"}],"role":"model"},"index":0}],"modelVersion":"gemini-3.8-flash","responseId":"id_REDACTED_1","usageMetadata":{"candidatesTokenCount":1,"promptTokenCount":795,"promptTokensDetails":[{"modality":"TEXT","tokenCount":795}],"serviceTier":"standard","thoughtsTokenCount":60,"totalTokenCount":856}}"#;
/// The text [`CONTENT_FRAME`] carries.
pub(super) const CONTENT_TEXT: &str = "pong";
/// The whole body Gemini answered a refused prompt with, same capture: one
/// feedback chunk, no candidates, then the stream closes (#2475).
pub(super) const BLOCKED_FRAME: &str = r#"{"promptFeedback":{"blockReason":"SAFETY"},"usageMetadata":{"promptTokenCount":795,"totalTokenCount":795}}"#;
/// An error envelope in band under HTTP 200: the envelope Gemini returned
/// for an overloaded model in the same capture, in the frame position the
/// adapter's unit tests pin (`in_band_http_errors_match_unary_classification`).
pub(super) const IN_BAND_ERROR_FRAME: &str = r#"{"error":{"code":503,"message":"The model is overloaded. Please try again later.","status":"UNAVAILABLE"}}"#;
/// Gemini's classification of an HTTP error envelope on the streamed path:
/// the transport's status, with the envelope preserved on the report (the
/// metadata-less funnel in `rig_core::provider_response`; the Responses
/// wire's request-id funnel classifies the same fault as `ProviderResponse`).
pub(super) fn http_error(status: u16) -> ErrorKind {
    ErrorKind::Http {
        status: Some(status),
    }
}
/// A key the scripted cells send: it must never reach a trace.
pub(super) const SCRIPTED_KEY: &str = "scripted-fault-key-7f3a9c";

/// Gemini's SSE framing of `frames`.
pub(super) fn gemini_sse(frames: &[&str]) -> Bytes {
    Bytes::from(
        frames
            .iter()
            .map(|frame| format!("data: {frame}\n\n"))
            .collect::<String>(),
    )
}

/// A client over a transport that answers one streaming request with
/// `chunks`, then EOF.
pub(super) fn scripted_client(chunks: Vec<Bytes>) -> gemini::Client<SequencedStreamingHttpClient> {
    gemini::Client::builder()
        .api_key(SCRIPTED_KEY)
        .http_client(scripted(chunks))
        .build()
        .expect("client should build")
}

/// The model refuses the request before any frame: the run fails with the
/// recorded 404 and its body, streams nothing, and records the one
/// completion as that failure.
#[tokio::test]
async fn setup_failure_fails_the_run_with_the_recorded_status() {
    // The fixture census wants the scenario as a literal at the wrapper call.
    with_gemini_cassette(
        "error_envelope/nonexistent_model_streaming_error_preserves_status_and_body",
        |client| async move {
            let agent = client
                .agent(MISSING_MODEL)
                .max_tokens(SETUP_MAX_TOKENS)
                .record_effects_with_events()
                .build();
            let mut stream = agent.prompt(SETUP_PROMPT).stream();
            let drained = drain(&mut stream).await;
            drop(stream);

            assert_eq!(drained.finals, 0, "no final response: {drained:?}");
            assert_eq!(drained.terminals, 0, "no terminal record: {drained:?}");
            assert!(drained.text.is_empty(), "no text: {drained:?}");
            assert_eq!(drained.errors.len(), 1, "one error item: {drained:?}");
            assert_setup_failure(&drained.errors[0], http_error(404), 404);

            let log = agent.take_effect_log().expect("recording");
            assert_setup_failure(sole_failed_completion(&log), http_error(404), 404);
            let errors = recorded_stream_errors(&log);
            assert_eq!(errors.len(), 1, "one recorded error item: {errors:?}");
            assert_eq!(errors[0].0, 0, "the error is the stream's first item");
        },
    )
    .await;
}

/// A refused prompt is the provider's verdict, not a transport truncation:
/// the run fails as a non-retryable provider error naming the block reason,
/// and the record holds that error rather than "ended before its terminal".
#[tokio::test]
async fn blocked_prompt_is_a_provider_refusal_not_a_truncation() {
    let client = scripted_client(vec![gemini_sse(&[BLOCKED_FRAME])]);
    let agent = client
        .agent(GEMINI_2_5_FLASH)
        .record_effects_with_events()
        .build();
    let mut stream = agent.prompt("pong?").stream();
    let drained = drain(&mut stream).await;
    drop(stream);

    assert_eq!(drained.finals, 0, "{drained:?}");
    assert!(drained.text.is_empty(), "{drained:?}");
    assert_eq!(drained.errors.len(), 1, "{drained:?}");
    let report = &drained.errors[0];
    assert_eq!(report.kind, ErrorKind::Provider, "{report:?}");
    assert!(!report.is_retryable(), "{report:?}");
    assert!(
        report.message.contains("block_reason=SAFETY"),
        "the block reason is named: {report:?}"
    );

    let log = agent.take_effect_log().expect("recording");
    let recorded = sole_failed_completion(&log);
    assert_eq!(recorded.kind, ErrorKind::Provider, "{recorded:?}");
    assert!(
        recorded.message.contains("block_reason=SAFETY"),
        "the record holds the refusal, not a truncation: {recorded:?}"
    );
}

/// The stream closes after content without a terminal record: the run
/// fails as a truncation, the text prefix was delivered, no final response
/// is assembled, and the record holds the truncation.
#[tokio::test]
async fn truncation_after_content_fails_the_run_and_keeps_the_prefix() {
    let client = scripted_client(vec![gemini_sse(&[CONTENT_FRAME])]);
    let agent = client
        .agent(GEMINI_2_5_FLASH)
        .record_effects_with_events()
        .build();
    let mut stream = agent.prompt("pong?").stream();
    let drained = drain(&mut stream).await;
    drop(stream);

    assert_eq!(drained.text, CONTENT_TEXT, "{drained:?}");
    assert_eq!(drained.finals, 0, "{drained:?}");
    assert_eq!(drained.terminals, 0, "{drained:?}");
    assert!(!drained.errors.is_empty(), "{drained:?}");
    for report in &drained.errors {
        assert_eq!(report.kind, ErrorKind::Response, "{report:?}");
        assert!(
            report.message.contains("terminal record"),
            "a truncation, by name: {report:?}"
        );
    }

    let log = agent.take_effect_log().expect("recording");
    let recorded = sole_failed_completion(&log);
    assert_eq!(recorded.kind, ErrorKind::Response, "{recorded:?}");
    assert_eq!(recorded.message, rig::serve::stream_truncated().message);
}

/// An error envelope after content: the run fails with the envelope's
/// classification (a 503, retryable, body preserved), the prefix was
/// delivered, and the error item sits after the content items.
#[tokio::test]
async fn in_band_error_after_content_fails_with_the_envelope() {
    let client = scripted_client(vec![gemini_sse(&[CONTENT_FRAME, IN_BAND_ERROR_FRAME])]);
    let agent = client
        .agent(GEMINI_2_5_FLASH)
        .record_effects_with_events()
        .build();
    let mut stream = agent.prompt("pong?").stream();
    let drained = drain(&mut stream).await;
    drop(stream);

    assert_eq!(drained.text, CONTENT_TEXT, "{drained:?}");
    assert_eq!(drained.finals, 0, "{drained:?}");
    assert_eq!(drained.terminals, 0, "{drained:?}");
    assert_eq!(drained.errors.len(), 1, "{drained:?}");
    let report = &drained.errors[0];
    assert_eq!(report.kind, http_error(503), "{report:?}");
    assert_eq!(report.http_status, Some(503), "{report:?}");
    assert!(report.is_retryable(), "{report:?}");
    assert_eq!(
        report
            .provider_response_body()
            .map(|body| serde_json::from_str::<serde_json::Value>(body).expect("JSON")),
        Some(serde_json::from_str(IN_BAND_ERROR_FRAME).expect("JSON")),
        "the envelope is preserved: {report:?}"
    );

    let log = agent.take_effect_log().expect("recording");
    let recorded = sole_failed_completion(&log);
    assert_eq!(recorded.kind, http_error(503), "{recorded:?}");
    assert_eq!(recorded.http_status, Some(503), "{recorded:?}");
    let errors = recorded_stream_errors(&log);
    assert_eq!(errors.len(), 1, "{errors:?}");
    let delivered = log.records[0]
        .events
        .as_ref()
        .map_or(0, |events| events.len());
    assert!(delivered > 0, "content items precede the error");
    assert_eq!(
        errors[0].0, delivered,
        "the error item sits after the delivered content"
    );
}
