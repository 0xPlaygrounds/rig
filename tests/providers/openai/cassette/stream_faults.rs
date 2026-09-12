//! OpenAI Responses stream faults through the runner: the recorded 400 on
//! stream setup, a consumer that drops the recorded stream mid-answer, and
//! scripted faults cut from the committed recordings — a stream that ends
//! before its terminal, one that ends after a complete tool call, one that
//! carries an error event after content — served to the real Responses
//! adapter. Native counterparts: `ecs_stream_faults.rs`. Every cell's
//! hypothesis is that the fault surfaces as its own kind, with nothing
//! recorded, committed or executed after it.

use bytes::Bytes;
use futures::StreamExt;
use rig::agent::MultiTurnStreamItem;
use rig::effect::EffectFamily;
use rig::error::ErrorKind;
use rig::prelude::*;
use rig::providers::openai::{self, GPT_4O};
use rig::streaming::{Delta, StreamEvent};
use rig::test_utils::SequencedStreamingHttpClient;

use super::super::support::with_openai_cassette;
use crate::{
    goldens::families,
    stream_faults::{
        CountedSubtract, Invocations, assert_setup_failure, drain, frame_data, frames_before,
        recorded_sse_frames, recorded_stream_errors, scripted, sole_failed_completion, sse_bytes,
    },
    support::{
        Adder, STREAMING_PREAMBLE, STREAMING_PROMPT, STREAMING_TOOLS_PREAMBLE,
        STREAMING_TOOLS_PROMPT,
    },
};

/// The recorded stream-setup failure, `error_envelope/nonexistent_model_streaming_error_preserves_status_and_body`,
/// carries this model, prompt and budget.
pub(super) const MISSING_MODEL: &str = "gpt-4o-mini-nonexistent-rig-test";
pub(super) const SETUP_PROMPT: &str = "Say hi.";
pub(super) const SETUP_MAX_TOKENS: u64 = 16;
/// The Responses adapter's classification of a rejected request: the
/// provider's response, preserved on the report.
pub(super) const SETUP_KIND: ErrorKind = ErrorKind::ProviderResponse;

/// The recorded text stream the cut cells derive from, and the recorded
/// tool-call turn (its first interaction: the call, then the terminal).
pub(super) const TEXT_STREAM: &str = "streaming/streaming_smoke";
pub(super) const TOOL_STREAM: &str = "streaming_tools/streaming_tools_smoke";
/// An error event, the shape the adapter's unit tests pin
/// (`streaming_error_event_preserves_full_payload`).
pub(super) const ERROR_EVENT: &str = r#"event: error
data: {"type":"error","error":{"message":"boom","code":"server_error","type":"server_error"}}"#;
/// A key the scripted cells send: it must never reach a trace.
pub(super) const SCRIPTED_KEY: &str = "sk-scripted-fault-key-7f3a9c";

/// The frames of the recorded text stream up to, not including, the text
/// item's completion: content deltas, then nothing.
pub(super) fn text_prefix_frames() -> Vec<String> {
    let frames = recorded_sse_frames("openai", TEXT_STREAM, 0);
    frames_before(&frames, |frame| {
        frame.starts_with("event: response.output_text.done")
    })
}

/// The text the deltas of `frames` carry.
pub(super) fn delta_text(frames: &[String]) -> String {
    frames
        .iter()
        .filter(|frame| frame.starts_with("event: response.output_text.delta"))
        .map(|frame| {
            frame_data(frame)["delta"]
                .as_str()
                .expect("a text delta")
                .to_owned()
        })
        .collect()
}

/// The frames of the recorded tool-call turn up to, not including, the
/// response's completion: the whole `subtract` call, then nothing.
pub(super) fn tool_call_prefix_frames() -> Vec<String> {
    let frames = recorded_sse_frames("openai", TOOL_STREAM, 0);
    let prefix = frames_before(&frames, |frame| {
        frame.starts_with("event: response.completed")
    });
    assert!(
        prefix
            .iter()
            .any(|frame| frame.starts_with("event: response.output_item.done")),
        "the cut keeps the completed call: {prefix:?}"
    );
    prefix
}

/// A client over a transport that answers one streaming request with
/// `chunks`, then EOF.
pub(super) fn scripted_client(chunks: Vec<Bytes>) -> openai::Client<SequencedStreamingHttpClient> {
    openai::Client::builder()
        .api_key(SCRIPTED_KEY)
        .http_client(scripted(chunks))
        .build()
        .expect("client should build")
}

/// The model refuses the request before any frame: the run fails with the
/// recorded 400 and its body, streams nothing, and records the one
/// completion as that failure.
#[tokio::test]
async fn setup_failure_fails_the_run_with_the_recorded_status() {
    // The fixture census wants the scenario as a literal at the wrapper call.
    with_openai_cassette(
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
            assert_setup_failure(&drained.errors[0], SETUP_KIND, 400);

            let log = agent.take_effect_log().expect("recording");
            assert_setup_failure(sole_failed_completion(&log), SETUP_KIND, 400);
            let errors = recorded_stream_errors(&log);
            assert_eq!(errors.len(), 1, "one recorded error item: {errors:?}");
            assert_eq!(errors[0].0, 0, "the error is the stream's first item");
        },
    )
    .await;
}

/// The stream closes after content without a terminal record: the run
/// fails as a truncation, the text prefix was delivered, no final response
/// is assembled, and the record holds the truncation.
#[tokio::test]
async fn truncation_after_content_fails_the_run_and_keeps_the_prefix() {
    let frames = text_prefix_frames();
    let prefix = delta_text(&frames);
    assert!(
        !prefix.is_empty(),
        "the recording streams text before its end"
    );
    let client = scripted_client(vec![sse_bytes(&frames)]);
    let agent = client
        .agent(GPT_4O)
        .preamble(STREAMING_PREAMBLE)
        .record_effects_with_events()
        .build();
    let mut stream = agent.prompt(STREAMING_PROMPT).stream();
    let drained = drain(&mut stream).await;
    drop(stream);

    assert_eq!(drained.text, prefix, "{drained:?}");
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

/// The stream closes after a complete tool call without a terminal record:
/// the call is not committed, the tool never runs, and the run fails as a
/// truncation with the one completion recorded.
#[tokio::test]
async fn truncation_after_a_complete_tool_call_never_runs_the_tool() {
    let frames = tool_call_prefix_frames();
    let invocations = Invocations::default();
    let client = scripted_client(vec![sse_bytes(&frames)]);
    let agent = client
        .agent(GPT_4O)
        .preamble(STREAMING_TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(CountedSubtract(invocations.clone()))
        .record_effects_with_events()
        .build();
    let mut stream = agent.prompt(STREAMING_TOOLS_PROMPT).max_turns(2).stream();
    let drained = drain(&mut stream).await;
    drop(stream);

    assert_eq!(drained.tool_calls, 0, "no call is committed: {drained:?}");
    assert_eq!(drained.finals, 0, "{drained:?}");
    assert_eq!(drained.terminals, 0, "{drained:?}");
    assert!(!drained.errors.is_empty(), "{drained:?}");
    for report in &drained.errors {
        assert_eq!(report.kind, ErrorKind::Response, "{report:?}");
    }
    assert_eq!(invocations.count(), 0, "the tool never ran");

    let log = agent.take_effect_log().expect("recording");
    assert_eq!(
        families(&log),
        [EffectFamily::Completion],
        "no tool effect follows the cut turn"
    );
    let recorded = sole_failed_completion(&log);
    assert_eq!(recorded.message, rig::serve::stream_truncated().message);
}

/// An error event after content: the run fails with the provider's error
/// (body preserved), the prefix was delivered, and the error item sits
/// after the content items.
#[tokio::test]
async fn error_event_after_content_fails_with_the_provider_error() {
    let mut frames = text_prefix_frames();
    let prefix = delta_text(&frames);
    frames.push(ERROR_EVENT.to_owned());
    let client = scripted_client(vec![sse_bytes(&frames)]);
    let agent = client
        .agent(GPT_4O)
        .preamble(STREAMING_PREAMBLE)
        .record_effects_with_events()
        .build();
    let mut stream = agent.prompt(STREAMING_PROMPT).stream();
    let drained = drain(&mut stream).await;
    drop(stream);

    assert_eq!(drained.text, prefix, "{drained:?}");
    assert_eq!(drained.finals, 0, "{drained:?}");
    assert_eq!(drained.terminals, 0, "{drained:?}");
    assert_eq!(drained.errors.len(), 1, "{drained:?}");
    let report = &drained.errors[0];
    assert_eq!(report.kind, ErrorKind::ProviderResponse, "{report:?}");
    assert!(
        report
            .provider_response_body()
            .is_some_and(|body| body.contains("boom")),
        "the event's payload is preserved: {report:?}"
    );

    let log = agent.take_effect_log().expect("recording");
    let recorded = sole_failed_completion(&log);
    assert_eq!(recorded.kind, ErrorKind::ProviderResponse, "{recorded:?}");
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

/// The consumer drops the recorded stream at its first text delta: the
/// completion is recorded as cancelled and no answer is assembled. The
/// replay marked the interaction consumed when it matched the request, so
/// `finish` still passes; the drop lands on the response body, which the
/// replay does not track.
#[tokio::test]
async fn dropping_the_stream_at_the_first_delta_records_a_cancel() {
    with_openai_cassette("streaming/streaming_smoke", |client| async move {
        let agent = client
            .agent(GPT_4O)
            .preamble(STREAMING_PREAMBLE)
            .record_effects_with_events()
            .build();
        {
            let mut stream = agent.prompt(STREAMING_PROMPT).stream();
            while let Some(item) = stream.next().await {
                match item {
                    Ok(MultiTurnStreamItem::StreamAssistantItem(StreamEvent::BlockDelta {
                        delta: Delta::Text { .. },
                        ..
                    })) => break,
                    Ok(MultiTurnStreamItem::FinalResponse(_)) => {
                        panic!("the answer arrived before the first delta")
                    }
                    Err(error) => panic!("no error before the drop: {error}"),
                    Ok(_) => {}
                }
            }
            // Dropped here, mid-answer.
        }
        for _ in 0..64 {
            tokio::task::yield_now().await;
        }
        let log = agent.take_effect_log().expect("recording");
        let recorded = sole_failed_completion(&log);
        assert_eq!(recorded.kind, ErrorKind::Cancelled, "{recorded:?}");
    })
    .await;
}
