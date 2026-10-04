//! OpenAI Responses stream faults through the runner: the recorded 400 on
//! stream setup, a consumer that drops the recorded stream mid-answer, and
//! scripted faults cut from the committed recordings — a stream that ends
//! before its terminal, one that ends after a complete tool call, one that
//! carries an error event after content — served to the real Responses
//! adapter. Native counterparts: `ecs_stream_faults.rs`. Every cell's
//! hypothesis is that the fault surfaces as its own kind, with nothing
//! recorded, committed or executed after it.

use bytes::Bytes;
use rig::effect::EffectFamily;
use rig::error::ErrorKind;
use rig::providers::openai::GPT_4O;
use rig::providers::openai::OpenAIConfig;
use rig::test_utils::SequencedStreamingHttpClient;
use rig_cassette::agent::AgentReplayExt;
use rig_test_support::cassette_models::OpenAiModels;

use crate::{
    goldens::families,
    stream_faults::{
        CountedSubtract, Invocations, SseShape, drain, recorded_sse_frames, recorded_stream_errors,
        scripted, sole_failed_completion, sse_bytes,
    },
    support::{
        Adder, STREAMING_PREAMBLE, STREAMING_PROMPT, STREAMING_TOOLS_PREAMBLE,
        STREAMING_TOOLS_PROMPT,
    },
};

/// The recorded text stream the cut cells derive from, and the recorded
/// tool-call turn (its first interaction: the call, then the terminal).
pub(super) const TEXT_STREAM: &str = "streaming/streaming_smoke";
pub(super) const TOOL_STREAM: &str = "streaming_tools/streaming_tools_smoke";
/// An error event, the shape the adapter's unit tests pin
/// (`streaming_error_event_preserves_full_payload`).
pub(super) const ERROR_EVENT: &str = crate::stream_faults::RESPONSES_ERROR_EVENT;
/// A key the scripted cells send: it must never reach a trace.
pub(super) const SCRIPTED_KEY: &str = "sk-scripted-fault-key-7f3a9c";

/// The frames of the recorded text stream up to, not including, the text
/// item's completion: content deltas, then nothing.
pub(super) fn text_prefix_frames() -> Vec<String> {
    SseShape::Responses.text_prefix(&recorded_sse_frames("openai", TEXT_STREAM, 0))
}

/// The text the deltas of `frames` carry.
pub(super) fn delta_text(frames: &[String]) -> String {
    SseShape::Responses.delta_text(frames)
}

/// The frames of the recorded tool-call turn up to, not including, the
/// response's completion: the whole `subtract` call, then nothing.
pub(super) fn tool_call_prefix_frames() -> Vec<String> {
    SseShape::Responses.tool_prefix(&recorded_sse_frames("openai", TOOL_STREAM, 0))
}

/// A client over a transport that answers one streaming request with
/// `chunks`, then EOF.
pub(super) fn scripted_client(chunks: Vec<Bytes>) -> (OpenAIConfig, SequencedStreamingHttpClient) {
    (OpenAIConfig::new(SCRIPTED_KEY), scripted(chunks))
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
    let (client, http) = scripted_client(vec![sse_bytes(&frames)]);
    let recorder = rig_cassette::effect_log::EffectLogRecorder::keeping_stream_events();
    let agent = rig::AgentBuilder::new(OpenAiModels::new(client, http.clone()).completion(GPT_4O))
        .preamble(STREAMING_PREAMBLE)
        .record_to(recorder.clone())
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
            report
                .message
                .contains("ended before the provider ended it"),
            "a truncation, by name: {report:?}"
        );
    }

    let log = agent.stamp(recorder.take());
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
    let (client, http) = scripted_client(vec![sse_bytes(&frames)]);
    let recorder = rig_cassette::effect_log::EffectLogRecorder::keeping_stream_events();
    let agent = rig::AgentBuilder::new(OpenAiModels::new(client, http.clone()).completion(GPT_4O))
        .preamble(STREAMING_TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(CountedSubtract(invocations.clone()))
        .record_to(recorder.clone())
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

    let log = agent.stamp(recorder.take());
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
    let (client, http) = scripted_client(vec![sse_bytes(&frames)]);
    let recorder = rig_cassette::effect_log::EffectLogRecorder::keeping_stream_events();
    let agent = rig::AgentBuilder::new(OpenAiModels::new(client, http.clone()).completion(GPT_4O))
        .preamble(STREAMING_PREAMBLE)
        .record_to(recorder.clone())
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

    let log = agent.stamp(recorder.take());
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
