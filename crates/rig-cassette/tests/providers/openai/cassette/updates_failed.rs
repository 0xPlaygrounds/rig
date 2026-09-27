//! `updates()` over replayed failures of the Responses adapter: streams cut
//! from the committed recordings (`streaming/streaming_smoke`,
//! `streaming_tools/streaming_tools_smoke`) and an error event after their
//! content, served to the real adapter. Each cell's hypothesis is that the
//! updates end with one `Failed` carrying the failure's kind and message,
//! and that its `partial` holds exactly the parts the recorded frames
//! finish before the fault. The expected text and call are read from the
//! recorded frames.

use rig::completion::CompletionRequest;
use rig::error::ErrorKind;
use rig::message::AssistantContent;
use rig::providers::openai::GPT_4O;
use rig::streaming::PartKind;
use rig_test_support::cassette_models::OpenAiModels;
use rig_test_support::updates::{assert_failed_update_contract, collect_updates};

use super::stream_faults::{
    ERROR_EVENT, delta_text, scripted_client, text_prefix_frames, tool_call_prefix_frames,
};
use crate::stream_faults::{frame_data, sse_bytes};
use crate::support::STREAMING_PROMPT;

/// The truncation error, as a caller receives it.
const TRUNCATED: &str = "ResponseError: provider stream ended without a terminal record; \
                         treating the turn as truncated";

/// The recorded frames of a stream that ends before its terminal: the text
/// the deltas carried ended when the stream did, and the updates fail as
/// truncated with that text in `partial`.
#[tokio::test]
async fn a_text_stream_cut_before_its_terminal_fails_with_the_text() {
    let frames = text_prefix_frames();
    let prefix = delta_text(&frames);
    let (client, http) = scripted_client(vec![sse_bytes(&frames)]);
    let model = OpenAiModels::new(client, http).completion(GPT_4O);
    let mut stream = model
        .stream(CompletionRequest::new(STREAMING_PROMPT))
        .expect("the stream opens");
    let updates = collect_updates(&mut stream).await;

    let (error, partial, delivered) = assert_failed_update_contract(&updates);
    assert_eq!(error.kind, ErrorKind::Response, "{error:?}");
    assert!(!error.retryable, "{error:?}");
    assert_eq!(error.message, TRUNCATED);
    assert_eq!(partial.choice, [AssistantContent::text(prefix.clone())]);
    assert_eq!(delivered[0].kind, PartKind::Text);
    assert_eq!(delivered[0].text, prefix, "the deltas carried the text");
    assert_eq!(partial, stream.partial());
}

/// The recorded tool-call turn cut before its completion: the call the
/// frames finished is in `partial`, and the updates fail as truncated.
#[tokio::test]
async fn a_tool_turn_cut_before_its_terminal_fails_with_the_finished_call() {
    let frames = tool_call_prefix_frames();
    let done = frames
        .iter()
        .find(|frame| frame.starts_with("event: response.output_item.done"))
        .map(|frame| frame_data(frame)["item"].clone())
        .expect("the recording finishes its call");
    let arguments: serde_json::Value = serde_json::from_str(
        done["arguments"]
            .as_str()
            .expect("the recorded call carries its arguments"),
    )
    .expect("the recorded arguments are JSON");
    let (client, http) = scripted_client(vec![sse_bytes(&frames)]);
    let model = OpenAiModels::new(client, http).completion(GPT_4O);
    let mut stream = model
        .stream(CompletionRequest::new(STREAMING_PROMPT))
        .expect("the stream opens");
    let updates = collect_updates(&mut stream).await;

    let (error, partial, delivered) = assert_failed_update_contract(&updates);
    assert_eq!(error.kind, ErrorKind::Response, "{error:?}");
    assert_eq!(error.message, TRUNCATED);
    let calls: Vec<_> = partial.tool_calls().collect();
    assert_eq!(calls.len(), 1, "{partial:?}");
    assert_eq!(
        Some(calls[0].function.name.as_str()),
        done["name"].as_str(),
        "the recorded call's name"
    );
    assert_eq!(
        calls[0].function.arguments, arguments,
        "the recorded arguments"
    );
    let name = done["name"].as_str().unwrap_or_default().to_owned();
    assert!(
        delivered
            .iter()
            .any(|part| part.kind == PartKind::ToolCall { name: name.clone() }),
        "{delivered:?}"
    );
}

/// An error event after the recorded text: the updates fail at the event
/// with the provider's error, and the driver closed the text before it, so
/// the text ended and is in `partial`.
#[tokio::test]
async fn an_error_event_after_content_fails_with_the_event_and_the_text() {
    let mut frames = text_prefix_frames();
    let prefix = delta_text(&frames);
    frames.push(ERROR_EVENT.to_owned());
    let (client, http) = scripted_client(vec![sse_bytes(&frames)]);
    let model = OpenAiModels::new(client, http).completion(GPT_4O);
    let mut stream = model
        .stream(CompletionRequest::new(STREAMING_PROMPT))
        .expect("the stream opens");
    let updates = collect_updates(&mut stream).await;

    let (error, partial, _) = assert_failed_update_contract(&updates);
    assert_eq!(error.kind, ErrorKind::ProviderResponse, "{error:?}");
    assert_eq!(error.code.as_deref(), Some("server_error"), "{error:?}");
    assert!(
        error
            .provider_response_body()
            .is_some_and(|body| body.contains("boom")),
        "the event's payload is preserved: {error:?}"
    );
    assert_eq!(partial.choice, [AssistantContent::text(prefix)]);
}
