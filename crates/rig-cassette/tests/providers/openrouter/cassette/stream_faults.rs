//! OpenRouter streams cut short before their terminal record, served to the
//! real chat and Responses adapters from committed recordings. Each cell's
//! hypothesis is that reasoning a truncated stream delivered keeps the
//! family of the model the request named, so it replays to that family and
//! no other, exactly as it would had the stream finished.

use futures::StreamExt;
use rig::completion::{CompletionModel, CompletionRequest};
use rig::message::{AssistantContent, Message, Reasoning};
use rig::prelude::*;
use rig::providers::openai::wire::{OPENROUTER, OpenAI};
use rig::streaming::stamp_reasoning;
use rig::test_utils::SequencedStreamingHttpClient;

use crate::stream_faults::{SseShape, recorded_sse_frames, scripted, sse_bytes};

/// A key the scripted cells send; the transport never forwards it.
const SCRIPTED_KEY: &str = "sk-scripted-fault-key-openrouter";

/// Claude thinking and a tool call over the chat route.
const CHAT_CLAUDE: &str = "reasoning_tool_order_matrix/streaming_single";
/// OpenAI reasoning over the chat route.
const CHAT_OPENAI: &str =
    "reasoning_usage_matrix/streaming_reasoning_tokens_reach_the_terminal_record";
/// Claude thinking over the Responses route.
const RESPONSES_CLAUDE: &str = "upstream_switch_matrix/responses_same_family_streamed";

#[derive(Clone, Copy, Debug)]
enum Route {
    Chat,
    Responses,
}

/// The recorded stream of `scenario`'s first exchange, up to and including
/// its last frame before the terminal that carries reasoning: the reasoning
/// arrived, the terminal record never does.
fn cut_after_reasoning(scenario: &str, route: Route) -> Vec<String> {
    let shape = match route {
        Route::Chat => SseShape::Chat,
        Route::Responses => SseShape::Responses,
    };
    let frames = recorded_sse_frames("openrouter", scenario, 0);
    let end = frames
        .iter()
        .position(|frame| shape.is_terminal(frame))
        .unwrap_or_else(|| panic!("{scenario}: the recording carries its terminal"));
    let last = frames[..end]
        .iter()
        .rposition(|frame| frame.contains("reasoning"))
        .unwrap_or_else(|| panic!("{scenario}: reasoning precedes the terminal"));
    frames[..=last].to_vec()
}

/// The model the recorded request named.
fn recorded_model(scenario: &str) -> String {
    crate::cassettes::recorded_json_request("openrouter", scenario)["model"]
        .as_str()
        .unwrap_or_else(|| panic!("{scenario}: the request names a model"))
        .to_owned()
}

fn scripted_client(frames: &[String]) -> Bound<OpenAI, SequencedStreamingHttpClient> {
    OpenAI::with_key(&OPENROUTER, SCRIPTED_KEY).bind(scripted(vec![sse_bytes(frames)]))
}

/// Stream the cut recording through `route` for the recorded model and
/// return the issuer the partial turn records and its stamped reasoning.
async fn partial_turn(scenario: &str, route: Route) -> (String, Vec<Reasoning>) {
    let client = scripted_client(&cut_after_reasoning(scenario, route));
    let model = recorded_model(scenario);
    let request = CompletionRequest {
        model: None,
        chat_history: vec![Message::user("Think it through.")],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };
    let mut stream = match route {
        Route::Chat => client.completion(model).stream(request).await,
        Route::Responses => client.responses(model).stream(request).await,
    }
    .expect("the stream opens");
    while stream.next().await.is_some() {}
    assert!(
        stream.response.is_none(),
        "{scenario}: the cut stream has no terminal record"
    );
    let issuer = stream
        .reasoning_issuer()
        .unwrap_or_else(|| panic!("{scenario}: a stream the adapter opened names an issuer"))
        .to_owned();
    let reasoning: Vec<Reasoning> = stamp_reasoning(stream.snapshot(), &issuer)
        .into_iter()
        .filter_map(|part| match part {
            AssistantContent::Reasoning(reasoning) => Some(reasoning),
            _ => None,
        })
        .collect();
    assert!(
        !reasoning.is_empty(),
        "{scenario}: the partial turn kept the delivered reasoning"
    );
    (issuer, reasoning)
}

/// Every reasoning part replays to `family` and to no other: not to the bare
/// gateway issuer, which an OpenRouter request for any family accepts, and
/// not to another family's issuer.
fn assert_scoped_to(reasoning: &[Reasoning], family: &str, others: &[&str]) {
    for part in reasoning {
        assert_eq!(part.provider.as_deref(), Some(family), "{part:?}");
        assert!(part.replayable_to(family), "{part:?}");
        assert!(!part.replayable_to("openrouter"), "{part:?}");
        for other in others {
            assert!(!part.replayable_to(other), "replays to {other}: {part:?}");
        }
    }
}

#[tokio::test]
async fn a_truncated_chat_stream_keeps_claude_thinking_to_anthropic() {
    let (issuer, reasoning) = partial_turn(CHAT_CLAUDE, Route::Chat).await;
    assert_eq!(issuer, "anthropic");
    assert_scoped_to(
        &reasoning,
        "anthropic",
        &["openrouter/openai", "openrouter/google"],
    );
}

#[tokio::test]
async fn a_truncated_chat_stream_keeps_openai_reasoning_to_openai() {
    let (issuer, reasoning) = partial_turn(CHAT_OPENAI, Route::Chat).await;
    assert_eq!(issuer, "openrouter/openai");
    assert_scoped_to(
        &reasoning,
        "openrouter/openai",
        &["anthropic", "openrouter/google"],
    );
}

#[tokio::test]
async fn a_truncated_responses_stream_keeps_claude_thinking_to_anthropic() {
    let (issuer, reasoning) = partial_turn(RESPONSES_CLAUDE, Route::Responses).await;
    assert_eq!(issuer, "anthropic");
    assert_scoped_to(
        &reasoning,
        "anthropic",
        &["openrouter/openai", "openrouter/google"],
    );
}
