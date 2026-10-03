//! The chat stream's hard cases, each a reply the shared pending-call buffer
//! must assemble: ids and names that arrive late or never, a reused index,
//! interleaved indices, whole calls in one chunk, `null` placeholders, a
//! length cut inside a call, late and repeated reasoning signatures, and a
//! provider id sent twice.

use bytes::Bytes;
use futures::StreamExt;
use serde_json::{Value, json};

use super::*;
use crate::completion::{CompletionResponse, FinishReason};
use crate::message::{AssistantContent, CallId, Reasoning, ToolCall};
use crate::providers::openai::wire::{Dialect, LLAMACPP, OPENAI, OPENROUTER, OpenAIConfig};
use crate::streaming::{Item, PartKind, StreamEvent};
use crate::test_utils::MockStreamingClient;

fn wire(dialect: &'static Dialect) -> Chat {
    OpenAIConfig::new("sk-test")
        .with_dialect(dialect)
        .chat("gpt-4.1-nano")
}

/// One `chat.completion.chunk` as an SSE event.
fn chunk(delta: Value, finish_reason: Option<&str>) -> String {
    let chunk = json!({
        "id": "chatcmpl-hard",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4.1-nano",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    });
    format!("data: {chunk}\n\n")
}

/// A `tool_calls` fragment; an absent field is the wire's `null`.
fn call(index: u32, id: Option<&str>, name: Option<&str>, arguments: Option<&str>) -> String {
    chunk(
        json!({"tool_calls": [{
            "index": index,
            "id": id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }]}),
        None,
    )
}

fn text(text: &str) -> String {
    chunk(json!({"content": text}), None)
}

fn finish(reason: &str) -> String {
    chunk(json!({}), Some(reason))
}

const DONE: &str = "data: [DONE]\n\n";

/// Stream `chunks` through `dialect`'s chat wire: the items, then what
/// `finish` returns.
async fn stream(
    dialect: &'static Dialect,
    chunks: &[String],
) -> (
    Vec<Result<Item<StreamEvent>, ProviderError>>,
    Result<CompletionResponse, ProviderError>,
) {
    let model = crate::driver::Model::new(
        wire(dialect),
        MockStreamingClient {
            sse_bytes: Bytes::from(chunks.concat()),
        },
    );
    let mut stream = model
        .stream(CompletionRequest::new("hard case"))
        .expect("the stream opens");
    let mut items = Vec::new();
    while let Some(item) = stream.next().await {
        items.push(item);
    }
    (items, stream.finish().await)
}

fn calls(response: &CompletionResponse) -> Vec<&ToolCall> {
    response
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect()
}

fn provider_id(call: &ToolCall) -> Option<&str> {
    call.id.provider().map(|provider| provider.as_str())
}

#[tokio::test]
async fn a_late_id_and_a_late_name_still_open_the_call() {
    let (_, response) = stream(
        &OPENAI,
        &[
            call(0, None, None, Some(r#"{"a":"#)),
            call(0, None, Some("add"), Some("1,")),
            call(0, Some("call_late"), None, Some(r#""b":2}"#)),
            finish("tool_calls"),
            DONE.to_owned(),
        ],
    )
    .await;
    let response = response.expect("the reply folds");
    let [call] = calls(&response)[..] else {
        panic!("one call: {:?}", response.choice);
    };
    assert_eq!(provider_id(call), Some("call_late"));
    assert_eq!(call.function.name, "add");
    assert_eq!(call.function.arguments_value(), json!({"a": 1, "b": 2}));
}

#[tokio::test]
async fn id_less_calls_get_distinct_rig_issued_ids() {
    let (_, response) = stream(
        &OPENAI,
        &[
            call(0, None, Some("add"), Some(r#"{"a":1}"#)),
            call(1, None, Some("add"), Some(r#"{"a":2}"#)),
            finish("tool_calls"),
            DONE.to_owned(),
        ],
    )
    .await;
    let response = response.expect("the reply folds");
    let [first, second] = calls(&response)[..] else {
        panic!("two calls: {:?}", response.choice);
    };
    assert!(first.id.is_local() && second.id.is_local());
    assert_ne!(first.id, second.id, "each id-less call has its own id");
    assert_eq!(first.function.arguments_value(), json!({"a": 1}));
    assert_eq!(second.function.arguments_value(), json!({"a": 2}));
}

#[tokio::test]
async fn a_reused_index_delivers_the_call_it_held() {
    let (_, response) = stream(
        &OPENAI,
        &[
            call(0, Some("call_1"), Some("add"), Some(r#"{"a":1}"#)),
            call(0, Some("call_2"), Some("subtract"), Some(r#"{"a":2}"#)),
            finish("tool_calls"),
            DONE.to_owned(),
        ],
    )
    .await;
    let response = response.expect("the reply folds");
    let ids: Vec<_> = calls(&response).into_iter().map(provider_id).collect();
    assert_eq!(ids, [Some("call_1"), Some("call_2")]);
}

#[tokio::test]
async fn interleaved_indices_assemble_their_own_arguments() {
    let (_, response) = stream(
        &OPENAI,
        &[
            call(0, Some("call_a"), Some("add"), Some(r#"{"a":"#)),
            call(1, Some("call_b"), Some("add"), Some(r#"{"a":"#)),
            call(0, None, None, Some("1}")),
            call(1, None, None, Some("2}")),
            finish("tool_calls"),
            DONE.to_owned(),
        ],
    )
    .await;
    let response = response.expect("the reply folds");
    let assembled: Vec<_> = calls(&response)
        .into_iter()
        .map(|call| (provider_id(call), call.function.arguments_value()))
        .collect();
    assert_eq!(
        assembled,
        [
            (Some("call_a"), json!({"a": 1})),
            (Some("call_b"), json!({"a": 2}))
        ]
    );
}

#[tokio::test]
async fn a_whole_call_in_one_chunk_folds_on_llamacpp() {
    let (_, response) = stream(
        &LLAMACPP,
        &[
            call(0, Some("call_whole"), Some("add"), Some(r#"{"a":1,"b":2}"#)),
            finish("tool_calls"),
            DONE.to_owned(),
        ],
    )
    .await;
    let response = response.expect("the reply folds");
    let [call] = calls(&response)[..] else {
        panic!("one call: {:?}", response.choice);
    };
    assert_eq!(provider_id(call), Some("call_whole"));
    assert_eq!(call.function.arguments_value(), json!({"a": 1, "b": 2}));
}

#[tokio::test]
async fn null_placeholders_are_no_fragments() {
    let (_, response) = stream(
        &OPENAI,
        &[
            chunk(json!({"content": null, "tool_calls": null}), None),
            call(0, Some("call_1"), Some("add"), None),
            call(0, None, None, None),
            call(0, None, None, Some(r#"{"a":1}"#)),
            finish("tool_calls"),
            DONE.to_owned(),
        ],
    )
    .await;
    let response = response.expect("the reply folds");
    assert_eq!(response.choice.len(), 1, "{:?}", response.choice);
    let [call] = calls(&response)[..] else {
        panic!("one call: {:?}", response.choice);
    };
    assert_eq!(call.function.arguments_value(), json!({"a": 1}));
}

#[tokio::test]
async fn a_length_cut_inside_a_call_keeps_what_it_states() {
    let (_, response) = stream(
        &OPENAI,
        &[
            text("noting"),
            call(
                0,
                Some("call_whole"),
                Some("record"),
                Some(r#"{"note":"first"}"#),
            ),
            call(1, Some("call_cut"), Some("record"), Some(r#"{"note":"The"#)),
            finish("length"),
            DONE.to_owned(),
        ],
    )
    .await;
    let response = response.expect("the reply folds");
    let ids: Vec<_> = calls(&response).into_iter().map(provider_id).collect();
    assert_eq!(ids, [Some("call_whole"), Some("call_cut")]);
    let cut = calls(&response)[1].function.clone();
    assert_eq!(cut.arguments_value(), json!({"note": "The"}));
    assert_eq!(cut.invalid_arguments.as_deref(), Some(r#"{"note":"The"#));
    assert!(
        response
            .choice
            .iter()
            .any(|block| block.canonical() == AssistantContent::text("noting"))
    );
    assert_eq!(response.finish_reason(), Some(FinishReason::Length));
}

fn reasoning_of(response: &CompletionResponse) -> Vec<Reasoning> {
    response
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Reasoning(reasoning) => Some(reasoning.clone()),
            _ => None,
        })
        .collect()
}

/// The `reasoning_details` a reasoning block holds in its provider item.
fn details_of(reasoning: &Reasoning) -> Value {
    reasoning
        .native
        .as_ref()
        .map(|native| native.item["reasoning_details"].clone())
        .unwrap_or_default()
}

fn signature(detail: &str) -> String {
    chunk(
        json!({"reasoning_details": [{
            "type": "reasoning.text",
            "text": "",
            "signature": detail,
            "index": 0,
        }]}),
        None,
    )
}

/// A block closes when the next one starts, so a signature that arrives
/// after the answer is reasoning of its own, and the message sent back
/// holds the reasoning and the signature as one message did.
#[tokio::test]
async fn a_late_signature_is_reasoning_after_the_answer() {
    let (_, response) = stream(
        &OPENROUTER,
        &[
            chunk(json!({"reasoning": "thinking"}), None),
            text("answer"),
            signature("sig-late"),
            finish("stop"),
            DONE.to_owned(),
        ],
    )
    .await;
    let response = response.expect("the reply folds");
    let [
        AssistantContent::Reasoning(first),
        AssistantContent::Text(answer),
        AssistantContent::Reasoning(late),
    ] = response.choice.as_slice()
    else {
        panic!(
            "reasoning, the answer and the signature: {:?}",
            response.choice
        );
    };
    assert_eq!(first.text, "thinking");
    assert_eq!(answer.text, "answer");
    let signed =
        json!([{"type": "reasoning.text", "text": "", "signature": "sig-late", "index": 0}]);
    assert_eq!(details_of(late), signed);
    let turn = crate::message::AssistantMessage {
        content: response.choice.clone(),
        ..response.head()
    };
    let Some(crate::providers::openai::completion::Message::Native(sent)) =
        crate::providers::openai::completion::assistant_message(turn)
    else {
        panic!("a rebuilt message");
    };
    assert_eq!(sent["reasoning"], "thinking");
    assert_eq!(sent["reasoning_details"], signed);
}

/// pi's merge: a second signature for the same detail fills nothing, as the
/// first already signed it.
#[tokio::test]
async fn a_second_signature_for_one_detail_keeps_the_first() {
    let (_, response) = stream(
        &OPENROUTER,
        &[
            chunk(json!({"reasoning": "thinking"}), None),
            signature("sig-1"),
            signature("sig-2"),
            text("answer"),
            finish("stop"),
            DONE.to_owned(),
        ],
    )
    .await;
    let response = response.expect("the reply folds");
    let reasoning = reasoning_of(&response);
    let [only] = &reasoning[..] else {
        panic!("one reasoning part: {:?}", response.choice);
    };
    assert_eq!(
        details_of(only),
        json!([{"type": "reasoning.text", "text": "", "signature": "sig-1", "index": 0}])
    );
}

#[tokio::test]
async fn a_provider_id_sent_twice_is_renamed() {
    let (_, response) = stream(
        &OPENAI,
        &[
            call(0, Some("call_same"), Some("add"), Some(r#"{"a":1}"#)),
            call(1, Some("call_same"), Some("add"), Some(r#"{"a":2}"#)),
            finish("tool_calls"),
            DONE.to_owned(),
        ],
    )
    .await;
    let response = response.expect("a duplicate id does not fail the reply");
    let ids: Vec<&CallId> = response.tool_calls().map(|call| &call.id).collect();
    let [first, second] = ids.as_slice() else {
        panic!("both calls are kept: {ids:?}");
    };
    assert_eq!(**first, CallId::from_wire("call_same"));
    assert!(matches!(second, CallId::Local(_)), "{second:?}");
}

/// Text streams as it arrives while a call waits in the buffer for its id:
/// the call surfaces only when it closes.
#[tokio::test]
async fn text_is_not_delayed_by_a_buffered_call() {
    let (items, response) = stream(
        &OPENAI,
        &[
            text("first "),
            call(0, None, Some("add"), Some(r#"{"a":1}"#)),
            text("second"),
            call(0, Some("call_1"), None, None),
            finish("tool_calls"),
            DONE.to_owned(),
        ],
    )
    .await;
    response.expect("the reply folds");
    let events: Vec<&StreamEvent> = items
        .iter()
        .filter_map(|item| match item {
            Ok(Item::Event(event)) => Some(event),
            _ => None,
        })
        .collect();
    let call_start = events
        .iter()
        .position(|event| {
            matches!(
                event,
                StreamEvent::Start {
                    kind: PartKind::ToolCall,
                    ..
                }
            )
        })
        .expect("the call surfaces");
    let texts: Vec<(usize, &str)> = events
        .iter()
        .enumerate()
        .filter_map(|(at, event)| match event {
            StreamEvent::Text { text, .. } => Some((at, text.as_str())),
            _ => None,
        })
        .collect();
    assert_eq!(
        texts.iter().map(|(_, text)| *text).collect::<Vec<_>>(),
        ["first ", "second"]
    );
    assert!(
        texts.iter().all(|(at, _)| *at < call_start),
        "both fragments stream before the call surfaces: {events:?}"
    );
}
