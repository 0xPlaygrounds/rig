//! Edge matrix for items a stateless Responses client must send back
//! unchanged (rig#2269).
//!
//! **Bug.** Things a client that manages its own conversation state could
//! not do: (a) send a `compaction` item back — `InputItem` rejected the tag;
//! (b) re-send an output message's `phase` — `OutputMessage` dropped it, a
//! streamed turn never stamped it, and a turn with several message items
//! replayed as one item under one id and one `phase`. OpenAI documents a
//! dropped `phase` as a quality regression on follow-ups.
//!
//! **Fix.** Every output item is one block of history holding the item
//! verbatim: a compaction item is an opaque block that replays, and each
//! message item, `phase` and id included, goes back as it came.
//!
//! **Fixtures.** Cell 1 is recorded live against `gpt-5.6-sol`, which
//! returns `phase: "final_answer"` on every message. Its second recorded
//! request is the proof: it carries the `phase` the first response
//! returned. Cell 2 is hand-derived from cell 1: a `compaction` item is
//! inserted at the head of turn 2's *request* `input[]` and at the head of
//! turn 1's *response* `output[]` (the shape `/responses/compact` returns),
//! with every other byte identical. Rig has no `/responses/compact` client
//! method — the reporter's fourth ask, deferred as maintainer-owned API
//! surface — so the item cannot be obtained live through rig; the derived
//! cell asserts the item is present in both places so a re-record cannot
//! silently drop it. Cells 3 to 5 are recorded live and streamed:
//! `gpt-5.4-nano` answers with `final_answer`, and `gpt-5.3-codex` sends a
//! `commentary` message before its answer or before a tool call.
//!
//! **How these cells fail on `origin/main`.** Cell 1 misses the mock on
//! turn 2 (rig sent no `phase`, so the recorded body differs) and its
//! post-replay assertion fails; cell 2 decodes the item only as `Output::Unknown` and the input side
//! rejects it with `unknown variant \`compaction\``. Cells 3 to 5 miss the
//! mock on their follow-up: streamed text carried no `phase`, and cell 4's
//! two messages were merged into one item.
//!
//! | # | cell | transport | proves | fixture |
//! |---|------|-----------|--------|---------|
//! | 1 | `phase_round_trips_on_follow_up` | blocking, 2 turns | turn-2 request carries turn-1's `phase` | recorded |
//! | 2 | `compaction_item_decodes_on_the_response` | blocking | compaction on `output[]` decodes typed; the same item re-serialized is accepted on the input side verbatim | derived from 1 |
//! | 3 | `streamed_phase_round_trips_on_follow_up` | streamed, 2 turns | a streamed turn's `phase` reaches its text and the follow-up, as unary text carries it | recorded |
//! | 4 | `commentary_and_final_answer_replay_as_two_items` | streamed, 2 turns | two message items keep their own id and `phase`, in order, and the follow-up is accepted | recorded |
//! | 5 | `commentary_before_a_tool_call_replays_with_its_phase` | streamed, 2 turns | a commentary message keeps its `phase` and its place before the call | recorded |
//!
//! Unit cells for the (de)serializers live in
//! `crates/rig-core/src/providers/openai/responses_api/stateless_replay_tests.rs`.

use futures::StreamExt;
use rig::completion::ToolDefinition;
use rig::message::{AssistantContent, Message, Text, ToolResultContent, UserContent};
use rig::providers::openai;
use rig::providers::openai::responses_api::{
    CompletionResponse as ProviderResponse, InputItem, Output,
};
use rig_test_support::cassette_models::OpenAiModels;
use serde::Deserialize;
use serde_json::Value;

use super::super::support::with_openai_cassette;
use rig::completion::CompletionRequest;

const TURN_ONE: &str = "Remember the codeword ALPHA-17. Reply exactly: ACK-1";
const TURN_TWO: &str = "Reply with exactly the remembered codeword.";
const PHASE: &str = "final_answer";

/// Every recorded request body of `scenario`, decoded, in wire order.
fn recorded_requests(scenario: &str) -> Vec<Value> {
    crate::cassettes::recorded_interaction_bodies("openai", scenario)
        .into_iter()
        .map(|(request, _)| serde_json::from_str(&request).expect("request body is JSON"))
        .collect()
}

/// Every recorded non-streaming response body of `scenario`, decoded.
fn recorded_responses(scenario: &str) -> Vec<Value> {
    crate::cassettes::recorded_interaction_bodies("openai", scenario)
        .into_iter()
        .filter_map(|(_, response)| serde_json::from_str(&response).ok())
        .collect()
}

fn assistant_items(request: &Value) -> Vec<&Value> {
    request["input"]
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter(|item| {
                    item.get("type").and_then(Value::as_str) == Some("message")
                        && item.get("role").and_then(Value::as_str) == Some("assistant")
                })
                .collect()
        })
        .unwrap_or_default()
}

/// The Responses API's own reply, read back out of
/// [`rig::completion::CompletionResponse::raw`] — the same value the
/// normalized response beside it was derived from.
fn provider_reply(response: &rig::completion::CompletionResponse) -> ProviderResponse {
    ProviderResponse::deserialize(&response.raw)
        .expect("`raw` is the serialized responses_api::CompletionResponse")
}

/// Two blocking turns, threading turn 1's normalized response back as history.
async fn two_turn_conversation(client: OpenAiModels) -> (ProviderResponse, ProviderResponse) {
    let model = client.completion(openai::GPT_5_6_SOL);
    let mut history = vec![Message::user(TURN_ONE)];
    let first = model
        .call(history.clone())
        .await
        .expect("turn 1 should succeed");
    let first_reply = provider_reply(&first);
    history.extend(first.message());
    history.push(Message::user(TURN_TWO));
    let second = model.call(history).await.expect("turn 2 should succeed");
    (first_reply, provider_reply(&second))
}

#[tokio::test]
async fn phase_round_trips_on_follow_up() {
    with_openai_cassette(
        "stateless_replay_matrix/phase_round_trips_on_follow_up",
        |client| async move {
            let (first, _second) = two_turn_conversation(client.openai).await;
            let phases: Vec<&str> = first
                .output
                .iter()
                .filter_map(|item| match item {
                    Output::Message(message) => message.phase.as_deref(),
                    _ => None,
                })
                .collect();
            assert_eq!(phases, [PHASE], "turn 1 must decode the wire's phase");
        },
    )
    .await;

    let requests = recorded_requests("stateless_replay_matrix/phase_round_trips_on_follow_up");
    assert_eq!(requests.len(), 2, "two turns recorded");
    let replayed = assistant_items(&requests[1]);
    assert_eq!(
        replayed.len(),
        1,
        "turn 2 replays exactly one assistant item"
    );
    assert_eq!(
        replayed[0]["phase"], PHASE,
        "turn 2's request must re-send the phase turn 1 returned: {}",
        replayed[0]
    );
    // Never on the block: the flatten would put it beside `text`.
    for block in replayed[0]["content"].as_array().expect("content array") {
        assert!(
            block.get("phase").is_none(),
            "phase leaked onto a block: {block}"
        );
    }
}

#[tokio::test]
async fn compaction_item_decodes_on_the_response() {
    if crate::cassettes::skip_when_recording(
        "cell 2 is hand-derived from cell 1: the compaction item on the response output is not what the API returned",
    ) {
        return;
    }
    with_openai_cassette(
        "stateless_replay_matrix/compaction_item_decodes_on_the_response",
        |client| async move {
            let model = client.openai.completion(openai::GPT_5_6_SOL);
            let response = model
                .call(CompletionRequest::new(TURN_ONE))
                .await
                .expect("a response carrying a compaction item must decode");
            let first = provider_reply(&response);
            let compaction = first
                .output
                .iter()
                .find_map(|item| match item {
                    Output::Unknown(item) if item["type"] == "compaction" => Some(item.clone()),
                    _ => None,
                })
                .expect("turn 1's output must keep the compaction item");
            assert_eq!(compaction.get("id"), Some(&Value::from("cmp_REDACTED_1")));

            // It reaches history as a block that replays, beside the
            // regular items.
            assert!(response.choice.iter().any(|block| matches!(
                block,
                AssistantContent::Opaque(opaque)
                    if opaque.replay && opaque.item == compaction
            )));

            // The same item is accepted on the input side byte-for-byte —
            // this is what a stateless client sends back.
            let wire = compaction;
            let input: InputItem =
                serde_json::from_value(wire.clone()).expect("the input side accepts the item");
            assert_eq!(
                serde_json::to_value(&input).expect("re-serializes"),
                wire,
                "the input item must re-emit the compaction item verbatim"
            );
        },
    )
    .await;

    let responses =
        recorded_responses("stateless_replay_matrix/compaction_item_decodes_on_the_response");
    assert!(
        responses[0]["output"][0]["type"] == "compaction",
        "derived fixture must carry the compaction item on the response output: {}",
        responses[0]["output"]
    );
}

/// A streamed turn: the reply the stream finished with.
async fn streamed(
    model: &rig::driver::Model<openai::responses_api::wire::Responses>,
    request: CompletionRequest,
) -> rig::completion::CompletionResponse {
    let mut stream = model.stream(request).expect("the stream starts");
    let mut items = Vec::new();
    while let Some(item) = stream.next().await {
        items.push(Ok(item.expect("every stream item decodes")));
    }
    let response = stream.finish().await.expect("the stream ends");
    rig_core::test_utils::streaming_conformance::assert_valid_event_stream(
        &items,
        &response.choice,
    );
    response
}

/// Every event a recorded SSE body carries, decoded.
fn sse_events(body: &str) -> Vec<Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .filter_map(|data| serde_json::from_str(data.trim()).ok())
        .collect()
}

/// The recorded stream's terminal response object.
fn terminal_response(body: &str) -> Value {
    sse_events(body)
        .into_iter()
        .find(|event| event["type"] == "response.completed")
        .map(|event| event["response"].clone())
        .expect("the recorded stream completed")
}

/// `(id, phase)` of each message item a recorded stream's `output_item.done`
/// events restate, in output order, with the phase `output_item.added` and
/// the terminal state for the same item.
fn recorded_messages(body: &str) -> Vec<(String, String)> {
    let events = sse_events(body);
    let stated = |kind: &str| -> Vec<(String, Value)> {
        events
            .iter()
            .filter(|event| event["type"] == kind && event["item"]["type"] == "message")
            .map(|event| {
                (
                    event["item"]["id"].as_str().unwrap_or_default().to_owned(),
                    event["item"]["phase"].clone(),
                )
            })
            .collect()
    };
    let done = stated("response.output_item.done");
    assert_eq!(
        stated("response.output_item.added"),
        done,
        "`output_item.added` states each message's phase before its text"
    );
    let terminal: Vec<(String, Value)> = terminal_response(body)["output"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|item| item["type"] == "message")
        .map(|item| {
            (
                item["id"].as_str().unwrap_or_default().to_owned(),
                item["phase"].clone(),
            )
        })
        .collect();
    assert_eq!(terminal, done, "the terminal restates each message's phase");
    done.into_iter()
        .map(|(id, phase)| {
            (
                id,
                phase
                    .as_str()
                    .expect("the recorded message states a phase")
                    .to_owned(),
            )
        })
        .collect()
}

/// `(id, phase)` of the message item each text block of a reply's choice
/// holds.
fn text_block_items(response: &rig::completion::CompletionResponse) -> Vec<(String, String)> {
    texts(&response.choice)
        .into_iter()
        .map(|text| {
            let field = |key: &str| {
                text.native
                    .as_ref()
                    .and_then(|native| native.item[key].as_str())
                    .unwrap_or_default()
                    .to_owned()
            };
            (field("id"), field("phase"))
        })
        .collect()
}

/// The text blocks the unary decoder makes of the recorded stream's own
/// terminal response: the same content, stated at once.
fn unary_text_of(model: &str, body: &str) -> Vec<Text> {
    let wire =
        openai::responses_api::wire::Responses::new(openai::OpenAIConfig::new("unused"), model);
    let reply = rig_test_support::history_survival::portability::decode_whole_reply(
        &wire,
        &terminal_response(body).to_string(),
    )
    .expect("the terminal response decodes as a unary reply");
    texts(&reply.choice)
}

fn texts(choice: &[AssistantContent]) -> Vec<Text> {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.clone()),
            _ => None,
        })
        .collect()
}

/// `(id, phase)` of each assistant message item a recorded request replays,
/// in order, with `-` for an absent field, and every item that carries a
/// `phase` at all.
fn replayed_messages(request: &Value) -> (Vec<(String, String)>, usize) {
    let items = request["input"].as_array().expect("the request has input");
    let phased = items
        .iter()
        .filter(|item| item.get("phase").is_some())
        .count();
    let messages = assistant_items(request)
        .into_iter()
        .map(|item| {
            let field = |key: &str| item[key].as_str().unwrap_or("-").to_owned();
            for block in item["content"].as_array().into_iter().flatten() {
                assert!(
                    block.get("phase").is_none() && block.get("message_id").is_none(),
                    "a message field leaked onto a content block: {block}"
                );
            }
            (field("id"), field("phase"))
        })
        .collect();
    (messages, phased)
}

/// The types of a recorded request's input items, in order.
fn input_types(request: &Value) -> Vec<String> {
    request["input"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|item| item["type"].as_str().unwrap_or("-").to_owned())
        .collect()
}

/// Assert the facts every streamed cell shares: the recorded turn-1 stream
/// states `expected_phases`, rig's streamed text carries each message's
/// phase and id as the unary decoder of the same content does, and the
/// follow-up request re-sends each on its own assistant item and nowhere
/// else, with no id repeated.
fn assert_streamed_phases_replay(
    scenario: &str,
    model: &str,
    first: &rig::completion::CompletionResponse,
    expected_phases: &[&str],
) -> Vec<(Value, String)> {
    let recorded = crate::cassettes::recorded_interaction_bodies("openai", scenario);
    assert_eq!(recorded.len(), 2, "two turns recorded");
    let messages = recorded_messages(&recorded[0].1);
    let phases: Vec<&str> = messages.iter().map(|(_, phase)| phase.as_str()).collect();
    assert_eq!(
        phases, expected_phases,
        "the recorded turn states these phases"
    );

    assert_eq!(
        text_block_items(first),
        messages,
        "each streamed text block carries its own message's phase and id"
    );
    assert_eq!(
        texts(&first.choice),
        unary_text_of(model, &recorded[0].1),
        "streamed and unary text carry equal extras"
    );

    let request: Value = serde_json::from_str(&recorded[1].0).expect("request is JSON");
    let (replayed, phased) = replayed_messages(&request);
    assert_eq!(
        replayed, messages,
        "the follow-up re-sends each message under its own id with its own phase, in order"
    );
    assert_eq!(phased, messages.len(), "no other item carries a phase");
    let ids: Vec<&str> = request["input"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|item| item["id"].as_str())
        .collect();
    let unique: std::collections::BTreeSet<&str> = ids.iter().copied().collect();
    assert_eq!(unique.len(), ids.len(), "no input item id repeats: {ids:?}");
    assert!(
        recorded[1].1.contains("\"type\":\"response.completed\""),
        "the provider accepted the follow-up"
    );
    recorded
        .into_iter()
        .map(|(request, response)| {
            (
                serde_json::from_str(&request).expect("request is JSON"),
                response,
            )
        })
        .collect()
}

const STREAMED_SCENARIO: &str = "stateless_replay_matrix/streamed_phase_round_trips_on_follow_up";

/// A streamed turn's `phase` reaches its text block and the follow-up
/// request, exactly as a unary turn's does.
#[tokio::test]
async fn streamed_phase_round_trips_on_follow_up() {
    let mut first = None;
    with_openai_cassette(
        "stateless_replay_matrix/streamed_phase_round_trips_on_follow_up",
        |client| {
            let first = &mut first;
            async move {
                let model = client.openai.responses(openai::GPT_5_4_NANO);
                let params =
                    serde_json::json!({ "store": false, "reasoning": { "effort": "low" } });
                let mut history = vec![Message::user(TURN_ONE)];
                let reply = streamed(
                    &model,
                    CompletionRequest::new(Message::user(TURN_ONE))
                        .max_tokens(256)
                        .additional_params(params.clone()),
                )
                .await;
                history.extend(reply.message());
                streamed(
                    &model,
                    CompletionRequest::new(TURN_TWO)
                        .messages(history)
                        .max_tokens(256)
                        .additional_params(params),
                )
                .await;
                *first = Some(reply);
            }
        },
    )
    .await;
    let first = first.expect("turn 1 ran");
    assert_streamed_phases_replay(STREAMED_SCENARIO, openai::GPT_5_4_NANO, &first, &[PHASE]);
}

/// The model OpenAI documents as sending `commentary`, which Rig names by
/// string: it has no constant.
const CODEX_MODEL: &str = "gpt-5.3-codex";

const TWO_MESSAGES_SCENARIO: &str =
    "stateless_replay_matrix/commentary_and_final_answer_replay_as_two_items";

/// A turn with a `commentary` message and a `final_answer` message, streamed:
/// each text block keeps its own message's phase and id, and the follow-up
/// sends both back as two items, in order, and is accepted.
#[tokio::test]
async fn commentary_and_final_answer_replay_as_two_items() {
    const PROMPT: &str = "Give me three fruit names.";
    let mut first = None;
    with_openai_cassette(
        "stateless_replay_matrix/commentary_and_final_answer_replay_as_two_items",
        |client| {
            let first = &mut first;
            async move {
                let model = client.openai.responses(CODEX_MODEL);
                let params =
                    serde_json::json!({ "store": false, "reasoning": { "effort": "low" } });
                let preamble = "Always start by sending the user a one-sentence progress update \
                as a separate commentary message, then send your final answer as a separate \
                message.";
                let reply = streamed(
                    &model,
                    CompletionRequest::new(PROMPT)
                        .preamble(preamble)
                        .max_tokens(800)
                        .additional_params(params.clone()),
                )
                .await;
                let mut history = vec![Message::user(PROMPT)];
                history.extend(reply.message());
                streamed(
                    &model,
                    CompletionRequest::new("Now name one more fruit. One word.")
                        .preamble(preamble)
                        .messages(history)
                        .max_tokens(800)
                        .additional_params(params),
                )
                .await;
                *first = Some(reply);
            }
        },
    )
    .await;
    let first = first.expect("turn 1 ran");
    let recorded = assert_streamed_phases_replay(
        TWO_MESSAGES_SCENARIO,
        CODEX_MODEL,
        &first,
        &["commentary", "final_answer"],
    );
    assert_eq!(
        input_types(&recorded[1].0)
            .into_iter()
            .filter(|kind| kind == "message")
            .count(),
        4,
        "user, commentary, final answer, user"
    );
}

const TOOL_SCENARIO: &str =
    "stateless_replay_matrix/commentary_before_a_tool_call_replays_with_its_phase";

/// A `commentary` message sent before a tool call keeps its phase and its
/// place: the follow-up replays reasoning, the commentary, then the call.
#[tokio::test]
async fn commentary_before_a_tool_call_replays_with_its_phase() {
    const PROMPT: &str = "What's the weather in Paris right now?";
    let mut first = None;
    with_openai_cassette(
        "stateless_replay_matrix/commentary_before_a_tool_call_replays_with_its_phase",
        |client| {
            let first = &mut first;
            async move {
                let model = client.openai.responses(CODEX_MODEL);
                let params =
                    serde_json::json!({ "store": false, "reasoning": { "effort": "low" } });
                let preamble = "Before you call any tool, first send the user one short sentence \
                saying what you are about to do. Then call the tool.";
                let tool = ToolDefinition {
                    name: rig_core::message::ToolName::new("get_weather").expect("tool name"),
                    description: "Look up the current weather for a city.".to_owned(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": { "city": { "type": "string" } },
                        "required": ["city"],
                    }),
                };
                let reply = streamed(
                    &model,
                    CompletionRequest::new(PROMPT)
                        .preamble(preamble)
                        .tool(tool.clone())
                        .max_tokens(600)
                        .additional_params(params.clone()),
                )
                .await;
                let call = reply
                    .tool_calls()
                    .next()
                    .cloned()
                    .expect("turn 1 calls the tool");
                let mut history = vec![Message::user(PROMPT)];
                history.extend(reply.message());
                history.push(Message::from(UserContent::tool_result(
                    call.id.clone(),
                    call.function.name.clone(),
                    vec![ToolResultContent::text("Sunny, 21 °C.")],
                )));
                streamed(
                    &model,
                    CompletionRequest::new("Answer in one sentence. Do not call any tool.")
                        .preamble(preamble)
                        .tool(tool)
                        .messages(history)
                        .max_tokens(600)
                        .additional_params(params),
                )
                .await;
                *first = Some(reply);
            }
        },
    )
    .await;
    let first = first.expect("turn 1 ran");
    let recorded =
        assert_streamed_phases_replay(TOOL_SCENARIO, CODEX_MODEL, &first, &["commentary"]);
    let types = input_types(&recorded[1].0);
    let commentary = recorded[1].0["input"]
        .as_array()
        .into_iter()
        .flatten()
        .position(|item| item["phase"] == "commentary")
        .expect("the commentary item is replayed");
    let call = types
        .iter()
        .position(|kind| kind == "function_call")
        .expect("the call is replayed");
    assert!(
        commentary < call,
        "the commentary item keeps its place before the call: {types:?}"
    );
    if let Some(reasoning) = types.iter().position(|kind| kind == "reasoning") {
        assert!(
            reasoning < commentary,
            "reasoning leads the turn: {types:?}"
        );
    }
    assert_eq!(
        types.get(call + 1).map(String::as_str),
        Some("function_call_output"),
        "the call is answered: {types:?}"
    );
}
