//! What an output-budget cut does to a tool call, on both transports.
//!
//! **Server**: the competent tier — `unsloth/Qwen3-8B-GGUF` Q4_K_M,
//! `--jinja --seed 42 --temp 0 -c 8192`, `llama-server` b10499-6d05498. A
//! model that reliably makes the call is the precondition for cutting one in
//! half.
//!
//! # Why this dimension exists
//!
//! [#2359](https://github.com/0xPlaygrounds/rig/pull/2359) established a
//! shared policy for the state llama.cpp produces here: a turn that ran out of
//! output budget mid-arguments comes back with `finish_reason: "length"` and
//! `tool_calls[].function.arguments` cut partway through the JSON object. The
//! decoder tolerates unparseable arguments **only** when the outer finish
//! reason maps to `Length`, drops the unusable call, and keeps the rest of the
//! turn; an ordinary completed `tool_calls` response with malformed JSON stays
//! a decode error.
//!
//! That policy was derived from DeepSeek and Mistral. llama.cpp is a third
//! wire that produces the same state, from a completely different server, and
//! nothing had checked the policy against it. Measured on b10499-6d05498 with
//! `tool_choice: required`, the cut lands exactly where the budget runs out:
//! 12 tokens gives `{`, 20 gives `{"note": "The`, 28 gives
//! `{"note": "The quarterly incident review found three unrelate`.
//!
//! | Cell | Transport | Pinned |
//! | --- | --- | --- |
//! | [`a_tool_call_cut_mid_arguments_does_not_destroy_the_turn`] | blocking | the response decodes; the unusable call is dropped, not returned half-parsed |
//! | [`the_streaming_path_drops_the_same_cut_call`] | streaming | the same boundary, and the stream still terminates cleanly |
//! | [`a_complete_call_under_the_same_cap_survives`] | blocking | the control: a cap large enough to finish yields a usable call |

use rig::client::CompletionClient;
use rig::completion::{CompletionModel, FinishReason};
use rig::message::AssistantContent;
use serde_json::Value;

use crate::cassettes::{recorded_sse_json_frames, recorded_statuses_and_bodies};

use super::super::cassette_support::*;

const NOTE_PROMPT: &str = "/no_think Record this note: The quarterly incident review found \
     three unrelated regressions in the billing pipeline.";

/// A budget measured to cut inside the argument object rather than before it.
const CUTTING_CAP: u64 = 20;
/// A budget measured to be enough for the whole call.
const COMPLETE_CAP: u64 = 256;

fn record_tool() -> rig::completion::ToolDefinition {
    rig::completion::ToolDefinition {
        name: "record".to_string(),
        description: "Record a long note.".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": { "note": { "type": "string" } },
            "required": ["note"],
        }),
    }
}

/// The recorded `(finish_reason, arguments)` of a blocking turn.
fn recorded_call(scenario: &str) -> (String, Option<String>) {
    let recorded = recorded_statuses_and_bodies("llamacpp", scenario);
    let (status, body) = recorded.last().expect("an interaction");
    assert_eq!(*status, 200, "{scenario}: {body}");
    let response: Value = serde_json::from_str(body).expect("response should be JSON");
    let choice = &response["choices"][0];
    (
        choice["finish_reason"]
            .as_str()
            .unwrap_or_default()
            .to_string(),
        choice["message"]["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .map(str::to_string),
    )
}

/// A tool call cut mid-arguments must not take the turn down with it.
#[tokio::test]
async fn a_tool_call_cut_mid_arguments_does_not_destroy_the_turn() {
    with_llamacpp_competent_cassette(
        "truncation_matrix/tool_call_cut_mid_arguments",
        |client| async move {
            let model = client.completion_model(CASSETTE_MODEL);
            let response = model
                .completion(
                    model
                        .completion_request(NOTE_PROMPT)
                        .tool(record_tool())
                        .tool_choice(rig::message::ToolChoice::Required)
                        .max_tokens(CUTTING_CAP)
                        .build(),
                )
                .await
                .expect(
                    "a budget cut mid-arguments must still decode — losing the turn's \
                     usage, id and finish reason with the unusable call is the defect \
                     #2359 fixed",
                );

            assert_eq!(
                response.finish_reason(),
                Some(FinishReason::Length),
                "the turn ran out of budget, and that is what authorizes the tolerance"
            );
            assert!(
                !response
                    .choice
                    .iter()
                    .any(|item| matches!(item, AssistantContent::ToolCall(_))),
                "a call whose arguments never parse is unusable and must be dropped, \
                 not handed to the caller half-formed: {:?}",
                response.choice
            );
            assert!(
                response.usage.output_tokens > 0,
                "the rest of the turn survives, usage included: {:?}",
                response.usage
            );
        },
    )
    .await;

    // The premise, from the bytes: the wire really did carry a truncated
    // argument string under `length`.
    let (finish_reason, arguments) = recorded_call("truncation_matrix/tool_call_cut_mid_arguments");
    assert_eq!(finish_reason, "length");
    let arguments = arguments.expect("the wire carried a tool call");
    assert!(
        serde_json::from_str::<Value>(&arguments).is_err(),
        "the recorded arguments must be *unparseable* for this cell to be about \
         truncation at all: {arguments:?}"
    );
    assert!(
        arguments.starts_with('{'),
        "and they must be a cut-off object rather than an empty string: {arguments:?}"
    );
}

/// The streaming path applies the same boundary.
#[tokio::test]
async fn the_streaming_path_drops_the_same_cut_call() {
    use futures::StreamExt as _;
    use rig::streaming::StreamedAssistantContent;

    with_llamacpp_competent_cassette(
        "truncation_matrix/streaming_tool_call_cut_mid_arguments",
        |client| async move {
            let model = client.completion_model(CASSETTE_MODEL);
            let mut stream = model
                .stream(
                    model
                        .completion_request(NOTE_PROMPT)
                        .tool(record_tool())
                        .tool_choice(rig::message::ToolChoice::Required)
                        .max_tokens(CUTTING_CAP)
                        .build(),
                )
                .await
                .expect("stream should start");

            let mut completed_calls = 0usize;
            let mut terminated = false;
            while let Some(item) = stream.next().await {
                match item.expect("no stream item may be an error") {
                    StreamedAssistantContent::ToolCall { .. } => completed_calls += 1,
                    StreamedAssistantContent::Final(_) => terminated = true,
                    _ => {}
                }
            }

            assert!(
                terminated,
                "the stream must terminate cleanly even though its only tool call was \
                 unusable"
            );
            assert_eq!(
                completed_calls, 0,
                "a call cut before its arguments parse must not be emitted as complete"
            );
        },
    )
    .await;

    let frames = recorded_sse_json_frames(
        "llamacpp",
        "truncation_matrix/streaming_tool_call_cut_mid_arguments",
    );
    let accumulated: String = frames
        .iter()
        .flat_map(|frame| {
            frame["choices"][0]["delta"]["tool_calls"]
                .as_array()
                .cloned()
                .unwrap_or_default()
        })
        .filter_map(|call| call["function"]["arguments"].as_str().map(str::to_string))
        .collect();
    assert!(
        !accumulated.is_empty() && serde_json::from_str::<Value>(&accumulated).is_err(),
        "the recorded stream must accumulate to an unparseable fragment: {accumulated:?}"
    );
    assert!(
        frames
            .iter()
            .any(|frame| frame["choices"][0]["finish_reason"] == serde_json::json!("length")),
        "and it must end under `length`, which is what authorizes dropping the call"
    );
}

/// The control: the same request with room to finish yields a usable call.
///
/// Without it, the two cells above would pass against a provider that had
/// simply stopped calling the tool.
#[tokio::test]
async fn a_complete_call_under_the_same_cap_survives() {
    with_llamacpp_competent_cassette(
        "truncation_matrix/complete_call_control",
        |client| async move {
            let model = client.completion_model(CASSETTE_MODEL);
            let response = model
                .completion(
                    model
                        .completion_request(NOTE_PROMPT)
                        .tool(record_tool())
                        .tool_choice(rig::message::ToolChoice::Required)
                        .max_tokens(COMPLETE_CAP)
                        .build(),
                )
                .await
                .expect("a generous budget should succeed");

            let call = response
                .choice
                .iter()
                .find_map(|item| match item {
                    AssistantContent::ToolCall(call) => Some(call.clone()),
                    _ => None,
                })
                .expect("the same request with room to finish must produce a call");
            assert_eq!(call.function.name, "record");
            assert!(
                call.function.arguments["note"]
                    .as_str()
                    .is_some_and(|note| !note.is_empty()),
                "and its arguments must be complete: {:?}",
                call.function.arguments
            );
        },
    )
    .await;

    let (finish_reason, arguments) = recorded_call("truncation_matrix/complete_call_control");
    assert_eq!(
        finish_reason, "tool_calls",
        "the control turn finished on its own"
    );
    serde_json::from_str::<Value>(&arguments.expect("a call"))
        .expect("the control's arguments must parse");
}
