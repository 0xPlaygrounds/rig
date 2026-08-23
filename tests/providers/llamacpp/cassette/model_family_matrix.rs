//! Cross-family tool calling: the dimension that tests
//! `EMITS_COMPLETE_SINGLE_CHUNK_TOOL_CALLS`.
//!
//! `--jinja` makes `llama-server` render the **model's own** chat template, so
//! the tool-call wire is a property of the *model family*, not of llama.cpp.
//! Qwen, Llama and Mistral each emit and parse tool calls through different
//! delimiters, and a claim about "what llama.cpp does with tool calls" that
//! was only ever checked against Qwen is a claim about Qwen.
//!
//! Coverage here is **targeted, not broad**, exactly as the corpus rules
//! require: one blocking tool cell and one streaming tool cell per non-Qwen
//! family. The rest of the matrix stays on Qwen.
//!
//! | Family | Model | Template tool support | Blocking | Streaming |
//! | --- | --- | --- | --- | --- |
//! | Qwen (smoke) | `unsloth/Qwen3-1.7B-GGUF` Q4_K_M | yes | `tools.rs`, `tool_matrix.rs` | `streaming_tools.rs` |
//! | Qwen (competent) | `unsloth/Qwen3-8B-GGUF` Q4_K_M | yes, parallel | `tool_matrix.rs` | — |
//! | Llama | `unsloth/Llama-3.2-3B-Instruct-GGUF` Q4_K_M | yes, no parallel | [`llama_family_calls_a_tool`] | [`llama_family_streams_tool_call_arguments_as_deltas`] |
//! | Mistral | `unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF` Q4_K_M | yes, parallel | [`mistral_family_calls_a_tool`] | [`mistral_family_streams_tool_call_arguments_as_deltas`] |
//! | Gemma | `unsloth/gemma-3-12b-it-GGUF` Q4_K_M | **none** | [`gemma_family_has_no_tool_calling_in_its_template`] | dropped — nothing to stream |
//!
//! Plus one cell that reads an existing fixture rather than recording:
//! [`even_a_zero_argument_call_streams_as_two_fragments`], the strongest case
//! against the const, since a two-character argument object is the one that
//! would arrive whole if any did.
//!
//! # The finding: llama.cpp does not emit single-chunk tool calls
//!
//! `EMITS_COMPLETE_SINGLE_CHUNK_TOOL_CALLS` asks whether the backend can put a
//! whole tool call — id, name and **complete** arguments — into one streaming
//! chunk. The provider inherited `true` for it. Measured against
//! `llama-server` b10499-6d05498 it is **false on every family that can call a
//! tool at all**: arguments stream one token at a time, so the first chunk
//! carries the name beside a lone `{` and the closing `}` arrives ten chunks
//! later. Even a *zero-argument* call streams as `{` then `}` rather than as
//! `{}`.
//!
//! The const is now `false`. Flipping it changes no output and that was
//! checked rather than argued: the shared accumulator's immediate-emit is a
//! probe (`UnparseableToolInput::Keep`) that finalizes a call only when its
//! accumulated arguments parse, so a lone `{` was already being declined, and
//! the whole recorded streaming corpus replays byte-identically either way.
//! What changes is that the const stops asserting something untrue — and that
//! a future build whose partial arguments happened to parse could not finalize
//! a truncated call.
//!
//! # Gemma has no tools, and that is a note rather than a finding
//!
//! `GET /props` on the Gemma server reports
//! `chat_template_caps.supports_tools: false` and
//! `supports_tool_calls: false`. Its template has no tool section, so
//! llama.cpp has nothing to render tool definitions into and nothing to parse
//! a call back out of; the model answers a tool prompt with a ```` ```tool_code ````
//! block as ordinary prose. A capability the loaded model lacks is not a rig
//! defect — it is the model choice — so the cell records the shape and the
//! streaming twin is dropped with this as its reason.

use rig::client::CompletionClient;
use rig::completion::CompletionModel;
use rig::message::AssistantContent;
use serde_json::Value;

use crate::cassettes::{
    recorded_json_request, recorded_sse_json_frames, recorded_statuses_and_bodies,
};
use crate::support::{Subtract, assistant_text_response};

use super::super::cassette_support::*;

const TOOL_PROMPT: &str = "Calculate 2 - 5 using the tool.";

/// The `(name, arguments)` pairs a recorded blocking response asked for.
fn recorded_tool_calls(scenario: &str) -> Vec<(String, String)> {
    let recorded = recorded_statuses_and_bodies("llamacpp", scenario);
    let (status, body) = recorded.last().expect("an interaction");
    assert_eq!(*status, 200, "{scenario}: {body}");
    let response: Value = serde_json::from_str(body).expect("response should be JSON");
    response["choices"][0]["message"]["tool_calls"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .map(|call| {
            (
                call["function"]["name"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
                call["function"]["arguments"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
            )
        })
        .collect()
}

/// Every tool-call fragment a recorded stream carried, in wire order.
fn recorded_tool_call_deltas(scenario: &str) -> Vec<(Option<String>, Option<String>)> {
    recorded_sse_json_frames("llamacpp", scenario)
        .into_iter()
        .flat_map(|frame| {
            frame["choices"][0]["delta"]["tool_calls"]
                .as_array()
                .cloned()
                .unwrap_or_default()
        })
        .map(|call| {
            (
                call["function"]["name"].as_str().map(str::to_string),
                call["function"]["arguments"].as_str().map(str::to_string),
            )
        })
        .collect()
}

/// The shared assertion: llama.cpp streamed this call as *deltas*, and rig
/// still reassembled it into one complete call.
///
/// Both halves matter. The first is the measurement that made
/// `EMITS_COMPLETE_SINGLE_CHUNK_TOOL_CALLS` false; the second is that being
/// false costs nothing.
fn assert_streamed_as_deltas_and_reassembled(scenario: &str, expected_tool: &str) {
    let deltas = recorded_tool_call_deltas(scenario);
    assert!(
        deltas.len() > 1,
        "{scenario}: llama.cpp streams tool-call arguments per token, so a \
         single fragment means the wire changed and \
         EMITS_COMPLETE_SINGLE_CHUNK_TOOL_CALLS should be re-measured: {deltas:?}"
    );

    let (first_name, first_arguments) = &deltas[0];
    assert_eq!(
        first_name.as_deref(),
        Some(expected_tool),
        "{scenario}: the opening fragment names the tool: {deltas:?}"
    );
    let first_arguments = first_arguments.clone().unwrap_or_default();
    assert!(
        serde_json::from_str::<Value>(&first_arguments).is_err(),
        "{scenario}: the opening fragment's arguments must be an *incomplete* \
         JSON prefix — if they parse, this family really does emit complete \
         single-chunk calls and the const is wrong the other way: \
         {first_arguments:?}"
    );

    // Reassembly: concatenating every fragment yields the whole object.
    let assembled = deltas
        .iter()
        .filter_map(|(_, arguments)| arguments.clone())
        .collect::<String>();
    let parsed: Value = serde_json::from_str(&assembled).unwrap_or_else(|error| {
        panic!("{scenario}: the concatenated fragments must parse: {error}: {assembled:?}")
    });
    assert!(
        parsed.is_object(),
        "{scenario}: tool arguments are an object: {parsed}"
    );
}

// ---------------------------------------------------------------------------
// Llama 3.2
// ---------------------------------------------------------------------------

#[tokio::test]
async fn llama_family_calls_a_tool() {
    with_llamacpp_llama_family_cassette(
        "model_family_matrix/llama_blocking_tool_call",
        |client| async move {
            let model = client.completion_model(CASSETTE_LLAMA_MODEL);
            let response = model
                .completion(
                    model
                        .completion_request(TOOL_PROMPT)
                        .tool(rig::tool::tool_definition(&Subtract))
                        .max_tokens(512)
                        .build(),
                )
                .await
                .expect("Llama 3.2's template supports tool calls");

            let calls = response
                .choice
                .iter()
                .filter_map(|item| match item {
                    AssistantContent::ToolCall(call) => Some(call.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(calls.len(), 1, "{:?}", response.choice);
            assert_eq!(calls[0].function.name, "subtract");
            assert!(
                calls[0].function.arguments.is_object(),
                "a different template must still produce object arguments: {:?}",
                calls[0].function.arguments
            );
        },
    )
    .await;

    let calls = recorded_tool_calls("model_family_matrix/llama_blocking_tool_call");
    assert_eq!(calls.len(), 1, "{calls:?}");
    assert_eq!(calls[0].0, "subtract");
    serde_json::from_str::<Value>(&calls[0].1)
        .expect("the wire's stringified arguments must be JSON");
}

#[tokio::test]
async fn llama_family_streams_tool_call_arguments_as_deltas() {
    with_llamacpp_llama_family_cassette(
        "model_family_matrix/llama_streaming_tool_call",
        |client| async move {
            let model = client.completion_model(CASSETTE_LLAMA_MODEL);
            let observation = crate::support::collect_raw_stream_observation(
                model
                    .stream(
                        model
                            .completion_request(TOOL_PROMPT)
                            .tool(rig::tool::tool_definition(&Subtract))
                            .max_tokens(512)
                            .build(),
                    )
                    .await
                    .expect("raw stream should start"),
            )
            .await;

            crate::support::assert_raw_stream_tool_call_arguments_are_objects(
                &observation,
                &["subtract"],
            );
        },
    )
    .await;

    assert_streamed_as_deltas_and_reassembled(
        "model_family_matrix/llama_streaming_tool_call",
        "subtract",
    );
}

// ---------------------------------------------------------------------------
// Mistral Small 3.2
// ---------------------------------------------------------------------------

#[tokio::test]
async fn mistral_family_calls_a_tool() {
    with_llamacpp_mistral_family_cassette(
        "model_family_matrix/mistral_blocking_tool_call",
        |client| async move {
            let model = client.completion_model(CASSETTE_MISTRAL_MODEL);
            let response = model
                .completion(
                    model
                        .completion_request(TOOL_PROMPT)
                        .tool(rig::tool::tool_definition(&Subtract))
                        .max_tokens(512)
                        .build(),
                )
                .await
                .expect("Mistral Small's template supports tool calls");

            let calls = response
                .choice
                .iter()
                .filter_map(|item| match item {
                    AssistantContent::ToolCall(call) => Some(call.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(calls.len(), 1, "{:?}", response.choice);
            assert_eq!(calls[0].function.name, "subtract");
            assert!(calls[0].function.arguments.is_object());
        },
    )
    .await;

    let calls = recorded_tool_calls("model_family_matrix/mistral_blocking_tool_call");
    assert_eq!(calls.len(), 1, "{calls:?}");
    assert_eq!(calls[0].0, "subtract");
}

#[tokio::test]
async fn mistral_family_streams_tool_call_arguments_as_deltas() {
    with_llamacpp_mistral_family_cassette(
        "model_family_matrix/mistral_streaming_tool_call",
        |client| async move {
            let model = client.completion_model(CASSETTE_MISTRAL_MODEL);
            let observation = crate::support::collect_raw_stream_observation(
                model
                    .stream(
                        model
                            .completion_request(TOOL_PROMPT)
                            .tool(rig::tool::tool_definition(&Subtract))
                            .max_tokens(512)
                            .build(),
                    )
                    .await
                    .expect("raw stream should start"),
            )
            .await;

            crate::support::assert_raw_stream_tool_call_arguments_are_objects(
                &observation,
                &["subtract"],
            );
        },
    )
    .await;

    assert_streamed_as_deltas_and_reassembled(
        "model_family_matrix/mistral_streaming_tool_call",
        "subtract",
    );
}

// ---------------------------------------------------------------------------
// Gemma 3 — no tool support in the template
// ---------------------------------------------------------------------------

/// Gemma's template declares no tool support, so a tool request degrades to
/// prose.
///
/// This is the "a capability the loaded model lacks is not a bug" row. rig
/// sends the tool definitions; llama.cpp renders a template with nowhere to
/// put them and no parser to read a call back out; the model answers with a
/// ```` ```tool_code ```` block as ordinary text. Nothing here is rig's to
/// fix, and the streaming twin is dropped because there is no tool-call stream
/// to observe.
#[tokio::test]
async fn gemma_family_has_no_tool_calling_in_its_template() {
    with_llamacpp_gemma_family_cassette(
        "model_family_matrix/gemma_tool_request_degrades_to_text",
        |client| async move {
            let model = client.completion_model(CASSETTE_GEMMA_MODEL);
            let response = model
                .completion(
                    model
                        .completion_request(TOOL_PROMPT)
                        .tool(rig::tool::tool_definition(&Subtract))
                        .max_tokens(256)
                        .build(),
                )
                .await
                .expect("a tool request against a tool-less template is not an error");

            assert!(
                !response
                    .choice
                    .iter()
                    .any(|item| matches!(item, AssistantContent::ToolCall(_))),
                "Gemma's template cannot produce a parsed tool call: {:?}",
                response.choice
            );
            let text = assistant_text_response(&response.choice).unwrap_or_default();
            assert!(
                !text.trim().is_empty(),
                "the turn still answers, in prose: {text:?}"
            );
        },
    )
    .await;

    // The premise: rig really did advertise the tool. Without this the cell
    // would pass against a request that never mentioned tools at all.
    let request = recorded_json_request(
        "llamacpp",
        "model_family_matrix/gemma_tool_request_degrades_to_text",
    );
    assert_eq!(
        request["tools"]
            .as_array()
            .map(std::vec::Vec::len)
            .unwrap_or_default(),
        1,
        "the tool definition must reach the wire: {request}"
    );
    assert!(
        recorded_tool_calls("model_family_matrix/gemma_tool_request_degrades_to_text").is_empty(),
        "and no parsed call comes back"
    );
}

/// Even a *zero-argument* call streams as `{` then `}`, never as `{}`.
///
/// The strongest possible case against
/// `EMITS_COMPLETE_SINGLE_CHUNK_TOOL_CALLS`: if any call could arrive whole in
/// one chunk it would be this one, whose entire argument object is two
/// characters. It does not.
///
/// Read out of `streaming_tools/raw_stream_emits_required_zero_arg_tool_call`
/// rather than re-recorded, because that fixture already exists — and because
/// its own cell asserts only on the reassembled `{}` and never looks at the
/// wire, which is exactly the gap this closes.
#[test]
fn even_a_zero_argument_call_streams_as_two_fragments() {
    let deltas =
        recorded_tool_call_deltas("streaming_tools/raw_stream_emits_required_zero_arg_tool_call");
    assert!(
        deltas.len() > 1,
        "a zero-argument call whose whole object is `{{}}` still arrives in \
         fragments: {deltas:?}"
    );
    let assembled = deltas
        .iter()
        .filter_map(|(_, arguments)| arguments.clone())
        .collect::<String>();
    assert_eq!(
        assembled, "{}",
        "and they reassemble to the empty object: {deltas:?}"
    );
    assert_ne!(
        deltas[0].1.clone().unwrap_or_default(),
        "{}",
        "the opening fragment is not the whole object — which is the claim \
         `EMITS_COMPLETE_SINGLE_CHUNK_TOOL_CALLS = true` would have made"
    );
}
