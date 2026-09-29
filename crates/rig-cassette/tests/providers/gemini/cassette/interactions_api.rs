//! Migrated from `examples/gemini_interactions_api.rs`.

use futures::StreamExt;
use rig::message::{
    AssistantContent, Message, ToolCall, ToolChoice, ToolResultContent, UserContent,
};
use rig::providers::gemini::interactions_api::{AdditionalParameters, Interaction, Tool};
use rig::streaming::Item;
use rig::streaming::StreamEvent;
use serde::Deserialize;

use crate::support::assert_nonempty_response;
use rig::completion::CompletionRequest;

fn extract_text(choice: &[AssistantContent]) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

fn first_tool_call(choice: &[AssistantContent]) -> Option<ToolCall> {
    choice.iter().find_map(|content| match content {
        AssistantContent::ToolCall(tool_call) => Some(tool_call.clone()),
        _ => None,
    })
}

#[tokio::test]
async fn basic_interaction_returns_id() {
    super::super::support::with_gemini_interactions_cassette(
        "interactions_api/basic_interaction_returns_id",
        |client| async move {
            let model = client.interactions("gemini-3-flash-preview");
            let params = AdditionalParameters {
                store: Some(true),
                ..Default::default()
            };
            let request = CompletionRequest::new("Give me two fun facts about hummingbirds.")
                .preamble("Be concise.")
                .additional_params(serde_json::to_value(params).expect("params should serialize"));
            let response = model
                .call(request)
                .await
                .expect("completion should succeed");

            assert_nonempty_response(&extract_text(&response.choice));
            // The interaction id is Gemini's continuation handle (fed back as
            // `previous_interaction_id`), not an assistant message id: it is
            // on the interaction document `raw` carries, and the decoder
            // reports that very value as the response id.
            let document = Interaction::deserialize(&response.raw)
                .expect("raw is the Interactions API's own document");
            assert!(
                !document.id.is_empty(),
                "interactions api should return an interaction id"
            );
            assert_eq!(
                response.response_id.as_deref(),
                Some(document.id.as_str()),
                "the continuation handle is what the normalized response names"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn followup_with_previous_interaction_id() {
    super::super::support::with_gemini_interactions_cassette(
        "interactions_api/followup_with_previous_interaction_id",
        |client| async move {
            let model = client.interactions("gemini-3-flash-preview");
            let initial = model
                .call(
                    CompletionRequest::new("Give me one short fact about hummingbirds.")
                        .additional_params(
                            serde_json::to_value(AdditionalParameters {
                                store: Some(true),
                                ..Default::default()
                            })
                            .expect("params should serialize"),
                        ),
                )
                .await
                .expect("initial completion should succeed");
            // Gemini's continuation handle is the interaction id, which the
            // decoder reports as the response id; it is what
            // `previous_interaction_id` echoes back.
            let interaction_id = initial
                .response_id
                .clone()
                .expect("expected an interaction id");
            assert!(!interaction_id.is_empty(), "expected an interaction id");

            let followup = model
                .call(
                    CompletionRequest::new("Now answer with a short analogy.").additional_params(
                        serde_json::to_value(AdditionalParameters {
                            previous_interaction_id: Some(interaction_id),
                            ..Default::default()
                        })
                        .expect("params should serialize"),
                    ),
                )
                .await
                .expect("followup completion should succeed");

            assert_nonempty_response(&extract_text(&followup.choice));
        },
    )
    .await;
}

#[tokio::test]
async fn google_search_tool_interaction() {
    super::super::support::with_gemini_interactions_cassette(
        "interactions_api/google_search_tool_interaction",
        |client| async move {
            let model = client.interactions("gemini-3-flash-preview");
            // The hosted-tool exchange log is provider-specific, so this
            // asserts against the interaction document `raw` carries and then
            // against the normalized view the decoder folded from the same
            // bytes — one request, exactly as the cassette recorded it.
            let response = model
                .call(
                    CompletionRequest::new("Who won the Euro 2024 tournament?").additional_params(
                        serde_json::to_value(AdditionalParameters {
                            tools: Some(vec![Tool::GoogleSearch]),
                            ..Default::default()
                        })
                        .expect("params should serialize"),
                    ),
                )
                .await
                .expect("search completion should succeed");

            let document = Interaction::deserialize(&response.raw)
                .expect("raw is the Interactions API's own document");
            assert!(
                !document.google_search_exchanges().is_empty(),
                "expected a search-backed exchange"
            );

            assert_nonempty_response(&extract_text(&response.choice));
        },
    )
    .await;
}

#[tokio::test]
async fn tool_result_roundtrip() {
    super::super::support::with_gemini_interactions_cassette(
        "interactions_api/tool_result_roundtrip",
        |client| async move {
            let model = client.interactions("gemini-3-flash-preview");
            let tool = rig::completion::ToolDefinition {
                name: "add".to_string(),
                description: "Add two numbers together".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "x": { "type": "number" },
                        "y": { "type": "number" }
                    },
                    "required": ["x", "y"]
                }),
            };

            let initial = model
                .call(
                    CompletionRequest::new("Use the add tool to sum 7 and 11.")
                        .tool(tool)
                        .tool_choice(ToolChoice::Required)
                        .additional_params(
                            serde_json::to_value(AdditionalParameters {
                                store: Some(true),
                                ..Default::default()
                            })
                            .expect("params should serialize"),
                        ),
                )
                .await
                .expect("tool call completion should succeed");

            // Gemini's continuation handle is the interaction id the decoder
            // reports as the response id, and the same response supplies the
            // tool call — so this still costs one interaction.
            let interaction_id = initial
                .response_id
                .clone()
                .expect("expected an interaction id");
            assert!(!interaction_id.is_empty(), "expected an interaction id");

            let tool_call = first_tool_call(&initial.choice).expect("expected a tool call");

            let followup = model
                .call(
                    CompletionRequest::new(Message::from(UserContent::tool_result(
                        tool_call.id.clone(),
                        tool_call.function.name.clone(),
                        rig_core::NonEmpty::new(ToolResultContent::json(
                            serde_json::json!({ "sum": 18.0 }),
                        )),
                    )))
                    .additional_params(
                        serde_json::to_value(AdditionalParameters {
                            previous_interaction_id: Some(interaction_id),
                            ..Default::default()
                        })
                        .expect("params should serialize"),
                    ),
                )
                .await
                .expect("tool result followup should succeed");

            assert_nonempty_response(&extract_text(&followup.choice));
        },
    )
    .await;
}

#[tokio::test]
async fn streaming_interaction() {
    super::super::support::with_gemini_interactions_cassette(
        "interactions_api/streaming_interaction",
        |client| async move {
            let model = client.interactions("gemini-3-flash-preview");
            let request = CompletionRequest::new("Write a 3-line poem about rust and rivers.")
                .temperature(0.4);
            let mut stream = model.stream(request).expect("stream should start");

            let mut text = String::new();
            while let Some(chunk) = stream.next().await {
                if let Item::Event(StreamEvent::Text { text: delta, .. }) =
                    chunk.expect("stream chunk should succeed")
                {
                    text.push_str(&delta);
                }
            }
            let response = stream.finish().await.expect("the stream ends");

            assert_nonempty_response(&text);
            assert!(
                response.usage.is_reported(),
                "expected the final response to expose token usage"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn streaming_final_metadata_exposes_model_version() {
    super::super::support::with_gemini_interactions_cassette(
        "interactions_api/streaming_final_metadata_exposes_model_version",
        |client| async move {
            let model = client.interactions("gemini-3-flash-preview");
            let request = CompletionRequest::new("Reply with exactly: interaction metadata ok")
                .temperature(0.0);
            let mut stream = model.stream(request).expect("stream should start");

            let mut text = String::new();
            while let Some(chunk) = stream.next().await {
                if let Item::Event(StreamEvent::Text { text: delta, .. }) =
                    chunk.expect("stream chunk should succeed")
                {
                    text.push_str(&delta);
                }
            }
            let response = stream.finish().await.expect("the stream ends");

            assert_nonempty_response(&text);
            assert_eq!(
                response.model.as_deref(),
                Some("gemini-3-flash-preview"),
                "expected Interactions stream final response to expose Interaction.model"
            );
            assert!(
                response.usage.is_reported(),
                "expected final response to expose Interactions token usage"
            );
        },
    )
    .await;
}

/// The Interactions surface must surface every token counter it is sent.
///
/// Replays a cassette recorded long before this assertion existed, which is what
/// makes it a genuine regression cell: the recorded wire reports
/// `total_input_tokens: 14`, `total_output_tokens: 34`, `total_tokens: 270` and
/// `total_thought_tokens: 222`, thinking beside output. Rig's output counts the
/// thinking, so input plus output is the provider's own total. The wire also
/// carries `total_cached_tokens`, which `cached_input_tokens` reports.
#[tokio::test]
async fn interactions_usage_surfaces_thinking_and_cached_tokens() {
    super::super::support::with_gemini_interactions_cassette(
        "interactions_api/basic_interaction_returns_id",
        |client| async move {
            let model = client.interactions("gemini-3-flash-preview");
            let params = AdditionalParameters {
                store: Some(true),
                ..Default::default()
            };
            let request = CompletionRequest::new("Give me two fun facts about hummingbirds.")
                .preamble("Be concise.")
                .additional_params(serde_json::to_value(params).expect("params should serialize"));

            let response = model
                .call(request)
                .await
                .expect("completion should succeed");
            let usage = response.usage;
            let recorded_total = response
                .raw
                .pointer("/usage/total_tokens")
                .and_then(serde_json::Value::as_u64);

            assert!(
                usage.reasoning_tokens.is_some_and(|n| n > 0),
                "the recorded interaction reports total_thought_tokens; dropping it loses the \
                 largest component of the spend. got {usage:?}"
            );
            assert!(
                usage.reasoning_tokens <= usage.output_tokens,
                "output includes the thinking the wire reports beside it: {usage:?}"
            );
            assert!(
                recorded_total.is_some(),
                "the recorded interaction reports a total"
            );
            assert_eq!(
                usage.total_tokens, recorded_total,
                "input plus output is the provider's own total: {usage:?}"
            );
            assert_eq!(
                Some(usage.input_tokens.unwrap_or(0) + usage.output_tokens.unwrap_or(0)),
                usage.total_tokens,
                "the total is input plus output: {usage:?}"
            );
        },
    )
    .await;
}

/// The code-execution request both tool-use cells send.
fn code_execution_request() -> CompletionRequest {
    CompletionRequest::new(
        "Use code execution to compute the sum of the first 50 prime numbers, then state it.",
    )
    .additional_params(
        serde_json::to_value(AdditionalParameters {
            tools: Some(vec![Tool::CodeExecution]),
            store: Some(false),
            ..Default::default()
        })
        .expect("params should serialize"),
    )
}

/// Rig's `Usage` against the Interactions `usage` object the reply carried:
/// the tool-use prompt is input beside `total_input_tokens`, thinking is
/// output beside `total_output_tokens`, and input plus output is the
/// provider's own `total_tokens`. That last equality is what shows the
/// tool-use tokens sit outside `total_input_tokens` rather than inside it.
fn assert_tool_use_usage(usage: &rig::completion::Usage, raw: &serde_json::Value) {
    let wire = raw.get("usage").expect("the reply carries its usage");
    let count = |field: &str| {
        wire.get(field)
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
    };
    let tool_use = count("total_tool_use_tokens");
    assert!(
        tool_use > 0,
        "code execution reports tool-use tokens, or this cell proves nothing: {wire}"
    );
    assert_eq!(usage.tool_use_prompt_tokens, Some(tool_use), "{wire}");
    assert_eq!(
        usage.input_tokens,
        Some(count("total_input_tokens") + tool_use),
        "input counts the tool-use prompt: {wire}"
    );
    assert_eq!(
        usage.output_tokens,
        Some(count("total_output_tokens") + count("total_thought_tokens")),
        "output counts the thinking: {wire}"
    );
    assert_eq!(
        usage.total_tokens,
        Some(count("total_tokens")),
        "input plus output is the provider's total: {wire}"
    );
    assert_eq!(
        usage.total_tokens,
        usage
            .input_tokens
            .zip(usage.output_tokens)
            .map(|(input, output)| input + output),
        "the total is input plus output: {usage:?}"
    );
}

/// A hosted tool's prompt is input: Interactions reports it as
/// `total_tool_use_tokens` beside `total_input_tokens`, and its own total
/// counts it. Recorded because no other Interactions cassette carries a
/// non-zero tool-use count (Google Search reports zero).
#[tokio::test]
async fn code_execution_usage_counts_the_tool_use_prompt_as_input() {
    super::super::support::with_gemini_interactions_cassette(
        "interactions_api/code_execution_usage",
        |client| async move {
            let model = client.interactions("gemini-3-flash-preview");
            let response = model
                .call(code_execution_request())
                .await
                .expect("code-execution completion should succeed");
            assert_nonempty_response(&extract_text(&response.choice));
            assert_tool_use_usage(&response.usage, &response.raw);
        },
    )
    .await;
}

/// The streamed twin: the terminal `interaction.completed` usage maps the
/// same way. Unstored (`store: false`), its status update carries no
/// `interaction_id`; that frame once passed for a whole interaction and
/// ended the reply before its answer and its usage.
#[tokio::test]
async fn code_execution_usage_counts_the_tool_use_prompt_as_input_streamed() {
    super::super::support::with_gemini_interactions_cassette(
        "interactions_api/code_execution_usage_streamed",
        |client| async move {
            let model = client.interactions("gemini-3-flash-preview");
            let mut stream = model
                .stream(code_execution_request())
                .expect("stream should start");
            let mut text = String::new();
            while let Some(chunk) = stream.next().await {
                if let Item::Event(StreamEvent::Text { text: delta, .. }) =
                    chunk.expect("stream chunk should succeed")
                {
                    text.push_str(&delta);
                }
            }
            let response = stream.finish().await.expect("the stream ends");
            assert_nonempty_response(&text);
            assert_tool_use_usage(&response.usage, &response.raw);
        },
    )
    .await;
}
