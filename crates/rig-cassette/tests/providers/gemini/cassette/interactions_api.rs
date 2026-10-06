//! Migrated from `examples/gemini_interactions_api.rs`.

use rig::message::{
    AssistantContent, Message, ToolCall, ToolChoice, ToolResultContent, UserContent,
};

use crate::support::assert_nonempty_response;
use rig::completion::CompletionRequest;

/// Whether the interaction's steps carry a Google Search call or result, as a
/// step of its own or as an item of a model output step.
fn has_google_search_exchange(steps: &[serde_json::Value]) -> bool {
    let is_search = |item: &serde_json::Value| {
        matches!(
            item["type"].as_str(),
            Some("google_search_call" | "google_search_result")
        )
    };
    steps.iter().any(|step| {
        is_search(step)
            || (step["type"] == "model_output"
                && step["content"]
                    .as_array()
                    .is_some_and(|items| items.iter().any(is_search)))
    })
}

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
            let params = serde_json::json!({ "store": true });
            let request = CompletionRequest::new("Give me two fun facts about hummingbirds.")
                .preamble("Be concise.")
                .additional_params(params);
            let response = model
                .call(request)
                .await
                .expect("completion should succeed");

            assert_nonempty_response(&extract_text(&response.choice));
            // The interaction id is Gemini's continuation handle (fed back as
            // `previous_interaction_id`), not an assistant message id: it is
            // on the interaction document `raw` carries, and the decoder
            // reports that very value as the response id.
            let id = response.raw["id"]
                .as_str()
                .expect("raw is the Interactions API's own document");
            assert!(
                !id.is_empty(),
                "interactions api should return an interaction id"
            );
            assert_eq!(
                response.response_id(),
                Some(id),
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
                        .additional_params(serde_json::json!({ "store": true })),
                )
                .await
                .expect("initial completion should succeed");
            // Gemini's continuation handle is the interaction id, which the
            // decoder reports as the response id; it is what
            // `previous_interaction_id` echoes back.
            let interaction_id = initial
                .response_id()
                .map(str::to_owned)
                .expect("expected an interaction id");
            assert!(!interaction_id.is_empty(), "expected an interaction id");

            let followup = model
                .call(
                    CompletionRequest::new("Now answer with a short analogy.").additional_params(
                        serde_json::json!({ "previous_interaction_id": interaction_id }),
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
                        serde_json::json!({ "tools": [{ "type": "google_search" }] }),
                    ),
                )
                .await
                .expect("search completion should succeed");

            let steps = response.raw["steps"]
                .as_array()
                .expect("raw is the Interactions API's own document");
            assert!(
                has_google_search_exchange(steps),
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
                name: rig_core::message::ToolName::new("add").expect("tool name"),
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
                        .additional_params(serde_json::json!({ "store": true })),
                )
                .await
                .expect("tool call completion should succeed");

            // Gemini's continuation handle is the interaction id the decoder
            // reports as the response id, and the same response supplies the
            // tool call — so this still costs one interaction.
            let interaction_id = initial
                .response_id()
                .map(str::to_owned)
                .expect("expected an interaction id");
            assert!(!interaction_id.is_empty(), "expected an interaction id");

            let tool_call = first_tool_call(&initial.choice).expect("expected a tool call");

            let followup = model
                .call(
                    CompletionRequest::new(Message::from(UserContent::tool_result(
                        tool_call.id.clone(),
                        tool_call.function.name.clone(),
                        vec![ToolResultContent::json(serde_json::json!({ "sum": 18.0 }))],
                    )))
                    .additional_params(
                        serde_json::json!({ "previous_interaction_id": interaction_id }),
                    ),
                )
                .await
                .expect("tool result followup should succeed");

            assert_nonempty_response(&extract_text(&followup.choice));
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
        serde_json::json!({ "store": false, "tools": [{ "type": "code_execution" }] }),
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

/// Every recorded whole interaction of this family.
const RECORDED_INTERACTIONS: &[&str] = &[
    "interactions_api/basic_interaction_returns_id",
    "interactions_api/code_execution_usage",
    "interactions_api/followup_with_previous_interaction_id",
    "interactions_api/google_search_tool_interaction",
    "interactions_api/tool_result_roundtrip",
    "interactions_raw_capture_matrix/raw_exposes_lifecycle_fields",
    "interactions_raw_capture_matrix/raw_roundtrips_interaction",
];

/// `document`, a whole interaction, restated as the events a stream of it
/// carries: each step starts bare, its content arrives as deltas, and it
/// stops.
fn restated(document: &serde_json::Value) -> Vec<rig_core::wire::WireFrame> {
    use serde_json::{Value, json};
    let frame = |value: Value| rig_core::wire::WireFrame::Text(value.to_string());
    let mut envelope = document.clone();
    let steps = envelope
        .as_object_mut()
        .and_then(|document| document.shift_remove("steps"))
        .unwrap_or_else(|| json!([]));
    let mut frames = Vec::new();
    for (index, step) in steps.as_array().into_iter().flatten().enumerate() {
        let mut head = step.as_object().cloned().unwrap_or_default();
        let deltas: Vec<Value> = match step["type"].as_str() {
            Some("thought") => {
                let summary = head.shift_remove("summary").unwrap_or_else(|| json!([]));
                let signature = head.shift_remove("signature");
                summary
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|content| json!({"type": "thought_summary", "content": content}))
                    .chain(signature.map(
                        |signature| json!({"type": "thought_signature", "signature": signature}),
                    ))
                    .collect()
            }
            Some("model_output") => head
                .shift_remove("content")
                .and_then(|content| content.as_array().cloned())
                .unwrap_or_default(),
            Some("function_call") => {
                let arguments = head.insert("arguments".to_owned(), json!({}));
                arguments
                    .map(|arguments| {
                        json!({"type": "arguments_delta", "arguments": arguments.to_string()})
                    })
                    .into_iter()
                    .collect()
            }
            _ => {
                head.retain(|key, _| key == "type");
                vec![step.clone()]
            }
        };
        frames.push(frame(
            json!({"event_type": "step.start", "index": index, "step": head}),
        ));
        for delta in deltas {
            frames.push(frame(
                json!({"event_type": "step.delta", "index": index, "delta": delta}),
            ));
        }
        frames.push(frame(json!({"event_type": "step.stop", "index": index})));
    }
    frames.push(frame(
        json!({"event_type": "interaction.completed", "interaction": envelope}),
    ));
    frames
}

/// Each recorded interaction decodes to the same turn whole and restated
/// as a stream, and the same model gets its steps back verbatim, hosted
/// search and code execution included.
#[test]
fn recorded_interactions_agree_in_both_modes_and_replay_verbatim() {
    use rig_core::wire::{Mode, Operation as _, Wire};

    let wire = rig::providers::gemini::interactions_api::Interactions::new(
        rig_core::providers::gemini::GeminiConfig::new("test-key"),
        "gemini-3-flash-preview",
    );
    let mut turns = 0;
    for scenario in RECORDED_INTERACTIONS {
        for (_, document) in crate::cassettes::recorded_json_turns("gemini", scenario) {
            turns += 1;
            let whole = [rig_core::wire::WireFrame::Text(document.to_string())];
            rig_core::test_utils::history::assert_restated_agrees(
                &wire,
                whole.clone(),
                restated(&document),
            );
            let response = rig_core::test_utils::history::decode(&wire, Mode::Unary, whole)
                .unwrap_or_else(|error| panic!("{scenario} decodes: {error}"));
            // The request declares the tools the turn called, so its calls
            // go back as calls.
            let tools = response
                .tool_calls()
                .map(|call| rig_core::completion::ToolDefinition {
                    name: call.function.name.clone(),
                    description: String::new(),
                    parameters: serde_json::json!({"type": "object", "properties": {}}),
                })
                .collect();
            let history = vec![
                Message::user("again"),
                response.message().expect("the reply is a turn"),
            ];
            let request = rig_core::operation::Completion::prepare(
                CompletionRequest::from(history).tools(tools),
                &wire.describe(),
            )
            .expect("the request is valid");
            let encoded = wire.encode(request, Mode::Unary).expect("it encodes");
            let rig_core::wire::Body::Bytes(body) = encoded.request.body() else {
                panic!("the request has a body");
            };
            let body: serde_json::Value = serde_json::from_slice(&body[..]).expect("a JSON body");
            let sent: Vec<_> = body["input"]
                .as_array()
                .into_iter()
                .flatten()
                .skip(1)
                .filter(|step| step["type"] != "function_result")
                .cloned()
                .collect();
            assert_eq!(
                Some(&sent),
                document["steps"].as_array(),
                "{scenario}: the same model gets its steps back verbatim"
            );
        }
    }
    assert_eq!(turns, 9, "every recorded turn was checked");
}

/// The recorded Google Search interaction cites its answer: each
/// `url_citation` covers whole bullet lines when its offsets count bytes,
/// as the API documents, and the en dash before them would shift them by
/// two had they counted characters. A stream that sends the annotations
/// after the text, as `text_annotation_delta`, cites the same and folds
/// into the same turn.
#[test]
fn the_recorded_search_interaction_cites_its_answer_in_both_modes() {
    use rig_core::wire::{Mode, WireFrame};
    use serde_json::json;

    let wire = rig::providers::gemini::interactions_api::Interactions::new(
        rig_core::providers::gemini::GeminiConfig::new("test-key"),
        "gemini-3-flash-preview",
    );
    let [(_, document)] = &crate::cassettes::recorded_json_turns(
        "gemini",
        "interactions_api/google_search_tool_interaction",
    )[..] else {
        panic!("the scenario records one turn");
    };
    // The same stream as `restated`, with each text item's annotations in
    // a delta of their own after its text.
    let streamed: Vec<WireFrame> = restated(document)
        .into_iter()
        .flat_map(|frame| {
            let WireFrame::Text(text) = &frame else {
                return vec![frame];
            };
            let mut event: serde_json::Value = serde_json::from_str(text).expect("JSON");
            let annotations = event["delta"]
                .as_object_mut()
                .filter(|delta| delta.get("type") == Some(&json!("text")))
                .and_then(|delta| delta.shift_remove("annotations"));
            let Some(annotations) = annotations else {
                return vec![frame];
            };
            let annotation = json!({
                "event_type": "step.delta",
                "index": event["index"],
                "delta": {"type": "text_annotation_delta", "annotations": annotations},
            });
            vec![
                WireFrame::Text(event.to_string()),
                WireFrame::Text(annotation.to_string()),
            ]
        })
        .collect();
    let whole = [WireFrame::Text(document.to_string())];
    rig_core::test_utils::history::assert_restated_agrees(&wire, whole.clone(), streamed.clone());
    for (mode, frames) in [(Mode::Unary, whole.to_vec()), (Mode::Streaming, streamed)] {
        let response = rig_core::test_utils::history::decode(&wire, mode, frames)
            .unwrap_or_else(|error| panic!("{mode:?} decodes: {error}"));
        let text = response
            .choice
            .iter()
            .find_map(|block| match block {
                AssistantContent::Text(text) if !text.citations().is_empty() => Some(text),
                _ => None,
            })
            .expect("a cited answer");
        let cited: Vec<_> = text
            .citations()
            .iter()
            .map(|citation| {
                let cited = text.cited(citation).expect("a span");
                let [source] = &citation.sources[..] else {
                    panic!("one source a citation");
                };
                let rig::message::SourceLocation::Url { url } = &source.location else {
                    panic!("a web source: {source:?}");
                };
                assert!(url.starts_with("https://vertexaisearch.cloud.google.com/"));
                (cited, source.title.as_deref())
            })
            .collect();
        let record = "*   **Record-Breaking Title:** This was Spain's fourth European Championship title (following wins in 1964, 2008, and 2012), making them the most successful team in the tournament's history.";
        let run = "*   **Perfect Run:** Spain became the first team to win all seven matches in a single European Championship tournament.";
        assert_eq!(
            cited,
            [
                (record, Some("wikipedia.org")),
                (run, Some("youtube.com")),
                (run, Some("youtube.com")),
                (run, Some("wikipedia.org")),
            ],
            "{mode:?}"
        );
    }
}
