//! Shared fixtures, tiny tools, and durable assertions for ignored smoke tests.
#![allow(dead_code)]

use futures::StreamExt;
use rig_core::{
    agent::{MultiTurnStreamItem, StreamingError, StreamingResult},
    completion::{AssistantContent, GetTokenUsage, ToolDefinition},
    embeddings::Embedding,
    streaming::{StreamedAssistantContent, StreamedUserContent, StreamingCompletionResponse},
    tool::Tool,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

pub(crate) const BASIC_PREAMBLE: &str = "You are a concise assistant. Answer directly.";
pub(crate) const BASIC_PROMPT: &str = "In one or two sentences, explain what Rust programming language is and why memory safety matters.";
pub(crate) const RAW_TEXT_RESPONSE_PREAMBLE: &str =
    "Return exactly the requested text as plain text with no bullets, quotes, or extra commentary.";
pub(crate) const RAW_TEXT_RESPONSE_PROMPT: &str =
    "Reply with exactly two short lines and nothing else. First line: cedar. Second line: maple.";

pub(crate) const CONTEXT_DOCS: [&str; 3] = [
    "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets.",
    "Definition of a *glarb-glarb*: A glarb-glarb is an ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.",
    "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.",
];
pub(crate) const CONTEXT_PROMPT: &str = "What does \"glarb-glarb\" mean?";

pub(crate) const TOOLS_PREAMBLE: &str = "You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.";
pub(crate) const TOOLS_PROMPT: &str = "Calculate 2 - 5.";

pub(crate) const LOADERS_GLOB: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/loaders/*.rs");
pub(crate) const LOADERS_PROMPT: &str = "Which fixture file builds an agent from the loaders test fixtures? Answer with just the file name.";

pub(crate) const STREAMING_PREAMBLE: &str =
    "You are a concise assistant. Answer directly in plain text.";
pub(crate) const STREAMING_PROMPT: &str =
    "In one short paragraph, explain what a solar eclipse is.";

pub(crate) const STREAMING_TOOLS_PREAMBLE: &str =
    "You are a calculator. Use the provided tools before answering arithmetic questions.";
pub(crate) const STREAMING_TOOLS_PROMPT: &str = "Calculate 2 - 5.";
pub(crate) const TWO_TOOL_STREAM_PREAMBLE: &str = "\
You are a precise assistant. When tools are available, you must use them instead of guessing. \
Call both `lookup_harbor_label` and `lookup_orchard_label` before writing any normal text. \
Never call the same tool twice once you already have its result.";
pub(crate) const TWO_TOOL_STREAM_PROMPT: &str = "\
Call `lookup_harbor_label` and `lookup_orchard_label` exactly once each before answering. \
After both tool results are available, stop calling tools and respond in one short sentence that includes both exact tool outputs.";
pub(crate) const ORDERED_TOOL_STREAM_PREAMBLE: &str = "\
You must call the requested tool before writing any normal text. \
After the tool result is available, do not call any more tools and answer in one short sentence that includes the exact tool output.";
pub(crate) const ORDERED_TOOL_STREAM_PROMPT: &str = "\
Call `lookup_harbor_label` exactly once before answering. \
After the tool result is available, answer in one short sentence that includes the exact tool output.";
pub(crate) const REQUIRED_ZERO_ARG_TOOL_PROMPT: &str =
    "Call the ping tool with no arguments. Do not answer with normal text before the tool call.";
pub(crate) const MULTI_TURN_STREAMING_PROMPT: &str =
    "Calculate ((10 - 4) * (3 + 5)) / 3 and describe the result in one short paragraph.";
pub(crate) const MULTI_TURN_STREAMING_EXPECTED_RESULT: i32 = 16;
pub(crate) const ALPHA_SIGNAL_OUTPUT: &str = "crimson-harbor";
pub(crate) const BETA_SIGNAL_OUTPUT: &str = "silver-orchard";

pub(crate) const STRUCTURED_OUTPUT_PROMPT: &str =
    "Return a concise event object for a local Rust meetup in Seattle.";

pub(crate) const EXTRACTOR_TEXT: &str =
    "Hello, my name is Ada Lovelace and I work as a mathematician.";

pub(crate) const IMAGE_PROMPT: &str =
    "A lighthouse on a rocky cliff at sunrise, painted in a clean illustrative style.";

pub(crate) const AUDIO_TEXT: &str = "The quick brown fox jumps over the lazy dog.";
pub(crate) const AUDIO_FIXTURE_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/data/en-us-natural-speech.mp3"
);
pub(crate) const IMAGE_FIXTURE_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/data/camponotus_flavomarginatus_ant.jpg"
);
pub(crate) const PDF_FIXTURE_PATH: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/pages.pdf");

pub(crate) const EMBEDDING_INPUTS: [&str; 3] = [
    "Rust values memory safety and predictable performance.",
    "Streaming responses arrive incrementally instead of all at once.",
    "Embeddings turn text into numeric vectors for similarity search.",
];

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
pub(crate) struct SmokeStructuredOutput {
    pub(crate) title: String,
    pub(crate) category: String,
    pub(crate) summary: String,
}

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
pub(crate) struct SmokePerson {
    #[schemars(required)]
    pub(crate) first_name: Option<String>,
    #[schemars(required)]
    pub(crate) last_name: Option<String>,
    #[schemars(required)]
    pub(crate) job: Option<String>,
}

#[derive(Deserialize)]
pub(crate) struct OperationArgs {
    pub(crate) x: i32,
    pub(crate) y: i32,
}

#[derive(Deserialize)]
pub(crate) struct EmptyArgs {}

#[derive(Debug, thiserror::Error)]
#[error("Math error")]
pub(crate) struct MathError;

#[derive(Deserialize, Serialize)]
pub(crate) struct Adder;

impl Tool for Adder {
    const NAME: &'static str = "add";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "add".to_string(),
            description: "Add x and y together".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "The first number to add"
                    },
                    "y": {
                        "type": "number",
                        "description": "The second number to add"
                    }
                },
                "required": ["x", "y"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.x + args.y)
    }
}

#[derive(Deserialize, Serialize)]
pub(crate) struct Subtract;

impl Tool for Subtract {
    const NAME: &'static str = "subtract";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "subtract".to_string(),
            description: "Subtract y from x (i.e.: x - y)".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "The number to subtract from"
                    },
                    "y": {
                        "type": "number",
                        "description": "The number to subtract"
                    }
                },
                "required": ["x", "y"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.x - args.y)
    }
}

#[derive(Deserialize, Serialize)]
pub(crate) struct AlphaSignal;

impl Tool for AlphaSignal {
    const NAME: &'static str = "lookup_harbor_label";
    type Error = MathError;
    type Args = EmptyArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Return the alpha signal marker.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {},
                "required": [],
            }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(ALPHA_SIGNAL_OUTPUT.to_string())
    }
}

#[derive(Deserialize, Serialize)]
pub(crate) struct BetaSignal;

impl Tool for BetaSignal {
    const NAME: &'static str = "lookup_orchard_label";
    type Error = MathError;
    type Args = EmptyArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Return the beta signal marker.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {},
                "required": [],
            }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(BETA_SIGNAL_OUTPUT.to_string())
    }
}

pub(crate) fn zero_arg_tool_definition(name: &str) -> ToolDefinition {
    ToolDefinition {
        name: name.to_owned(),
        description: format!("A zero-argument tool named {name}."),
        parameters: json!({
            "type": "object",
            "properties": {},
            "required": [],
        }),
    }
}

pub(crate) fn assert_nonempty_response(response: &str) {
    let trimmed = response.trim();

    assert!(
        !trimmed.is_empty(),
        "Response was empty or whitespace-only."
    );
}

pub(crate) fn assistant_text_response(
    choice: &rig_core::OneOrMany<AssistantContent>,
) -> Option<String> {
    let response = choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");

    if response.is_empty() {
        None
    } else {
        Some(response)
    }
}

pub(crate) fn assert_contains_any_case_insensitive(response: &str, expected: &[&str]) {
    assert_nonempty_response(response);

    let response_lower = response.to_ascii_lowercase();
    let matched = expected
        .iter()
        .any(|needle| response_lower.contains(&needle.to_ascii_lowercase()));

    assert!(
        matched,
        "Response {:?} did not contain any of {:?}.",
        response, expected
    );
}

pub(crate) fn assert_contains_all_case_insensitive(response: &str, expected: &[&str]) {
    assert_nonempty_response(response);

    let response_lower = response.to_ascii_lowercase();
    let missing: Vec<&str> = expected
        .iter()
        .copied()
        .filter(|needle| !response_lower.contains(&needle.to_ascii_lowercase()))
        .collect();

    assert!(
        missing.is_empty(),
        "Response {:?} did not contain all of {:?}; missing {:?}.",
        response,
        expected,
        missing
    );
}

pub(crate) fn assert_mentions_expected_number(response: &str, expected: i32) {
    assert_nonempty_response(response);

    let response_lower = response.to_ascii_lowercase();
    let abs = expected.abs();
    let mut candidates = vec![expected.to_string()];

    if expected < 0 {
        candidates.push(format!("minus {abs}"));
        candidates.push(format!("negative {abs}"));
    }

    let matched = candidates
        .iter()
        .any(|candidate| response_lower.contains(&candidate.to_ascii_lowercase()));

    assert!(
        matched,
        "Response {:?} did not mention the expected number {:?}.",
        response, expected
    );
}

pub(crate) fn assert_weather_tool_roundtrip_response(
    city: &str,
    weather: &str,
    expected_city: &str,
) {
    assert_nonempty_response(city);
    assert_nonempty_response(weather);

    assert_eq!(
        city.trim().to_ascii_lowercase(),
        expected_city.trim().to_ascii_lowercase(),
        "expected city {:?}, got {:?}",
        expected_city,
        city
    );

    assert!(
        weather.to_ascii_lowercase().contains("fire and brimstone"),
        "expected the weather description to preserve the tool result, got {:?}",
        weather
    );
}

pub(crate) fn assert_nonempty_bytes(bytes: &[u8]) {
    assert!(!bytes.is_empty(), "Expected non-empty bytes.");
}

pub(crate) fn assert_embeddings_nonempty_and_consistent(
    embeddings: &[Embedding],
    expected_count: usize,
) {
    assert_eq!(
        embeddings.len(),
        expected_count,
        "Expected {expected_count} embeddings but received {}.",
        embeddings.len()
    );

    let mut expected_dims = None;

    for embedding in embeddings {
        assert!(
            !embedding.vec.is_empty(),
            "Embedding for {:?} was empty.",
            embedding.document
        );

        let dims = embedding.vec.len();
        match expected_dims {
            Some(previous_dims) => assert_eq!(
                dims, previous_dims,
                "Expected consistent embedding dimensionality."
            ),
            None => expected_dims = Some(dims),
        }
    }
}

pub(crate) async fn collect_stream_final_response<R>(
    stream: &mut StreamingResult<R>,
) -> Result<String, StreamingError> {
    let mut final_response = None;

    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::FinalResponse(response) = item? {
            final_response = Some(response.response().to_owned());
        }
    }

    Ok(final_response.expect("stream should yield a final response"))
}

pub(crate) async fn assert_stream_contains_zero_arg_tool_call_named<R>(
    mut stream: StreamingCompletionResponse<R>,
    expected_name: &str,
    expect_final_response: bool,
) where
    R: Clone + Unpin + GetTokenUsage,
{
    let mut saw_final = false;
    let mut saw_matching_tool_call = false;

    while let Some(chunk) = stream.next().await {
        match chunk.expect("stream item should be ok") {
            StreamedAssistantContent::Final(_) => saw_final = true,
            StreamedAssistantContent::ToolCall { tool_call, .. } => {
                if tool_call.function.name == expected_name {
                    assert_eq!(tool_call.function.arguments, json!({}));
                    saw_matching_tool_call = true;
                }
            }
            _ => {}
        }
    }

    if expect_final_response {
        assert!(saw_final, "stream should still yield a final response");
    }

    assert!(
        saw_matching_tool_call,
        "expected stream to emit a zero-argument tool call named {expected_name}"
    );
}

pub(crate) struct StreamObservation {
    pub(crate) all_streamed_text: String,
    pub(crate) final_turn_text: String,
    pub(crate) final_response_text: Option<String>,
    pub(crate) tool_calls: Vec<String>,
    pub(crate) tool_call_records: Vec<ToolCallRecord>,
    pub(crate) tool_results: usize,
    pub(crate) errors: Vec<String>,
    pub(crate) got_final_response: bool,
    pub(crate) events: Vec<&'static str>,
}

impl StreamObservation {
    fn new() -> Self {
        Self {
            all_streamed_text: String::new(),
            final_turn_text: String::new(),
            final_response_text: None,
            tool_calls: Vec::new(),
            tool_call_records: Vec::new(),
            tool_results: 0,
            errors: Vec::new(),
            got_final_response: false,
            events: Vec::new(),
        }
    }
}

pub(crate) struct ToolCallRecord {
    pub(crate) name: String,
    pub(crate) signature: Option<String>,
    pub(crate) additional_params: Option<serde_json::Value>,
}

pub(crate) struct RawStreamObservation {
    pub(crate) text: String,
    pub(crate) tool_calls: Vec<rig_core::message::ToolCall>,
    pub(crate) tool_call_records: Vec<ToolCallRecord>,
    pub(crate) errors: Vec<String>,
    pub(crate) got_final: bool,
    pub(crate) events: Vec<&'static str>,
}

impl RawStreamObservation {
    fn new() -> Self {
        Self {
            text: String::new(),
            tool_calls: Vec::new(),
            tool_call_records: Vec::new(),
            errors: Vec::new(),
            got_final: false,
            events: Vec::new(),
        }
    }
}

pub(crate) async fn collect_stream_observation<R>(
    stream: &mut StreamingResult<R>,
) -> StreamObservation {
    let mut observation = StreamObservation::new();

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(content)) => match content {
                StreamedAssistantContent::Text(text) => {
                    observation.all_streamed_text.push_str(&text.text);
                    observation.final_turn_text.push_str(&text.text);
                    observation.events.push("text");
                }
                StreamedAssistantContent::ToolCall { tool_call, .. } => {
                    observation.tool_calls.push(tool_call.function.name.clone());
                    observation.tool_call_records.push(ToolCallRecord {
                        name: tool_call.function.name,
                        signature: tool_call.signature,
                        additional_params: tool_call.additional_params,
                    });
                    observation.events.push("tool_call");
                }
                StreamedAssistantContent::ToolCallDelta { .. } => {
                    observation.events.push("tool_call_delta");
                }
                StreamedAssistantContent::Reasoning(_) => {
                    observation.events.push("reasoning");
                }
                StreamedAssistantContent::ReasoningDelta { .. } => {
                    observation.events.push("reasoning_delta");
                }
                StreamedAssistantContent::Final(_) => {
                    observation.events.push("stream_final");
                }
            },
            Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult { .. })) => {
                observation.tool_results += 1;
                observation.final_turn_text.clear();
                observation.events.push("tool_result");
            }
            Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                observation.final_response_text = Some(response.response().to_owned());
                observation.got_final_response = true;
                observation.events.push("final_response");
            }
            Ok(_) => {}
            Err(error) => {
                observation.errors.push(error.to_string());
                observation.events.push("error");
            }
        }
    }

    observation
}

pub(crate) async fn collect_raw_stream_observation<R>(
    mut stream: StreamingCompletionResponse<R>,
) -> RawStreamObservation
where
    R: Clone + Unpin + GetTokenUsage,
{
    let mut observation = RawStreamObservation::new();

    while let Some(item) = stream.next().await {
        match item {
            Ok(StreamedAssistantContent::Text(text)) => {
                observation.text.push_str(&text.text);
                observation.events.push("text");
            }
            Ok(StreamedAssistantContent::ToolCall { tool_call, .. }) => {
                observation.tool_calls.push(tool_call.clone());
                observation.tool_call_records.push(ToolCallRecord {
                    name: tool_call.function.name,
                    signature: tool_call.signature,
                    additional_params: tool_call.additional_params,
                });
                observation.events.push("tool_call");
            }
            Ok(StreamedAssistantContent::ToolCallDelta { .. }) => {
                observation.events.push("tool_call_delta");
            }
            Ok(StreamedAssistantContent::Reasoning(_)) => {
                observation.events.push("reasoning");
            }
            Ok(StreamedAssistantContent::ReasoningDelta { .. }) => {
                observation.events.push("reasoning_delta");
            }
            Ok(StreamedAssistantContent::Final(_)) => {
                observation.got_final = true;
                observation.events.push("final");
            }
            Err(error) => {
                observation.errors.push(error.to_string());
                observation.events.push("error");
            }
        }
    }

    observation
}

fn first_event_index(events: &[&'static str], expected: &'static str) -> Option<usize> {
    events.iter().position(|event| *event == expected)
}

fn event_count_before(events: &[&'static str], expected: &'static str, end_index: usize) -> usize {
    events
        .iter()
        .take(end_index)
        .filter(|event| **event == expected)
        .count()
}

fn first_unique_tool_calls(tool_calls: &[String]) -> Vec<&str> {
    let mut unique = Vec::new();

    for name in tool_calls {
        if !unique.contains(&name.as_str()) {
            unique.push(name.as_str());
        }
    }

    unique
}

pub(crate) fn assert_two_tool_roundtrip_contract(
    observation: &StreamObservation,
    expected_tools: &[&str],
    expected_markers: &[&str],
) {
    assert!(
        observation.errors.is_empty(),
        "stream should not emit errors: {:?}",
        observation.errors
    );
    assert!(
        observation.got_final_response,
        "stream should emit a final response"
    );
    assert_eq!(
        observation.final_response_text.as_deref(),
        Some(observation.final_turn_text.as_str()),
        "FinalResponse.response() should match the final turn's streamed text"
    );
    assert!(
        observation.tool_results >= expected_tools.len(),
        "expected at least {} tool-result events, got {}",
        expected_tools.len(),
        observation.tool_results
    );

    let first_text = first_event_index(&observation.events, "text")
        .expect("stream should emit final text after the tool roundtrip");
    let tool_calls_before_text = event_count_before(&observation.events, "tool_call", first_text);
    let tool_results_before_text =
        event_count_before(&observation.events, "tool_result", first_text);

    assert!(
        tool_calls_before_text >= expected_tools.len(),
        "expected at least {} tool-call events before the first text chunk, got {}. Events: {:?}",
        expected_tools.len(),
        tool_calls_before_text,
        observation.events
    );
    assert!(
        tool_results_before_text >= expected_tools.len(),
        "expected at least {} tool-result events before the first text chunk, got {}. Events: {:?}",
        expected_tools.len(),
        tool_results_before_text,
        observation.events
    );

    for expected_tool in expected_tools {
        assert!(
            observation
                .tool_calls
                .iter()
                .any(|name| name == expected_tool),
            "expected tool call for {expected_tool}, saw {:?}",
            observation.tool_calls
        );
    }

    let first_unique = first_unique_tool_calls(&observation.tool_calls);
    assert!(
        first_unique.len() >= expected_tools.len(),
        "expected at least {} unique tool calls, saw {:?}",
        expected_tools.len(),
        observation.tool_calls
    );

    for expected_tool in expected_tools {
        assert!(
            first_unique
                .iter()
                .take(expected_tools.len())
                .any(|name| name == expected_tool),
            "expected the initial unique tool-call phase to include {expected_tool}, saw {:?}",
            first_unique
        );
    }

    let response = observation
        .final_response_text
        .as_deref()
        .expect("stream should produce a final response string");
    assert_contains_all_case_insensitive(response, expected_markers);
}

pub(crate) fn assert_tool_call_precedes_later_text(
    observation: &StreamObservation,
    expected_tool: &str,
    expected_markers: &[&str],
) {
    assert!(
        observation.errors.is_empty(),
        "stream should not emit errors: {:?}",
        observation.errors
    );
    assert!(
        observation.got_final_response,
        "stream should emit a final response"
    );
    assert_eq!(
        observation.final_response_text.as_deref(),
        Some(observation.final_turn_text.as_str()),
        "FinalResponse.response() should match the final turn's streamed text"
    );
    assert!(
        observation
            .tool_calls
            .iter()
            .any(|name| name == expected_tool),
        "expected tool call for {expected_tool}, saw {:?}",
        observation.tool_calls
    );
    assert!(
        observation.tool_results >= 1,
        "expected at least one tool-result event, got {}",
        observation.tool_results
    );

    let first_tool_call = first_event_index(&observation.events, "tool_call")
        .expect("stream should emit a tool call event");
    let first_tool_result = first_event_index(&observation.events, "tool_result")
        .expect("stream should emit a tool result event");
    let first_text = first_event_index(&observation.events, "text")
        .expect("stream should emit text after tools");

    assert!(
        first_tool_call < first_text,
        "expected a tool call before later text, saw events {:?}",
        observation.events
    );
    assert!(
        first_tool_result < first_text,
        "expected a tool result before later text, saw events {:?}",
        observation.events
    );

    let response = observation
        .final_response_text
        .as_deref()
        .expect("stream should produce a final response string");
    assert_contains_all_case_insensitive(response, expected_markers);
}

pub(crate) fn assert_raw_stream_tool_call_precedes_text(
    observation: &RawStreamObservation,
    expected_tool: &str,
) {
    assert!(
        observation.errors.is_empty(),
        "raw stream should not emit errors: {:?}",
        observation.errors
    );
    assert!(
        observation.got_final,
        "raw stream should emit a final response"
    );

    let record = observation
        .tool_call_records
        .iter()
        .find(|record| record.name == expected_tool)
        .unwrap_or_else(|| {
            panic!(
                "expected raw stream tool call for {expected_tool}, saw {:?}",
                observation
                    .tool_call_records
                    .iter()
                    .map(|record| record.name.as_str())
                    .collect::<Vec<_>>()
            )
        });

    assert!(
        first_event_index(&observation.events, "tool_call").is_some(),
        "expected a tool_call event for {expected_tool}, saw {:?}",
        observation.events
    );

    if let Some(first_text) = first_event_index(&observation.events, "text") {
        let first_tool_call = first_event_index(&observation.events, "tool_call")
            .expect("raw stream should emit a tool_call event");
        assert!(
            first_tool_call < first_text,
            "expected the raw stream to emit a tool call before any text, saw events {:?}",
            observation.events
        );
    }

    let _ = record;
}

pub(crate) fn assert_raw_stream_contains_distinct_tool_calls_before_text(
    observation: &RawStreamObservation,
    expected_tools: &[&str],
) {
    assert!(
        observation.errors.is_empty(),
        "raw stream should not emit errors: {:?}",
        observation.errors
    );
    assert!(
        observation.got_final,
        "raw stream should emit a final response"
    );
    assert!(
        !observation.tool_calls.is_empty(),
        "raw stream should emit tool calls"
    );

    let tool_call_names = observation
        .tool_calls
        .iter()
        .map(|tool_call| tool_call.function.name.clone())
        .collect::<Vec<_>>();

    for expected_tool in expected_tools {
        assert!(
            tool_call_names.iter().any(|name| name == expected_tool),
            "expected raw stream tool call for {expected_tool}, saw {:?}",
            tool_call_names
        );
    }

    let first_unique = first_unique_tool_calls(&tool_call_names);
    assert!(
        first_unique.len() >= expected_tools.len(),
        "expected at least {} unique raw stream tool calls, saw {:?}",
        expected_tools.len(),
        tool_call_names
    );

    for expected_tool in expected_tools {
        assert!(
            first_unique
                .iter()
                .take(expected_tools.len())
                .any(|name| name == expected_tool),
            "expected the initial unique raw tool-call phase to include {expected_tool}, saw {:?}",
            first_unique
        );
    }

    if let Some(first_text) = first_event_index(&observation.events, "text") {
        let tool_calls_before_text =
            event_count_before(&observation.events, "tool_call", first_text);

        assert!(
            tool_calls_before_text >= expected_tools.len(),
            "expected at least {} raw tool-call events before the first text chunk, got {}. Events: {:?}",
            expected_tools.len(),
            tool_calls_before_text,
            observation.events
        );
    }
}

pub(crate) fn assert_raw_stream_text_contains(
    observation: &RawStreamObservation,
    expected: &[&str],
) {
    assert!(
        observation.errors.is_empty(),
        "raw stream should not emit errors: {:?}",
        observation.errors
    );
    assert!(
        observation.got_final,
        "raw stream should emit a final response"
    );
    assert_contains_all_case_insensitive(&observation.text, expected);
}

pub(crate) fn assert_loader_answer_is_relevant(response: &str) {
    assert_contains_any_case_insensitive(
        response,
        &[
            "agent_with_loaders",
            "agent_with_loaders.rs",
            "agent with loaders",
        ],
    );
}

pub(crate) fn assert_smoke_structured_output(output: &SmokeStructuredOutput) {
    assert_nonempty_response(&output.title);
    assert_nonempty_response(&output.category);
    assert_nonempty_response(&output.summary);
}
