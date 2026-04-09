//! Shared fixtures, tiny tools, and durable assertions for ignored smoke tests.
#![allow(dead_code)]

use futures::StreamExt;
use rig::{
    agent::{MultiTurnStreamItem, StreamingError, StreamingResult},
    completion::ToolDefinition,
    embeddings::Embedding,
    streaming::{StreamedAssistantContent, StreamedUserContent},
    tool::Tool,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

pub(crate) const BASIC_PREAMBLE: &str = "You are a concise assistant. Answer directly.";
pub(crate) const BASIC_PROMPT: &str = "In one or two sentences, explain what Rust programming language is and why memory safety matters.";

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
pub(crate) const MULTI_TURN_STREAMING_PROMPT: &str =
    "Calculate ((10 - 4) * (3 + 5)) / 3 and describe the result in one short paragraph.";
pub(crate) const MULTI_TURN_STREAMING_EXPECTED_RESULT: i32 = 16;

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

pub(crate) fn assert_nonempty_response(response: &str) {
    let trimmed = response.trim();

    assert!(
        !trimmed.is_empty(),
        "Response was empty or whitespace-only."
    );
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

pub(crate) struct StreamObservation {
    pub(crate) all_streamed_text: String,
    pub(crate) final_turn_text: String,
    pub(crate) final_response_text: Option<String>,
    pub(crate) tool_calls: Vec<String>,
    pub(crate) tool_results: usize,
    pub(crate) errors: Vec<String>,
    pub(crate) got_final_response: bool,
}

impl StreamObservation {
    fn new() -> Self {
        Self {
            all_streamed_text: String::new(),
            final_turn_text: String::new(),
            final_response_text: None,
            tool_calls: Vec::new(),
            tool_results: 0,
            errors: Vec::new(),
            got_final_response: false,
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
                }
                StreamedAssistantContent::ToolCall { tool_call, .. } => {
                    observation.tool_calls.push(tool_call.function.name);
                }
                _ => {}
            },
            Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult { .. })) => {
                observation.tool_results += 1;
                observation.final_turn_text.clear();
            }
            Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                observation.final_response_text = Some(response.response().to_owned());
                observation.got_final_response = true;
            }
            Ok(_) => {}
            Err(error) => observation.errors.push(error.to_string()),
        }
    }

    observation
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
