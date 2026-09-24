//! Shared fixtures, tiny tools, and durable assertions for ignored smoke tests.
#![allow(dead_code)]

use futures::StreamExt;

use rig_agent::agent::MultiTurnStreamItem;

use rig_agent::agent::StreamingError;

use rig_agent::agent::StreamingResult;

use rig_agent::completion::AssistantContent;

use rig_agent::completion::ToolDefinition;

use rig_core::embeddings::Embedding;

use rig_core::streaming::Delta;

use rig_core::streaming::StreamEvent;

use rig_core::streaming::StreamedUserContent;

use rig_core::streaming::StreamingCompletionResponse;

use rig_core::tool::PortableTool;

use rig_core::tool::Tool;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};

/// System instruction for the basic completion smoke scenario.
pub const BASIC_PREAMBLE: &str = "You are a concise assistant. Answer directly.";
/// Rust and memory-safety question used by basic completion smoke tests.
pub const BASIC_PROMPT: &str = "In one or two sentences, explain what Rust programming language is and why memory safety matters.";
/// System instruction for inspecting a raw text response.
pub const RAW_TEXT_RESPONSE_PREAMBLE: &str =
    "Return exactly the requested text as plain text with no bullets, quotes, or extra commentary.";
/// Prompt used by raw text-response smoke tests.
pub const RAW_TEXT_RESPONSE_PROMPT: &str =
    "Reply with exactly two short lines and nothing else. First line: cedar. Second line: maple.";

/// Fixed context documents defining the fictional glarb-glarb.
pub const CONTEXT_DOCS: [&str; 3] = [
    "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets.",
    "Definition of a *glarb-glarb*: A glarb-glarb is an ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.",
    "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.",
];
/// Question answered from the fixed context documents.
pub const CONTEXT_PROMPT: &str = "What does \"glarb-glarb\" mean?";

/// System instruction requiring arithmetic tool use.
pub const TOOLS_PREAMBLE: &str = "You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.";
/// Subtraction question used by tool-call smoke tests.
pub const TOOLS_PROMPT: &str = "Calculate 2 - 5.";

/// Manifest-anchored glob for loader fixture source files.
pub const LOADERS_GLOB: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../..",
    "/tests/data/loaders/*.rs"
);
/// Question that identifies the agent-building loader fixture.
pub const LOADERS_PROMPT: &str = "Which fixture file builds an agent from the loaders test fixtures? Answer with just the file name.";

/// Plain-text system instruction used by streaming smoke tests.
pub const STREAMING_PREAMBLE: &str = "You are a concise assistant. Answer directly in plain text.";
/// Solar-eclipse question used by streaming smoke tests.
pub const STREAMING_PROMPT: &str = "In one short paragraph, explain what a solar eclipse is.";

/// System instruction for a streamed arithmetic tool roundtrip.
pub const STREAMING_TOOLS_PREAMBLE: &str =
    "You are a calculator. Use the provided tools before answering arithmetic questions.";
/// Subtraction question used by streamed tool tests.
pub const STREAMING_TOOLS_PROMPT: &str = "Calculate 2 - 5.";
/// System instruction requiring both deterministic signal tools.
pub const TWO_TOOL_STREAM_PREAMBLE: &str = "\
You are a precise assistant. When tools are available, you must use them instead of guessing. \
Call both `lookup_harbor_label` and `lookup_orchard_label` before writing any normal text. \
Never call the same tool twice once you already have its result.";
/// Prompt exercising two signal-tool calls before the final answer.
pub const TWO_TOOL_STREAM_PROMPT: &str = "\
Call `lookup_harbor_label` and `lookup_orchard_label` exactly once each before answering. \
After both tool results are available, stop calling tools and respond in one short sentence that includes both exact tool outputs.";
/// System instruction used to assert tool and text event ordering.
pub const ORDERED_TOOL_STREAM_PREAMBLE: &str = "\
You must call the requested tool before writing any normal text. \
After the tool result is available, do not call any more tools and answer in one short sentence that includes the exact tool output.";
/// Prompt used to assert tool and text event ordering.
pub const ORDERED_TOOL_STREAM_PROMPT: &str = "\
Call `lookup_harbor_label` exactly once before answering. \
After the tool result is available, answer in one short sentence that includes the exact tool output.";
/// Prompt requiring a zero-argument tool invocation.
pub const REQUIRED_ZERO_ARG_TOOL_PROMPT: &str =
    "Call the ping tool with no arguments. Do not answer with normal text before the tool call.";
/// Arithmetic prompt whose tool results feed a later model turn.
pub const MULTI_TURN_STREAMING_PROMPT: &str =
    "Calculate ((10 - 4) * (3 + 5)) / 3 and describe the result in one short paragraph.";
/// Expected arithmetic result of the multi-turn streaming prompt.
pub const MULTI_TURN_STREAMING_EXPECTED_RESULT: i32 = 16;
/// Deterministic result returned by the alpha signal tool.
pub const ALPHA_SIGNAL_OUTPUT: &str = "crimson-harbor";
/// Deterministic result returned by the beta signal tool.
pub const BETA_SIGNAL_OUTPUT: &str = "silver-orchard";

/// Prompt requesting the structured smoke-test response.
pub const STRUCTURED_OUTPUT_PROMPT: &str =
    "Return a concise event object for a local Rust meetup in Seattle.";

/// Source text for the person-extraction smoke test.
pub const EXTRACTOR_TEXT: &str = "Hello, my name is Ada Lovelace and I work as a mathematician.";

/// Question about the committed image fixture.
pub const IMAGE_PROMPT: &str =
    "A lighthouse on a rocky cliff at sunrise, painted in a clean illustrative style.";

/// Fixed text used by audio-generation smoke tests.
pub const AUDIO_TEXT: &str = "The quick brown fox jumps over the lazy dog.";
/// Manifest-anchored path to the committed audio fixture.
pub const AUDIO_FIXTURE_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../..",
    "/tests/data/en-us-natural-speech.mp3"
);
/// Manifest-anchored path to the committed image fixture.
pub const IMAGE_FIXTURE_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../..",
    "/tests/data/camponotus_flavomarginatus_ant.jpg"
);
/// Manifest-anchored path to the committed PDF fixture.
pub const PDF_FIXTURE_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../..",
    "/tests/data/pages.pdf"
);
/// Manifest-anchored path to the committed video fixture.
pub const VIDEO_FIXTURE_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../..",
    "/tests/data/sample_video.mp4"
);

/// Fixed text inputs used to check embedding count and dimensions.
pub const EMBEDDING_INPUTS: [&str; 3] = [
    "Rust values memory safety and predictable performance.",
    "Streaming responses arrive incrementally instead of all at once.",
    "Embeddings turn text into numeric vectors for similarity search.",
];

/// Small structured response shared by provider smoke tests.
#[derive(Debug, Deserialize, JsonSchema, Serialize)]
// Rust documentation must not change the schemas embedded in recorded requests.
#[schemars(description = "")]
pub struct SmokeStructuredOutput {
    /// Title extracted or generated by the model.
    #[schemars(description = "")]
    pub title: String,
    /// Category extracted or generated by the model.
    #[schemars(description = "")]
    pub category: String,
    /// Summary extracted or generated by the model.
    #[schemars(description = "")]
    pub summary: String,
}

/// Return a deterministic example of the structured smoke response.
pub fn smoke_structured_output_value() -> serde_json::Value {
    json!({
        "title": "Seattle Rust Meetup",
        "category": "Technology",
        "summary": "A focused local meetup for Rust developers."
    })
}

/// Derive the synthetic output-tool name from the serialized schema's SHA-256 prefix.
pub fn ecs_synthetic_output_tool_name<T>() -> String
where
    T: JsonSchema,
{
    let schema = schemars::schema_for!(T);
    let mut hasher = Sha256::new();
    hasher.update(schema.as_value().to_string().as_bytes());
    let prefix = hasher
        .finalize()
        .iter()
        .take(4)
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("__rig_output_{prefix}")
}

/// Person fields used to test required extraction properties.
#[derive(Debug, Deserialize, JsonSchema, Serialize)]
#[schemars(description = "")]
pub struct SmokePerson {
    /// Extracted first name.
    #[schemars(required, description = "")]
    pub first_name: Option<String>,
    /// Extracted last name.
    #[schemars(required, description = "")]
    pub last_name: Option<String>,
    /// Extracted occupation.
    #[schemars(required, description = "")]
    pub job: Option<String>,
}

#[cfg(test)]
mod tests;

#[derive(Deserialize)]
/// Integer operands shared by arithmetic tools.
pub struct OperationArgs {
    /// First operand, and minuend for subtraction.
    pub x: i32,
    /// Second operand, and subtrahend for subtraction.
    pub y: i32,
}

#[derive(Deserialize)]
/// Empty object accepted by zero-argument tools.
pub struct EmptyArgs {}

#[derive(Debug, thiserror::Error)]
#[error("Math error")]
/// Error type required by the deterministic arithmetic and signal tools.
pub struct MathError;

#[derive(Deserialize, Serialize)]
/// Tool that adds its two integer arguments.
pub struct Adder;

impl Tool for Adder {
    const NAME: &'static str = "add";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    fn description(&self) -> String {
        "Add x and y together".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::from_str(
                r#"{"type":"object","properties":{"x":{"type":"number","description":"The first number to add"},"y":{"type":"number","description":"The second number to add"}},"required":["x","y"]}"#,
            )
            .expect("adder schema should deserialize")
    }

    async fn call(
        &self,
        _context: &mut rig_core::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(args.x + args.y)
    }
}

#[derive(Deserialize, Serialize)]
/// Tool that subtracts its second integer argument from the first.
pub struct Subtract;

impl Tool for Subtract {
    const NAME: &'static str = "subtract";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    fn description(&self) -> String {
        "Subtract y from x (i.e.: x - y)".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::from_str(
                r#"{"type":"object","properties":{"x":{"type":"number","description":"The number to subtract from"},"y":{"type":"number","description":"The number to subtract"}},"required":["x","y"]}"#,
            )
            .expect("subtract schema should deserialize")
    }

    async fn call(
        &self,
        _context: &mut rig_core::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(args.x - args.y)
    }
}

#[derive(Clone, Copy, Deserialize, Serialize)]
/// Portable arithmetic tool that adds its integer arguments.
pub struct PortableAdder;

impl PortableTool for PortableAdder {
    const NAME: &'static str = "add";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    fn description(&self) -> String {
        "Add x and y together".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "The first number to add"},
                "y": {"type": "number", "description": "The second number to add"}
            },
            "required": ["x", "y"]
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.x + args.y)
    }
}

#[derive(Clone, Copy, Deserialize, Serialize)]
/// Portable arithmetic tool that subtracts its second argument from the first.
pub struct PortableSubtract;

impl PortableTool for PortableSubtract {
    const NAME: &'static str = "subtract";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    fn description(&self) -> String {
        "Subtract y from x (i.e.: x - y)".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "The number to subtract from"},
                "y": {"type": "number", "description": "The number to subtract"}
            },
            "required": ["x", "y"]
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.x - args.y)
    }
}

#[derive(Deserialize, Serialize)]
/// Zero-argument tool returning the fixed alpha signal marker.
pub struct AlphaSignal;

impl Tool for AlphaSignal {
    const NAME: &'static str = "lookup_harbor_label";
    type Error = MathError;
    type Args = EmptyArgs;
    type Output = String;

    fn description(&self) -> String {
        "Return the alpha signal marker.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {},
            "required": [],
        })
    }

    async fn call(
        &self,
        _context: &mut rig_core::tool::ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(ALPHA_SIGNAL_OUTPUT.to_string())
    }
}

#[derive(Deserialize, Serialize)]
/// Zero-argument tool returning the fixed beta signal marker.
pub struct BetaSignal;

impl Tool for BetaSignal {
    const NAME: &'static str = "lookup_orchard_label";
    type Error = MathError;
    type Args = EmptyArgs;
    type Output = String;

    fn description(&self) -> String {
        "Return the beta signal marker.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {},
            "required": [],
        })
    }

    async fn call(
        &self,
        _context: &mut rig_core::tool::ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(BETA_SIGNAL_OUTPUT.to_string())
    }
}

/// Build a named tool definition accepting an empty JSON object.
pub fn zero_arg_tool_definition(name: &str) -> ToolDefinition {
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

/// Assert that a response contains non-whitespace text.
pub fn assert_nonempty_response(response: &str) {
    let trimmed = response.trim();

    assert!(
        !trimmed.is_empty(),
        "Response was empty or whitespace-only."
    );
}

/// Join assistant text blocks with newlines, returning none for empty text.
pub fn assistant_text_response(choice: &[AssistantContent]) -> Option<String> {
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

/// Assert a nonempty response contains at least one expected substring, ignoring ASCII case.
pub fn assert_contains_any_case_insensitive(response: &str, expected: &[&str]) {
    assert_nonempty_response(response);

    let response_lower = response.to_ascii_lowercase();
    let matched = expected
        .iter()
        .any(|needle| response_lower.contains(&needle.to_ascii_lowercase()));

    assert!(
        matched,
        "Response {response:?} did not contain any of {expected:?}."
    );
}

/// Assert a nonempty response contains every expected substring, ignoring ASCII case.
pub fn assert_contains_all_case_insensitive(response: &str, expected: &[&str]) {
    assert_nonempty_response(response);

    let response_lower = response.to_ascii_lowercase();
    let missing: Vec<&str> = expected
        .iter()
        .copied()
        .filter(|needle| !response_lower.contains(&needle.to_ascii_lowercase()))
        .collect();

    assert!(
        missing.is_empty(),
        "Response {response:?} did not contain all of {expected:?}; missing {missing:?}."
    );
}

/// Assert a response mentions the number, accepting minus/negative wording for negatives.
pub fn assert_mentions_expected_number(response: &str, expected: i32) {
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
        "Response {response:?} did not mention the expected number {expected:?}."
    );
}

/// Assert the expected city and the fixed weather-tool result survive the roundtrip.
pub fn assert_weather_tool_roundtrip_response(city: &str, weather: &str, expected_city: &str) {
    assert_nonempty_response(city);
    assert_nonempty_response(weather);

    assert_eq!(
        city.trim().to_ascii_lowercase(),
        expected_city.trim().to_ascii_lowercase(),
        "expected city {expected_city:?}, got {city:?}"
    );

    assert!(
        weather.to_ascii_lowercase().contains("fire and brimstone"),
        "expected the weather description to preserve the tool result, got {weather:?}"
    );
}

/// Assert that a generated byte payload is nonempty.
pub fn assert_nonempty_bytes(bytes: &[u8]) {
    assert!(!bytes.is_empty(), "Expected non-empty bytes.");
}

/// The container an image payload's leading bytes declare.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageContainer {
    Png,
    Jpeg,
    Webp,
    Gif,
}

/// Classify decoded image bytes by their magic number, or `None` when they
/// are not an image at all (a fixture placeholder decodes to `hello`).
pub fn image_container(bytes: &[u8]) -> Option<ImageContainer> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some(ImageContainer::Png)
    } else if bytes.starts_with(b"\xff\xd8\xff") {
        Some(ImageContainer::Jpeg)
    } else if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        Some(ImageContainer::Webp)
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Some(ImageContainer::Gif)
    } else {
        None
    }
}

/// Assert the bytes are a real image: a recognized container with a body
/// behind the header, not a placeholder.
pub fn assert_image_bytes(bytes: &[u8]) -> ImageContainer {
    let container = image_container(bytes).unwrap_or_else(|| {
        panic!(
            "expected image bytes, got {} bytes starting {:?}",
            bytes.len(),
            &bytes[..bytes.len().min(8)]
        )
    });
    assert!(
        bytes.len() > 64,
        "expected an image body behind the {container:?} header, got {} bytes",
        bytes.len()
    );
    container
}

/// Assert the expected embedding count and nonzero, consistent vector dimensions.
pub fn assert_embeddings_nonempty_and_consistent(embeddings: &[Embedding], expected_count: usize) {
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

// ---------------------------------------------------------------------------
// Model-turn termination metadata (rig#2184).
//
// One implementation, driven by every provider suite that covers the feature.
// That is the portability claim as code: if a provider needed its own probe,
// the metadata would not be provider-neutral.
// ---------------------------------------------------------------------------

/// One turn's termination as a hook sees it: why it stopped, and the effective
/// output-token cap that attempt ran under.
pub type ObservedTermination = (Option<rig_agent::completion::FinishReason>, Option<u64>);

/// Records `ModelTurnFinished`'s normalized termination metadata for every
/// accepted turn, naming no provider and touching no raw response type.
#[derive(Clone, Debug, Default)]
pub struct TurnTerminationProbe {
    observations: std::sync::Arc<std::sync::Mutex<Vec<ObservedTermination>>>,
}

impl TurnTerminationProbe {
    /// Return a snapshot of the observed model-turn termination metadata.
    pub fn observations(&self) -> Vec<ObservedTermination> {
        self.observations.lock().expect("observations").clone()
    }

    /// The reason reported for the first accepted turn.
    pub fn first_reason(&self) -> Option<rig_agent::completion::FinishReason> {
        self.observations()
            .first()
            .and_then(|(reason, _)| reason.clone())
    }

    /// The effective cap reported for the first accepted turn.
    pub fn first_max_tokens(&self) -> Option<u64> {
        self.observations().first().and_then(|(_, cap)| *cap)
    }
}

impl rig_agent::agent::AgentHook for TurnTerminationProbe {
    async fn on_model_turn_finished(
        &self,
        _ctx: &rig_agent::agent::HookContext,
        event: rig_agent::agent::ModelTurnFinished<'_>,
    ) -> rig_agent::agent::ModelTurnAction {
        self.observations
            .lock()
            .expect("observations")
            .push((event.finish_reason.cloned(), event.max_tokens));
        rig_agent::agent::ModelTurnAction::continue_run()
    }
}

/// Raises the output-token cap whenever the provider cut a turn short, then
/// retries it — the policy rig#2184 exists to make portable. It reads only
/// `FinishReason` and the reported cap, so the same instance drives every
/// provider.
///
/// Register this *before* any hook that may return a non-continue action:
/// such an action short-circuits the hooks behind it.
#[derive(Clone, Debug)]
pub struct EscalateCapOnTruncation {
    cap: std::sync::Arc<std::sync::atomic::AtomicU64>,
    escalated_to: std::sync::Arc<std::sync::Mutex<Vec<u64>>>,
    grown_cap: u64,
    max_retries: u32,
    retries: std::sync::Arc<std::sync::atomic::AtomicU32>,
}

impl EscalateCapOnTruncation {
    /// Start every attempt at `start_cap`; on a truncated, tool-free turn,
    /// re-run it once at `grown_cap`.
    pub fn new(start_cap: u64, grown_cap: u64) -> Self {
        Self {
            cap: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(start_cap)),
            escalated_to: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            grown_cap,
            max_retries: 1,
            retries: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
        }
    }

    /// The caps this hook asked for, in order — one entry per escalation.
    pub fn escalations(&self) -> Vec<u64> {
        self.escalated_to.lock().expect("escalated_to").clone()
    }

    /// Return the number of retries requested by this hook.
    pub fn retries(&self) -> u32 {
        self.retries.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl rig_agent::agent::AgentHook for EscalateCapOnTruncation {
    /// Every attempt is prepared afresh, so the current cap is applied here and
    /// reported back on that attempt's `ModelTurnFinished`.
    async fn on_completion_call(
        &self,
        _ctx: &rig_agent::agent::HookContext,
        _event: rig_agent::agent::CompletionCallEvent<'_>,
    ) -> rig_agent::agent::CompletionCallAction {
        rig_agent::agent::CompletionCallAction::patch(
            rig_agent::agent::RequestPatch::new()
                .max_tokens(self.cap.load(std::sync::atomic::Ordering::SeqCst)),
        )
    }

    async fn on_model_turn_finished(
        &self,
        _ctx: &rig_agent::agent::HookContext,
        event: rig_agent::agent::ModelTurnFinished<'_>,
    ) -> rig_agent::agent::ModelTurnAction {
        let truncated = event
            .finish_reason
            .is_some_and(rig_agent::completion::FinishReason::truncated_output);
        // Retrying a turn carrying tool calls is rejected, so a policy that may
        // meet one has to check before asking.
        let has_tool_call = event
            .content
            .iter()
            .any(|content| matches!(content, AssistantContent::ToolCall(_)));

        if truncated
            && !has_tool_call
            && self.retries.load(std::sync::atomic::Ordering::SeqCst) < self.max_retries
        {
            self.retries
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.cap
                .store(self.grown_cap, std::sync::atomic::Ordering::SeqCst);
            self.escalated_to
                .lock()
                .expect("escalated_to")
                .push(self.grown_cap);
            return rig_agent::agent::ModelTurnAction::repeat();
        }
        rig_agent::agent::ModelTurnAction::continue_run()
    }
}

/// Drain the stream, propagating errors and requiring an agent final response.
pub async fn collect_stream_final_response(
    stream: &mut StreamingResult,
) -> Result<String, StreamingError> {
    let mut final_response = None;

    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::FinalResponse(response) = item? {
            final_response = Some(response.output().to_owned());
        }
    }

    Ok(final_response.expect("stream should yield a final response"))
}

/// Drain the stream, propagating errors and requiring agent and provider final responses.
pub async fn collect_stream_final_response_and_provider_final(
    stream: &mut StreamingResult,
) -> Result<(String, rig_core::streaming::StreamFinal), StreamingError> {
    let mut final_response = None;
    let mut provider_final = None;

    while let Some(item) = stream.next().await {
        match item? {
            MultiTurnStreamItem::StreamAssistantItem(StreamEvent::Final(final_)) => {
                provider_final = Some(final_);
            }
            MultiTurnStreamItem::FinalResponse(response) => {
                final_response = Some(response.output().to_owned());
            }
            _ => {}
        }
    }

    Ok((
        final_response.expect("stream should yield a final response"),
        provider_final.expect("stream should yield a typed provider final"),
    ))
}

/// Assert a successful raw stream contains the named empty-argument call and, optionally, a final event.
pub async fn assert_stream_contains_zero_arg_tool_call_named(
    mut stream: StreamingCompletionResponse,
    expected_name: &str,
    expect_final_response: bool,
) {
    let mut saw_final = false;
    let mut saw_matching_tool_call = false;

    while let Some(chunk) = stream.next().await {
        match chunk.expect("stream item should be ok") {
            StreamEvent::Final(_) => saw_final = true,
            StreamEvent::BlockEnd {
                block: Some(AssistantContent::ToolCall(tool_call)),
                ..
            } if tool_call.function.name == expected_name => {
                assert_eq!(tool_call.function.arguments, json!({}));
                saw_matching_tool_call = true;
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

/// Text, tool, error, and termination observations from an agent stream.
pub struct StreamObservation {
    /// Text accumulated across all model turns.
    pub all_streamed_text: String,
    /// Text accumulated for the final model turn.
    pub final_turn_text: String,
    /// Output carried by the agent's final response, when present.
    pub final_response_text: Option<String>,
    /// Tool-call names in stream order.
    pub tool_calls: Vec<String>,
    /// Tool-call metadata in stream order.
    pub tool_call_records: Vec<ToolCallRecord>,
    /// Number of tool-result events observed.
    pub tool_results: usize,
    /// Formatted stream errors in observation order.
    pub errors: Vec<String>,
    /// Whether an agent final response was observed.
    pub got_final_response: bool,
    /// Event-kind labels in stream order for ordering assertions.
    pub events: Vec<&'static str>,
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

/// Provider tool-call metadata retained for signature and parameter checks.
pub struct ToolCallRecord {
    /// Tool name emitted by the provider.
    pub name: String,
    /// Provider signature associated with the tool call, when present.
    pub signature: Option<String>,
    /// Additional provider parameters associated with the call, when present.
    pub additional_params: Option<serde_json::Value>,
}

/// Observations from a raw provider completion stream before agent tool execution.
pub struct RawStreamObservation {
    /// Text accumulated from provider text deltas.
    pub text: String,
    /// Completed tool calls in stream order.
    pub tool_calls: Vec<rig_core::message::ToolCall>,
    /// Tool-call metadata in stream order.
    pub tool_call_records: Vec<ToolCallRecord>,
    /// Formatted stream errors in observation order.
    pub errors: Vec<String>,
    /// Whether a provider final event was observed.
    pub got_final: bool,
    /// Event-kind labels in stream order for ordering assertions.
    pub events: Vec<&'static str>,
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

/// Drain an agent stream into text, tool, error, and final-response observations.
pub async fn collect_stream_observation(stream: &mut StreamingResult) -> StreamObservation {
    let mut observation = StreamObservation::new();

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(event)) => match event {
                StreamEvent::BlockDelta {
                    delta: Delta::Text { text },
                    ..
                } => {
                    observation.all_streamed_text.push_str(&text);
                    observation.final_turn_text.push_str(&text);
                    observation.events.push("text");
                }
                StreamEvent::BlockDelta {
                    delta: Delta::ToolName { .. } | Delta::ToolArguments { .. },
                    ..
                } => {
                    observation.events.push("tool_call_delta");
                }
                StreamEvent::BlockEnd {
                    block: Some(AssistantContent::Reasoning(_)),
                    ..
                } => {
                    observation.events.push("reasoning");
                }
                StreamEvent::BlockDelta {
                    delta: Delta::Reasoning { .. },
                    ..
                } => {
                    observation.events.push("reasoning_delta");
                }
                StreamEvent::Final(_) => {
                    observation.events.push("stream_final");
                }
                StreamEvent::Unknown(_) => {
                    observation.events.push("unknown");
                }
                StreamEvent::BlockStart { .. }
                | StreamEvent::BlockDelta {
                    delta: Delta::TextMeta { .. },
                    ..
                }
                | StreamEvent::BlockEnd { .. } => {}
            },
            // The engine reports the model's completed calls itself once the
            // turn commits; the provider-level `BlockEnd` is not forwarded.
            Ok(MultiTurnStreamItem::ToolCall { tool_call, .. }) => {
                observation.tool_calls.push(tool_call.function.name.clone());
                observation.tool_call_records.push(ToolCallRecord {
                    name: tool_call.function.name,
                    signature: tool_call.signature,
                    additional_params: tool_call.additional_params,
                });
                observation.events.push("tool_call");
            }
            Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult { .. })) => {
                observation.tool_results += 1;
                observation.final_turn_text.clear();
                observation.events.push("tool_result");
            }
            Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                observation.final_response_text = Some(response.output().to_owned());
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

/// Drive a raw provider stream to exhaustion, keeping both the visible text
/// and the terminal record.
///
/// The observation helpers above drop the terminal record; matrices that are
/// about what the terminal *carries* (a finish reason, usage) need it.
pub async fn collect_text_and_terminal(
    mut stream: StreamingCompletionResponse,
) -> (String, Option<rig_core::streaming::StreamFinal>) {
    let mut text = String::new();
    let mut terminal = None;

    while let Some(item) = stream.next().await {
        match item.expect("stream item should not be an error") {
            StreamEvent::BlockDelta {
                delta: Delta::Text { text: chunk },
                ..
            } => text.push_str(&chunk),
            StreamEvent::Final(final_record) => terminal = Some(final_record),
            _ => {}
        }
    }

    (text, terminal)
}

/// Drive a raw provider stream to exhaustion and return the terminal record
/// it must have ended with, discarding the visible text.
///
/// Providers that repeat their accounting across closing frames emit more
/// than one terminal event; this keeps the last, exactly as
/// [`collect_text_and_terminal`] reports it.
pub async fn collect_required_terminal(
    stream: StreamingCompletionResponse,
) -> rig_core::streaming::StreamFinal {
    let (_, terminal) = collect_text_and_terminal(stream).await;
    terminal.expect("stream should end with a terminal record")
}

/// Drive a raw provider stream to exhaustion, keeping the visible text and
/// requiring that it ended with exactly one terminal record.
///
/// The two claims belong together for the dialects whose contract is one
/// terminal record *and* whose cells are about what the stream said:
/// [`collect_sole_terminal`] drops the text, [`collect_text_and_terminal`]
/// keeps the last of several records.
pub async fn collect_text_and_sole_terminal(
    mut stream: StreamingCompletionResponse,
) -> (String, rig_core::streaming::StreamFinal) {
    let mut text = String::new();
    let mut finals = Vec::new();

    while let Some(item) = stream.next().await {
        match item.expect("stream item should be ok") {
            StreamEvent::BlockDelta {
                delta: Delta::Text { text: chunk },
                ..
            } => text.push_str(&chunk),
            StreamEvent::Final(record) => finals.push(record),
            _ => {}
        }
    }

    assert_eq!(
        finals.len(),
        1,
        "stream should yield exactly one terminal record"
    );
    (text, finals.remove(0))
}

/// Drive a raw provider stream to exhaustion and return its one terminal
/// record.
///
/// Exactly one terminal record per stream is the contract, so a stream that
/// emitted none — or more than one — fails here instead of silently handing
/// back the last.
pub async fn collect_sole_terminal(
    stream: StreamingCompletionResponse,
) -> rig_core::streaming::StreamFinal {
    collect_text_and_sole_terminal(stream).await.1
}

/// Where a matrix cell parks the observation its recorded turn produced.
///
/// The cassette wrappers take the turn as a closure and return the closure's
/// `Result`, so anything else the turn observed has to leave through shared
/// state. The wrapper call itself stays inline in every cell with its own
/// scenario literal — the fixture scan reads that literal out of the AST — so
/// what is shared here is the parking, never the call, the request or the
/// expectations.
#[derive(Debug)]
pub struct Observed<T>(std::sync::Arc<std::sync::Mutex<Option<T>>>);

impl<T> Observed<T> {
    /// A sink that has observed nothing yet.
    pub fn new() -> Self {
        Self(std::sync::Arc::new(std::sync::Mutex::new(None)))
    }

    /// Park what the turn produced, replacing any earlier observation.
    pub fn put(&self, observation: T) {
        *self.0.lock().expect("observation lock") = Some(observation);
    }

    /// Take what the turn parked, requiring that the turn actually ran.
    pub fn take(&self) -> T {
        self.0
            .lock()
            .expect("observation lock")
            .take()
            .expect("the cell should observe a value")
    }
}

impl<T> Clone for Observed<T> {
    fn clone(&self) -> Self {
        Self(std::sync::Arc::clone(&self.0))
    }
}

impl<T> Default for Observed<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Concatenate the assistant text blocks of a completion choice.
///
/// Unlike [`assistant_text_response`] this joins with nothing and always
/// returns a string: a matrix comparing against a recorded `content` field
/// wants the text exactly as the wire carried it, empty included.
pub fn assistant_text(choice: &[AssistantContent]) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

/// The normalized completion response serialized with its raw capture
/// cleared, so an assertion about the normalized surface cannot be satisfied
/// by something `raw` happens to carry.
pub fn normalized_without_raw(
    mut response: rig_core::completion::CompletionResponse,
) -> serde_json::Value {
    response.raw = serde_json::Value::Null;
    serde_json::to_value(&response).expect("normalized response should serialize")
}

/// Whether any object anywhere inside a JSON value carries the named key.
pub fn json_contains_key(value: &serde_json::Value, needle: &str) -> bool {
    match value {
        serde_json::Value::Object(map) => map
            .iter()
            .any(|(key, value)| key == needle || json_contains_key(value, needle)),
        serde_json::Value::Array(items) => items.iter().any(|item| json_contains_key(item, needle)),
        _ => false,
    }
}

/// Compare one field of a live reply against the recording at the strength
/// the current mode supports.
///
/// Replay serves the fixture back, so the values must be equal. A recording
/// pass sees a value the provider just minted while the fixture holds the
/// scrubbed one, so the claim there is that both carry the field and agree on
/// its JSON type.
pub fn assert_wire_value_matches(
    live: &serde_json::Value,
    recorded: &serde_json::Value,
    field: &str,
) {
    wire_value_matches_in(
        crate::cassettes::CassetteMode::current(),
        live,
        recorded,
        field,
    );
}

fn wire_value_matches_in(
    mode: crate::cassettes::CassetteMode,
    live: &serde_json::Value,
    recorded: &serde_json::Value,
    field: &str,
) {
    let (live_value, recorded_value) = (live.get(field), recorded.get(field));
    match mode {
        crate::cassettes::CassetteMode::Replay => assert_eq!(
            live_value, recorded_value,
            "{field}: replayed value must equal the recorded wire value"
        ),
        crate::cassettes::CassetteMode::Record => {
            let (Some(live_value), Some(recorded_value)) = (live_value, recorded_value) else {
                panic!("{field}: both the live value and the recording must carry it");
            };
            assert_eq!(
                std::mem::discriminant(live_value),
                std::mem::discriminant(recorded_value),
                "{field}: live and recorded values must share a JSON type"
            );
        }
    }
}

/// Compare a live reply document with its recorded fixture, key for key.
///
/// Both must hold the same keys at every depth, apart from the top-level
/// `ignore` keys, which the caller checks another way (a minted id), and
/// arrays of the same length. Keys the recorder normalizes
/// ([`crate::cassettes::is_volatile_json_key`], at any depth) compare
/// through [`assert_wire_value_matches`]: by JSON type in a recording pass,
/// exactly in replay. Every other value compares exactly in both modes.
///
/// # Panics
///
/// Panics on the first difference, naming `context` and the path to it
/// (`.data[3].created_at`).
pub fn assert_matches_recorded_document(
    live: &serde_json::Value,
    recorded: &serde_json::Value,
    ignore: &[&str],
    context: &str,
) {
    matches_recorded_document_in(
        crate::cassettes::CassetteMode::current(),
        live,
        recorded,
        ignore,
        context,
    );
}

pub(crate) fn matches_recorded_document_in(
    mode: crate::cassettes::CassetteMode,
    live: &serde_json::Value,
    recorded: &serde_json::Value,
    ignore: &[&str],
    context: &str,
) {
    compare_recorded(mode, live, recorded, ignore, context, "");
}

fn compare_recorded(
    mode: crate::cassettes::CassetteMode,
    live: &serde_json::Value,
    recorded: &serde_json::Value,
    ignore: &[&str],
    context: &str,
    at: &str,
) {
    use serde_json::Value;
    match (live, recorded) {
        (Value::Object(live_map), Value::Object(recorded_map)) => {
            let keys =
                |map: &serde_json::Map<String, Value>| -> std::collections::BTreeSet<String> {
                    map.keys()
                        .filter(|key| !ignore.contains(&key.as_str()))
                        .cloned()
                        .collect()
                };
            let recorded_keys = keys(recorded_map);
            assert_eq!(
                keys(live_map),
                recorded_keys,
                "{context}: the keys of `{at}`"
            );
            for key in &recorded_keys {
                let path = format!("{at}.{key}");
                if crate::cassettes::is_volatile_json_key(key) {
                    let (live_value, recorded_value) = (&live_map[key], &recorded_map[key]);
                    match mode {
                        crate::cassettes::CassetteMode::Replay => assert_eq!(
                            live_value, recorded_value,
                            "{context}: `{path}` must equal the recorded wire value in replay"
                        ),
                        crate::cassettes::CassetteMode::Record => assert_eq!(
                            std::mem::discriminant(live_value),
                            std::mem::discriminant(recorded_value),
                            "{context}: `{path}` must share the recorded value's JSON type"
                        ),
                    }
                } else {
                    compare_recorded(
                        mode,
                        &live_map[key],
                        &recorded_map[key],
                        &[],
                        context,
                        &path,
                    );
                }
            }
        }
        (Value::Array(live_items), Value::Array(recorded_items)) => {
            assert_eq!(
                live_items.len(),
                recorded_items.len(),
                "{context}: the length of `{at}`"
            );
            for (index, (live_item, recorded_item)) in
                live_items.iter().zip(recorded_items).enumerate()
            {
                compare_recorded(
                    mode,
                    live_item,
                    recorded_item,
                    &[],
                    context,
                    &format!("{at}[{index}]"),
                );
            }
        }
        _ => assert_eq!(
            live, recorded,
            "{context}: `{at}` must be carried unchanged"
        ),
    }
}

/// Compare a generated token (response id, system fingerprint, request id)
/// observed by a test with the value its fixture holds.
///
/// On the recording pass the fixture being compared against is the previous
/// recording (or a legacy placeholder), so the live token cannot equal it;
/// both are then required to be present and non-empty. On replay the harness
/// serves the recorded bytes back, so equality is exact, which is what CI
/// runs. Presence must agree in both modes.
pub fn assert_matches_recorded_token(actual: Option<&str>, recorded: Option<&str>, context: &str) {
    match crate::cassettes::CassetteMode::current() {
        crate::cassettes::CassetteMode::Replay => {
            assert_eq!(
                actual, recorded,
                "{context}: replay serves the fixture's token back"
            );
        }
        crate::cassettes::CassetteMode::Record => {
            assert_eq!(
                actual.is_some(),
                recorded.is_some(),
                "{context}: live and recorded token presence must agree"
            );
            if let (Some(actual), Some(recorded)) = (actual, recorded) {
                assert!(
                    !actual.trim().is_empty() && !recorded.trim().is_empty(),
                    "{context}: live and recorded token must both be non-empty"
                );
            }
        }
    }
}

/// Drain a raw provider stream into text, tool, error, and final-event observations.
pub async fn collect_raw_stream_observation(
    mut stream: StreamingCompletionResponse,
) -> RawStreamObservation
where
{
    let mut observation = RawStreamObservation::new();

    while let Some(item) = stream.next().await {
        match item {
            Ok(StreamEvent::BlockDelta {
                delta: Delta::Text { text },
                ..
            }) => {
                observation.text.push_str(&text);
                observation.events.push("text");
            }
            Ok(StreamEvent::BlockEnd {
                block: Some(AssistantContent::ToolCall(tool_call)),
                ..
            }) => {
                observation.tool_calls.push(tool_call.clone());
                observation.tool_call_records.push(ToolCallRecord {
                    name: tool_call.function.name,
                    signature: tool_call.signature,
                    additional_params: tool_call.additional_params,
                });
                observation.events.push("tool_call");
            }
            Ok(StreamEvent::BlockDelta {
                delta: Delta::ToolName { .. } | Delta::ToolArguments { .. },
                ..
            }) => {
                observation.events.push("tool_call_delta");
            }
            Ok(StreamEvent::BlockEnd {
                block: Some(AssistantContent::Reasoning(_)),
                ..
            }) => {
                observation.events.push("reasoning");
            }
            Ok(StreamEvent::BlockDelta {
                delta: Delta::Reasoning { .. },
                ..
            }) => {
                observation.events.push("reasoning_delta");
            }
            Ok(StreamEvent::Final(_)) => {
                observation.got_final = true;
                observation.events.push("final");
            }
            Ok(StreamEvent::Unknown(_)) => {
                observation.events.push("unknown");
            }
            Ok(StreamEvent::BlockStart { .. })
            | Ok(StreamEvent::BlockDelta {
                delta: Delta::TextMeta { .. },
                ..
            })
            | Ok(StreamEvent::BlockEnd { .. }) => {}
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

/// Assert both tool roundtrips precede text and their markers appear in a consistent final answer.
pub fn assert_two_tool_roundtrip_contract(
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
        "FinalResponse.output() should match the final turn's streamed text"
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
            "expected the initial unique tool-call phase to include {expected_tool}, saw {first_unique:?}"
        );
    }

    let response = observation
        .final_response_text
        .as_deref()
        .expect("stream should produce a final response string");
    assert_contains_all_case_insensitive(response, expected_markers);
}

/// Assert the expected tool executes before later text and the final answer contains its markers.
pub fn assert_tool_call_precedes_later_text(
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
        "FinalResponse.output() should match the final turn's streamed text"
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

/// Assert a successful raw stream contains the expected tool before any text that is emitted.
pub fn assert_raw_stream_tool_call_precedes_text(
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

/// Assert distinct expected tool calls precede any text in a successful raw stream.
pub fn assert_raw_stream_contains_distinct_tool_calls_before_text(
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
            "expected raw stream tool call for {expected_tool}, saw {tool_call_names:?}"
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
            "expected the initial unique raw tool-call phase to include {expected_tool}, saw {first_unique:?}"
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

/// Every tool call surfaced on the raw stream must carry a JSON **object** as
/// its `function.arguments` (never a bare string), the invariant fixed in #1958:
/// a downstream object-typed serializer (e.g. Anthropic's `tool_use.input`)
/// rejects a string input. This guards the streaming aggregator end-to-end on
/// real provider traffic, complementing the in-crate eviction unit tests.
pub fn assert_raw_stream_tool_call_arguments_are_objects(
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
        observation.tool_calls.len() >= expected_tools.len(),
        "expected at least {} raw stream tool calls, saw {:?}",
        expected_tools.len(),
        observation
            .tool_calls
            .iter()
            .map(|tool_call| tool_call.function.name.clone())
            .collect::<Vec<_>>(),
    );

    for tool_call in &observation.tool_calls {
        assert!(
            tool_call.function.arguments.is_object(),
            "tool call `{}` must surface object arguments, got {:?}",
            tool_call.function.name,
            tool_call.function.arguments,
        );
    }
}

/// Assert a successful raw stream's text contains every expected substring.
pub fn assert_raw_stream_text_contains(observation: &RawStreamObservation, expected: &[&str]) {
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

/// Assert the loader answer identifies the agent-building fixture.
pub fn assert_loader_answer_is_relevant(response: &str) {
    assert_contains_any_case_insensitive(
        response,
        &[
            "agent_with_loaders",
            "agent_with_loaders.rs",
            "agent with loaders",
        ],
    );
}

/// Assert every structured smoke-response field contains non-whitespace text.
pub fn assert_smoke_structured_output(output: &SmokeStructuredOutput) {
    assert_nonempty_response(&output.title);
    assert_nonempty_response(&output.category);
    assert_nonempty_response(&output.summary);
}

/// Shared identity-observing hook for the rig#2265 response-identity
/// cassettes: captures each event's [`rig_agent::completion::ResponseIdentity`]
/// so provider suites can assert per-attempt identity on every observer
/// surface. `CompletionResponse` fires once per accepted model turn on both
/// drivers, so `responses` fills on blocking and streamed runs alike.
#[derive(Clone, Default)]
pub struct IdentityProbe {
    /// Response identities captured at completion-response events.
    pub responses: std::sync::Arc<std::sync::Mutex<Vec<rig_agent::completion::ResponseIdentity>>>,
    /// Response identities captured at model-turn events.
    pub turns: std::sync::Arc<std::sync::Mutex<Vec<rig_agent::completion::ResponseIdentity>>>,
}

impl IdentityProbe {
    /// Return a snapshot of the model-turn response identities.
    pub fn turn_identities(&self) -> Vec<rig_agent::completion::ResponseIdentity> {
        self.turns.lock().expect("turn identities").clone()
    }

    /// Return a snapshot of the completion-response identities.
    pub fn response_identities(&self) -> Vec<rig_agent::completion::ResponseIdentity> {
        self.responses.lock().expect("response identities").clone()
    }
}

impl rig_agent::agent::AgentHook for IdentityProbe {
    async fn on_outcome(
        &self,
        _ctx: &rig_agent::agent::HookContext,
        event: rig_agent::agent::OutcomeEvent<'_>,
    ) -> rig_agent::agent::OutcomeAction {
        if let Some(response) = event.completion() {
            self.responses
                .lock()
                .expect("response identities")
                .push(response.identity());
        }
        rig_agent::agent::OutcomeAction::proceed()
    }

    async fn on_model_turn_finished(
        &self,
        _ctx: &rig_agent::agent::HookContext,
        event: rig_agent::agent::ModelTurnFinished<'_>,
    ) -> rig_agent::agent::ModelTurnAction {
        self.turns
            .lock()
            .expect("turn identities")
            .push(event.identity.clone());
        rig_agent::agent::ModelTurnAction::continue_run()
    }
}

/// Assert a transport request id is populated and non-empty.
pub fn assert_transport_request_id(id: Option<&str>, context: &str) {
    assert!(
        id.is_some_and(|id| !id.trim().is_empty()),
        "{context}: provider_request_id must be populated"
    );
}

/// Per-provider expectations for the recorded embedding matrix
/// (`tests/providers/<provider>/cassette/embedding_matrix.rs`). The cells are
/// shared; what a provider's wire actually reports — usage, a model echo, a
/// transport request id — is data, asserted from the recordings.
pub struct EmbeddingMatrixExpectations {
    /// Stable descriptor name stamped on `EmbeddingResponse::provider`.
    pub provider: &'static str,
    /// Whether the recorded wire reports non-zero token usage.
    pub reports_usage: bool,
    /// Whether the recorded wire echoes a model identifier.
    pub reports_model: bool,
    /// Whether the provider has a transport request-id header contract on
    /// this endpoint.
    pub reports_request_id: bool,
}

/// The shared "normalized response is complete" cell: embeddings in input
/// order, provider attribution, and each metadata axis present exactly when
/// the provider's wire reports it — `None`/zero is the documented outcome on
/// the axes it does not.
pub fn assert_normalized_embedding_response(
    response: &rig_core::embeddings::EmbeddingResponse,
    inputs: &[&str],
    expectations: &EmbeddingMatrixExpectations,
) {
    assert_embeddings_nonempty_and_consistent(&response.embeddings, inputs.len());
    for (embedding, input) in response.embeddings.iter().zip(inputs) {
        assert_eq!(
            embedding.document, *input,
            "embeddings must preserve input order"
        );
    }
    assert_eq!(response.provider, expectations.provider);
    assert_eq!(
        response.usage.is_reported(),
        expectations.reports_usage,
        "usage mismatch for {}: got {:?}",
        expectations.provider,
        response.usage
    );
    assert_eq!(
        response.model.is_some(),
        expectations.reports_model,
        "model echo mismatch for {}: got {:?}",
        expectations.provider,
        response.model
    );
    assert_eq!(
        response.provider_request_id.is_some(),
        expectations.reports_request_id,
        "request-id mismatch for {}: got {:?}",
        expectations.provider,
        response.provider_request_id
    );
    assert_eq!(
        response.identity().provider_request_id,
        response.provider_request_id
    );
    assert!(
        !response.raw.is_null(),
        "every HTTP provider seam populates `raw`"
    );
}

/// Wire-level probe middleware for the run-lifecycle cassette matrix.
///
/// Counts each `HttpMiddleware` phase, records the last observed response
/// status and request-body length, and injects a benign
/// `x-rig-lifecycle-probe` header (deliberately outside the harness's
/// recorded-header allowlist, so cassette matching is identical with and
/// without it). Everything it observes holds in both cassette modes: on
/// replay the same phases fire against the replay server.
#[derive(Clone, Default)]
pub struct WireProbe {
    /// Number of request-header middleware invocations.
    pub header_phases: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    /// Number of request-body middleware invocations.
    pub body_phases: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    /// Number of response middleware invocations.
    pub response_phases: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    /// Most recently observed response status code.
    pub last_status: std::sync::Arc<std::sync::atomic::AtomicU16>,
    /// Byte length of the most recently observed request body.
    pub last_body_len: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl rig_core::http_client::HttpMiddleware for WireProbe {
    fn before_request_headers<'a>(
        &'a self,
        _method: &'a rig_core::http_client::Method,
        _uri: &'a rig_core::http_client::Uri,
        headers: &'a mut rig_core::http_client::HeaderMap,
    ) -> rig_core::wasm_compat::WasmBoxedFuture<'a, rig_core::http_client::Result<()>> {
        Box::pin(async move {
            self.header_phases
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            headers.insert(
                "x-rig-lifecycle-probe",
                rig_core::http_client::HeaderValue::from_static("1"),
            );
            Ok(())
        })
    }

    fn before_request_body<'a>(
        &'a self,
        _method: &'a rig_core::http_client::Method,
        _uri: &'a rig_core::http_client::Uri,
        headers: &'a rig_core::http_client::HeaderMap,
        body: bytes::Bytes,
    ) -> rig_core::wasm_compat::WasmBoxedFuture<'a, rig_core::http_client::Result<bytes::Bytes>>
    {
        Box::pin(async move {
            self.body_phases
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            // The body phase runs after the header phase mutated the map.
            assert!(
                headers.contains_key("x-rig-lifecycle-probe"),
                "body hooks see the final headers"
            );
            self.last_body_len
                .store(body.len(), std::sync::atomic::Ordering::SeqCst);
            Ok(body)
        })
    }

    fn after_response<'a>(
        &'a self,
        _method: &'a rig_core::http_client::Method,
        _uri: &'a rig_core::http_client::Uri,
        status: rig_core::http_client::StatusCode,
        _headers: &'a rig_core::http_client::HeaderMap,
    ) -> rig_core::wasm_compat::WasmBoxedFuture<'a, rig_core::http_client::Result<()>> {
        Box::pin(async move {
            self.response_phases
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.last_status
                .store(status.as_u16(), std::sync::atomic::Ordering::SeqCst);
            Ok(())
        })
    }
}

impl WireProbe {
    /// Assert the counters of a completed single-request exchange.
    pub fn assert_single_exchange(&self) {
        use std::sync::atomic::Ordering::SeqCst;
        assert_eq!(self.header_phases.load(SeqCst), 1, "one header phase");
        assert_eq!(self.body_phases.load(SeqCst), 1, "one body phase");
        assert_eq!(self.response_phases.load(SeqCst), 1, "one response phase");
        assert_eq!(
            self.last_status.load(SeqCst),
            200,
            "success status observed"
        );
        assert!(
            self.last_body_len.load(SeqCst) > 0,
            "the serialized provider payload was visible to the body phase"
        );
    }
}

/// Entry-log probe for the run-lifecycle cassette matrix: appends one
/// `"phase"` entry per lifecycle event — `"run_start"` at `on_run_start`
/// (turn 0, before any model call) and `"completion_call"` per model call —
/// and captures the full replayed log at settle. What it pins on real
/// provider traffic: append order across lifecycle events, turn stamping
/// (0 pre-run, then the one-based call index), and that the entry log is
/// storage, not context — the replay server matches request bodies
/// byte-exactly, so replay staying green proves entries never reach the wire.
#[derive(Clone, Default)]
pub struct EntryLogProbe {
    /// Run-record entries observed when the run settled.
    pub settled: std::sync::Arc<std::sync::Mutex<Vec<rig_agent::agent::RunEntry>>>,
}

impl EntryLogProbe {
    /// The settled `"phase"` log as `(turn, value)` pairs.
    pub fn settled_phases(&self) -> Vec<(usize, String)> {
        self.settled
            .lock()
            .expect("settled")
            .iter()
            .map(|entry| {
                (
                    entry.turn,
                    entry.value.as_str().unwrap_or_default().to_string(),
                )
            })
            .collect()
    }

    /// Assert the settled log of a completed run: a turn-0 `run_start` first,
    /// then one `completion_call` per model call with consecutive one-based
    /// turn stamps, at least `min_calls` of them.
    pub fn assert_phases(&self, min_calls: usize) {
        let phases = self.settled_phases();
        assert!(
            phases.len() > min_calls,
            "expected run_start plus at least {min_calls} completion calls: {phases:?}"
        );
        assert_eq!(
            phases[0],
            (0, "run_start".to_string()),
            "the pre-run append is stamped turn 0: {phases:?}"
        );
        for (index, (turn, value)) in phases[1..].iter().enumerate() {
            assert_eq!(
                (*turn, value.as_str()),
                (index + 1, "completion_call"),
                "per-call snapshots are turn-stamped in call order: {phases:?}"
            );
        }
    }
}

impl rig_agent::agent::AgentHook for EntryLogProbe {
    async fn on_run_start(
        &self,
        ctx: &rig_agent::agent::HookContext,
        _event: rig_agent::agent::RunStart<'_>,
    ) -> rig_agent::agent::RunStartAction {
        ctx.append_entry("phase", &"run_start")
            .expect("a str serializes");
        rig_agent::agent::RunStartAction::Continue
    }

    async fn on_completion_call(
        &self,
        ctx: &rig_agent::agent::HookContext,
        _event: rig_agent::agent::CompletionCallEvent<'_>,
    ) -> rig_agent::agent::CompletionCallAction {
        ctx.append_entry("phase", &"completion_call")
            .expect("a str serializes");
        rig_agent::agent::CompletionCallAction::Continue
    }

    async fn on_run_settled(
        &self,
        ctx: &rig_agent::agent::HookContext,
        _event: rig_agent::agent::RunSettled<'_>,
    ) {
        *self.settled.lock().expect("settled") = ctx.entries("phase");
    }
}

/// Agent-hook probe for the run-lifecycle cassette matrix: counts
/// `on_run_start` firings (optionally rewriting the prompt), appends one
/// `"completion_calls"` snapshot entry to the run's record per model call,
/// and records every `on_run_settled` outcome plus the entries visible at
/// settle time.
#[derive(Clone, Default)]
pub struct LifecycleHookProbe {
    /// Optional replacement for the prompt at run start.
    pub rewrite_to: Option<String>,
    /// Number of observed run-start events.
    pub starts: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    /// Formatted outcomes observed at run settlement.
    pub settles: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    /// Run-record entries visible at settlement.
    pub settled_entries: std::sync::Arc<std::sync::Mutex<Vec<rig_agent::agent::RunEntry>>>,
}

impl LifecycleHookProbe {
    /// Create a lifecycle probe that rewrites the starting prompt.
    pub fn rewriting_to(prompt: &str) -> Self {
        Self {
            rewrite_to: Some(prompt.to_string()),
            ..Self::default()
        }
    }

    /// The settle outcomes observed so far ("response" or "error:…").
    pub fn settle_outcomes(&self) -> Vec<String> {
        self.settles.lock().expect("settles").clone()
    }

    /// The durable completion-call counter as seen at settle time: the
    /// last-wins read of the `"completion_calls"` snapshot entries the hook
    /// appended to the run's record.
    pub fn exported_completion_calls(&self) -> Option<u64> {
        self.settled_entries
            .lock()
            .expect("entries")
            .iter()
            .rev()
            .find(|entry| entry.kind == "completion_calls")
            .and_then(|entry| entry.value.as_u64())
    }
}

impl rig_agent::agent::AgentHook for LifecycleHookProbe {
    async fn on_run_start(
        &self,
        _ctx: &rig_agent::agent::HookContext,
        _event: rig_agent::agent::RunStart<'_>,
    ) -> rig_agent::agent::RunStartAction {
        self.starts
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        match &self.rewrite_to {
            Some(prompt) => rig_agent::agent::RunStartAction::rewrite(
                rig_agent::completion::Message::user(prompt),
            ),
            None => rig_agent::agent::RunStartAction::Continue,
        }
    }

    async fn on_completion_call(
        &self,
        ctx: &rig_agent::agent::HookContext,
        _event: rig_agent::agent::CompletionCallEvent<'_>,
    ) -> rig_agent::agent::CompletionCallAction {
        // Snapshot + last-wins: append the running count per model call; the
        // settle-time read takes the most recent snapshot. Entries land in
        // the run's serializable record — and never on the wire, which the
        // cassette replay proves byte-exactly (the replay server matches
        // request bodies).
        let calls = ctx
            .last_entry("completion_calls")
            .and_then(|entry| entry.value.as_u64())
            .unwrap_or(0)
            + 1;
        ctx.append_entry("completion_calls", &calls)
            .expect("a u64 serializes");
        rig_agent::agent::CompletionCallAction::Continue
    }

    async fn on_run_settled(
        &self,
        ctx: &rig_agent::agent::HookContext,
        event: rig_agent::agent::RunSettled<'_>,
    ) {
        *self.settled_entries.lock().expect("entries") = ctx.entries("completion_calls");
        let outcome = match event.outcome {
            rig_agent::agent::SettledOutcome::Response(_) => "response".to_string(),
            rig_agent::agent::SettledOutcome::Error(reason) => format!("error:{reason}"),
        };
        self.settles.lock().expect("settles").push(outcome);
    }
}
