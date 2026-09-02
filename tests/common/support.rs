//! Shared fixtures, tiny tools, and durable assertions for ignored smoke tests.
#![allow(dead_code)]

use futures::StreamExt;
use rig::{
    agent::{MultiTurnStreamItem, StreamingError, StreamingResult},
    completion::{AssistantContent, ToolDefinition},
    embeddings::Embedding,
    streaming::{Delta, StreamEvent, StreamedUserContent, StreamingCompletionResponse},
    tool::PortableTool,
    tool::Tool,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};

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
pub(crate) const VIDEO_FIXTURE_PATH: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/sample_video.mp4");

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

pub(crate) fn smoke_structured_output_value() -> serde_json::Value {
    json!({
        "title": "Seattle Rust Meetup",
        "category": "Technology",
        "summary": "A focused local meetup for Rust developers."
    })
}

pub(crate) fn ecs_synthetic_output_tool_name<T>() -> String
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
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
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
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(args.x - args.y)
    }
}

#[derive(Clone, Copy, Deserialize, Serialize)]
pub(crate) struct PortableAdder;

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
pub(crate) struct PortableSubtract;

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
pub(crate) struct AlphaSignal;

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
        _context: &mut rig::tool::ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
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
        _context: &mut rig::tool::ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
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

pub(crate) fn assistant_text_response(choice: &[AssistantContent]) -> Option<String> {
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
        "Response {response:?} did not contain any of {expected:?}."
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
        "Response {response:?} did not contain all of {expected:?}; missing {missing:?}."
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
        "Response {response:?} did not mention the expected number {expected:?}."
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
        "expected city {expected_city:?}, got {city:?}"
    );

    assert!(
        weather.to_ascii_lowercase().contains("fire and brimstone"),
        "expected the weather description to preserve the tool result, got {weather:?}"
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

// ---------------------------------------------------------------------------
// Model-turn termination metadata (rig#2184).
//
// One implementation, driven by every provider suite that covers the feature.
// That is the portability claim as code: if a provider needed its own probe,
// the metadata would not be provider-neutral.
// ---------------------------------------------------------------------------

/// One turn's termination as a hook sees it: why it stopped, and the effective
/// output-token cap that attempt ran under.
pub(crate) type ObservedTermination = (Option<rig::completion::FinishReason>, Option<u64>);

/// Records `ModelTurnFinished`'s normalized termination metadata for every
/// accepted turn, naming no provider and touching no raw response type.
#[derive(Clone, Debug, Default)]
pub(crate) struct TurnTerminationProbe {
    observations: std::sync::Arc<std::sync::Mutex<Vec<ObservedTermination>>>,
}

impl TurnTerminationProbe {
    pub(crate) fn observations(&self) -> Vec<ObservedTermination> {
        self.observations.lock().expect("observations").clone()
    }

    /// The reason reported for the first accepted turn.
    pub(crate) fn first_reason(&self) -> Option<rig::completion::FinishReason> {
        self.observations()
            .first()
            .and_then(|(reason, _)| reason.clone())
    }

    /// The effective cap reported for the first accepted turn.
    pub(crate) fn first_max_tokens(&self) -> Option<u64> {
        self.observations().first().and_then(|(_, cap)| *cap)
    }
}

impl rig::agent::AgentHook for TurnTerminationProbe {
    async fn on_model_turn_finished(
        &self,
        _ctx: &rig::agent::HookContext,
        event: rig::agent::ModelTurnFinished<'_>,
    ) -> rig::agent::ModelTurnAction {
        self.observations
            .lock()
            .expect("observations")
            .push((event.finish_reason.cloned(), event.max_tokens));
        rig::agent::ModelTurnAction::continue_run()
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
pub(crate) struct EscalateCapOnTruncation {
    cap: std::sync::Arc<std::sync::atomic::AtomicU64>,
    escalated_to: std::sync::Arc<std::sync::Mutex<Vec<u64>>>,
    grown_cap: u64,
    max_retries: u32,
    retries: std::sync::Arc<std::sync::atomic::AtomicU32>,
}

impl EscalateCapOnTruncation {
    /// Start every attempt at `start_cap`; on a truncated, tool-free turn,
    /// re-run it once at `grown_cap`.
    pub(crate) fn new(start_cap: u64, grown_cap: u64) -> Self {
        Self {
            cap: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(start_cap)),
            escalated_to: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            grown_cap,
            max_retries: 1,
            retries: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
        }
    }

    /// The caps this hook asked for, in order — one entry per escalation.
    pub(crate) fn escalations(&self) -> Vec<u64> {
        self.escalated_to.lock().expect("escalated_to").clone()
    }

    pub(crate) fn retries(&self) -> u32 {
        self.retries.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl rig::agent::AgentHook for EscalateCapOnTruncation {
    /// Every attempt is prepared afresh, so the current cap is applied here and
    /// reported back on that attempt's `ModelTurnFinished`.
    async fn on_completion_call(
        &self,
        _ctx: &rig::agent::HookContext,
        _event: rig::agent::CompletionCallEvent<'_>,
    ) -> rig::agent::CompletionCallAction {
        rig::agent::CompletionCallAction::patch(
            rig::agent::RequestPatch::new()
                .max_tokens(self.cap.load(std::sync::atomic::Ordering::SeqCst)),
        )
    }

    async fn on_model_turn_finished(
        &self,
        _ctx: &rig::agent::HookContext,
        event: rig::agent::ModelTurnFinished<'_>,
    ) -> rig::agent::ModelTurnAction {
        let truncated = event
            .finish_reason
            .is_some_and(rig::completion::FinishReason::truncated_output);
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
            return rig::agent::ModelTurnAction::repeat();
        }
        rig::agent::ModelTurnAction::continue_run()
    }
}

pub(crate) async fn collect_stream_final_response(
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

pub(crate) async fn collect_stream_final_response_and_provider_final(
    stream: &mut StreamingResult,
) -> Result<(String, rig::streaming::StreamFinal), StreamingError> {
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

pub(crate) async fn assert_stream_contains_zero_arg_tool_call_named(
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
            } => {
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
    pub(crate) tool_calls: Vec<rig::message::ToolCall>,
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

pub(crate) async fn collect_stream_observation(stream: &mut StreamingResult) -> StreamObservation {
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
pub(crate) async fn collect_text_and_terminal(
    mut stream: StreamingCompletionResponse,
) -> (String, Option<rig::streaming::StreamFinal>) {
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

pub(crate) async fn collect_raw_stream_observation(
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
pub(crate) fn assert_raw_stream_tool_call_arguments_are_objects(
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

/// Shared identity-observing hook for the rig#2265 response-identity
/// cassettes: captures each event's [`rig::completion::ResponseIdentity`]
/// so provider suites can assert per-attempt identity on every observer
/// surface. `CompletionResponse` fires once per accepted model turn on both
/// drivers, so `responses` fills on blocking and streamed runs alike.
#[derive(Clone, Default)]
pub(crate) struct IdentityProbe {
    pub(crate) responses: std::sync::Arc<std::sync::Mutex<Vec<rig::completion::ResponseIdentity>>>,
    pub(crate) turns: std::sync::Arc<std::sync::Mutex<Vec<rig::completion::ResponseIdentity>>>,
}

impl IdentityProbe {
    pub(crate) fn turn_identities(&self) -> Vec<rig::completion::ResponseIdentity> {
        self.turns.lock().expect("turn identities").clone()
    }

    pub(crate) fn response_identities(&self) -> Vec<rig::completion::ResponseIdentity> {
        self.responses.lock().expect("response identities").clone()
    }
}

impl rig::agent::AgentHook for IdentityProbe {
    async fn on_outcome(
        &self,
        _ctx: &rig::agent::HookContext,
        event: rig::agent::OutcomeEvent<'_>,
    ) -> rig::agent::OutcomeAction {
        if let Some(response) = event.completion() {
            self.responses
                .lock()
                .expect("response identities")
                .push(response.identity());
        }
        rig::agent::OutcomeAction::proceed()
    }

    async fn on_model_turn_finished(
        &self,
        _ctx: &rig::agent::HookContext,
        event: rig::agent::ModelTurnFinished<'_>,
    ) -> rig::agent::ModelTurnAction {
        self.turns
            .lock()
            .expect("turn identities")
            .push(event.identity.clone());
        rig::agent::ModelTurnAction::continue_run()
    }
}

/// Assert a transport request id is populated and non-empty.
pub(crate) fn assert_transport_request_id(id: Option<&str>, context: &str) {
    assert!(
        id.is_some_and(|id| !id.trim().is_empty()),
        "{context}: provider_request_id must be populated"
    );
}

/// Per-provider expectations for the recorded embedding matrix
/// (`tests/providers/<provider>/cassette/embedding_matrix.rs`). The cells are
/// shared; what a provider's wire actually reports — usage, a model echo, a
/// transport request id — is data, asserted from the recordings.
pub(crate) struct EmbeddingMatrixExpectations {
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
pub(crate) fn assert_normalized_embedding_response(
    response: &rig::embeddings::EmbeddingResponse,
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
        response.usage.has_values(),
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
pub(crate) struct WireProbe {
    pub(crate) header_phases: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    pub(crate) body_phases: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    pub(crate) response_phases: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    pub(crate) last_status: std::sync::Arc<std::sync::atomic::AtomicU16>,
    pub(crate) last_body_len: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl rig::http_client::HttpMiddleware for WireProbe {
    fn before_request_headers<'a>(
        &'a self,
        _method: &'a rig::http_client::Method,
        _uri: &'a rig::http_client::Uri,
        headers: &'a mut rig::http_client::HeaderMap,
    ) -> rig::wasm_compat::WasmBoxedFuture<'a, rig::http_client::Result<()>> {
        Box::pin(async move {
            self.header_phases
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            headers.insert(
                "x-rig-lifecycle-probe",
                rig::http_client::HeaderValue::from_static("1"),
            );
            Ok(())
        })
    }

    fn before_request_body<'a>(
        &'a self,
        _method: &'a rig::http_client::Method,
        _uri: &'a rig::http_client::Uri,
        headers: &'a rig::http_client::HeaderMap,
        body: bytes::Bytes,
    ) -> rig::wasm_compat::WasmBoxedFuture<'a, rig::http_client::Result<bytes::Bytes>> {
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
        _method: &'a rig::http_client::Method,
        _uri: &'a rig::http_client::Uri,
        status: rig::http_client::StatusCode,
        _headers: &'a rig::http_client::HeaderMap,
    ) -> rig::wasm_compat::WasmBoxedFuture<'a, rig::http_client::Result<()>> {
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
    pub(crate) fn assert_single_exchange(&self) {
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
pub(crate) struct EntryLogProbe {
    pub(crate) settled: std::sync::Arc<std::sync::Mutex<Vec<rig::agent::RunEntry>>>,
}

impl EntryLogProbe {
    /// The settled `"phase"` log as `(turn, value)` pairs.
    pub(crate) fn settled_phases(&self) -> Vec<(usize, String)> {
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
    pub(crate) fn assert_phases(&self, min_calls: usize) {
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

impl rig::agent::AgentHook for EntryLogProbe {
    async fn on_run_start(
        &self,
        ctx: &rig::agent::HookContext,
        _event: rig::agent::RunStart<'_>,
    ) -> rig::agent::RunStartAction {
        ctx.append_entry("phase", &"run_start")
            .expect("a str serializes");
        rig::agent::RunStartAction::Continue
    }

    async fn on_completion_call(
        &self,
        ctx: &rig::agent::HookContext,
        _event: rig::agent::CompletionCallEvent<'_>,
    ) -> rig::agent::CompletionCallAction {
        ctx.append_entry("phase", &"completion_call")
            .expect("a str serializes");
        rig::agent::CompletionCallAction::Continue
    }

    async fn on_run_settled(
        &self,
        ctx: &rig::agent::HookContext,
        _event: rig::agent::RunSettled<'_>,
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
pub(crate) struct LifecycleHookProbe {
    pub(crate) rewrite_to: Option<String>,
    pub(crate) starts: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    pub(crate) settles: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    pub(crate) settled_entries: std::sync::Arc<std::sync::Mutex<Vec<rig::agent::RunEntry>>>,
}

impl LifecycleHookProbe {
    pub(crate) fn rewriting_to(prompt: &str) -> Self {
        Self {
            rewrite_to: Some(prompt.to_string()),
            ..Self::default()
        }
    }

    /// The settle outcomes observed so far ("response" or "error:…").
    pub(crate) fn settle_outcomes(&self) -> Vec<String> {
        self.settles.lock().expect("settles").clone()
    }

    /// The durable completion-call counter as seen at settle time: the
    /// last-wins read of the `"completion_calls"` snapshot entries the hook
    /// appended to the run's record.
    pub(crate) fn exported_completion_calls(&self) -> Option<u64> {
        self.settled_entries
            .lock()
            .expect("entries")
            .iter()
            .rev()
            .find(|entry| entry.kind == "completion_calls")
            .and_then(|entry| entry.value.as_u64())
    }
}

impl rig::agent::AgentHook for LifecycleHookProbe {
    async fn on_run_start(
        &self,
        _ctx: &rig::agent::HookContext,
        _event: rig::agent::RunStart<'_>,
    ) -> rig::agent::RunStartAction {
        self.starts
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        match &self.rewrite_to {
            Some(prompt) => {
                rig::agent::RunStartAction::rewrite(rig::completion::Message::user(prompt))
            }
            None => rig::agent::RunStartAction::Continue,
        }
    }

    async fn on_completion_call(
        &self,
        ctx: &rig::agent::HookContext,
        _event: rig::agent::CompletionCallEvent<'_>,
    ) -> rig::agent::CompletionCallAction {
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
        rig::agent::CompletionCallAction::Continue
    }

    async fn on_run_settled(
        &self,
        ctx: &rig::agent::HookContext,
        event: rig::agent::RunSettled<'_>,
    ) {
        *self.settled_entries.lock().expect("entries") = ctx.entries("completion_calls");
        let outcome = match event.outcome {
            rig::agent::SettledOutcome::Response(_) => "response".to_string(),
            rig::agent::SettledOutcome::Error(reason) => format!("error:{reason}"),
        };
        self.settles.lock().expect("settles").push(outcome);
    }
}
