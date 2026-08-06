//! Wire-sequence conformance scenarios for provider streaming pipelines.
//!
//! The streaming sibling of `rig-agent`'s `model_conformance`: each scenario
//! drives raw wire bytes (SSE or NDJSON) through a provider's *complete*
//! streaming path — bytes → decode → normalize → aggregated
//! [`StreamingCompletionResponse`](crate::streaming::StreamingCompletionResponse)
//! — and asserts the [`StreamFinal`] contract
//! table documented on that type. Scenarios state the contract; a per-provider
//! [`ProviderWireFixture`] supplies the frames, since each wire format spells
//! the same event differently.
//!
//! Every sequence family here pins a shipped bug from the #2257 review rounds
//! (`rig-2257-code-review-findings-*.md`); the per-scenario comments cite the
//! specific finding.
//!
//! Suites are expanded per wire family by
//! [`streaming_conformance_suite!`](crate::streaming_conformance_suite).
//! Scenarios a wire cannot spell return an explicit
//! [`ScenarioOutcome::Skipped`] that the macro cross-checks against the
//! suite's declared [`SuiteCapabilities`] — a skip is always visible and can
//! never masquerade as a pass, so the executed count is exactly the declared
//! grid minus the named skips (#2258 review, F8 corpus honesty).

use bytes::Bytes;
use futures::StreamExt;
use futures::future::BoxFuture;

use crate::{
    OneOrMany,
    completion::{CompletionError, FinishReason},
    http_client,
    message::AssistantContent,
    streaming::{StreamFinal, StreamedAssistantContent},
};

/// Typed failure from a wire-conformance scenario.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ConformanceError {
    /// Opening the stream failed before any wire frame was consumed.
    #[error(transparent)]
    Completion(#[from] CompletionError),
    /// The pipeline violated the streaming contract table.
    #[error("{scenario} conformance failed for {provider}: {details}")]
    Contract {
        /// Stable scenario name.
        scenario: &'static str,
        /// Provider driver under test.
        provider: &'static str,
        /// Actionable observation explaining the failure.
        details: String,
    },
}

impl ConformanceError {
    fn contract(
        scenario: &'static str,
        provider: &'static str,
        details: impl Into<String>,
    ) -> Self {
        Self::Contract {
            scenario,
            provider,
            details: details.into(),
        }
    }
}

/// Outcome of a passing wire-conformance scenario.
#[derive(Debug)]
pub struct ScenarioReport {
    /// Stable scenario name.
    pub name: &'static str,
    /// Provider driver the scenario ran against.
    pub provider: &'static str,
    /// Human-readable observations, one per verified sub-case.
    pub observations: Vec<String>,
}

/// What a capability-gated scenario did: ran its assertions, or skipped
/// because the wire family cannot spell the sequence shape.
///
/// A skip is an explicit, named outcome — never a silent pass. The
/// [`streaming_conformance_suite!`](crate::streaming_conformance_suite)
/// macro cross-checks it against the suite's declared capability flags via
/// [`check_gated_outcome`], so a fixture cannot vacuously pass a scenario its
/// capabilities claim to cover (#2258 review, F8 corpus-honesty batch).
#[derive(Debug)]
pub enum ScenarioOutcome {
    /// The scenario ran and its assertions held.
    Ran(ScenarioReport),
    /// The fixture lacks the sequence shape; nothing was asserted.
    Skipped {
        /// Stable scenario name.
        name: &'static str,
        /// Provider driver under test.
        provider: &'static str,
        /// Why the wire family cannot spell the shape.
        reason: &'static str,
    },
}

/// Streaming-relevant capability flags for one wire family's conformance
/// suite: which optional sequence shapes the wire can spell.
///
/// Each flag mirrors an `Option` field on [`ProviderWireFixture`], and the
/// only constructor is [`ProviderWireFixture::capabilities`] — suites never
/// hand-write flags, so a flag structurally cannot drift from the wire
/// fixture that backs it (it *is* the fixture's populated-field set).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SuiteCapabilities {
    /// The wire streams tool-call arguments incrementally
    /// (`partial_tool_call_frames`).
    pub partial_tool_args: bool,
    /// The wire has a genuine terminal that can omit usage metrics
    /// (`zero_usage_terminal_frames`).
    pub zero_usage_terminal: bool,
    /// The wire has a data-less terminal signal (`bare_terminal_frames`).
    pub bare_terminal: bool,
    /// A frame-level decode failure can be spelled (`malformed_frame`).
    pub malformed_frame: bool,
    /// An unknown event type can be spelled (`unknown_event_frame`).
    pub unknown_event_frame: bool,
    /// A known event with a schema-defective payload can be spelled
    /// (`defective_known_frame`).
    pub defective_known_frame: bool,
    /// The wire has a delta-less choice prelude shape
    /// (`delta_less_prelude_frame`).
    pub delta_less_prelude: bool,
    /// The wire has a refusal channel (`refusal`).
    pub refusal: bool,
}

impl SuiteCapabilities {
    /// Build a capability set from manifest names. An unknown name is an
    /// error so a typo in a suite's `manifest:` list fails loudly rather
    /// than silently asserting an empty flag; the macro-expanded test
    /// asserts on it.
    pub fn from_names(names: &[&str]) -> Result<Self, String> {
        let mut caps = Self::default();
        for name in names {
            match *name {
                "partial_tool_args" => caps.partial_tool_args = true,
                "zero_usage_terminal" => caps.zero_usage_terminal = true,
                "bare_terminal" => caps.bare_terminal = true,
                "malformed_frame" => caps.malformed_frame = true,
                "unknown_event_frame" => caps.unknown_event_frame = true,
                "defective_known_frame" => caps.defective_known_frame = true,
                "delta_less_prelude" => caps.delta_less_prelude = true,
                "refusal" => caps.refusal = true,
                other => {
                    return Err(format!(
                        "unknown capability name in suite manifest: {other}"
                    ));
                }
            }
        }
        Ok(caps)
    }
}

/// The canonical fixture-driven scenario set every wire-family suite must
/// expand — one named test each, compared against the macro's emitted list by
/// its `suite_is_complete` test (langchain's anti-tamper precedent).
pub const CANONICAL_SCENARIOS: &[&str] = &[
    "truncation_preserves_content_without_terminal",
    "transport_error_after_tool_call_yields_err_then_end",
    "malformed_frame_surfaces_err_and_terminal_still_completes",
    "unknown_event_is_skipped",
    "defective_known_event_surfaces_err",
    "delta_less_choice_prelude_is_a_noop",
    "refusal_frames_deliver_text_without_error",
    "bare_terminal_after_only_unparseable_frames_fabricates_nothing",
    "usage_variants_are_reported_or_zero_sentinel",
];

/// Every streaming wire family in the workspace. The workspace registry test
/// (`all_wire_families_have_conformance_suites`) fails CI when any family
/// lacks a [`streaming_conformance_suite!`](crate::streaming_conformance_suite)
/// invocation naming it.
pub const WIRE_FAMILIES: &[&str] = &[
    "openai_chat",
    "openai_responses",
    "openai_responses_websocket",
    "chatgpt",
    "anthropic",
    "gemini_rest",
    "gemini_interactions",
    "gemini_grpc",
    "cohere",
    "ollama",
    "xai",
    "copilot",
    "bedrock",
    "candle",
];

/// The sanctioned reason for an expected-failure scenario, from `xfail`
/// entries of the form `"scenario_name: reason (finding reference)"`.
pub fn xfail_reason<'a>(xfail: &[&'a str], scenario: &str) -> Option<&'a str> {
    xfail.iter().find_map(|entry| {
        let (name, reason) = entry.split_once(':')?;
        (name.trim() == scenario).then(|| reason.trim())
    })
}

/// `xfail` entries that do not name a canonical scenario or carry no reason.
pub fn invalid_xfail_entries(xfail: &[&str]) -> Vec<String> {
    xfail
        .iter()
        .filter(|entry| match entry.split_once(':') {
            Some((name, reason)) => {
                !CANONICAL_SCENARIOS.contains(&name.trim()) || reason.trim().is_empty()
            }
            None => true,
        })
        .map(|entry| entry.to_string())
        .collect()
}

/// Enforce a capability-gated scenario's outcome against the suite's declared
/// capability flag and its `xfail` list.
///
/// A `Skipped` outcome passes only when the capability is disclaimed; a `Ran`
/// outcome passes only when it is declared — so a vacuous pass (fixture lacks
/// the shape but the suite claims to cover it) is impossible, and the skip is
/// visible in the test output.
pub fn check_gated_outcome(
    scenario: &'static str,
    capability: bool,
    xfail: &[&str],
    outcome: Result<ScenarioOutcome, ConformanceError>,
) -> Result<(), String> {
    match (xfail_reason(xfail, scenario), outcome) {
        (Some(reason), Err(error)) => {
            eprintln!("xfail {scenario}: {reason} ({error})");
            Ok(())
        }
        (Some(reason), Ok(_)) => Err(format!(
            "{scenario} passed but is listed as xfail ({reason}); remove the xfail entry"
        )),
        (None, Err(error)) => Err(format!("{scenario} failed: {error}")),
        (None, Ok(ScenarioOutcome::Ran(_))) => {
            if capability {
                Ok(())
            } else {
                Err(format!(
                    "{scenario} ran but the suite disclaims the capability; set the flag to true"
                ))
            }
        }
        (None, Ok(ScenarioOutcome::Skipped { reason, .. })) => {
            if capability {
                Err(format!(
                    "{scenario} skipped ({reason}) but the suite declares the capability; \
                     a declared capability's scenario must run"
                ))
            } else {
                eprintln!("skipped {scenario}: {reason}");
                Ok(())
            }
        }
    }
}

/// Enforce an always-runnable scenario's result against the `xfail` list.
pub fn check_ungated_outcome(
    scenario: &'static str,
    xfail: &[&str],
    result: Result<ScenarioReport, ConformanceError>,
) -> Result<(), String> {
    match (xfail_reason(xfail, scenario), result) {
        (Some(reason), Err(error)) => {
            eprintln!("xfail {scenario}: {reason} ({error})");
            Ok(())
        }
        (Some(reason), Ok(_)) => Err(format!(
            "{scenario} passed but is listed as xfail ({reason}); remove the xfail entry"
        )),
        (None, Err(error)) => Err(format!("{scenario} failed: {error}")),
        (None, Ok(_)) => Ok(()),
    }
}

/// One scripted wire input frame.
///
/// Byte-transport wires (SSE, NDJSON, websocket) script raw bytes fed through
/// the provider's HTTP layer; typed-event wires (bedrock, candle,
/// gemini-grpc) script already-typed SDK events fed to the adapter directly —
/// events-first, no mock transport — which the typed driver downcasts back.
#[derive(Clone)]
pub enum WireInput {
    /// A raw wire byte frame.
    Bytes(Bytes),
    /// An already-typed SDK event for a typed-event wire.
    Event(std::sync::Arc<dyn std::any::Any + Send + Sync>),
}

impl WireInput {
    /// The frame's raw bytes, when it is a byte frame.
    pub fn as_bytes(&self) -> Option<&Bytes> {
        match self {
            Self::Bytes(bytes) => Some(bytes),
            Self::Event(_) => None,
        }
    }

    /// The frame's typed event, when it is an event frame of type `T`.
    pub fn downcast_event<T: 'static>(&self) -> Option<&T> {
        match self {
            Self::Bytes(_) => None,
            Self::Event(event) => event.downcast_ref(),
        }
    }
}

impl From<Bytes> for WireInput {
    fn from(bytes: Bytes) -> Self {
        Self::Bytes(bytes)
    }
}

impl std::fmt::Debug for WireInput {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bytes(bytes) => formatter.debug_tuple("Bytes").field(bytes).finish(),
            Self::Event(_) => formatter.write_str("Event(..)"),
        }
    }
}

/// Build a typed-event fixture frame.
pub fn event_frame<T: Send + Sync + 'static>(event: T) -> WireInput {
    WireInput::Event(std::sync::Arc::new(event))
}

/// The wire frames a driver feeds into the provider's pipeline. An `Err`
/// chunk models a mid-stream transport failure.
pub type WireChunks = Vec<http_client::Result<WireInput>>;

/// Build the chunk list for an all-delivered frame sequence.
pub fn ok_chunks(frames: impl IntoIterator<Item = impl Into<WireInput>>) -> WireChunks {
    frames.into_iter().map(|frame| Ok(frame.into())).collect()
}

/// A scripted mid-stream transport failure chunk.
pub fn transport_error_chunk() -> http_client::Result<WireInput> {
    Err(http_client::Error::InvalidStatusCodeWithMessage(
        http::StatusCode::BAD_GATEWAY,
        "connection reset".to_string(),
    ))
}

/// Executable stream-lifecycle validator (#2258 C1).
///
/// The invariants every normalized stream must satisfy, stated once and run
/// over every recorded cassette and corpus fixture that drains through
/// [`fixtures::drain`] — the langchain `assert_valid_event_stream` move:
/// prose contracts scattered across N adapters become one executable
/// artifact. Panics with the violated law.
///
/// Laws (universal — they hold for truncated and errored streams too):
///
/// 1. **Terminal latch.** At most one [`StreamedAssistantContent::Final`],
///    and no content item (text, reasoning, tool call or delta) follows it —
///    only in-band errors and `Unknown` passthrough may.
/// 2. **Text conservation.** The aggregated text is exactly the
///    concatenation of the yielded text deltas: accumulated delta content
///    equals the payload the aggregate delivers.
/// 3. **Completed-call conservation.** Every completed tool call yielded on
///    the stream appears in the aggregated choice exactly once, and vice
///    versa (counts match; aggregation neither drops nor duplicates).
/// 4. **Delta-before-completion.** A completed call correlated with
///    fragments (same `internal_call_id`) never precedes its own deltas.
/// 5. **Reasoning provenance.** The aggregate contains a reasoning part only
///    if the stream yielded reasoning items; and when only deltas were
///    yielded (no full block), the aggregated reasoning text is exactly
///    their concatenation.
pub fn assert_valid_event_stream(
    items: &[Result<crate::streaming::StreamedAssistantContent, CompletionError>],
    choice: &OneOrMany<AssistantContent>,
) {
    use crate::message::AssistantContent;
    use crate::streaming::StreamedAssistantContent as Item;

    let ok_items: Vec<&Item> = items.iter().filter_map(|item| item.as_ref().ok()).collect();

    // Law 1: terminal latch.
    let final_count = ok_items
        .iter()
        .filter(|item| matches!(item, Item::Final(_)))
        .count();
    assert!(
        final_count <= 1,
        "law 1 (terminal latch): {final_count} terminal records yielded"
    );
    if let Some(final_index) = ok_items
        .iter()
        .position(|item| matches!(item, Item::Final(_)))
    {
        for item in ok_items.get(final_index + 1..).unwrap_or_default() {
            assert!(
                matches!(item, Item::Unknown(_)),
                "law 1 (terminal latch): content item after the terminal record: {item:?}"
            );
        }
    }

    // Law 2: text conservation.
    let streamed_text: String = ok_items
        .iter()
        .filter_map(|item| match item {
            Item::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect();
    let aggregated_text: String = choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        aggregated_text, streamed_text,
        "law 2 (text conservation): aggregated text differs from the streamed deltas"
    );

    // Law 3: completed-call conservation.
    let yielded_calls = ok_items
        .iter()
        .filter(|item| matches!(item, Item::ToolCall { .. }))
        .count();
    let aggregated_calls = choice
        .iter()
        .filter(|content| matches!(content, AssistantContent::ToolCall(_)))
        .count();
    assert_eq!(
        aggregated_calls, yielded_calls,
        "law 3 (completed-call conservation): {yielded_calls} calls yielded, \
         {aggregated_calls} aggregated"
    );

    // Law 4: delta-before-completion.
    let mut seen_delta_ids: Vec<&str> = Vec::new();
    let mut completed_ids: Vec<&str> = Vec::new();
    for item in &ok_items {
        match item {
            Item::ToolCallDelta {
                internal_call_id, ..
            } => {
                assert!(
                    !completed_ids.contains(&internal_call_id.as_str()),
                    "law 4: a delta for internal id {internal_call_id} arrived after its \
                     completed call"
                );
                seen_delta_ids.push(internal_call_id);
            }
            Item::ToolCall {
                internal_call_id, ..
            } => completed_ids.push(internal_call_id),
            _ => {}
        }
    }

    // Law 5: reasoning provenance.
    let yielded_reasoning = ok_items
        .iter()
        .any(|item| matches!(item, Item::Reasoning(_) | Item::ReasoningDelta { .. }));
    let aggregated_reasoning = choice
        .iter()
        .any(|content| matches!(content, AssistantContent::Reasoning(_)));
    assert!(
        yielded_reasoning || !aggregated_reasoning,
        "law 5 (reasoning provenance): aggregated reasoning with no reasoning yielded"
    );
    let yielded_full_block = ok_items
        .iter()
        .any(|item| matches!(item, Item::Reasoning(_)));
    if yielded_reasoning && !yielded_full_block {
        let streamed_reasoning: String = ok_items
            .iter()
            .filter_map(|item| match item {
                Item::ReasoningDelta { reasoning, .. } => Some(reasoning.as_str()),
                _ => None,
            })
            .collect();
        let aggregated_reasoning_text: String = choice
            .iter()
            .filter_map(|content| match content {
                AssistantContent::Reasoning(reasoning) => Some(reasoning.content.iter()),
                _ => None,
            })
            .flatten()
            .filter_map(|part| match part {
                crate::message::ReasoningContent::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            aggregated_reasoning_text, streamed_reasoning,
            "law 5 (reasoning conservation): with no full block, the aggregated reasoning \
             must be exactly the concatenated deltas"
        );
    }
}

/// Everything the consumer observed from one full pipeline run: the yielded
/// items in order, plus the aggregated choice and terminal record.
#[derive(Debug)]
pub struct DrainedStream {
    /// Every item the stream yielded, in order.
    pub items: Vec<Result<StreamedAssistantContent, CompletionError>>,
    /// The final aggregated assistant message.
    pub choice: OneOrMany<AssistantContent>,
    /// The normalized terminal record, absent on truncation or terminal error.
    pub response: Option<StreamFinal>,
}

impl DrainedStream {
    /// Text deltas yielded to the consumer, in order.
    pub fn texts(&self) -> Vec<&str> {
        self.items
            .iter()
            .filter_map(|item| match item {
                Ok(StreamedAssistantContent::Text(text)) => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }

    /// Names of the complete tool calls yielded to the consumer, in order.
    pub fn tool_call_names(&self) -> Vec<&str> {
        self.items
            .iter()
            .filter_map(|item| match item {
                Ok(StreamedAssistantContent::ToolCall { tool_call, .. }) => {
                    Some(tool_call.function.name.as_str())
                }
                _ => None,
            })
            .collect()
    }

    /// Raw payloads of the `Unknown` passthrough items the stream yielded,
    /// in order.
    pub fn unknown_values(&self) -> Vec<&serde_json::Value> {
        self.items
            .iter()
            .filter_map(|item| match item {
                Ok(StreamedAssistantContent::Unknown(value)) => Some(value),
                _ => None,
            })
            .collect()
    }

    /// Number of `Err` items the stream yielded.
    pub fn error_count(&self) -> usize {
        self.items.iter().filter(|item| item.is_err()).count()
    }

    /// Number of terminal records the stream yielded.
    pub fn final_count(&self) -> usize {
        self.items
            .iter()
            .filter(|item| matches!(item, Ok(StreamedAssistantContent::Final(_))))
            .count()
    }

    /// Index of the first `Err` item, if any.
    fn first_error_index(&self) -> Option<usize> {
        self.items.iter().position(|item| item.is_err())
    }

    /// Text blocks in the aggregated choice, in order.
    pub fn choice_texts(&self) -> Vec<&str> {
        self.choice
            .iter()
            .filter_map(|content| match content {
                AssistantContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }

    /// Reasoning items in the aggregated choice, in order.
    pub fn choice_reasoning(&self) -> Vec<&crate::message::Reasoning> {
        self.choice
            .iter()
            .filter_map(|content| match content {
                AssistantContent::Reasoning(reasoning) => Some(reasoning),
                _ => None,
            })
            .collect()
    }

    /// Names of the tool calls in the aggregated choice, in order.
    pub fn choice_tool_call_names(&self) -> Vec<&str> {
        self.choice
            .iter()
            .filter_map(|content| match content {
                AssistantContent::ToolCall(tool_call) => Some(tool_call.function.name.as_str()),
                _ => None,
            })
            .collect()
    }
}

type DriveFn = Box<
    dyn Fn(WireChunks) -> BoxFuture<'static, Result<DrainedStream, CompletionError>> + Send + Sync,
>;

/// One provider's full streaming pipeline over scripted wire chunks.
///
/// The closure builds a fresh provider client over a scripted HTTP double
/// (`SequencedStreamingHttpClient`), opens `CompletionModel::stream`, drains
/// it, and returns everything the consumer observed.
pub struct WireDriver {
    /// Stable descriptor name of the provider under test.
    pub provider: &'static str,
    drive: DriveFn,
}

impl WireDriver {
    /// Wrap a provider pipeline closure.
    pub fn new(
        provider: &'static str,
        drive: impl Fn(WireChunks) -> BoxFuture<'static, Result<DrainedStream, CompletionError>>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self {
            provider,
            drive: Box::new(drive),
        }
    }

    /// Run the provider's full pipeline over `chunks` and drain it.
    pub async fn drive(&self, chunks: WireChunks) -> Result<DrainedStream, CompletionError> {
        (self.drive)(chunks).await
    }
}

/// Refusal frames and the text the pipeline must deliver for them.
pub struct RefusalFixture {
    /// Frames carrying the refusal content.
    pub frames: Vec<WireInput>,
    /// Text the consumer must observe.
    pub expected_text: &'static str,
}

type BufferedDriveFn = Box<
    dyn Fn(String) -> BoxFuture<'static, Result<OneOrMany<AssistantContent>, CompletionError>>
        + Send
        + Sync,
>;

/// A buffered-body pipeline (the ChatGPT backend shape): the full SSE body is
/// re-parsed after the fact and merged with the terminal response body.
pub struct BufferedBodyDriver {
    /// Stable descriptor name of the provider under test.
    pub provider: &'static str,
    drive: BufferedDriveFn,
}

impl BufferedBodyDriver {
    /// Wrap a buffered pipeline closure.
    pub fn new(
        provider: &'static str,
        drive: impl Fn(
            String,
        )
            -> BoxFuture<'static, Result<OneOrMany<AssistantContent>, CompletionError>>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self {
            provider,
            drive: Box::new(drive),
        }
    }

    /// Run the buffered pipeline over a complete SSE body.
    pub async fn drive(
        &self,
        body: String,
    ) -> Result<OneOrMany<AssistantContent>, CompletionError> {
        (self.drive)(body).await
    }
}

/// Per-provider wire frames for the shared scenario set.
///
/// `Option` fields cover sequence shapes a wire family cannot spell (e.g.
/// ollama's NDJSON has no event types, so no "unknown event type" frame).
pub struct ProviderWireFixture {
    /// The provider's full pipeline.
    pub driver: WireDriver,
    /// Frames that deliver exactly the text deltas in `expected_texts`.
    pub text_frames: Vec<WireInput>,
    /// The text deltas `text_frames` delivers, in order.
    pub expected_texts: Vec<&'static str>,
    /// Frames that fully deliver one tool call (including any completion
    /// signal the wire needs, but no stream terminal).
    pub tool_call_frames: Vec<WireInput>,
    /// Name of the tool call `tool_call_frames` delivers.
    pub expected_tool_name: &'static str,
    /// Frames that leave a tool call mid-arguments, where the wire streams
    /// arguments incrementally.
    pub partial_tool_call_frames: Option<Vec<WireInput>>,
    /// The provider's genuine stream terminal, carrying usage.
    pub terminal_frames: Vec<WireInput>,
    /// Total tokens `terminal_frames` reports.
    pub expected_usage_total: u64,
    /// Finish reason `terminal_frames` reports.
    pub expected_finish_reason: Option<FinishReason>,
    /// A genuine terminal that reports no usage metrics at all.
    pub zero_usage_terminal_frames: Option<Vec<WireInput>>,
    /// A terminal signal that carries no data of its own (e.g. a bare
    /// `[DONE]`), for wires that have one.
    pub bare_terminal_frames: Option<Vec<WireInput>>,
    /// A frame that fails the wire decode entirely. `None` only for
    /// typed-event wires, whose SDK surfaces decode failures as transport
    /// errors — a frame-level corrupt input cannot be spelled there.
    pub malformed_frame: Option<WireInput>,
    /// An event type this client does not know, for typed-event wires.
    pub unknown_event_frame: Option<WireInput>,
    /// A known event whose payload is schema-defective.
    pub defective_known_frame: Option<WireInput>,
    /// A delta-less choice prelude (the Azure `prompt_filter_results` shape).
    pub delta_less_prelude_frame: Option<WireInput>,
    /// Refusal content frames, where the wire has a refusal channel.
    pub refusal: Option<RefusalFixture>,
}

impl ProviderWireFixture {
    /// The capability set this fixture's populated optional fields spell —
    /// the descriptor the suite macro gates scenarios on.
    ///
    /// Deriving flags here (instead of hand-writing them per suite
    /// invocation) makes flag/fixture drift structurally impossible: a shape
    /// the fixture supplies is a declared capability, a shape it lacks is a
    /// visible named skip, and there is nothing else to keep in sync.
    pub fn capabilities(&self) -> SuiteCapabilities {
        SuiteCapabilities {
            partial_tool_args: self.partial_tool_call_frames.is_some(),
            zero_usage_terminal: self.zero_usage_terminal_frames.is_some(),
            bare_terminal: self.bare_terminal_frames.is_some(),
            malformed_frame: self.malformed_frame.is_some(),
            unknown_event_frame: self.unknown_event_frame.is_some(),
            defective_known_frame: self.defective_known_frame.is_some(),
            delta_less_prelude: self.delta_less_prelude_frame.is_some(),
            refusal: self.refusal.is_some(),
        }
    }
}

fn concat_frames(parts: &[&[WireInput]]) -> Vec<WireInput> {
    parts
        .iter()
        .flat_map(|frames| frames.iter().cloned())
        .collect()
}

/// Truncation at every position — EOF before content, mid-text, mid-tool-args,
/// after a fully-delivered tool call — must preserve delivered content and
/// never produce a terminal record.
///
/// Pins the truncation family from round one (`rig-2257-code-review-findings-ec9f2625.md`):
/// EOF without the provider's end event must not synthesize a successful
/// zero-usage terminal.
pub async fn truncation_preserves_content_without_terminal(
    fixture: &ProviderWireFixture,
) -> Result<ScenarioReport, ConformanceError> {
    const SCENARIO: &str = "truncation_preserves_content_without_terminal";
    let provider = fixture.driver.provider;
    let mut observations = Vec::new();

    // EOF before any content.
    let drained = fixture.driver.drive(Vec::new()).await?;
    if drained.response.is_some() || drained.final_count() != 0 {
        return Err(ConformanceError::contract(
            SCENARIO,
            provider,
            "an empty stream must not synthesize a terminal record",
        ));
    }
    observations.push("EOF before content: no terminal".to_string());

    // EOF after text deltas.
    let drained = fixture
        .driver
        .drive(ok_chunks(fixture.text_frames.clone()))
        .await?;
    if drained.texts() != fixture.expected_texts {
        return Err(ConformanceError::contract(
            SCENARIO,
            provider,
            format!(
                "text delivered before truncation must be preserved: expected {:?}, observed {:?}",
                fixture.expected_texts,
                drained.texts()
            ),
        ));
    }
    if drained.response.is_some() || drained.final_count() != 0 {
        return Err(ConformanceError::contract(
            SCENARIO,
            provider,
            "EOF after text deltas must not synthesize a terminal record",
        ));
    }
    observations.push("EOF mid-text: content preserved, no terminal".to_string());

    // EOF mid-tool-arguments, where the wire streams arguments.
    if let Some(partial) = &fixture.partial_tool_call_frames {
        let drained = fixture.driver.drive(ok_chunks(partial.clone())).await?;
        if drained.response.is_some() || drained.final_count() != 0 {
            return Err(ConformanceError::contract(
                SCENARIO,
                provider,
                "EOF mid-tool-arguments must not synthesize a terminal record",
            ));
        }
        observations.push("EOF mid-tool-args: no terminal".to_string());
    }

    // EOF after a fully-delivered tool call, before the stream terminal.
    let drained = fixture
        .driver
        .drive(ok_chunks(fixture.tool_call_frames.clone()))
        .await?;
    if drained.tool_call_names() != vec![fixture.expected_tool_name] {
        return Err(ConformanceError::contract(
            SCENARIO,
            provider,
            format!(
                "a fully-delivered tool call must survive truncation: observed {:?}",
                drained.tool_call_names()
            ),
        ));
    }
    if drained.response.is_some() || drained.final_count() != 0 {
        return Err(ConformanceError::contract(
            SCENARIO,
            provider,
            "EOF after a delivered tool call must not synthesize a terminal record",
        ));
    }
    observations.push("EOF after tool-complete: tool call preserved, no terminal".to_string());

    Ok(ScenarioReport {
        name: SCENARIO,
        provider,
        observations,
    })
}

/// A transport failure after a fully-delivered tool call must yield the tool
/// call, then the `Err`, then end — with no terminal record after the error.
///
/// Pins the flush-before-terminal-error ordering from round five
/// (`rig-2257-code-review-findings-5c73639c.md`): a first-`Err`-stop consumer
/// must still see delivered tool calls.
pub async fn transport_error_after_tool_call_yields_err_then_end(
    fixture: &ProviderWireFixture,
) -> Result<ScenarioReport, ConformanceError> {
    const SCENARIO: &str = "transport_error_after_tool_call_yields_err_then_end";
    let provider = fixture.driver.provider;

    let mut chunks = ok_chunks(fixture.tool_call_frames.clone());
    chunks.push(transport_error_chunk());
    let drained = fixture.driver.drive(chunks).await?;

    if drained.tool_call_names() != vec![fixture.expected_tool_name] {
        return Err(ConformanceError::contract(
            SCENARIO,
            provider,
            format!(
                "the delivered tool call must precede the transport error: observed {:?}",
                drained.tool_call_names()
            ),
        ));
    }
    let error_index = drained.first_error_index().ok_or_else(|| {
        ConformanceError::contract(
            SCENARIO,
            provider,
            "the transport failure must reach the consumer",
        )
    })?;
    if error_index + 1 != drained.items.len() {
        return Err(ConformanceError::contract(
            SCENARIO,
            provider,
            "nothing may follow the terminal transport error",
        ));
    }
    if drained.response.is_some() || drained.final_count() != 0 {
        return Err(ConformanceError::contract(
            SCENARIO,
            provider,
            "a transport failure must not be papered over with a terminal record",
        ));
    }

    Ok(ScenarioReport {
        name: SCENARIO,
        provider,
        observations: vec!["tool call, then Err, then end; no terminal".to_string()],
    })
}

/// A malformed frame between valid content and the genuine terminal must
/// surface as an `Err` item while the stream keeps consuming, so the terminal
/// still completes it.
///
/// Pins the malformed-frame policy row of the [`StreamFinal`] contract table
/// (round four, `rig-2257-code-review-findings-1e5a7ad8.md`).
pub async fn malformed_frame_surfaces_err_and_terminal_still_completes(
    fixture: &ProviderWireFixture,
) -> Result<ScenarioOutcome, ConformanceError> {
    const SCENARIO: &str = "malformed_frame_surfaces_err_and_terminal_still_completes";
    let provider = fixture.driver.provider;
    let Some(malformed) = &fixture.malformed_frame else {
        return Ok(ScenarioOutcome::Skipped {
            name: SCENARIO,
            provider,
            reason: "wire family cannot spell a frame-level decode failure",
        });
    };

    let frames = concat_frames(&[
        &fixture.text_frames,
        std::slice::from_ref(malformed),
        &fixture.terminal_frames,
    ]);
    let drained = fixture.driver.drive(ok_chunks(frames)).await?;

    if drained.error_count() != 1 {
        return Err(ConformanceError::contract(
            SCENARIO,
            provider,
            format!(
                "the malformed frame must surface as exactly one Err item, observed {}",
                drained.error_count()
            ),
        ));
    }
    if drained.texts() != fixture.expected_texts {
        return Err(ConformanceError::contract(
            SCENARIO,
            provider,
            "content around the malformed frame must be preserved",
        ));
    }
    if drained.response.is_none() {
        return Err(ConformanceError::contract(
            SCENARIO,
            provider,
            "the genuine terminal after a recoverable parse error must still complete the stream",
        ));
    }

    Ok(ScenarioOutcome::Ran(ScenarioReport {
        name: SCENARIO,
        provider,
        observations: vec!["Err surfaced, terminal still completed".to_string()],
    }))
}

/// An event type the client does not know must be skipped without an error,
/// and the stream must still complete.
///
/// Pins the unknown-event forward-compatibility policy (round three,
/// `rig-2257-code-review-findings-8a2f41c7.md`).
pub async fn unknown_event_is_skipped(
    fixture: &ProviderWireFixture,
) -> Result<ScenarioOutcome, ConformanceError> {
    const SCENARIO: &str = "unknown_event_is_skipped";
    let provider = fixture.driver.provider;
    let Some(unknown) = &fixture.unknown_event_frame else {
        return Ok(ScenarioOutcome::Skipped {
            name: SCENARIO,
            provider,
            reason: "wire family cannot spell an unknown event type",
        });
    };

    let frames = concat_frames(&[
        &fixture.text_frames,
        std::slice::from_ref(unknown),
        &fixture.terminal_frames,
    ]);
    let drained = fixture.driver.drive(ok_chunks(frames)).await?;

    if drained.error_count() != 0 {
        return Err(ConformanceError::contract(
            SCENARIO,
            provider,
            "an unknown event type must be skipped, not surfaced as an error",
        ));
    }
    if drained.texts() != fixture.expected_texts || drained.response.is_none() {
        return Err(ConformanceError::contract(
            SCENARIO,
            provider,
            "the stream must deliver its content and complete around the skipped event",
        ));
    }
    // The frame is skipped semantically but observable verbatim on the raw
    // passthrough channel (openai-agents' raw-event precedent, #2258 item 5).
    if drained.unknown_values().len() != 1 {
        return Err(ConformanceError::contract(
            SCENARIO,
            provider,
            format!(
                "exactly one Unknown passthrough item must surface for the unknown frame, \
                 observed {}",
                drained.unknown_values().len()
            ),
        ));
    }

    // Control run without the unknown frame: the aggregated assistant choice
    // must be byte-identical — the passthrough item is never folded in.
    let control_frames = concat_frames(&[&fixture.text_frames, &fixture.terminal_frames]);
    let control = fixture.driver.drive(ok_chunks(control_frames)).await?;
    if drained.choice != control.choice {
        return Err(ConformanceError::contract(
            SCENARIO,
            provider,
            "the unknown frame must not perturb the aggregated assistant choice",
        ));
    }

    Ok(ScenarioOutcome::Ran(ScenarioReport {
        name: SCENARIO,
        provider,
        observations: vec![
            "unknown event skipped semantically, surfaced on the raw channel, \
             choice unchanged, stream completed"
                .to_string(),
        ],
    }))
}

/// A *known* event whose payload is schema-defective must surface as an `Err`
/// item (and the stream keeps consuming to the genuine terminal).
///
/// Pins the round-5 known-type strictness policy and its silent revert for
/// OpenAI Responses content parts — the open P2 in
/// `rig-2257-code-review-findings-34ee8ba5.md` ("Round-5 known-type strictness
/// silently reverted for content parts").
pub async fn defective_known_event_surfaces_err(
    fixture: &ProviderWireFixture,
) -> Result<ScenarioOutcome, ConformanceError> {
    const SCENARIO: &str = "defective_known_event_surfaces_err";
    let provider = fixture.driver.provider;
    let Some(defective) = &fixture.defective_known_frame else {
        return Ok(ScenarioOutcome::Skipped {
            name: SCENARIO,
            provider,
            reason: "wire family cannot spell a known event with a schema-defective payload",
        });
    };

    let frames = concat_frames(&[
        &fixture.text_frames,
        std::slice::from_ref(defective),
        &fixture.terminal_frames,
    ]);
    let drained = fixture.driver.drive(ok_chunks(frames)).await?;

    if drained.error_count() != 1 {
        return Err(ConformanceError::contract(
            SCENARIO,
            provider,
            format!(
                "a known event with a schema defect must surface exactly one Err item, observed {}",
                drained.error_count()
            ),
        ));
    }
    if drained.response.is_none() {
        return Err(ConformanceError::contract(
            SCENARIO,
            provider,
            "the genuine terminal must still complete the stream after the defective frame",
        ));
    }

    Ok(ScenarioOutcome::Ran(ScenarioReport {
        name: SCENARIO,
        provider,
        observations: vec!["defective known event surfaced as Err; stream completed".to_string()],
    }))
}

/// A delta-less choice (the Azure `prompt_filter_results` prelude) must be a
/// no-op — no error, no content, and the rest of the stream unaffected.
///
/// Pins the Azure prelude no-op from round two
/// (`rig-2257-code-review-findings-b91d03aa.md`).
pub async fn delta_less_choice_prelude_is_a_noop(
    fixture: &ProviderWireFixture,
) -> Result<ScenarioOutcome, ConformanceError> {
    const SCENARIO: &str = "delta_less_choice_prelude_is_a_noop";
    let provider = fixture.driver.provider;
    let Some(prelude) = &fixture.delta_less_prelude_frame else {
        return Ok(ScenarioOutcome::Skipped {
            name: SCENARIO,
            provider,
            reason: "wire family has no delta-less prelude shape",
        });
    };

    let frames = concat_frames(&[
        std::slice::from_ref(prelude),
        &fixture.text_frames,
        &fixture.terminal_frames,
    ]);
    let drained = fixture.driver.drive(ok_chunks(frames)).await?;

    if drained.error_count() != 0 {
        return Err(ConformanceError::contract(
            SCENARIO,
            provider,
            "the delta-less prelude must not surface an error",
        ));
    }
    if drained.texts() != fixture.expected_texts || drained.response.is_none() {
        return Err(ConformanceError::contract(
            SCENARIO,
            provider,
            "the prelude must not perturb content delivery or the terminal",
        ));
    }

    Ok(ScenarioOutcome::Ran(ScenarioReport {
        name: SCENARIO,
        provider,
        observations: vec!["delta-less prelude ignored; stream unaffected".to_string()],
    }))
}

/// Refusal frames must deliver their text to the consumer without an error.
///
/// Pins the refusal-delta handling from round three
/// (`rig-2257-code-review-findings-8a2f41c7.md`).
pub async fn refusal_frames_deliver_text_without_error(
    fixture: &ProviderWireFixture,
) -> Result<ScenarioOutcome, ConformanceError> {
    const SCENARIO: &str = "refusal_frames_deliver_text_without_error";
    let provider = fixture.driver.provider;
    let Some(refusal) = &fixture.refusal else {
        return Ok(ScenarioOutcome::Skipped {
            name: SCENARIO,
            provider,
            reason: "wire family has no refusal channel",
        });
    };

    let frames = concat_frames(&[&refusal.frames, &fixture.terminal_frames]);
    let drained = fixture.driver.drive(ok_chunks(frames)).await?;

    if drained.error_count() != 0 {
        return Err(ConformanceError::contract(
            SCENARIO,
            provider,
            "refusal content must not surface as an error",
        ));
    }
    let delivered = drained.texts().concat();
    if delivered != refusal.expected_text {
        return Err(ConformanceError::contract(
            SCENARIO,
            provider,
            format!(
                "refusal text must be delivered: expected {:?}, observed {delivered:?}",
                refusal.expected_text
            ),
        ));
    }
    if drained.response.is_none() {
        return Err(ConformanceError::contract(
            SCENARIO,
            provider,
            "a refused turn still ends with the provider's genuine terminal",
        ));
    }

    Ok(ScenarioOutcome::Ran(ScenarioReport {
        name: SCENARIO,
        provider,
        observations: vec!["refusal text delivered without error".to_string()],
    }))
}

/// On the buffered-body pipeline (the ChatGPT backend), a terminal whose body
/// carries text never seen as a delta must merge that text into the choice
/// exactly once, and a body restating streamed deltas must not duplicate them.
///
/// Pins the terminal-body/delta per-kind merge from round five
/// (`rig-2257-code-review-findings-5c73639c.md`) and the empty-delta merge
/// direction verified in round six (`rig-2257-code-review-findings-34ee8ba5.md`
/// P3-2).
pub async fn terminal_body_content_merges_per_kind(
    driver: &BufferedBodyDriver,
    cases: Vec<(&'static str, String)>,
    expected_text: &str,
) -> Result<ScenarioReport, ConformanceError> {
    const SCENARIO: &str = "terminal_body_content_merges_per_kind";
    let provider = driver.provider;
    let mut observations = Vec::new();

    for (label, body) in cases {
        let choice = driver.drive(body).await?;
        let choice_text: String = choice
            .iter()
            .filter_map(|content| match content {
                AssistantContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect();
        let occurrences = choice_text.matches(expected_text).count();
        if occurrences != 1 {
            return Err(ConformanceError::contract(
                SCENARIO,
                provider,
                format!(
                    "{label}: terminal-body text must appear exactly once in the choice, observed {occurrences} in {choice_text:?}"
                ),
            ));
        }
        observations.push(format!("{label}: text merged exactly once"));
    }

    Ok(ScenarioReport {
        name: SCENARIO,
        provider,
        observations,
    })
}

/// A bare terminal signal after only-unparseable frames must not fabricate a
/// successful terminal record: the parse errors were already surfaced, and a
/// default-usage terminal would dress the failure up as success.
///
/// Pins the bare-`[DONE]` guard from round six
/// (`rig-2257-code-review-findings-5c73639c.md`, carried into `34ee8ba5`).
pub async fn bare_terminal_after_only_unparseable_frames_fabricates_nothing(
    fixture: &ProviderWireFixture,
) -> Result<ScenarioOutcome, ConformanceError> {
    const SCENARIO: &str = "bare_terminal_after_only_unparseable_frames_fabricates_nothing";
    let provider = fixture.driver.provider;
    let Some(bare_terminal) = &fixture.bare_terminal_frames else {
        return Ok(ScenarioOutcome::Skipped {
            name: SCENARIO,
            provider,
            reason: "wire family has no data-less terminal signal",
        });
    };
    let Some(malformed) = &fixture.malformed_frame else {
        return Ok(ScenarioOutcome::Skipped {
            name: SCENARIO,
            provider,
            reason: "wire family cannot spell a frame-level decode failure",
        });
    };

    let frames = concat_frames(&[std::slice::from_ref(malformed), bare_terminal]);
    let drained = fixture.driver.drive(ok_chunks(frames)).await?;

    if drained.error_count() == 0 {
        return Err(ConformanceError::contract(
            SCENARIO,
            provider,
            "the unparseable frame must surface as an Err item",
        ));
    }
    if drained.response.is_some() || drained.final_count() != 0 {
        return Err(ConformanceError::contract(
            SCENARIO,
            provider,
            "a bare terminal with no decoded frame must not fabricate a terminal record",
        ));
    }

    Ok(ScenarioOutcome::Ran(ScenarioReport {
        name: SCENARIO,
        provider,
        observations: vec!["no fabricated terminal after only-unparseable frames".to_string()],
    }))
}

/// The genuine terminal must report the provider's usage; a terminal without
/// usage metrics must complete with the documented zero-usage sentinel rather
/// than being suppressed or invented.
///
/// Pins the zero-usage-sentinel contract on [`StreamFinal::usage`]
/// (round one, `rig-2257-code-review-findings-ec9f2625.md`).
pub async fn usage_variants_are_reported_or_zero_sentinel(
    fixture: &ProviderWireFixture,
) -> Result<ScenarioReport, ConformanceError> {
    const SCENARIO: &str = "usage_variants_are_reported_or_zero_sentinel";
    let provider = fixture.driver.provider;
    let mut observations = Vec::new();

    let frames = concat_frames(&[&fixture.text_frames, &fixture.terminal_frames]);
    let drained = fixture.driver.drive(ok_chunks(frames)).await?;
    let response = drained.response.as_ref().ok_or_else(|| {
        ConformanceError::contract(
            SCENARIO,
            provider,
            "the genuine terminal must produce a record",
        )
    })?;
    if response.usage.total_tokens != fixture.expected_usage_total {
        return Err(ConformanceError::contract(
            SCENARIO,
            provider,
            format!(
                "terminal usage must be preserved: expected total {}, observed {}",
                fixture.expected_usage_total, response.usage.total_tokens
            ),
        ));
    }
    if response.finish_reason != fixture.expected_finish_reason {
        return Err(ConformanceError::contract(
            SCENARIO,
            provider,
            format!(
                "terminal finish reason must be normalized: expected {:?}, observed {:?}",
                fixture.expected_finish_reason, response.finish_reason
            ),
        ));
    }
    observations.push(format!(
        "usage total {} and finish reason {:?} preserved",
        fixture.expected_usage_total, fixture.expected_finish_reason
    ));

    if let Some(zero_usage) = &fixture.zero_usage_terminal_frames {
        let frames = concat_frames(&[&fixture.text_frames, zero_usage]);
        let drained = fixture.driver.drive(ok_chunks(frames)).await?;
        let response = drained.response.as_ref().ok_or_else(|| {
            ConformanceError::contract(
                SCENARIO,
                provider,
                "a usage-less genuine terminal must still complete the stream",
            )
        })?;
        if response.usage.total_tokens != 0 {
            return Err(ConformanceError::contract(
                SCENARIO,
                provider,
                "missing usage metrics must be the zero-usage sentinel, not invented values",
            ));
        }
        observations.push("usage-less terminal completed with the zero sentinel".to_string());
    }

    Ok(ScenarioReport {
        name: SCENARIO,
        provider,
        observations,
    })
}

/// Reasoning-summary deltas followed by the item's full `output_item.done`
/// block must aggregate to the summary exactly once — the full block
/// supersedes its own deltas, never duplicates them.
///
/// Pins the open P1 in `rig-2257-code-review-findings-34ee8ba5.md` ("OpenAI
/// Responses reasoning-summary streams duplicate reasoning content"):
/// `reasoning_summary_text.delta` drops `item_id`, so the strict same-item
/// table appends the full block beside the delta-built item.
pub async fn reasoning_summary_deltas_are_superseded_without_duplication(
    driver: &WireDriver,
    frames: Vec<WireInput>,
    summary_text: &str,
) -> Result<ScenarioReport, ConformanceError> {
    const SCENARIO: &str = "reasoning_summary_deltas_are_superseded_without_duplication";
    let provider = driver.provider;

    let drained = driver.drive(ok_chunks(frames)).await?;
    if drained.error_count() != 0 || drained.response.is_none() {
        return Err(ConformanceError::contract(
            SCENARIO,
            provider,
            "the reasoning stream must complete without errors",
        ));
    }
    let reasoning = drained.choice_reasoning();
    let occurrences: usize = reasoning
        .iter()
        .flat_map(|item| item.content.iter())
        .filter(|content| match content {
            crate::message::ReasoningContent::Summary(text)
            | crate::message::ReasoningContent::Text { text, .. } => text.contains(summary_text),
            _ => false,
        })
        .count();
    if occurrences != 1 {
        return Err(ConformanceError::contract(
            SCENARIO,
            provider,
            format!(
                "the summary must appear exactly once in the aggregated choice, observed {occurrences} across {reasoning:?}"
            ),
        ));
    }
    if reasoning.len() != 1 {
        return Err(ConformanceError::contract(
            SCENARIO,
            provider,
            format!(
                "deltas and their full block must collapse to one reasoning item, observed {}",
                reasoning.len()
            ),
        ));
    }

    Ok(ScenarioReport {
        name: SCENARIO,
        provider,
        observations: vec!["summary aggregated exactly once".to_string()],
    })
}

/// A reasoning item whose `output_item.done` carries several parts under one
/// item id (summary parts, text, encrypted) must keep every part, in order —
/// same-id sibling blocks append, they never replace each other.
///
/// Pins the open P1 in `rig-2257-code-review-findings-34ee8ba5.md` ("The by-id
/// fallback collapses multi-part same-id reasoning items"): the `rposition`
/// fallback replaces the just-appended same-id sibling.
pub async fn multi_part_same_id_reasoning_keeps_every_part(
    driver: &WireDriver,
    frames: Vec<WireInput>,
    expected_parts: &[&str],
) -> Result<ScenarioReport, ConformanceError> {
    const SCENARIO: &str = "multi_part_same_id_reasoning_keeps_every_part";
    let provider = driver.provider;

    let drained = driver.drive(ok_chunks(frames)).await?;
    if drained.error_count() != 0 || drained.response.is_none() {
        return Err(ConformanceError::contract(
            SCENARIO,
            provider,
            "the reasoning stream must complete without errors",
        ));
    }
    let observed: Vec<String> = drained
        .choice_reasoning()
        .iter()
        .flat_map(|item| item.content.iter())
        .map(|content| match content {
            crate::message::ReasoningContent::Summary(text) => text.clone(),
            crate::message::ReasoningContent::Text { text, .. } => text.clone(),
            crate::message::ReasoningContent::Encrypted(data) => data.clone(),
            crate::message::ReasoningContent::Redacted { data } => data.clone(),
        })
        .collect();
    if observed != expected_parts {
        return Err(ConformanceError::contract(
            SCENARIO,
            provider,
            format!(
                "every same-id reasoning part must survive in order: expected {expected_parts:?}, observed {observed:?}"
            ),
        ));
    }

    Ok(ScenarioReport {
        name: SCENARIO,
        provider,
        observations: vec![format!(
            "all {} reasoning parts survived",
            expected_parts.len()
        )],
    })
}

/// Reasoning deltas interleaved with a tool call, then the item's completed
/// block, must aggregate to exactly one reasoning item carrying the block's
/// content.
///
/// Pins the interleaved-reasoning replacement contract on
/// [`StreamedAssistantContent::Reasoning`] (round six,
/// `rig-2257-code-review-findings-34ee8ba5.md`, "Verified sound" section).
pub async fn interleaved_reasoning_aggregates_to_one_item(
    driver: &WireDriver,
    frames: Vec<WireInput>,
    expected_text: &str,
) -> Result<ScenarioReport, ConformanceError> {
    const SCENARIO: &str = "interleaved_reasoning_aggregates_to_one_item";
    let provider = driver.provider;

    let drained = driver.drive(ok_chunks(frames)).await?;
    if drained.error_count() != 0 || drained.response.is_none() {
        return Err(ConformanceError::contract(
            SCENARIO,
            provider,
            "the interleaved stream must complete without errors",
        ));
    }
    let reasoning = drained.choice_reasoning();
    if reasoning.len() != 1 {
        return Err(ConformanceError::contract(
            SCENARIO,
            provider,
            format!(
                "interleaved deltas and their completed block must collapse to one reasoning item, observed {}",
                reasoning.len()
            ),
        ));
    }
    let carries_text = reasoning
        .iter()
        .flat_map(|item| item.content.iter())
        .any(|content| match content {
            crate::message::ReasoningContent::Summary(text)
            | crate::message::ReasoningContent::Text { text, .. } => text == expected_text,
            _ => false,
        });
    if !carries_text {
        return Err(ConformanceError::contract(
            SCENARIO,
            provider,
            format!("the reasoning item must carry the completed block's text {expected_text:?}"),
        ));
    }

    Ok(ScenarioReport {
        name: SCENARIO,
        provider,
        observations: vec!["exactly one reasoning item with the completed content".to_string()],
    })
}

/// On a constant-id wire (a boundary-minted per-stream reasoning id), other
/// output closes the open reasoning item: thought → tool call → thought must
/// aggregate as `[Reasoning(first), ToolCall, Reasoning(second)]` — two items
/// in arrival order, never one merged item that misorders history on replay.
///
/// Pins the F1b ordering dimension of the #2258 review (main's
/// "other output closes the reasoning item" boundary, lost when identity
/// became the per-stream constant).
pub async fn interleaved_constant_id_reasoning_preserves_order(
    driver: &WireDriver,
    frames: Vec<WireInput>,
    first: &str,
    tool_name: &str,
    second: &str,
) -> Result<ScenarioReport, ConformanceError> {
    const SCENARIO: &str = "interleaved_constant_id_reasoning_preserves_order";
    let provider = driver.provider;

    let drained = driver.drive(ok_chunks(frames)).await?;
    if drained.error_count() != 0 || drained.response.is_none() {
        return Err(ConformanceError::contract(
            SCENARIO,
            provider,
            "the interleaved stream must complete without errors",
        ));
    }
    assert_reasoning_tool_reasoning(SCENARIO, provider, &drained, first, tool_name, second)?;

    Ok(ScenarioReport {
        name: SCENARIO,
        provider,
        observations: vec!["boundary kept: reasoning, tool call, reasoning in order".to_string()],
    })
}

/// On a constant-id wire whose completed reasoning block arrives as a signed
/// full restatement (gemini `thoughtSignature`), a full block *after*
/// interleaved output must not replace-and-discard the thought accumulated
/// before the boundary: the choice keeps `[Reasoning(first), ToolCall,
/// Reasoning(second, signed)]`.
///
/// Pins the F1b erasure dimension of the #2258 review, on top of the F1
/// adapter fix (the signed chunk restates only post-boundary fragments).
pub async fn interleaved_signed_full_reasoning_does_not_erase_prior_thought(
    driver: &WireDriver,
    frames: Vec<WireInput>,
    first: &str,
    tool_name: &str,
    second: &str,
) -> Result<ScenarioReport, ConformanceError> {
    const SCENARIO: &str = "interleaved_signed_full_reasoning_does_not_erase_prior_thought";
    let provider = driver.provider;

    let drained = driver.drive(ok_chunks(frames)).await?;
    if drained.error_count() != 0 || drained.response.is_none() {
        return Err(ConformanceError::contract(
            SCENARIO,
            provider,
            "the interleaved stream must complete without errors",
        ));
    }
    assert_reasoning_tool_reasoning(SCENARIO, provider, &drained, first, tool_name, second)?;
    let signed = drained.choice_reasoning().last().is_some_and(|reasoning| {
        reasoning.content.iter().any(|content| {
            matches!(
                content,
                crate::message::ReasoningContent::Text {
                    signature: Some(_),
                    ..
                }
            )
        })
    });
    if !signed {
        return Err(ConformanceError::contract(
            SCENARIO,
            provider,
            "the post-boundary block must keep its signature",
        ));
    }

    Ok(ScenarioReport {
        name: SCENARIO,
        provider,
        observations: vec![
            "pre-boundary thought survived; signed block completed the post-boundary part"
                .to_string(),
        ],
    })
}

/// Shared assertion: the aggregated choice is exactly
/// `[Reasoning(first), ToolCall(tool_name), Reasoning(second…)]`.
fn assert_reasoning_tool_reasoning(
    scenario: &'static str,
    provider: &'static str,
    drained: &DrainedStream,
    first: &str,
    tool_name: &str,
    second: &str,
) -> Result<(), ConformanceError> {
    let shape: Vec<String> = drained
        .choice
        .iter()
        .map(|content| match content {
            AssistantContent::Reasoning(reasoning) => {
                let text: String = reasoning
                    .content
                    .iter()
                    .filter_map(|content| match content {
                        crate::message::ReasoningContent::Summary(text)
                        | crate::message::ReasoningContent::Text { text, .. } => {
                            Some(text.as_str())
                        }
                        _ => None,
                    })
                    .collect();
                format!("reasoning:{text}")
            }
            AssistantContent::ToolCall(tool_call) => {
                format!("tool:{}", tool_call.function.name)
            }
            AssistantContent::Text(text) => format!("text:{}", text.text),
            AssistantContent::Image(_) => "image".to_string(),
        })
        .collect();
    let expected = vec![
        format!("reasoning:{first}"),
        format!("tool:{tool_name}"),
        format!("reasoning:{second}"),
    ];
    if shape != expected {
        return Err(ConformanceError::contract(
            scenario,
            provider,
            format!(
                "the boundary must survive aggregation: expected {expected:?}, observed {shape:?}"
            ),
        ));
    }
    Ok(())
}

/// Drain one OpenAI Responses *websocket* turn's server events into
/// everything a streaming consumer would observe, through the SAME decode
/// state machine the production session drives
/// (`RawChoiceAccumulator` + `normalize_responses_stream`).
///
/// The websocket pipeline is request/response: `next_event` has no in-band
/// `Err` channel, so the caller collects events (stopping at the first
/// terminal or session error) and this helper replays them. One policy the
/// helper supplies that the buffered session cannot: tool calls the provider
/// fully delivered flush before a session error, mirroring the SSE loop's
/// flush-before-terminal-error contract (`RawChoiceAccumulator::take_tool_calls`).
#[cfg(all(not(target_family = "wasm"), feature = "websocket"))]
pub async fn drain_openai_responses_websocket_events(
    provider: &'static str,
    events: Vec<
        Result<
            crate::providers::openai::responses_api::websocket::ResponsesWebSocketEvent,
            CompletionError,
        >,
    >,
) -> DrainedStream {
    use crate::providers::openai::responses_api::ResponsesUsage;
    use crate::providers::openai::responses_api::streaming::{
        RawChoiceAccumulator, ResponseChunkKind, ResponsesStreamOptions, normalize_responses_stream,
    };
    use crate::providers::openai::responses_api::websocket::ResponsesWebSocketEvent;

    let mut accumulator = RawChoiceAccumulator::new(ResponsesUsage::new());
    let mut raw = Vec::new();
    let mut errored = false;
    for event in events {
        match event {
            Ok(ResponsesWebSocketEvent::Item(chunk)) => raw.extend(
                accumulator
                    .decode_item_chunk(chunk, ResponsesStreamOptions::strict())
                    .into_iter()
                    .map(Ok),
            ),
            Ok(ResponsesWebSocketEvent::Response(chunk)) => {
                let terminal = matches!(
                    chunk.kind,
                    ResponseChunkKind::ResponseCompleted
                        | ResponseChunkKind::ResponseFailed
                        | ResponseChunkKind::ResponseIncomplete
                );
                if let Err(error) =
                    accumulator.record_response_chunk(chunk.kind, chunk.response, "")
                {
                    raw.extend(accumulator.take_tool_calls().into_iter().map(Ok));
                    raw.push(Err(error));
                    errored = true;
                    break;
                }
                if terminal {
                    break;
                }
            }
            // Semantic skip, raw passthrough: an unknown frame never reaches
            // the accumulator but is still yielded verbatim.
            Ok(ResponsesWebSocketEvent::Unknown(value)) => {
                raw.push(Ok(crate::streaming::RawStreamingChoice::Unknown(value)));
            }
            // `response.done` / `error` envelopes are websocket-only shapes the
            // fixtures never script; the production session maps them to a
            // terminal or a provider error before this replay runs.
            Ok(ResponsesWebSocketEvent::Done(_)) => {}
            Ok(ResponsesWebSocketEvent::Error(error)) => {
                raw.extend(accumulator.take_tool_calls().into_iter().map(Ok));
                raw.push(Err(CompletionError::ProviderError(error.to_string())));
                errored = true;
                break;
            }
            Err(error) => {
                raw.extend(accumulator.take_tool_calls().into_iter().map(Ok));
                raw.push(Err(error));
                errored = true;
                break;
            }
        }
    }
    if !errored {
        raw.extend(accumulator.finish().into_iter().map(Ok));
    }

    let stream = normalize_responses_stream(provider, Box::pin(futures::stream::iter(raw)));
    fixtures::drain(stream).await
}

/// Per-provider wire fixtures for the shared scenario set.
pub mod fixtures {
    use super::*;
    use crate::client::CompletionClient;
    use crate::completion::CompletionModel;
    use crate::test_utils::SequencedStreamingHttpClient;
    use serde_json::json;

    /// Drain a full normalized stream into everything the consumer observed.
    /// Public so provider-crate conformance suites (the typed-event wires)
    /// can reuse it in their drivers.
    pub async fn drain(mut stream: crate::streaming::StreamingCompletionResponse) -> DrainedStream {
        let mut items = Vec::new();
        while let Some(item) = stream.next().await {
            items.push(item);
        }
        let drained = DrainedStream {
            items,
            choice: stream.choice.clone(),
            response: stream.response.clone(),
        };
        // Every fixture and cassette that drains through this helper runs
        // the lifecycle validator — the prose invariants as one executable
        // artifact (#2258 C1).
        super::assert_valid_event_stream(&drained.items, &drained.choice);
        drained
    }

    /// Lower fixture frames onto the byte transport a `SequencedStreamingHttpClient`
    /// replays. Only byte frames are valid here — an event frame in a
    /// byte-driver fixture is a fixture authoring error.
    fn byte_chunks(chunks: WireChunks) -> Result<Vec<http_client::Result<Bytes>>, CompletionError> {
        chunks
            .into_iter()
            .map(|chunk| match chunk {
                Ok(WireInput::Bytes(bytes)) => Ok(Ok(bytes)),
                Ok(WireInput::Event(_)) => Err(CompletionError::ProviderError(
                    "typed-event frame fed to a byte-transport driver".to_string(),
                )),
                Err(error) => Ok(Err(error)),
            })
            .collect()
    }

    fn sse(frame: &serde_json::Value) -> WireInput {
        WireInput::Bytes(Bytes::from(format!("data: {frame}\n\n")))
    }

    fn sse_raw(data: &str) -> WireInput {
        WireInput::Bytes(Bytes::from(format!("data: {data}\n\n")))
    }

    fn ndjson(frame: &serde_json::Value) -> WireInput {
        WireInput::Bytes(Bytes::from(format!("{frame}\n")))
    }

    /// The frame's SSE text, for buffered-body pipelines that re-parse a
    /// whole body string.
    fn frame_text(frame: &WireInput) -> String {
        frame
            .as_bytes()
            .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
            .unwrap_or_default()
    }

    /// OpenAI chat-completions wire (the shared OpenAI-compatible SSE path).
    pub mod openai_chat {
        use super::*;

        fn driver() -> WireDriver {
            WireDriver::new("openai", |chunks| {
                Box::pin(async move {
                    let client = crate::providers::openai::Client::builder()
                        .http_client(SequencedStreamingHttpClient::new(byte_chunks(chunks)?))
                        .api_key("test-key")
                        .build()?
                        .completions_api();
                    let model = client.completion_model("gpt-4o");
                    let request = model.completion_request("hello").build();
                    let stream = model.stream(request).await?;
                    Ok(drain(stream).await)
                })
            })
        }

        /// The chat-completions fixture.
        pub fn fixture() -> ProviderWireFixture {
            ProviderWireFixture {
                driver: driver(),
                text_frames: vec![sse(&json!({
                    "id": "chatcmpl-1",
                    "model": "gpt-4o-2024-08-06",
                    "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": null}],
                    "usage": null,
                }))],
                expected_texts: vec!["hi"],
                tool_call_frames: vec![
                    sse(&json!({
                        "choices": [{"index": 0, "delta": {"tool_calls": [{
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": ""},
                        }]}, "finish_reason": null}],
                    })),
                    sse(&json!({
                        "choices": [{"index": 0, "delta": {"tool_calls": [{
                            "index": 0,
                            "function": {"arguments": "{\"city\":\"Tokyo\"}"},
                        }]}, "finish_reason": null}],
                    })),
                    // No `finish_reason` chunk: on the chat wire that IS the
                    // terminal signal, and these frames must stop short of it.
                    // EOF/error cleanup still flushes the completed call.
                ],
                expected_tool_name: "get_weather",
                partial_tool_call_frames: Some(vec![sse(&json!({
                    "choices": [{"index": 0, "delta": {"tool_calls": [{
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"cit"},
                    }]}, "finish_reason": null}],
                }))]),
                terminal_frames: vec![
                    sse(&json!({
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        "usage": null,
                    })),
                    sse(&json!({
                        "choices": [],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    })),
                    sse_raw("[DONE]"),
                ],
                expected_usage_total: 15,
                expected_finish_reason: Some(FinishReason::Stop),
                zero_usage_terminal_frames: Some(vec![
                    sse(&json!({
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        "usage": null,
                    })),
                    sse_raw("[DONE]"),
                ]),
                bare_terminal_frames: Some(vec![sse_raw("[DONE]")]),
                malformed_frame: Some(sse_raw("{not json")),
                unknown_event_frame: None,
                // A wrongly-typed `content` is tolerated by the lenient delta
                // decode; a wrongly-typed `choices` is a genuine schema defect
                // of the known chunk shape.
                defective_known_frame: Some(sse_raw(r#"{"choices": 42}"#)),
                // The Azure `prompt_filter_results` prelude: a choice with no
                // `delta` at all.
                delta_less_prelude_frame: Some(sse_raw(
                    r#"{"id":"","object":"","choices":[{"prompt_index":0,"content_filter_results":{"hate":{"filtered":false,"severity":"safe"}}}]}"#,
                )),
                refusal: None,
            }
        }
    }

    /// OpenAI Responses API wire.
    pub mod openai_responses {
        use super::*;

        /// The driver alone, for the reasoning-specific scenarios.
        pub fn driver() -> WireDriver {
            WireDriver::new("openai", |chunks| {
                Box::pin(async move {
                    let client = crate::providers::openai::Client::builder()
                        .http_client(SequencedStreamingHttpClient::new(byte_chunks(chunks)?))
                        .api_key("test-key")
                        .build()?;
                    let model = client.completion_model("gpt-5.4");
                    let request = model.completion_request("hello").build();
                    let stream = model.stream(request).await?;
                    Ok(drain(stream).await)
                })
            })
        }

        fn completed_response(
            usage: Option<serde_json::Value>,
            output: serde_json::Value,
        ) -> serde_json::Value {
            json!({
                "id": "resp_1",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "gpt-5.4",
                "output": output,
                "tools": [],
                "usage": usage,
            })
        }

        fn terminal(usage: Option<serde_json::Value>, output: serde_json::Value) -> WireInput {
            sse(&json!({
                "type": "response.completed",
                "sequence_number": 99,
                "response": completed_response(usage, output),
            }))
        }

        fn usage_json() -> serde_json::Value {
            json!({
                "input_tokens": 10,
                "output_tokens": 5,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 15,
            })
        }

        fn text_delta(text: &str) -> WireInput {
            sse(&json!({
                "type": "response.output_text.delta",
                "content_index": 0,
                "delta": text,
                "item_id": "msg_1",
                "output_index": 0,
                "sequence_number": 1,
            }))
        }

        fn tool_call_done() -> WireInput {
            sse(&json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "sequence_number": 2,
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "arguments": "{\"city\":\"Tokyo\"}",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "status": "completed",
                },
            }))
        }

        /// Synthetic twin of the recorded
        /// `openai/streaming_grammar/incomplete_mid_tool_call` cassette: a
        /// forced tool call cut by `max_output_tokens` mid-arguments. The wire
        /// restates the call on `response.output_item.done` with the
        /// arguments truncated mid-JSON and item status `incomplete`, then
        /// ends with a genuine `response.incomplete` terminal.
        pub fn incomplete_mid_tool_call_frames() -> Vec<WireInput> {
            vec![
                sse(&json!({
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "sequence_number": 1,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "arguments": "",
                        "call_id": "call_1",
                        "name": "add",
                        "status": "in_progress",
                    },
                })),
                sse(&json!({
                    "type": "response.function_call_arguments.delta",
                    "item_id": "fc_1",
                    "output_index": 0,
                    "sequence_number": 2,
                    "delta": "{\"x",
                })),
                sse(&json!({
                    "type": "response.function_call_arguments.delta",
                    "item_id": "fc_1",
                    "output_index": 0,
                    "sequence_number": 3,
                    "delta": "\":48151",
                })),
                sse(&json!({
                    "type": "response.function_call_arguments.done",
                    "item_id": "fc_1",
                    "output_index": 0,
                    "sequence_number": 4,
                    "arguments": "{\"x\":48151",
                })),
                sse(&json!({
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "sequence_number": 5,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "arguments": "{\"x\":48151",
                        "call_id": "call_1",
                        "name": "add",
                        "status": "incomplete",
                    },
                })),
                sse(&json!({
                    "type": "response.incomplete",
                    "sequence_number": 6,
                    "response": {
                        "id": "resp_1",
                        "object": "response",
                        "created_at": 0,
                        "status": "incomplete",
                        "incomplete_details": {"reason": "max_output_tokens"},
                        "model": "gpt-5.4",
                        "output": [{
                            "type": "function_call",
                            "id": "fc_1",
                            "arguments": "{\"x\":48151",
                            "call_id": "call_1",
                            "name": "add",
                            "status": "incomplete",
                        }],
                        "tools": [],
                        "usage": usage_json(),
                    },
                })),
            ]
        }

        fn reasoning_done_item(
            id: &str,
            summary: serde_json::Value,
            content: serde_json::Value,
            encrypted: Option<&str>,
        ) -> WireInput {
            let mut item = json!({
                "type": "reasoning",
                "id": id,
                "summary": summary,
                "content": content,
                "status": "completed",
            });
            if let (Some(encrypted), Some(object)) = (encrypted, item.as_object_mut()) {
                object.insert("encrypted_content".to_string(), json!(encrypted));
            }
            sse(&json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "sequence_number": 3,
                "item": item,
            }))
        }

        /// The Responses-API fixture.
        pub fn fixture() -> ProviderWireFixture {
            ProviderWireFixture {
                driver: driver(),
                text_frames: vec![text_delta("hi")],
                expected_texts: vec!["hi"],
                tool_call_frames: vec![tool_call_done()],
                expected_tool_name: "get_weather",
                partial_tool_call_frames: Some(vec![
                    sse(&json!({
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "sequence_number": 1,
                        "item": {
                            "type": "function_call",
                            "id": "fc_1",
                            "arguments": "",
                            "call_id": "call_1",
                            "name": "get_weather",
                            "status": "in_progress",
                        },
                    })),
                    sse(&json!({
                        "type": "response.function_call_arguments.delta",
                        "item_id": "fc_1",
                        "output_index": 0,
                        "sequence_number": 2,
                        "delta": "{\"cit",
                    })),
                ]),
                terminal_frames: vec![terminal(Some(usage_json()), json!([]))],
                expected_usage_total: 15,
                expected_finish_reason: Some(FinishReason::Stop),
                zero_usage_terminal_frames: Some(vec![terminal(None, json!([]))]),
                bare_terminal_frames: None,
                malformed_frame: Some(sse_raw("{not json")),
                unknown_event_frame: Some(sse(&json!({
                    "type": "response.web_search_call.searching",
                    "output_index": 0,
                    "sequence_number": 4,
                    "item_id": "ws_1",
                }))),
                // The P2 probe shape from `rig-2257-code-review-findings-34ee8ba5.md`:
                // a known part tag (`output_text`) with a schema-defective payload.
                defective_known_frame: Some(sse(&json!({
                    "type": "response.content_part.added",
                    "item_id": "msg_1",
                    "output_index": 0,
                    "content_index": 0,
                    "sequence_number": 5,
                    "part": {"type": "output_text", "text": 42},
                }))),
                delta_less_prelude_frame: None,
                refusal: Some(RefusalFixture {
                    frames: vec![sse(&json!({
                        "type": "response.refusal.delta",
                        "content_index": 0,
                        "delta": "I cannot help with that.",
                        "item_id": "msg_1",
                        "output_index": 0,
                        "sequence_number": 1,
                    }))],
                    expected_text: "I cannot help with that.",
                }),
            }
        }

        /// The buffered-body pipeline the ChatGPT backend uses: the SSE body
        /// is re-parsed after the fact and merged with the terminal response
        /// body, per content kind.
        ///
        /// Drives the *real* entry — `CompletionModel::completion` on a
        /// ChatGPT client whose HTTP double answers the `/responses` POST
        /// with the scripted SSE body — so the scenario exercises
        /// `normalized_completion` itself rather than a mirrored copy of its
        /// fallback logic (#2258 review, F8 drift risk).
        pub fn buffered_driver() -> BufferedBodyDriver {
            BufferedBodyDriver::new("chatgpt", |body| {
                Box::pin(async move {
                    let client = crate::providers::chatgpt::Client::builder()
                        .api_key(crate::providers::chatgpt::ChatGPTAuth::AccessToken {
                            access_token: "test-token".to_string(),
                            account_id: Some("account-id".to_string()),
                        })
                        .http_client(crate::test_utils::RecordingHttpClient::new(body))
                        .build()?;
                    let model = client.completion_model("gpt-5.4");
                    let request = model.completion_request("hello").build();
                    let response = model.completion(request).await?;
                    Ok(response.choice)
                })
            })
        }

        fn message_output(text: &str) -> serde_json::Value {
            json!([{
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }])
        }

        /// A terminal whose body carries text never seen as a delta.
        pub fn terminal_body_only_sse_body(text: &str) -> String {
            frame_text(&terminal(Some(usage_json()), message_output(text)))
        }

        /// A streamed delta plus a terminal body restating the same text.
        pub fn terminal_body_and_delta_sse_body(text: &str) -> String {
            let frames = [
                text_delta(text),
                terminal(Some(usage_json()), message_output(text)),
            ];
            frames.iter().map(frame_text).collect()
        }

        /// A streamed delta whose terminal body carries no output items — the
        /// gpt-5.x shape the buffered fallback exists for.
        pub fn delta_only_sse_body(text: &str) -> String {
            let frames = [text_delta(text), terminal(Some(usage_json()), json!([]))];
            frames.iter().map(frame_text).collect()
        }

        /// The ChatGPT envelope-less replay shape (#2258 F3): a summary delta
        /// with NO envelope bookkeeping at all (repair injects
        /// `output_index: 0`, minting the `output-0` identity), then the
        /// item's envelope-full `output_item.done` restating the summary
        /// under its real `rs_*` id, then the terminal. The done item must
        /// adopt the minted per-slot identity and supersede the delta build.
        pub fn envelope_less_reasoning_supersede_sse_body() -> (String, &'static str) {
            let delta = json!({
                "type": "response.reasoning_summary_text.delta",
                "delta": "step 1",
            });
            let frames = [
                sse(&delta),
                reasoning_done_item(
                    "rs_1",
                    json!([{"type": "summary_text", "text": "step 1"}]),
                    json!([]),
                    None,
                ),
                terminal(Some(usage_json()), json!([])),
            ];
            (frames.iter().map(frame_text).collect(), "step 1")
        }

        /// Summary deltas followed by their item's full `output_item.done`
        /// block, then the terminal. The deltas carry `item_id` on the wire;
        /// the full block restates the summary.
        pub fn reasoning_summary_supersede_frames() -> (Vec<WireInput>, &'static str) {
            let frames = vec![
                sse(&json!({
                    "type": "response.reasoning_summary_text.delta",
                    "item_id": "rs_1",
                    "output_index": 0,
                    "summary_index": 0,
                    "sequence_number": 1,
                    "delta": "step 1",
                })),
                reasoning_done_item(
                    "rs_1",
                    json!([{"type": "summary_text", "text": "step 1"}]),
                    json!([]),
                    None,
                ),
                terminal(Some(usage_json()), json!([])),
            ];
            (frames, "step 1")
        }

        /// One reasoning item done-block carrying two summary parts, visible
        /// text, and encrypted content under a single item id.
        pub fn multi_part_reasoning_frames() -> (Vec<WireInput>, Vec<&'static str>) {
            let frames = vec![
                reasoning_done_item(
                    "rs_1",
                    json!([
                        {"type": "summary_text", "text": "s1"},
                        {"type": "summary_text", "text": "s2"},
                    ]),
                    json!([{"type": "reasoning_text", "text": "visible"}]),
                    Some("enc_blob"),
                ),
                terminal(Some(usage_json()), json!([])),
            ];
            (frames, vec!["s1", "s2", "visible", "enc_blob"])
        }

        /// A reasoning delta, an interleaved tool call, then the reasoning
        /// item's completed block and the terminal.
        pub fn interleaved_reasoning_frames() -> (Vec<WireInput>, &'static str) {
            let frames = vec![
                sse(&json!({
                    "type": "response.reasoning_text.delta",
                    "item_id": "rs_2",
                    "output_index": 0,
                    "content_index": 0,
                    "sequence_number": 1,
                    "delta": "thinking",
                })),
                tool_call_done(),
                reasoning_done_item(
                    "rs_2",
                    json!([]),
                    json!([{"type": "reasoning_text", "text": "full reasoning"}]),
                    None,
                ),
                terminal(Some(usage_json()), json!([])),
            ];
            (frames, "full reasoning")
        }
    }

    /// Gemini REST (`streamGenerateContent`) SSE wire.
    pub mod gemini_rest {
        use super::*;

        fn driver() -> WireDriver {
            WireDriver::new("gemini", |chunks| {
                Box::pin(async move {
                    let client = crate::providers::gemini::Client::builder()
                        .api_key("test-key")
                        .http_client(SequencedStreamingHttpClient::new(byte_chunks(chunks)?))
                        .build()?;
                    let model = client.completion_model(
                        crate::providers::gemini::completion::GEMINI_2_5_PRO_PREVIEW_06_05,
                    );
                    let request = model.completion_request("hello").build();
                    let stream = model.stream(request).await?;
                    Ok(drain(stream).await)
                })
            })
        }

        /// The Gemini REST fixture.
        pub fn fixture() -> ProviderWireFixture {
            ProviderWireFixture {
                driver: driver(),
                text_frames: vec![sse(&json!({
                    "candidates": [{"content": {"parts": [{"text": "hi"}], "role": "model"}}],
                    "responseId": "resp-1",
                    "modelVersion": "gemini-2.5-pro",
                }))],
                expected_texts: vec!["hi"],
                tool_call_frames: vec![sse(&json!({
                    "candidates": [{"content": {"parts": [{
                        "functionCall": {"name": "get_weather", "args": {"city": "Tokyo"}},
                    }], "role": "model"}}],
                    "responseId": "resp-1",
                    "modelVersion": "gemini-2.5-pro",
                }))],
                expected_tool_name: "get_weather",
                // Gemini delivers tool calls whole; arguments never stream.
                partial_tool_call_frames: None,
                terminal_frames: vec![sse(&json!({
                    "candidates": [{
                        "content": {"parts": [], "role": "model"},
                        "finishReason": "STOP",
                    }],
                    "usageMetadata": {
                        "promptTokenCount": 5,
                        "candidatesTokenCount": 2,
                        "totalTokenCount": 7,
                    },
                    "responseId": "resp-1",
                    "modelVersion": "gemini-2.5-pro",
                }))],
                expected_usage_total: 7,
                expected_finish_reason: Some(FinishReason::Stop),
                zero_usage_terminal_frames: Some(vec![sse(&json!({
                    "candidates": [{
                        "content": {"parts": [], "role": "model"},
                        "finishReason": "STOP",
                    }],
                    "responseId": "resp-1",
                    "modelVersion": "gemini-2.5-pro",
                }))]),
                bare_terminal_frames: None,
                malformed_frame: Some(sse_raw("{not json")),
                // The wire has no event tag; valid JSON carrying neither
                // `candidates` nor `usageMetadata` is unrecognizable and must
                // be warn-skipped, not silently decoded as an empty chunk.
                unknown_event_frame: Some(sse_raw(r#"{"noise":true}"#)),
                defective_known_frame: Some(sse_raw(r#"{"candidates": 42}"#)),
                delta_less_prelude_frame: None,
                refusal: None,
            }
        }

        fn chunk(parts: serde_json::Value) -> WireInput {
            sse(&json!({
                "candidates": [{"content": {"parts": parts, "role": "model"}}],
                "responseId": "resp-1",
                "modelVersion": "gemini-2.5-pro",
            }))
        }

        fn terminal_frame() -> WireInput {
            sse(&json!({
                "candidates": [{
                    "content": {"parts": [], "role": "model"},
                    "finishReason": "STOP",
                }],
                "usageMetadata": {
                    "promptTokenCount": 5,
                    "candidatesTokenCount": 2,
                    "totalTokenCount": 7,
                },
                "responseId": "resp-1",
                "modelVersion": "gemini-2.5-pro",
            }))
        }

        /// Thought delta, interleaved tool call, thought delta, terminal —
        /// the constant-id (`reasoning-0`) interleaving shape.
        pub fn interleaved_thought_frames()
        -> (Vec<WireInput>, &'static str, &'static str, &'static str) {
            let frames = vec![
                chunk(json!([{"text": "before tool", "thought": true}])),
                chunk(json!([{
                    "functionCall": {"name": "get_weather", "args": {"city": "Tokyo"}},
                }])),
                chunk(json!([{"text": "after tool", "thought": true}])),
                terminal_frame(),
            ];
            (frames, "before tool", "get_weather", "after tool")
        }

        /// Thought delta, interleaved tool call, then a signed full thought
        /// chunk carrying non-empty text — the F1 erasure shape.
        pub fn interleaved_signed_thought_frames()
        -> (Vec<WireInput>, &'static str, &'static str, &'static str) {
            let frames = vec![
                chunk(json!([{"text": "before tool", "thought": true}])),
                chunk(json!([{
                    "functionCall": {"name": "get_weather", "args": {"city": "Tokyo"}},
                }])),
                chunk(json!([{
                    "text": "signed conclusion",
                    "thought": true,
                    "thoughtSignature": "sig-1",
                }])),
                terminal_frame(),
            ];
            (frames, "before tool", "get_weather", "signed conclusion")
        }
    }

    /// Gemini Interactions SSE wire (`event_type`-tagged events).
    pub mod interactions {
        use super::*;

        fn driver() -> WireDriver {
            WireDriver::new("gemini", |chunks| {
                Box::pin(async move {
                    let client = crate::providers::gemini::Client::builder()
                        .api_key("test-key")
                        .http_client(SequencedStreamingHttpClient::new(byte_chunks(chunks)?))
                        .build()?
                        .interactions_api();
                    let model = client.completion_model("gemini-2.5-pro");
                    let request = model.completion_request("hello").build();
                    let stream = model.stream(request).await?;
                    Ok(drain(stream).await)
                })
            })
        }

        fn completed(usage: Option<serde_json::Value>) -> WireInput {
            let mut interaction = json!({
                "id": "int-1",
                "model": "gemini-2.5-pro",
                "status": "completed",
            });
            if let (Some(usage), Some(object)) = (usage, interaction.as_object_mut()) {
                object.insert("usage".to_string(), usage);
            }
            sse(&json!({
                "event_type": "interaction.completed",
                "interaction": interaction,
            }))
        }

        /// The Interactions fixture.
        pub fn fixture() -> ProviderWireFixture {
            ProviderWireFixture {
                driver: driver(),
                text_frames: vec![sse(&json!({
                    "event_type": "step.delta",
                    "index": 0,
                    "delta": {"type": "text", "text": "hi"},
                }))],
                expected_texts: vec!["hi"],
                tool_call_frames: vec![sse(&json!({
                    "event_type": "step.delta",
                    "index": 0,
                    "delta": {
                        "type": "function_call",
                        "name": "get_weather",
                        "arguments": {"city": "Tokyo"},
                        "id": "call-1",
                    },
                }))],
                expected_tool_name: "get_weather",
                // The Interactions wire delivers function calls whole;
                // arguments never stream.
                partial_tool_call_frames: None,
                terminal_frames: vec![completed(Some(json!({
                    "total_input_tokens": 5,
                    "total_output_tokens": 2,
                    "total_tokens": 7,
                })))],
                expected_usage_total: 7,
                expected_finish_reason: Some(FinishReason::Stop),
                zero_usage_terminal_frames: Some(vec![completed(None)]),
                bare_terminal_frames: None,
                malformed_frame: Some(sse_raw("{not json")),
                unknown_event_frame: Some(sse(&json!({
                    "event_type": "future.event",
                    "index": 0,
                }))),
                // A known tag (`step.delta`) with a schema-defective payload
                // must classify `Corrupt`, never `Unknown`.
                defective_known_frame: Some(sse_raw(
                    r#"{"event_type":"step.delta","index":0,"delta":42}"#,
                )),
                delta_less_prelude_frame: None,
                refusal: None,
            }
        }

        /// Thought-summary delta, interleaved function call, thought-summary
        /// delta, terminal — the constant-id (`reasoning-0`) interleaving
        /// shape on the Interactions wire.
        pub fn interleaved_thought_frames()
        -> (Vec<WireInput>, &'static str, &'static str, &'static str) {
            let frames = vec![
                sse(&json!({
                    "event_type": "step.delta",
                    "index": 0,
                    "delta": {
                        "type": "thought_summary",
                        "content": {"text": "before tool"},
                    },
                })),
                sse(&json!({
                    "event_type": "step.delta",
                    "index": 0,
                    "delta": {
                        "type": "function_call",
                        "name": "get_weather",
                        "arguments": {"city": "Tokyo"},
                        "id": "call-1",
                    },
                })),
                sse(&json!({
                    "event_type": "step.delta",
                    "index": 0,
                    "delta": {
                        "type": "thought_summary",
                        "content": {"text": "after tool"},
                    },
                })),
                completed(Some(json!({
                    "total_input_tokens": 5,
                    "total_output_tokens": 2,
                    "total_tokens": 7,
                }))),
            ];
            (frames, "before tool", "get_weather", "after tool")
        }
    }

    /// Anthropic Messages SSE wire (`type`-tagged events, index-as-id blocks).
    pub mod anthropic {
        use super::*;

        fn driver() -> WireDriver {
            WireDriver::new("anthropic", |chunks| {
                Box::pin(async move {
                    let client = crate::providers::anthropic::Client::builder()
                        .api_key("test-key")
                        .http_client(SequencedStreamingHttpClient::new(byte_chunks(chunks)?))
                        .build()?;
                    let model = client.completion_model(
                        crate::providers::anthropic::completion::CLAUDE_SONNET_4_6,
                    );
                    let request = model.completion_request("hello").build();
                    let stream = model.stream(request).await?;
                    Ok(drain(stream).await)
                })
            })
        }

        fn message_start() -> WireInput {
            sse(&json!({
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "role": "assistant",
                    "content": [],
                    "model": "claude-sonnet-4-6",
                    "stop_reason": null,
                    "stop_sequence": null,
                    "usage": {"input_tokens": 5, "output_tokens": 0},
                },
            }))
        }

        /// The Anthropic fixture.
        pub fn fixture() -> ProviderWireFixture {
            ProviderWireFixture {
                driver: driver(),
                text_frames: vec![
                    message_start(),
                    sse(&json!({
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    })),
                    sse(&json!({
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": "hi"},
                    })),
                ],
                expected_texts: vec!["hi"],
                tool_call_frames: vec![
                    sse(&json!({
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "get_weather",
                            "input": {},
                        },
                    })),
                    sse(&json!({
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "input_json_delta", "partial_json": "{\"city\":\"Tokyo\"}"},
                    })),
                    // `content_block_stop` completes the call; the stream
                    // terminal (`message_delta`) is deliberately absent.
                    sse(&json!({"type": "content_block_stop", "index": 0})),
                ],
                expected_tool_name: "get_weather",
                partial_tool_call_frames: Some(vec![
                    sse(&json!({
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "get_weather",
                            "input": {},
                        },
                    })),
                    sse(&json!({
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "input_json_delta", "partial_json": "{\"cit"},
                    })),
                ]),
                terminal_frames: vec![sse(&json!({
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": null},
                    "usage": {"output_tokens": 4},
                }))],
                // input 5 (from message_start) + output 4.
                expected_usage_total: 9,
                expected_finish_reason: Some(FinishReason::Stop),
                // The Anthropic terminal always carries `usage`; there is no
                // usage-less genuine terminal to spell on this wire.
                zero_usage_terminal_frames: None,
                // `message_stop` carries no data of its own and must not
                // fabricate a terminal record.
                bare_terminal_frames: Some(vec![sse(&json!({"type": "message_stop"}))]),
                malformed_frame: Some(sse_raw("{not json")),
                unknown_event_frame: Some(sse(&json!({
                    "type": "content_block_heartbeat",
                    "index": 0,
                }))),
                // A known tag (`content_block_delta`) with a schema-defective
                // payload must classify `Corrupt`, never `Unknown`.
                defective_known_frame: Some(sse_raw(
                    r#"{"type":"content_block_delta","index":0,"delta":42}"#,
                )),
                delta_less_prelude_frame: None,
                refusal: None,
            }
        }
    }

    /// Cohere v2 chat SSE wire.
    pub mod cohere {
        use super::*;

        fn driver() -> WireDriver {
            WireDriver::new("cohere", |chunks| {
                Box::pin(async move {
                    let client = crate::providers::cohere::Client::builder()
                        .api_key("test-key")
                        .http_client(SequencedStreamingHttpClient::new(byte_chunks(chunks)?))
                        .build()?;
                    let model = client.completion_model(crate::providers::cohere::COMMAND_R);
                    let request = model.completion_request("hello").build();
                    let stream = model.stream(request).await?;
                    Ok(drain(stream).await)
                })
            })
        }

        /// The Cohere fixture.
        pub fn fixture() -> ProviderWireFixture {
            ProviderWireFixture {
                driver: driver(),
                text_frames: vec![
                    sse(&json!({"type": "message-start", "id": "msg_1"})),
                    sse(&json!({
                        "type": "content-delta",
                        "delta": {"message": {"content": {"text": "hi"}}},
                    })),
                ],
                expected_texts: vec!["hi"],
                tool_call_frames: vec![
                    sse(&json!({
                        "type": "tool-call-start",
                        "delta": {"message": {"tool_calls": {
                            "id": "call_1",
                            "function": {"name": "get_weather", "arguments": ""},
                        }}},
                    })),
                    sse(&json!({
                        "type": "tool-call-delta",
                        "delta": {"message": {"tool_calls": {
                            "function": {"arguments": "{\"city\":\"Tokyo\"}"},
                        }}},
                    })),
                    sse(&json!({"type": "tool-call-end"})),
                ],
                expected_tool_name: "get_weather",
                partial_tool_call_frames: Some(vec![sse(&json!({
                    "type": "tool-call-start",
                    "delta": {"message": {"tool_calls": {
                        "id": "call_1",
                        "function": {"name": "get_weather", "arguments": "{\"cit"},
                    }}},
                }))]),
                terminal_frames: vec![sse(&json!({
                    "type": "message-end",
                    "delta": {
                        "finish_reason": "COMPLETE",
                        "usage": {"tokens": {"input_tokens": 10, "output_tokens": 4}},
                    },
                }))],
                expected_usage_total: 14,
                expected_finish_reason: Some(FinishReason::Stop),
                zero_usage_terminal_frames: Some(vec![sse(&json!({"type": "message-end"}))]),
                bare_terminal_frames: None,
                malformed_frame: Some(sse_raw("{not json")),
                unknown_event_frame: Some(sse(&json!({
                    "type": "citation-start",
                    "delta": {"message": {"citations": {}}},
                }))),
                defective_known_frame: Some(sse_raw(r#"{"type":"content-delta","delta":42}"#)),
                delta_less_prelude_frame: None,
                refusal: None,
            }
        }
    }

    /// Ollama `/api/chat` NDJSON wire.
    pub mod ollama {
        use super::*;

        fn driver() -> WireDriver {
            WireDriver::new("ollama", |chunks| {
                Box::pin(async move {
                    let client = crate::providers::ollama::Client::builder()
                        .api_key("test-key")
                        .http_client(SequencedStreamingHttpClient::new(byte_chunks(chunks)?))
                        .build()?;
                    let model = client.completion_model("llama3.2");
                    let request = model.completion_request("hello").build();
                    let stream = model.stream(request).await?;
                    Ok(drain(stream).await)
                })
            })
        }

        /// The Ollama fixture.
        pub fn fixture() -> ProviderWireFixture {
            ProviderWireFixture {
                driver: driver(),
                text_frames: vec![ndjson(&json!({
                    "model": "llama3.2",
                    "created_at": "2023-08-04T19:22:45.499127Z",
                    "message": {"role": "assistant", "content": "hi"},
                    "done": false,
                }))],
                expected_texts: vec!["hi"],
                tool_call_frames: vec![ndjson(&json!({
                    "model": "llama3.2",
                    "created_at": "2023-08-04T19:22:45.499127Z",
                    "message": {"role": "assistant", "content": "", "tool_calls": [{
                        "function": {"name": "get_weather", "arguments": {"city": "Tokyo"}},
                    }]},
                    "done": false,
                }))],
                expected_tool_name: "get_weather",
                // NDJSON delivers tool calls whole; arguments never stream.
                partial_tool_call_frames: None,
                terminal_frames: vec![ndjson(&json!({
                    "model": "llama3.2",
                    "created_at": "2023-08-04T19:22:47.499127Z",
                    "message": {"role": "assistant", "content": ""},
                    "done": true,
                    "done_reason": "stop",
                    "prompt_eval_count": 10,
                    "eval_count": 4,
                }))],
                expected_usage_total: 14,
                expected_finish_reason: Some(FinishReason::Stop),
                zero_usage_terminal_frames: Some(vec![ndjson(&json!({
                    "model": "llama3.2",
                    "created_at": "2023-08-04T19:22:47.499127Z",
                    "message": {"role": "assistant", "content": ""},
                    "done": true,
                    "done_reason": "stop",
                }))]),
                bare_terminal_frames: None,
                malformed_frame: Some(WireInput::Bytes(Bytes::from_static(b"{not json\n"))),
                unknown_event_frame: None,
                defective_known_frame: Some(ndjson(&json!({
                    "model": "llama3.2",
                    "created_at": "2023-08-04T19:22:46.499127Z",
                    "message": {"role": "assistant", "content": 42},
                    "done": false,
                }))),
                delta_less_prelude_frame: None,
                refusal: None,
            }
        }

        /// Thinking delta, interleaved tool call, thinking delta, terminal —
        /// the constant-id (`reasoning-0`) interleaving shape on NDJSON.
        pub fn interleaved_thinking_frames()
        -> (Vec<WireInput>, &'static str, &'static str, &'static str) {
            let frames = vec![
                ndjson(&json!({
                    "model": "llama3.2",
                    "created_at": "2023-08-04T19:22:45.499127Z",
                    "message": {"role": "assistant", "content": "", "thinking": "before tool"},
                    "done": false,
                })),
                ndjson(&json!({
                    "model": "llama3.2",
                    "created_at": "2023-08-04T19:22:45.599127Z",
                    "message": {"role": "assistant", "content": "", "tool_calls": [{
                        "function": {"name": "get_weather", "arguments": {"city": "Tokyo"}},
                    }]},
                    "done": false,
                })),
                ndjson(&json!({
                    "model": "llama3.2",
                    "created_at": "2023-08-04T19:22:45.699127Z",
                    "message": {"role": "assistant", "content": "", "thinking": "after tool"},
                    "done": false,
                })),
                ndjson(&json!({
                    "model": "llama3.2",
                    "created_at": "2023-08-04T19:22:47.499127Z",
                    "message": {"role": "assistant", "content": ""},
                    "done": true,
                    "done_reason": "stop",
                    "prompt_eval_count": 10,
                    "eval_count": 4,
                })),
            ];
            (frames, "before tool", "get_weather", "after tool")
        }
    }
}
