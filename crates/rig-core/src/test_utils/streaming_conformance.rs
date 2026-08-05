//! Wire-sequence conformance scenarios for provider streaming pipelines.
//!
//! The streaming sibling of `rig-agent`'s `model_conformance`: each scenario
//! drives raw wire bytes (SSE or NDJSON) through a provider's *complete*
//! streaming path — bytes → decode → normalize → aggregated
//! [`StreamingCompletionResponse`](crate::streaming::StreamingCompletionResponse)
//! — and asserts the [`StreamFinal`](crate::streaming::StreamFinal) contract
//! table documented on that type. Scenarios state the contract; a per-provider
//! [`ProviderWireFixture`] supplies the frames, since each wire format spells
//! the same event differently.
//!
//! Every sequence family here pins a shipped bug from the #2257 review rounds
//! (`rig-2257-code-review-findings-*.md`); the per-scenario comments cite the
//! specific finding.

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

/// The wire chunks a driver feeds into the provider's HTTP layer. An `Err`
/// chunk models a mid-stream transport failure.
pub type WireChunks = Vec<http_client::Result<Bytes>>;

/// Build the chunk list for an all-delivered byte sequence.
pub fn ok_chunks(frames: impl IntoIterator<Item = Bytes>) -> WireChunks {
    frames.into_iter().map(Ok).collect()
}

/// A scripted mid-stream transport failure chunk.
pub fn transport_error_chunk() -> http_client::Result<Bytes> {
    Err(http_client::Error::InvalidStatusCodeWithMessage(
        http::StatusCode::BAD_GATEWAY,
        "connection reset".to_string(),
    ))
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
    pub frames: Vec<Bytes>,
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
    pub text_frames: Vec<Bytes>,
    /// The text deltas `text_frames` delivers, in order.
    pub expected_texts: Vec<&'static str>,
    /// Frames that fully deliver one tool call (including any completion
    /// signal the wire needs, but no stream terminal).
    pub tool_call_frames: Vec<Bytes>,
    /// Name of the tool call `tool_call_frames` delivers.
    pub expected_tool_name: &'static str,
    /// Frames that leave a tool call mid-arguments, where the wire streams
    /// arguments incrementally.
    pub partial_tool_call_frames: Option<Vec<Bytes>>,
    /// The provider's genuine stream terminal, carrying usage.
    pub terminal_frames: Vec<Bytes>,
    /// Total tokens `terminal_frames` reports.
    pub expected_usage_total: u64,
    /// Finish reason `terminal_frames` reports.
    pub expected_finish_reason: Option<FinishReason>,
    /// A genuine terminal that reports no usage metrics at all.
    pub zero_usage_terminal_frames: Option<Vec<Bytes>>,
    /// A terminal signal that carries no data of its own (e.g. a bare
    /// `[DONE]`), for wires that have one.
    pub bare_terminal_frames: Option<Vec<Bytes>>,
    /// A frame that fails the wire decode entirely.
    pub malformed_frame: Bytes,
    /// An event type this client does not know, for typed-event wires.
    pub unknown_event_frame: Option<Bytes>,
    /// A known event whose payload is schema-defective.
    pub defective_known_frame: Option<Bytes>,
    /// A delta-less choice prelude (the Azure `prompt_filter_results` shape).
    pub delta_less_prelude_frame: Option<Bytes>,
    /// Refusal content frames, where the wire has a refusal channel.
    pub refusal: Option<RefusalFixture>,
}

fn concat_frames(parts: &[&[Bytes]]) -> Vec<Bytes> {
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
) -> Result<ScenarioReport, ConformanceError> {
    const SCENARIO: &str = "malformed_frame_surfaces_err_and_terminal_still_completes";
    let provider = fixture.driver.provider;

    let frames = concat_frames(&[
        &fixture.text_frames,
        std::slice::from_ref(&fixture.malformed_frame),
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

    Ok(ScenarioReport {
        name: SCENARIO,
        provider,
        observations: vec!["Err surfaced, terminal still completed".to_string()],
    })
}

/// An event type the client does not know must be skipped without an error,
/// and the stream must still complete.
///
/// Pins the unknown-event forward-compatibility policy (round three,
/// `rig-2257-code-review-findings-8a2f41c7.md`).
pub async fn unknown_event_is_skipped(
    fixture: &ProviderWireFixture,
) -> Result<ScenarioReport, ConformanceError> {
    const SCENARIO: &str = "unknown_event_is_skipped";
    let provider = fixture.driver.provider;
    let Some(unknown) = &fixture.unknown_event_frame else {
        return Ok(ScenarioReport {
            name: SCENARIO,
            provider,
            observations: vec!["wire family has no event types; nothing to skip".to_string()],
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

    Ok(ScenarioReport {
        name: SCENARIO,
        provider,
        observations: vec!["unknown event skipped, stream completed".to_string()],
    })
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
) -> Result<ScenarioReport, ConformanceError> {
    const SCENARIO: &str = "defective_known_event_surfaces_err";
    let provider = fixture.driver.provider;
    let Some(defective) = &fixture.defective_known_frame else {
        return Ok(ScenarioReport {
            name: SCENARIO,
            provider,
            observations: vec!["fixture supplies no defective known frame".to_string()],
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

    Ok(ScenarioReport {
        name: SCENARIO,
        provider,
        observations: vec!["defective known event surfaced as Err; stream completed".to_string()],
    })
}

/// A delta-less choice (the Azure `prompt_filter_results` prelude) must be a
/// no-op — no error, no content, and the rest of the stream unaffected.
///
/// Pins the Azure prelude no-op from round two
/// (`rig-2257-code-review-findings-b91d03aa.md`).
pub async fn delta_less_choice_prelude_is_a_noop(
    fixture: &ProviderWireFixture,
) -> Result<ScenarioReport, ConformanceError> {
    const SCENARIO: &str = "delta_less_choice_prelude_is_a_noop";
    let provider = fixture.driver.provider;
    let Some(prelude) = &fixture.delta_less_prelude_frame else {
        return Ok(ScenarioReport {
            name: SCENARIO,
            provider,
            observations: vec!["wire family has no delta-less prelude shape".to_string()],
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

    Ok(ScenarioReport {
        name: SCENARIO,
        provider,
        observations: vec!["delta-less prelude ignored; stream unaffected".to_string()],
    })
}

/// Refusal frames must deliver their text to the consumer without an error.
///
/// Pins the refusal-delta handling from round three
/// (`rig-2257-code-review-findings-8a2f41c7.md`).
pub async fn refusal_frames_deliver_text_without_error(
    fixture: &ProviderWireFixture,
) -> Result<ScenarioReport, ConformanceError> {
    const SCENARIO: &str = "refusal_frames_deliver_text_without_error";
    let provider = fixture.driver.provider;
    let Some(refusal) = &fixture.refusal else {
        return Ok(ScenarioReport {
            name: SCENARIO,
            provider,
            observations: vec!["wire family has no refusal channel".to_string()],
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

    Ok(ScenarioReport {
        name: SCENARIO,
        provider,
        observations: vec!["refusal text delivered without error".to_string()],
    })
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
) -> Result<ScenarioReport, ConformanceError> {
    const SCENARIO: &str = "bare_terminal_after_only_unparseable_frames_fabricates_nothing";
    let provider = fixture.driver.provider;
    let Some(bare_terminal) = &fixture.bare_terminal_frames else {
        return Ok(ScenarioReport {
            name: SCENARIO,
            provider,
            observations: vec!["wire family has no data-less terminal signal".to_string()],
        });
    };

    let frames = concat_frames(&[
        std::slice::from_ref(&fixture.malformed_frame),
        bare_terminal,
    ]);
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

    Ok(ScenarioReport {
        name: SCENARIO,
        provider,
        observations: vec!["no fabricated terminal after only-unparseable frames".to_string()],
    })
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
    frames: Vec<Bytes>,
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
    frames: Vec<Bytes>,
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
    frames: Vec<Bytes>,
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
    frames: Vec<Bytes>,
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
    frames: Vec<Bytes>,
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

/// Per-provider wire fixtures for the shared scenario set.
pub mod fixtures {
    use super::*;
    use crate::client::CompletionClient;
    use crate::completion::CompletionModel;
    use crate::test_utils::SequencedStreamingHttpClient;
    use serde_json::json;

    async fn drain(mut stream: crate::streaming::StreamingCompletionResponse) -> DrainedStream {
        let mut items = Vec::new();
        while let Some(item) = stream.next().await {
            items.push(item);
        }
        DrainedStream {
            items,
            choice: stream.choice.clone(),
            response: stream.response.clone(),
        }
    }

    fn sse(frame: &serde_json::Value) -> Bytes {
        Bytes::from(format!("data: {frame}\n\n"))
    }

    fn sse_raw(data: &str) -> Bytes {
        Bytes::from(format!("data: {data}\n\n"))
    }

    fn ndjson(frame: &serde_json::Value) -> Bytes {
        Bytes::from(format!("{frame}\n"))
    }

    /// OpenAI chat-completions wire (the shared OpenAI-compatible SSE path).
    pub mod openai_chat {
        use super::*;

        fn driver() -> WireDriver {
            WireDriver::new("openai", |chunks| {
                Box::pin(async move {
                    let client = crate::providers::openai::Client::builder()
                        .http_client(SequencedStreamingHttpClient::new(chunks))
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
                malformed_frame: sse_raw("{not json"),
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
                        .http_client(SequencedStreamingHttpClient::new(chunks))
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

        fn terminal(usage: Option<serde_json::Value>, output: serde_json::Value) -> Bytes {
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

        fn text_delta(text: &str) -> Bytes {
            sse(&json!({
                "type": "response.output_text.delta",
                "content_index": 0,
                "delta": text,
                "item_id": "msg_1",
                "output_index": 0,
                "sequence_number": 1,
            }))
        }

        fn tool_call_done() -> Bytes {
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

        fn reasoning_done_item(
            id: &str,
            summary: serde_json::Value,
            content: serde_json::Value,
            encrypted: Option<&str>,
        ) -> Bytes {
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
                malformed_frame: sse_raw("{not json"),
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
        pub fn buffered_driver() -> BufferedBodyDriver {
            BufferedBodyDriver::new("chatgpt", |body| {
                Box::pin(async move {
                    let raw_response =
                        crate::providers::openai::responses_api::streaming::parse_sse_completion_body(
                            &body, "ChatGPT",
                        )?;
                    // Mirror the ChatGPT backend's `normalized_completion`:
                    // the terminal body is authoritative when it carries
                    // output items; an empty `output` falls back to replaying
                    // the raw event stream and merging per content kind.
                    use crate::completion::NormalizeCompletionResponse as _;
                    let response = match raw_response.clone().normalize("chatgpt") {
                        Ok(response) => response,
                        Err(CompletionError::ResponseError(_)) if raw_response.output.is_empty() => {
                            crate::providers::openai::responses_api::streaming::completion_response_from_sse_body(
                                "chatgpt", &body, raw_response,
                            )
                            .await?
                        }
                        Err(error) => return Err(error),
                    };
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
            String::from_utf8_lossy(&terminal(Some(usage_json()), message_output(text)))
                .into_owned()
        }

        /// A streamed delta plus a terminal body restating the same text.
        pub fn terminal_body_and_delta_sse_body(text: &str) -> String {
            let frames = [
                text_delta(text),
                terminal(Some(usage_json()), message_output(text)),
            ];
            frames
                .iter()
                .map(|frame| String::from_utf8_lossy(frame).into_owned())
                .collect()
        }

        /// A streamed delta whose terminal body carries no output items — the
        /// gpt-5.x shape the buffered fallback exists for.
        pub fn delta_only_sse_body(text: &str) -> String {
            let frames = [text_delta(text), terminal(Some(usage_json()), json!([]))];
            frames
                .iter()
                .map(|frame| String::from_utf8_lossy(frame).into_owned())
                .collect()
        }

        /// Summary deltas followed by their item's full `output_item.done`
        /// block, then the terminal. The deltas carry `item_id` on the wire;
        /// the full block restates the summary.
        pub fn reasoning_summary_supersede_frames() -> (Vec<Bytes>, &'static str) {
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
        pub fn multi_part_reasoning_frames() -> (Vec<Bytes>, Vec<&'static str>) {
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
        pub fn interleaved_reasoning_frames() -> (Vec<Bytes>, &'static str) {
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
                        .http_client(SequencedStreamingHttpClient::new(chunks))
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
                malformed_frame: sse_raw("{not json"),
                unknown_event_frame: None,
                defective_known_frame: Some(sse_raw(r#"{"candidates": 42}"#)),
                delta_less_prelude_frame: None,
                refusal: None,
            }
        }

        fn chunk(parts: serde_json::Value) -> Bytes {
            sse(&json!({
                "candidates": [{"content": {"parts": parts, "role": "model"}}],
                "responseId": "resp-1",
                "modelVersion": "gemini-2.5-pro",
            }))
        }

        fn terminal_frame() -> Bytes {
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
        pub fn interleaved_thought_frames() -> (Vec<Bytes>, &'static str, &'static str, &'static str)
        {
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
        -> (Vec<Bytes>, &'static str, &'static str, &'static str) {
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

    /// Cohere v2 chat SSE wire.
    pub mod cohere {
        use super::*;

        fn driver() -> WireDriver {
            WireDriver::new("cohere", |chunks| {
                Box::pin(async move {
                    let client = crate::providers::cohere::Client::builder()
                        .api_key("test-key")
                        .http_client(SequencedStreamingHttpClient::new(chunks))
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
                malformed_frame: sse_raw("{not json"),
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
                        .http_client(SequencedStreamingHttpClient::new(chunks))
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
                malformed_frame: Bytes::from_static(b"{not json\n"),
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
        -> (Vec<Bytes>, &'static str, &'static str, &'static str) {
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
