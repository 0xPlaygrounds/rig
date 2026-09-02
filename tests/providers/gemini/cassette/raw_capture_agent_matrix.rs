//! Matrix for raw provider response capture through the agent on the Gemini
//! REST route: the hook events, `PromptResponse::completion_calls`, and the
//! streamed terminal record.
//!
//! # The feature
//!
//! Capture is always on. The provider populates `CompletionResponse::raw` /
//! `StreamFinal::raw` on every response, and the agent exposes that payload —
//! **per attempt**, never a previous attempt's — as `raw` on the
//! `CompletionResponse` and `ModelTurnFinished` hook events, on each
//! `CompletionCall` the run records, and on the streamed
//! `StreamedAssistantContent::Final`. `raw` is `Value::Null` only on a value
//! built by hand, with no provider response behind it; `Value::Null` never
//! means "not requested".
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `hooks_observe_raw_blocking` | `agent.prompt` | `CompletionResponse` and `ModelTurnFinished` see `raw`; `responseId` matches fixture | recorded |
//! | 2 | `hooks_observe_raw_streamed` | `agent.stream_prompt` | `CompletionResponse` and `ModelTurnFinished` see `raw`; `response_id` matches fixture | recorded |
//! | 3 | `multi_turn_tool_run_records_distinct_raw_blocking` | tool run, `agent.prompt` | two `completion_calls`, two different `raw["responseId"]`s equal to the interactions' ids in order; the first carries the `functionCall` | recorded |
//! | 4 | `multi_turn_tool_run_records_distinct_raw_streamed` | tool run, `agent.stream_prompt` | two `CompletionCall` items, two different `raw["response_id"]`s equal to the interactions' ids in order; the forwarded last `Final.raw` is the final turn's | recorded |
//!
//! Both surfaces fire the same two events per accepted model turn:
//! `CompletionResponse` — after the unary call returns on the blocking
//! surface, after the whole stream is assembled on the streamed one, with
//! `HookContext::is_streaming` telling them apart — and the medium-neutral
//! `ModelTurnFinished`. Tool-only turns fire both. Cells 1–2 pin exactly
//! which events fire and what each carries, and cells 3–4 that every attempt
//! of a tool run fires them, so a hook observing either event alone provably
//! sees the payload for every accepted call on both surfaces.
//!
//! # Identity on this route
//!
//! Gemini names a response by `responseId` (the streamed terminal carries it
//! as `response_id`) and sends no request-id header. The harness scrubs
//! `responseId` into a stable per-value placeholder on the way into the
//! fixture, so the comparison against the recording is structural in record
//! mode (the observed ids repeat / differ exactly as the fixture's do) and
//! exact on replay, when the observed ids *are* the fixture's placeholders.
//!
//! Every recorded cell re-derives its premise from its own fixture: the
//! recorded interactions carry non-empty `responseId`s (distinct per attempt
//! where the cell needs two), and the multi-turn cells' first interaction is a
//! `functionCall` turn.

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use rig::agent::{
    AgentHook, HookContext, ModelTurnAction, ModelTurnFinished, MultiTurnStreamItem,
    ObservationAction,
};
use rig::completion::Message;
use rig::message::AssistantContent;
use rig::prelude::*;
use rig::streaming::StreamEvent;
use rig::tool::Tool;
use serde_json::Value;

use super::super::support::with_gemini_cassette;
use super::super::tools_support::FORCE_TOOLS_PREAMBLE;
use crate::support::Adder;

const GEMINI_PROVIDER: &str = "gemini";

/// Cheap, non-thinking, so the recorded bodies stay small.
const MODEL: &str = "gemini-2.5-flash-lite";

const TEXT_PROMPT: &str = "Reply with exactly: agent raw probe";
const TOOL_PROMPT: &str = "Use the add tool to add 2 and 3, then state the result.";

/// One `CompletionResponse` observation: which driver fired it, whether the
/// canonical content carried a tool call, and the attempt's `raw`.
#[derive(Clone, Debug, PartialEq)]
struct ResponseSeen {
    streaming: bool,
    tool_call: bool,
    raw: Value,
}

/// Records what each hook event carried as `raw`, per event, in fire order.
#[derive(Clone, Default)]
struct RawProbe {
    completion_response: Arc<Mutex<Vec<ResponseSeen>>>,
    model_turn_finished: Arc<Mutex<Vec<Value>>>,
}

impl RawProbe {
    /// Every `CompletionResponse` event's `raw`, in fire order.
    fn completion_responses(&self) -> Vec<Value> {
        self.response_events()
            .into_iter()
            .map(|seen| seen.raw)
            .collect()
    }

    /// Every `CompletionResponse` event's `HookContext::is_streaming`.
    fn response_streaming_flags(&self) -> Vec<bool> {
        self.response_events()
            .iter()
            .map(|seen| seen.streaming)
            .collect()
    }

    /// Whether each `CompletionResponse` event's content carried a tool call.
    fn response_tool_call_flags(&self) -> Vec<bool> {
        self.response_events()
            .iter()
            .map(|seen| seen.tool_call)
            .collect()
    }

    fn response_events(&self) -> Vec<ResponseSeen> {
        self.completion_response.lock().expect("probe").clone()
    }

    fn model_turns(&self) -> Vec<Value> {
        self.model_turn_finished.lock().expect("probe").clone()
    }
}

impl AgentHook for RawProbe {
    async fn on_completion_response(
        &self,
        ctx: &HookContext,
        event: rig::agent::CompletionResponseEvent<'_>,
    ) -> ObservationAction {
        self.completion_response
            .lock()
            .expect("probe")
            .push(ResponseSeen {
                streaming: ctx.is_streaming(),
                tool_call: event
                    .content
                    .iter()
                    .any(|content| matches!(content, AssistantContent::ToolCall(_))),
                raw: event.raw.clone(),
            });
        ObservationAction::continue_run()
    }

    async fn on_model_turn_finished(
        &self,
        _ctx: &HookContext,
        event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        self.model_turn_finished
            .lock()
            .expect("probe")
            .push(event.raw.clone());
        ModelTurnAction::continue_run()
    }
}

/// What a streamed run yielded: the per-call records and every terminal
/// record, in order.
#[derive(Default)]
struct StreamedRun {
    completion_calls: Vec<rig::agent::CompletionCall>,
    finals: Vec<rig::streaming::StreamFinal>,
    output: Option<String>,
}

async fn drain(mut stream: rig::agent::StreamingResult) -> StreamedRun {
    let mut run = StreamedRun::default();
    while let Some(item) = stream.next().await {
        match item.expect("stream item should succeed") {
            MultiTurnStreamItem::CompletionCall(call) => run.completion_calls.push(call),
            MultiTurnStreamItem::StreamAssistantItem(StreamEvent::Final(final_)) => {
                run.finals.push(final_);
            }
            MultiTurnStreamItem::FinalResponse(response) => {
                run.output = Some(response.output().to_owned());
            }
            _ => {}
        }
    }
    run
}

/// JSON `data:` frames of one recorded `streamGenerateContent` body.
fn sse_json_frames(body: &str) -> Vec<Value> {
    body.lines()
        .filter_map(|line| line.trim().strip_prefix("data:"))
        .map(str::trim)
        .filter(|payload| !payload.is_empty())
        .map(|payload| serde_json::from_str(payload).expect("recorded SSE frame should be JSON"))
        .collect()
}

/// Every recorded interaction's response, parsed: the unary body itself, or
/// the streamed body's frames.
fn recorded_responses(scenario: &str, streamed: bool) -> Vec<Vec<Value>> {
    crate::cassettes::recorded_interaction_bodies(GEMINI_PROVIDER, scenario)
        .iter()
        .map(|(_, response)| {
            if streamed {
                sse_json_frames(response)
            } else {
                vec![
                    serde_json::from_str::<Value>(response)
                        .expect("recorded blocking body should be JSON"),
                ]
            }
        })
        .collect()
}

/// `responseId` of every recorded interaction, in wire order — from the
/// blocking body, or from the first streamed frame that names one.
fn recorded_response_ids(scenario: &str, streamed: bool) -> Vec<Option<String>> {
    recorded_responses(scenario, streamed)
        .iter()
        .map(|frames| {
            frames
                .iter()
                .find_map(|frame| frame["responseId"].as_str())
                .map(str::to_string)
        })
        .collect()
}

/// Whether each recorded interaction's model turn carries a `functionCall`
/// part, in wire order.
fn recorded_function_call_turns(scenario: &str, streamed: bool) -> Vec<bool> {
    recorded_responses(scenario, streamed)
        .iter()
        .map(|frames| {
            frames.iter().any(|frame| {
                frame
                    .pointer("/candidates/0/content/parts")
                    .and_then(Value::as_array)
                    .is_some_and(|parts| {
                        parts.iter().any(|part| part.get("functionCall").is_some())
                    })
            })
        })
        .collect()
}

/// `usageMetadata.totalTokenCount` of every recorded interaction's last
/// usage-bearing frame, in wire order — Gemini's streamed usage is cumulative,
/// so this is the total the terminal must report.
fn recorded_total_tokens(scenario: &str, streamed: bool) -> Vec<Option<u64>> {
    recorded_responses(scenario, streamed)
        .iter()
        .map(|frames| {
            frames
                .iter()
                .rev()
                .find_map(|frame| frame.pointer("/usageMetadata/totalTokenCount"))
                .and_then(Value::as_u64)
        })
        .collect()
}

fn ids_of(raws: &[Value], key: &str) -> Vec<Option<String>> {
    raws.iter()
        .map(|raw| raw[key].as_str().map(str::to_string))
        .collect()
}

/// Every payload in `raws` has a provider response behind it — none is the
/// hand-built `Value::Null`.
fn assert_all_populated(raws: &[Value], context: &str) {
    assert!(
        raws.iter().all(|raw| !raw.is_null()),
        "{context}: every call carries raw, got {raws:?}"
    );
}

/// The observed ids repeat / differ exactly as the recorded ones do, and on
/// replay they *are* the recorded ones (the fixture's scrubbed placeholders).
///
/// Structural in record mode because the harness rewrites `responseId` into
/// a placeholder on the way to disk, so the live ids can never equal the
/// fixture bytes textually; a per-value placeholder keeps the identity
/// structure (same id → same placeholder) that this compares.
fn assert_ids_match_recording(
    observed: &[Option<String>],
    recorded: &[Option<String>],
    context: &str,
) {
    fn ranks(ids: &[Option<String>]) -> Vec<Option<usize>> {
        let mut seen: Vec<&str> = Vec::new();
        ids.iter()
            .map(|id| {
                id.as_deref().map(|id| {
                    seen.iter()
                        .position(|known| *known == id)
                        .unwrap_or_else(|| {
                            seen.push(id);
                            seen.len() - 1
                        })
                })
            })
            .collect()
    }

    assert_eq!(
        observed.len(),
        recorded.len(),
        "{context}: observed {observed:?} and recorded {recorded:?} must have one id per interaction"
    );
    assert_eq!(
        ranks(observed),
        ranks(recorded),
        "{context}: observed {observed:?} must repeat/differ exactly as the recording {recorded:?}"
    );
    if crate::cassettes::CassetteMode::current() == crate::cassettes::CassetteMode::Replay {
        assert_eq!(
            observed, recorded,
            "{context}: on replay the observed ids are the fixture's own"
        );
    }
}

fn assert_distinct_response_ids(ids: &[Option<String>], context: &str) {
    assert!(
        ids.iter()
            .all(|id| id.as_deref().is_some_and(|id| !id.is_empty())),
        "{context}: premise — every recorded attempt names a responseId, got {ids:?}"
    );
    for (i, a) in ids.iter().enumerate() {
        for b in &ids[i + 1..] {
            assert_ne!(a, b, "{context}: premise — attempts have distinct ids");
        }
    }
}

// ---------------------------------------------------------------------------
// 1–2: the hook events
// ---------------------------------------------------------------------------

#[tokio::test]
async fn hooks_observe_raw_blocking() {
    let scenario = "raw_capture_agent_matrix/hooks_observe_raw_blocking";
    let probe = RawProbe::default();
    let hook = probe.clone();
    with_gemini_cassette(
        "raw_capture_agent_matrix/hooks_observe_raw_blocking",
        move |client| async move {
            let agent = client.agent(MODEL).temperature(0.0).add_hook(hook).build();
            agent
                .prompt(TEXT_PROMPT)
                .await
                .expect("prompt should succeed");
        },
    )
    .await;

    let responses = probe.completion_responses();
    let turns = probe.model_turns();
    assert_eq!(responses.len(), 1, "one CompletionResponse event");
    assert_eq!(
        probe.response_streaming_flags(),
        [false],
        "the blocking driver fires it with is_streaming() == false"
    );
    assert_eq!(turns.len(), 1, "one ModelTurnFinished event");
    let raw = &responses[0];
    assert!(!raw.is_null(), "CompletionResponse.raw is populated");
    assert_eq!(&turns[0], raw, "both events observe the same payload");
    // The payload is this attempt's own response.
    let recorded = recorded_response_ids(scenario, false);
    assert_eq!(recorded.len(), 1);
    assert_distinct_response_ids(&recorded, scenario);
    assert_ids_match_recording(&ids_of(&responses, "responseId"), &recorded, scenario);
    let body = recorded_responses(scenario, false).remove(0).remove(0);
    assert_eq!(
        raw.pointer("/candidates/0/finishReason"),
        body.pointer("/candidates/0/finishReason")
    );
    assert_eq!(raw["modelVersion"], body["modelVersion"]);
    assert_eq!(
        raw.pointer("/usageMetadata/totalTokenCount"),
        body.pointer("/usageMetadata/totalTokenCount")
    );
}

#[tokio::test]
async fn hooks_observe_raw_streamed() {
    let scenario = "raw_capture_agent_matrix/hooks_observe_raw_streamed";
    let probe = RawProbe::default();
    let hook = probe.clone();
    with_gemini_cassette(
        "raw_capture_agent_matrix/hooks_observe_raw_streamed",
        move |client| async move {
            let agent = client.agent(MODEL).temperature(0.0).add_hook(hook).build();
            let run = drain(
                agent
                    .stream_prompt(Message::user(TEXT_PROMPT))
                    .stream()
                    .await,
            )
            .await;
            assert!(run.output.is_some(), "the run finished");
            assert_eq!(run.finals.len(), 1, "one text turn, one terminal record");
            assert!(
                !run.finals[0].raw.is_null(),
                "the streamed terminal carries raw"
            );
        },
    )
    .await;

    let responses = probe.completion_responses();
    let turns = probe.model_turns();
    assert_eq!(
        responses.len(),
        1,
        "the streamed surface fires CompletionResponse once the stream is assembled"
    );
    assert_eq!(
        probe.response_streaming_flags(),
        [true],
        "the streaming driver fires it with is_streaming() == true"
    );
    assert_eq!(turns.len(), 1, "one ModelTurnFinished event");
    let raw = &responses[0];
    assert!(!raw.is_null(), "CompletionResponse.raw is populated");
    assert_eq!(&turns[0], raw, "both events observe the same payload");
    // The streamed payload is Gemini's *terminal* record, assembled by rig
    // from the frames: `response_id` + `usage_metadata`, not `candidates`.
    assert!(raw.get("response_id").is_some() && raw.get("usage_metadata").is_some());
    assert!(raw.get("candidates").is_none());
    let recorded = recorded_response_ids(scenario, true);
    assert_eq!(recorded.len(), 1);
    assert_distinct_response_ids(&recorded, scenario);
    assert_ids_match_recording(&ids_of(&responses, "response_id"), &recorded, scenario);
    assert_eq!(
        responses[0]
            .pointer("/usage_metadata/totalTokenCount")
            .and_then(Value::as_u64),
        recorded_total_tokens(scenario, true)[0],
        "the terminal usage is the last frame's cumulative total"
    );
}

// ---------------------------------------------------------------------------
// 3–4: multi-turn tool runs
// ---------------------------------------------------------------------------

#[tokio::test]
async fn multi_turn_tool_run_records_distinct_raw_blocking() {
    let scenario = "raw_capture_agent_matrix/multi_turn_tool_run_records_distinct_raw_blocking";
    let probe = RawProbe::default();
    let hook = probe.clone();
    let observed: Arc<Mutex<Vec<Value>>> = Default::default();
    let sink = observed.clone();
    with_gemini_cassette(
        "raw_capture_agent_matrix/multi_turn_tool_run_records_distinct_raw_blocking",
        move |client| async move {
            let agent = client
                .agent(MODEL)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool(Adder)
                .add_hook(hook)
                .build();
            let response = agent
                .prompt(TOOL_PROMPT)
                .max_turns(3)
                .await
                .expect("tool run should succeed");
            let calls = &response.completion_calls;
            assert_eq!(calls.len(), 2, "a tool turn then a text turn");
            *sink.lock().expect("sink") = calls.iter().map(|call| call.raw.clone()).collect();
        },
    )
    .await;

    let raws = observed.lock().expect("sink").clone();
    assert_all_populated(&raws, scenario);
    assert_ne!(raws[0], raws[1], "two attempts, two different payloads");
    let ids = ids_of(&raws, "responseId");
    let recorded = recorded_response_ids(scenario, false);
    assert_distinct_response_ids(&recorded, scenario);
    assert_ids_match_recording(&ids, &recorded, scenario);
    // Each payload is its own attempt's: the first is the functionCall turn,
    // the second the text turn.
    assert_eq!(
        recorded_function_call_turns(scenario, false),
        [true, false],
        "premise: a functionCall turn then a text turn"
    );
    assert!(
        raws[0]
            .pointer("/candidates/0/content/parts/0/functionCall/name")
            .is_some_and(|name| name == Adder::NAME),
        "the first payload carries the wire's functionCall: {:?}",
        raws[0]
    );
    assert!(
        raws[1]
            .pointer("/candidates/0/content/parts/0/text")
            .is_some(),
        "the second payload carries the wire's text part: {:?}",
        raws[1]
    );
    assert_eq!(
        raws.iter()
            .map(|raw| raw
                .pointer("/usageMetadata/totalTokenCount")
                .and_then(Value::as_u64))
            .collect::<Vec<_>>(),
        recorded_total_tokens(scenario, false)
    );
    // The hooks saw the same two payloads in the same order; the first
    // CompletionResponse is the tool-call turn, the second the text turn.
    assert_eq!(probe.completion_responses(), raws);
    assert_eq!(probe.response_tool_call_flags(), [true, false]);
    assert_eq!(probe.model_turns(), raws);
}

#[tokio::test]
async fn multi_turn_tool_run_records_distinct_raw_streamed() {
    let scenario = "raw_capture_agent_matrix/multi_turn_tool_run_records_distinct_raw_streamed";
    let probe = RawProbe::default();
    let hook = probe.clone();
    let observed: Arc<Mutex<Vec<Value>>> = Default::default();
    let sink = observed.clone();
    let last_final: Arc<Mutex<Option<Value>>> = Default::default();
    let final_sink = last_final.clone();
    with_gemini_cassette(
        "raw_capture_agent_matrix/multi_turn_tool_run_records_distinct_raw_streamed",
        move |client| async move {
            let agent = client
                .agent(MODEL)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool(Adder)
                .add_hook(hook)
                .build();
            let run = drain(
                agent
                    .stream_prompt(Message::user(TOOL_PROMPT))
                    .max_turns(3)
                    .stream()
                    .await,
            )
            .await;
            assert!(run.output.is_some(), "the run finished");
            assert_eq!(
                run.completion_calls.len(),
                2,
                "a tool turn then a text turn"
            );
            assert!(!run.finals.is_empty(), "the stream yields terminal records");
            // The last forwarded terminal record is the final turn's, and its
            // payload is the one the final call recorded.
            let last = run.finals.last().expect("terminal");
            assert_eq!(last.raw, run.completion_calls[1].raw);
            *final_sink.lock().expect("sink") = Some(last.raw.clone());
            *sink.lock().expect("sink") = run
                .completion_calls
                .iter()
                .map(|call| call.raw.clone())
                .collect();
        },
    )
    .await;

    let raws = observed.lock().expect("sink").clone();
    assert_all_populated(&raws, scenario);
    assert_ne!(raws[0], raws[1], "two attempts, two different payloads");
    let recorded = recorded_response_ids(scenario, true);
    assert_distinct_response_ids(&recorded, scenario);
    assert_ids_match_recording(&ids_of(&raws, "response_id"), &recorded, scenario);
    // Each payload is its own attempt's: the terminal totals land in the
    // interactions' order, and the first interaction is the functionCall turn.
    assert_eq!(
        recorded_function_call_turns(scenario, true),
        [true, false],
        "premise: a functionCall turn then a text turn"
    );
    assert_eq!(
        raws.iter()
            .map(|raw| raw
                .pointer("/usage_metadata/totalTokenCount")
                .and_then(Value::as_u64))
            .collect::<Vec<_>>(),
        recorded_total_tokens(scenario, true)
    );
    // The forwarded Final is the *last* interaction's, not the first's.
    let last = last_final
        .lock()
        .expect("sink")
        .clone()
        .expect("last Final");
    assert_ids_match_recording(&ids_of(&[last], "response_id"), &recorded[1..], scenario);
    // ModelTurnFinished fires for both attempts and sees each attempt's own.
    assert_eq!(probe.model_turns(), raws);
    // CompletionResponse fires for both attempts too — the tool-only turn
    // included — and each firing carries its own attempt's payload: the
    // first is the tool-call turn, the second the text turn.
    assert_eq!(probe.completion_responses(), raws);
    assert_eq!(probe.response_tool_call_flags(), [true, false]);
    assert_eq!(probe.response_streaming_flags(), [true, true]);
}
