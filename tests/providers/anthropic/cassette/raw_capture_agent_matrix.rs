//! Matrix for raw provider response capture through the agent: the hook
//! events, `PromptResponse::completion_calls`, and the streamed terminal
//! record.
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
//! | 1 | `hooks_observe_raw_blocking` | `agent.prompt` | `CompletionResponse` and `ModelTurnFinished` see `raw`; `id` matches fixture | recorded |
//! | 2 | `hooks_observe_raw_streamed` | `agent.stream_prompt` | `CompletionResponse` and `ModelTurnFinished` see `raw`; `message_id` matches fixture | recorded |
//! | 3 | `multi_turn_tool_run_records_distinct_raw_blocking` | tool run, `agent.prompt` | two `completion_calls`, two different `raw["id"]`s equal to the interactions' ids in order | recorded |
//! | 4 | `multi_turn_tool_run_records_distinct_raw_streamed` | tool run, `agent.stream_prompt` | two `CompletionCall` items, two different `raw["message_id"]`s equal to the interactions' ids in order | recorded |
//! | 5 | `streamed_final_carries_final_turn_raw` | tool run, `agent.stream_prompt` | the last `StreamedAssistantContent::Final.raw["message_id"]` is the last interaction's id | recorded |
//! | 6 | `retried_turn_records_retried_attempt_raw_blocking` | `ModelTurnFinished` → `Retry` once, `agent.prompt` | second recorded call / second event carry the second interaction's `id` | recorded |
//! | 7 | `retried_turn_records_retried_attempt_raw_streamed` | same, `agent.stream_prompt` | same, by `message_id` | recorded |
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
//! Every recorded cell re-derives its premise from its own fixture: the
//! recorded interactions carry `msg_…` ids (distinct per attempt where the
//! cell needs two), and the multi-turn cells' first interaction is a
//! `tool_use` turn.

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use rig::agent::{
    AgentHook, HookContext, ModelTurnAction, ModelTurnFinished, MultiTurnStreamItem,
    ObservationAction,
};
use rig::completion::Message;
use rig::message::AssistantContent;
use rig::prelude::*;
use rig::providers::anthropic::completion::CLAUDE_HAIKU_4_5;
use rig::streaming::StreamedAssistantContent;
use rig::tool::Tool;
use serde_json::Value;

use super::super::support::{
    assert_ids_match_recording, recorded_response_body, sse_json_frames, with_anthropic_cassette,
};
use crate::support::{Adder, TOOLS_PREAMBLE};

const ANTHROPIC_PROVIDER: &str = "anthropic";
const TEXT_PROMPT: &str = "Reply with exactly: agent raw probe";
const TOOL_PROMPT: &str = "What is 2 + 3? Use the tool, then state the result.";

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
    /// When set, the first `ModelTurnFinished` is rejected with `Repeat`.
    retry_once: bool,
}

impl RawProbe {
    fn retrying_once() -> Self {
        Self {
            retry_once: true,
            ..Self::default()
        }
    }

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
        let mut seen = self.model_turn_finished.lock().expect("probe");
        seen.push(event.raw.clone());
        if self.retry_once && seen.len() == 1 {
            ModelTurnAction::repeat()
        } else {
            ModelTurnAction::continue_run()
        }
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
            MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Final(final_)) => {
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

/// Message ids of every recorded interaction, in wire order — from the
/// blocking body's `id`, or from the stream's `message_start`.
fn recorded_message_ids(scenario: &str, streamed: bool) -> Vec<Option<String>> {
    crate::cassettes::recorded_interaction_bodies(ANTHROPIC_PROVIDER, scenario)
        .iter()
        .map(|(_, response)| {
            if streamed {
                sse_json_frames(response)
                    .iter()
                    .find(|frame| frame["type"] == "message_start")
                    .and_then(|frame| frame["message"]["id"].as_str())
                    .map(str::to_string)
            } else {
                serde_json::from_str::<Value>(response)
                    .expect("recorded blocking body should be JSON")["id"]
                    .as_str()
                    .map(str::to_string)
            }
        })
        .collect()
}

/// Stop reasons of every recorded interaction, in wire order.
fn recorded_stop_reasons(scenario: &str, streamed: bool) -> Vec<Option<String>> {
    crate::cassettes::recorded_interaction_bodies(ANTHROPIC_PROVIDER, scenario)
        .iter()
        .map(|(_, response)| {
            if streamed {
                sse_json_frames(response)
                    .iter()
                    .find(|frame| frame["type"] == "message_delta")
                    .and_then(|frame| frame["delta"]["stop_reason"].as_str())
                    .map(str::to_string)
            } else {
                serde_json::from_str::<Value>(response)
                    .expect("recorded blocking body should be JSON")["stop_reason"]
                    .as_str()
                    .map(str::to_string)
            }
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

fn assert_distinct_msg_ids(ids: &[Option<String>], context: &str) {
    assert!(
        ids.iter()
            .all(|id| id.as_deref().is_some_and(|id| id.starts_with("msg_"))),
        "{context}: premise — every recorded attempt names a msg_ id, got {ids:?}"
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
    with_anthropic_cassette(
        "raw_capture_agent_matrix/hooks_observe_raw_blocking",
        move |client| async move {
            let agent = client
                .agent(CLAUDE_HAIKU_4_5)
                .max_tokens(32)
                .add_hook(hook)
                .build();
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
    let recorded = recorded_message_ids(scenario, false);
    assert_eq!(recorded.len(), 1);
    assert_ids_match_recording(&ids_of(&responses, "id"), &recorded, scenario);
    let body = recorded_response_body(scenario);
    assert_eq!(raw["stop_reason"], body["stop_reason"]);
    assert_eq!(raw["model"], body["model"]);
    assert_eq!(
        raw["usage"]["output_tokens"],
        body["usage"]["output_tokens"]
    );
}

#[tokio::test]
async fn hooks_observe_raw_streamed() {
    let scenario = "raw_capture_agent_matrix/hooks_observe_raw_streamed";
    let probe = RawProbe::default();
    let hook = probe.clone();
    with_anthropic_cassette(
        "raw_capture_agent_matrix/hooks_observe_raw_streamed",
        move |client| async move {
            let agent = client
                .agent(CLAUDE_HAIKU_4_5)
                .max_tokens(32)
                .add_hook(hook)
                .build();
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
    // The streamed payload is the Anthropic *terminal* record.
    assert!(raw.get("message_id").is_some() && raw.get("usage").is_some());
    let recorded = recorded_message_ids(scenario, true);
    assert_eq!(recorded.len(), 1);
    assert_ids_match_recording(&ids_of(&responses, "message_id"), &recorded, scenario);
    assert_eq!(
        ids_of(&responses, "stop_reason"),
        recorded_stop_reasons(scenario, true)
    );
}

// ---------------------------------------------------------------------------
// 3–5: multi-turn tool runs
// ---------------------------------------------------------------------------

#[tokio::test]
async fn multi_turn_tool_run_records_distinct_raw_blocking() {
    let scenario = "raw_capture_agent_matrix/multi_turn_tool_run_records_distinct_raw_blocking";
    let probe = RawProbe::default();
    let hook = probe.clone();
    let observed: Arc<Mutex<Vec<Value>>> = Default::default();
    let sink = observed.clone();
    with_anthropic_cassette(
        "raw_capture_agent_matrix/multi_turn_tool_run_records_distinct_raw_blocking",
        move |client| async move {
            let agent = client
                .agent(CLAUDE_HAIKU_4_5)
                .preamble(TOOLS_PREAMBLE)
                .max_tokens(1024)
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
    let ids = ids_of(&raws, "id");
    let recorded = recorded_message_ids(scenario, false);
    assert_distinct_msg_ids(&recorded, scenario);
    assert_ids_match_recording(&ids, &recorded, scenario);
    // Each payload is its own attempt's: the first stopped for tool use.
    let stop_reasons = recorded_stop_reasons(scenario, false);
    assert_eq!(stop_reasons[0].as_deref(), Some("tool_use"), "premise");
    assert_eq!(ids_of(&raws, "stop_reason"), stop_reasons);
    assert_eq!(raws[0]["content"][0]["type"], "tool_use");
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
    with_anthropic_cassette(
        "raw_capture_agent_matrix/multi_turn_tool_run_records_distinct_raw_streamed",
        move |client| async move {
            let agent = client
                .agent(CLAUDE_HAIKU_4_5)
                .preamble(TOOLS_PREAMBLE)
                .max_tokens(1024)
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
    let recorded = recorded_message_ids(scenario, true);
    assert_distinct_msg_ids(&recorded, scenario);
    assert_ids_match_recording(&ids_of(&raws, "message_id"), &recorded, scenario);
    let stop_reasons = recorded_stop_reasons(scenario, true);
    assert_eq!(stop_reasons[0].as_deref(), Some("tool_use"), "premise");
    assert_eq!(ids_of(&raws, "stop_reason"), stop_reasons);
    // ModelTurnFinished fires for both attempts and sees each attempt's own.
    assert_eq!(probe.model_turns(), raws);
    // CompletionResponse fires for both attempts too — the tool-only turn
    // included — and each firing carries its own attempt's payload: the
    // first is the tool-call turn, the second the text turn.
    assert_eq!(probe.completion_responses(), raws);
    assert_eq!(probe.response_tool_call_flags(), [true, false]);
    assert_eq!(probe.response_streaming_flags(), [true, true]);
}

#[tokio::test]
async fn streamed_final_carries_final_turn_raw() {
    let scenario = "raw_capture_agent_matrix/streamed_final_carries_final_turn_raw";
    let observed: Arc<Mutex<Vec<Value>>> = Default::default();
    let sink = observed.clone();
    with_anthropic_cassette(
        "raw_capture_agent_matrix/streamed_final_carries_final_turn_raw",
        move |client| async move {
            let agent = client
                .agent(CLAUDE_HAIKU_4_5)
                .preamble(TOOLS_PREAMBLE)
                .max_tokens(1024)
                .tool(Adder)
                .build();
            let run = drain(
                agent
                    .stream_prompt(Message::user(TOOL_PROMPT))
                    .max_turns(3)
                    .stream()
                    .await,
            )
            .await;
            assert_eq!(
                run.completion_calls.len(),
                2,
                "a tool turn then a text turn"
            );
            assert!(!run.finals.is_empty(), "the stream yields terminal records");
            let last = run.finals.last().expect("terminal");
            // The last terminal record is the final turn's, and its payload is
            // the one the final call recorded.
            assert_eq!(last.raw, run.completion_calls[1].raw);
            *sink.lock().expect("sink") = vec![last.raw.clone()];
        },
    )
    .await;

    let raws = observed.lock().expect("sink").clone();
    let recorded = recorded_message_ids(scenario, true);
    assert_distinct_msg_ids(&recorded, scenario);
    assert_eq!(recorded.len(), 2);
    assert_ids_match_recording(&ids_of(&raws, "message_id"), &recorded[1..], scenario);
    // …and it is the *last* interaction, not the first: on replay the two
    // placeholders differ, and in record mode the rank check above cannot
    // tell first from last, so pin it by the stop reason too.
    let stop_reasons = recorded_stop_reasons(scenario, true);
    assert_eq!(
        stop_reasons,
        [Some("tool_use".into()), Some("end_turn".into())]
    );
    assert_eq!(ids_of(&raws, "stop_reason"), vec![Some("end_turn".into())]);
}

// ---------------------------------------------------------------------------
// 6–7: a retried turn
// ---------------------------------------------------------------------------

#[tokio::test]
async fn retried_turn_records_retried_attempt_raw_blocking() {
    let scenario = "raw_capture_agent_matrix/retried_turn_records_retried_attempt_raw_blocking";
    let probe = RawProbe::retrying_once();
    let hook = probe.clone();
    let observed: Arc<Mutex<Vec<Value>>> = Default::default();
    let sink = observed.clone();
    with_anthropic_cassette(
        "raw_capture_agent_matrix/retried_turn_records_retried_attempt_raw_blocking",
        move |client| async move {
            let agent = client
                .agent(CLAUDE_HAIKU_4_5)
                .max_tokens(32)
                .add_hook(hook)
                .build();
            let response = agent
                .prompt(TEXT_PROMPT)
                .max_turns(3)
                .await
                .expect("retried run should succeed");
            assert_eq!(
                response.completion_calls.len(),
                2,
                "the rejected attempt and the retried attempt are both recorded"
            );
            *sink.lock().expect("sink") = response
                .completion_calls
                .iter()
                .map(|call| call.raw.clone())
                .collect();
        },
    )
    .await;

    let raws = observed.lock().expect("sink").clone();
    assert_all_populated(&raws, scenario);
    let recorded = recorded_message_ids(scenario, false);
    assert_distinct_msg_ids(&recorded, scenario);
    assert_eq!(recorded.len(), 2, "premise: two attempts were made");
    // The retried attempt's record carries the retried attempt's payload —
    // the second interaction's id, not the first's.
    assert_ne!(raws[0], raws[1]);
    assert_ids_match_recording(&ids_of(&raws, "id"), &recorded, scenario);
    // And the hook that asked for the retry saw each attempt's own payload.
    assert_eq!(probe.model_turns(), raws);
    assert_eq!(probe.completion_responses(), raws);
}

#[tokio::test]
async fn retried_turn_records_retried_attempt_raw_streamed() {
    let scenario = "raw_capture_agent_matrix/retried_turn_records_retried_attempt_raw_streamed";
    let probe = RawProbe::retrying_once();
    let hook = probe.clone();
    let observed: Arc<Mutex<Vec<Value>>> = Default::default();
    let sink = observed.clone();
    with_anthropic_cassette(
        "raw_capture_agent_matrix/retried_turn_records_retried_attempt_raw_streamed",
        move |client| async move {
            let agent = client
                .agent(CLAUDE_HAIKU_4_5)
                .max_tokens(32)
                .add_hook(hook)
                .build();
            let run = drain(
                agent
                    .stream_prompt(Message::user(TEXT_PROMPT))
                    .max_turns(3)
                    .stream()
                    .await,
            )
            .await;
            assert!(run.output.is_some());
            assert_eq!(run.completion_calls.len(), 2, "both attempts recorded");
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
    let recorded = recorded_message_ids(scenario, true);
    assert_distinct_msg_ids(&recorded, scenario);
    assert_eq!(recorded.len(), 2, "premise: two attempts were made");
    assert_ne!(raws[0], raws[1]);
    assert_ids_match_recording(&ids_of(&raws, "message_id"), &recorded, scenario);
    assert_eq!(probe.model_turns(), raws);
    // As on the blocking surface, CompletionResponse fired for the rejected
    // attempt and the retried one alike, each with its own payload.
    assert_eq!(probe.completion_responses(), raws);
}

// Keeps `Tool` in scope for `Adder`'s definition even if a future edit stops
// naming it directly.
#[allow(dead_code)]
fn _tool_trait_in_scope() -> &'static str {
    Adder::NAME
}
