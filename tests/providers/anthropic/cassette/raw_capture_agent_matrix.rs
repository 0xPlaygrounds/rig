//! Matrix for opt-in raw provider response capture through the agent: the
//! hook events, `PromptResponse::completion_calls`, the streamed terminal
//! record, and the three toggles that decide whether a run captures.
//!
//! # The feature
//!
//! `AgentBuilder::capture_raw_response`, overridable per run
//! (`AgentRunner::capture_raw_response`) and per request
//! (`PromptRequest::capture_raw_response`), travels on the concrete
//! `CompletionRequest` to the provider, which populates
//! `CompletionResponse::raw` / `StreamFinal::raw`. The agent then exposes that
//! payload — **per attempt**, never a previous attempt's — as `raw` on the
//! `CompletionResponse`, `StreamResponseFinish`, and `ModelTurnFinished` hook
//! events, on each `CompletionCall` the run records, and on the streamed
//! `StreamedAssistantContent::Final`. Off (the default) means `None`
//! everywhere, and the flag never changes the request on the wire.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `hooks_observe_raw_blocking_on` | `agent.prompt`, agent-level on | `CompletionResponse` and `ModelTurnFinished` see `raw`; `id` matches fixture | recorded |
//! | 2 | `hooks_observe_none_blocking_off` | `agent.prompt`, default | both events see `None` | recorded |
//! | 3 | `hooks_observe_raw_streamed_on` | `agent.stream_prompt`, agent-level on | `StreamResponseFinish` and `ModelTurnFinished` see `raw`; `message_id` matches fixture | recorded |
//! | 4 | `hooks_observe_none_streamed_off` | `agent.stream_prompt`, default | both events see `None` | recorded |
//! | 5 | `multi_turn_tool_run_records_distinct_raw_blocking` | tool run, `agent.prompt` | two `completion_calls`, two different `raw["id"]`s equal to the interactions' ids in order | recorded |
//! | 6 | `multi_turn_tool_run_records_distinct_raw_streamed` | tool run, `agent.stream_prompt` | two `CompletionCall` items, two different `raw["message_id"]`s equal to the interactions' ids in order | recorded |
//! | 7 | `streamed_final_carries_final_turn_raw` | tool run, `agent.stream_prompt` | the last `StreamedAssistantContent::Final.raw["message_id"]` is the last interaction's id | recorded |
//! | 8 | `retried_turn_records_retried_attempt_raw_blocking` | `ModelTurnFinished` → `Retry` once, `agent.prompt` | second recorded call / second event carry the second interaction's `id` | recorded |
//! | 9 | `retried_turn_records_retried_attempt_raw_streamed` | same, `agent.stream_prompt` | same, by `message_id` | recorded |
//! | 10 | `toggle_agent_level_on` | `AgentBuilder::capture_raw_response(true)` | `completion_calls[0].raw` present | recorded |
//! | 11 | `toggle_per_run_override_off` | agent on, `AgentRunner::capture_raw_response(false)` | `completion_calls[0].raw` absent | recorded |
//! | 12 | `toggle_per_request_override_on` | agent default, `PromptRequest::capture_raw_response(true)` | `completion_calls[0].raw` present | recorded |
//! | 13 | `toggle_default_off` | agent default, request default | `completion_calls[0].raw` absent | recorded |
//! | 14 | `toggle_request_invariant` | recorded request bodies of 10–13 | byte-identical | recorded (derived from cells 10–13) |
//!
//! Each surface fires its own response event — `CompletionResponse` on the
//! blocking surface, `StreamResponseFinish` on the streamed one — and both
//! fire the medium-neutral `ModelTurnFinished`; cells 1–4 pin exactly which
//! events fire and what each carries, so a hook observing `ModelTurnFinished`
//! alone provably sees the payload for every accepted call on both surfaces.
//!
//! Cell 14 makes no request of its own; it reads the four toggle fixtures,
//! which send the same prompt for exactly that comparison. Every recorded cell
//! re-derives its premise from its own fixture: the recorded interactions
//! carry `msg_…` ids (distinct per attempt where the cell needs two), and the
//! multi-turn cells' first interaction is a `tool_use` turn.

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use rig::agent::{
    AgentHook, HookContext, ModelTurnAction, ModelTurnFinished, MultiTurnStreamItem,
    ObservationAction, StreamResponseFinish,
};
use rig::completion::Message;
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

const TOGGLE_SCENARIOS: [&str; 4] = [
    "raw_capture_agent_matrix/toggle_agent_level_on",
    "raw_capture_agent_matrix/toggle_per_run_override_off",
    "raw_capture_agent_matrix/toggle_per_request_override_on",
    "raw_capture_agent_matrix/toggle_default_off",
];

/// Records what each hook event carried as `raw`, per event, in fire order.
#[derive(Clone, Default)]
struct RawProbe {
    completion_response: Arc<Mutex<Vec<Option<Value>>>>,
    stream_response_finish: Arc<Mutex<Vec<Option<Value>>>>,
    model_turn_finished: Arc<Mutex<Vec<Option<Value>>>>,
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

    fn completion_responses(&self) -> Vec<Option<Value>> {
        self.completion_response.lock().expect("probe").clone()
    }

    fn stream_finishes(&self) -> Vec<Option<Value>> {
        self.stream_response_finish.lock().expect("probe").clone()
    }

    fn model_turns(&self) -> Vec<Option<Value>> {
        self.model_turn_finished.lock().expect("probe").clone()
    }
}

impl AgentHook for RawProbe {
    async fn on_completion_response(
        &self,
        _ctx: &HookContext,
        event: rig::agent::CompletionResponseEvent<'_>,
    ) -> ObservationAction {
        self.completion_response
            .lock()
            .expect("probe")
            .push(event.raw.cloned());
        ObservationAction::continue_run()
    }

    async fn on_stream_response_finish(
        &self,
        _ctx: &HookContext,
        event: StreamResponseFinish<'_>,
    ) -> ObservationAction {
        self.stream_response_finish
            .lock()
            .expect("probe")
            .push(event.raw.cloned());
        ObservationAction::continue_run()
    }

    async fn on_model_turn_finished(
        &self,
        _ctx: &HookContext,
        event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        let mut seen = self.model_turn_finished.lock().expect("probe");
        seen.push(event.raw.cloned());
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
                run.finals.push(final_)
            }
            MultiTurnStreamItem::FinalResponse(response) => {
                run.output = Some(response.output().to_owned())
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

fn ids_of(raws: &[Option<Value>], key: &str) -> Vec<Option<String>> {
    raws.iter()
        .map(|raw| {
            raw.as_ref()
                .and_then(|raw| raw[key].as_str())
                .map(str::to_string)
        })
        .collect()
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
// 1–4: the hook events
// ---------------------------------------------------------------------------

#[tokio::test]
async fn hooks_observe_raw_blocking_on() {
    let scenario = "raw_capture_agent_matrix/hooks_observe_raw_blocking_on";
    let probe = RawProbe::default();
    let hook = probe.clone();
    with_anthropic_cassette(
        "raw_capture_agent_matrix/hooks_observe_raw_blocking_on",
        move |client| async move {
            let agent = client
                .agent(CLAUDE_HAIKU_4_5)
                .max_tokens(32)
                .capture_raw_response(true)
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
    assert_eq!(turns.len(), 1, "one ModelTurnFinished event");
    assert!(
        probe.stream_finishes().is_empty(),
        "the blocking surface fires no StreamResponseFinish"
    );
    let raw = responses[0].as_ref().expect("CompletionResponse.raw on");
    assert_eq!(
        turns[0].as_ref(),
        Some(raw),
        "both events observe the same payload"
    );
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
async fn hooks_observe_none_blocking_off() {
    let scenario = "raw_capture_agent_matrix/hooks_observe_none_blocking_off";
    let probe = RawProbe::default();
    let hook = probe.clone();
    with_anthropic_cassette(
        "raw_capture_agent_matrix/hooks_observe_none_blocking_off",
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

    assert_eq!(probe.completion_responses(), vec![None]);
    assert_eq!(probe.model_turns(), vec![None]);
    assert!(probe.stream_finishes().is_empty());
    // The fixture did record a full response — `None` is policy, not absence.
    let body = recorded_response_body(scenario);
    assert!(body["id"].as_str().is_some_and(|id| id.starts_with("msg_")));
}

#[tokio::test]
async fn hooks_observe_raw_streamed_on() {
    let scenario = "raw_capture_agent_matrix/hooks_observe_raw_streamed_on";
    let probe = RawProbe::default();
    let hook = probe.clone();
    with_anthropic_cassette(
        "raw_capture_agent_matrix/hooks_observe_raw_streamed_on",
        move |client| async move {
            let agent = client
                .agent(CLAUDE_HAIKU_4_5)
                .max_tokens(32)
                .capture_raw_response(true)
                .add_hook(hook)
                .build();
            let run = drain(agent.stream_prompt(Message::user(TEXT_PROMPT)).await).await;
            assert!(run.output.is_some(), "the run finished");
            assert_eq!(run.finals.len(), 1, "one text turn, one terminal record");
            assert!(
                run.finals[0].raw.is_some(),
                "the streamed terminal carries raw"
            );
        },
    )
    .await;

    let finishes = probe.stream_finishes();
    let turns = probe.model_turns();
    assert!(
        probe.completion_responses().is_empty(),
        "the streamed surface fires StreamResponseFinish, not CompletionResponse"
    );
    assert_eq!(finishes.len(), 1, "one StreamResponseFinish event");
    assert_eq!(turns.len(), 1, "one ModelTurnFinished event");
    let raw = finishes[0].as_ref().expect("StreamResponseFinish.raw on");
    assert_eq!(
        turns[0].as_ref(),
        Some(raw),
        "both events observe the same payload"
    );
    // The streamed payload is the Anthropic *terminal* record.
    assert!(raw.get("message_id").is_some() && raw.get("usage").is_some());
    let recorded = recorded_message_ids(scenario, true);
    assert_eq!(recorded.len(), 1);
    assert_ids_match_recording(&ids_of(&finishes, "message_id"), &recorded, scenario);
    assert_eq!(
        ids_of(&finishes, "stop_reason"),
        recorded_stop_reasons(scenario, true)
    );
}

#[tokio::test]
async fn hooks_observe_none_streamed_off() {
    let scenario = "raw_capture_agent_matrix/hooks_observe_none_streamed_off";
    let probe = RawProbe::default();
    let hook = probe.clone();
    with_anthropic_cassette(
        "raw_capture_agent_matrix/hooks_observe_none_streamed_off",
        move |client| async move {
            let agent = client
                .agent(CLAUDE_HAIKU_4_5)
                .max_tokens(32)
                .add_hook(hook)
                .build();
            let run = drain(agent.stream_prompt(Message::user(TEXT_PROMPT)).await).await;
            assert_eq!(run.finals.len(), 1);
            assert!(run.finals[0].raw.is_none(), "terminal raw off by default");
            assert_eq!(run.completion_calls.len(), 1);
            assert!(run.completion_calls[0].raw.is_none());
        },
    )
    .await;

    assert!(probe.completion_responses().is_empty());
    assert_eq!(probe.stream_finishes(), vec![None]);
    assert_eq!(probe.model_turns(), vec![None]);
    let recorded = recorded_message_ids(scenario, true);
    assert_distinct_msg_ids(&recorded, scenario);
}

// ---------------------------------------------------------------------------
// 5–7: multi-turn tool runs
// ---------------------------------------------------------------------------

#[tokio::test]
async fn multi_turn_tool_run_records_distinct_raw_blocking() {
    let scenario = "raw_capture_agent_matrix/multi_turn_tool_run_records_distinct_raw_blocking";
    let probe = RawProbe::default();
    let hook = probe.clone();
    let observed: Arc<Mutex<Vec<Option<Value>>>> = Default::default();
    let sink = observed.clone();
    with_anthropic_cassette(
        "raw_capture_agent_matrix/multi_turn_tool_run_records_distinct_raw_blocking",
        move |client| async move {
            let agent = client
                .agent(CLAUDE_HAIKU_4_5)
                .preamble(TOOLS_PREAMBLE)
                .max_tokens(1024)
                .tool(Adder)
                .capture_raw_response(true)
                .add_hook(hook)
                .build();
            let response = agent
                .prompt(TOOL_PROMPT)
                .max_turns(3)
                .extended_details()
                .await
                .expect("tool run should succeed");
            let calls = &response.completion_calls;
            assert_eq!(calls.len(), 2, "a tool turn then a text turn");
            *sink.lock().expect("sink") = calls
                .iter()
                .map(|call| call.raw.as_deref().cloned())
                .collect();
        },
    )
    .await;

    let raws = observed.lock().expect("sink").clone();
    assert!(raws.iter().all(Option::is_some), "every call carries raw");
    assert_ne!(raws[0], raws[1], "two attempts, two different payloads");
    let ids = ids_of(&raws, "id");
    let recorded = recorded_message_ids(scenario, false);
    assert_distinct_msg_ids(&recorded, scenario);
    assert_ids_match_recording(&ids, &recorded, scenario);
    // Each payload is its own attempt's: the first stopped for tool use.
    let stop_reasons = recorded_stop_reasons(scenario, false);
    assert_eq!(stop_reasons[0].as_deref(), Some("tool_use"), "premise");
    assert_eq!(ids_of(&raws, "stop_reason"), stop_reasons);
    assert_eq!(
        raws[0].as_ref().expect("raw")["content"][0]["type"],
        "tool_use"
    );
    // The hooks saw the same two payloads in the same order.
    assert_eq!(probe.completion_responses(), raws);
    assert_eq!(probe.model_turns(), raws);
}

#[tokio::test]
async fn multi_turn_tool_run_records_distinct_raw_streamed() {
    let scenario = "raw_capture_agent_matrix/multi_turn_tool_run_records_distinct_raw_streamed";
    let probe = RawProbe::default();
    let hook = probe.clone();
    let observed: Arc<Mutex<Vec<Option<Value>>>> = Default::default();
    let sink = observed.clone();
    with_anthropic_cassette(
        "raw_capture_agent_matrix/multi_turn_tool_run_records_distinct_raw_streamed",
        move |client| async move {
            let agent = client
                .agent(CLAUDE_HAIKU_4_5)
                .preamble(TOOLS_PREAMBLE)
                .max_tokens(1024)
                .tool(Adder)
                .capture_raw_response(true)
                .add_hook(hook)
                .build();
            let run = drain(
                agent
                    .stream_prompt(Message::user(TOOL_PROMPT))
                    .max_turns(3)
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
                .map(|call| call.raw.as_deref().cloned())
                .collect();
        },
    )
    .await;

    let raws = observed.lock().expect("sink").clone();
    assert!(raws.iter().all(Option::is_some), "every call carries raw");
    assert_ne!(raws[0], raws[1], "two attempts, two different payloads");
    let recorded = recorded_message_ids(scenario, true);
    assert_distinct_msg_ids(&recorded, scenario);
    assert_ids_match_recording(&ids_of(&raws, "message_id"), &recorded, scenario);
    let stop_reasons = recorded_stop_reasons(scenario, true);
    assert_eq!(stop_reasons[0].as_deref(), Some("tool_use"), "premise");
    assert_eq!(ids_of(&raws, "stop_reason"), stop_reasons);
    // ModelTurnFinished fires for both attempts and sees each attempt's own.
    assert_eq!(probe.model_turns(), raws);
    // The tool-only turn fires no StreamResponseFinish; the text turn's
    // finish carries the second attempt's payload.
    assert_eq!(probe.stream_finishes(), vec![raws[1].clone()]);
}

#[tokio::test]
async fn streamed_final_carries_final_turn_raw() {
    let scenario = "raw_capture_agent_matrix/streamed_final_carries_final_turn_raw";
    let observed: Arc<Mutex<Vec<Option<Value>>>> = Default::default();
    let sink = observed.clone();
    with_anthropic_cassette(
        "raw_capture_agent_matrix/streamed_final_carries_final_turn_raw",
        move |client| async move {
            let agent = client
                .agent(CLAUDE_HAIKU_4_5)
                .preamble(TOOLS_PREAMBLE)
                .max_tokens(1024)
                .tool(Adder)
                .capture_raw_response(true)
                .build();
            let run = drain(
                agent
                    .stream_prompt(Message::user(TOOL_PROMPT))
                    .max_turns(3)
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
            *sink.lock().expect("sink") = vec![last.raw.as_deref().cloned()];
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
// 8–9: a retried turn
// ---------------------------------------------------------------------------

#[tokio::test]
async fn retried_turn_records_retried_attempt_raw_blocking() {
    let scenario = "raw_capture_agent_matrix/retried_turn_records_retried_attempt_raw_blocking";
    let probe = RawProbe::retrying_once();
    let hook = probe.clone();
    let observed: Arc<Mutex<Vec<Option<Value>>>> = Default::default();
    let sink = observed.clone();
    with_anthropic_cassette(
        "raw_capture_agent_matrix/retried_turn_records_retried_attempt_raw_blocking",
        move |client| async move {
            let agent = client
                .agent(CLAUDE_HAIKU_4_5)
                .max_tokens(32)
                .capture_raw_response(true)
                .add_hook(hook)
                .build();
            let response = agent
                .prompt(TEXT_PROMPT)
                .max_turns(3)
                .extended_details()
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
                .map(|call| call.raw.as_deref().cloned())
                .collect();
        },
    )
    .await;

    let raws = observed.lock().expect("sink").clone();
    assert!(raws.iter().all(Option::is_some));
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
    let observed: Arc<Mutex<Vec<Option<Value>>>> = Default::default();
    let sink = observed.clone();
    with_anthropic_cassette(
        "raw_capture_agent_matrix/retried_turn_records_retried_attempt_raw_streamed",
        move |client| async move {
            let agent = client
                .agent(CLAUDE_HAIKU_4_5)
                .max_tokens(32)
                .capture_raw_response(true)
                .add_hook(hook)
                .build();
            let run = drain(
                agent
                    .stream_prompt(Message::user(TEXT_PROMPT))
                    .max_turns(3)
                    .await,
            )
            .await;
            assert!(run.output.is_some());
            assert_eq!(run.completion_calls.len(), 2, "both attempts recorded");
            *sink.lock().expect("sink") = run
                .completion_calls
                .iter()
                .map(|call| call.raw.as_deref().cloned())
                .collect();
        },
    )
    .await;

    let raws = observed.lock().expect("sink").clone();
    assert!(raws.iter().all(Option::is_some));
    let recorded = recorded_message_ids(scenario, true);
    assert_distinct_msg_ids(&recorded, scenario);
    assert_eq!(recorded.len(), 2, "premise: two attempts were made");
    assert_ne!(raws[0], raws[1]);
    assert_ids_match_recording(&ids_of(&raws, "message_id"), &recorded, scenario);
    assert_eq!(probe.model_turns(), raws);
}

// ---------------------------------------------------------------------------
// 10–14: the toggles
// ---------------------------------------------------------------------------

#[tokio::test]
async fn toggle_agent_level_on() {
    let scenario = "raw_capture_agent_matrix/toggle_agent_level_on";
    with_anthropic_cassette(
        "raw_capture_agent_matrix/toggle_agent_level_on",
        |client| async move {
            let agent = client
                .agent(CLAUDE_HAIKU_4_5)
                .max_tokens(32)
                .capture_raw_response(true)
                .build();
            let response = agent
                .prompt(TEXT_PROMPT)
                .extended_details()
                .await
                .expect("prompt should succeed");
            assert_eq!(response.completion_calls.len(), 1);
            assert!(
                response.completion_calls[0].raw.is_some(),
                "agent-level opt-in reaches the recorded call"
            );
        },
    )
    .await;
    assert!(
        recorded_response_body(scenario)["id"]
            .as_str()
            .is_some_and(|id| id.starts_with("msg_"))
    );
}

#[tokio::test]
async fn toggle_per_run_override_off() {
    let scenario = "raw_capture_agent_matrix/toggle_per_run_override_off";
    with_anthropic_cassette(
        "raw_capture_agent_matrix/toggle_per_run_override_off",
        |client| async move {
            let agent = client
                .agent(CLAUDE_HAIKU_4_5)
                .max_tokens(32)
                .capture_raw_response(true)
                .build();
            let response = agent
                .runner(Message::user(TEXT_PROMPT))
                .capture_raw_response(false)
                .run()
                .await
                .expect("run should succeed");
            assert_eq!(response.completion_calls.len(), 1);
            assert!(
                response.completion_calls[0].raw.is_none(),
                "a per-run override turns an agent-level opt-in off"
            );
        },
    )
    .await;
    assert!(
        recorded_response_body(scenario)["id"]
            .as_str()
            .is_some_and(|id| id.starts_with("msg_"))
    );
}

#[tokio::test]
async fn toggle_per_request_override_on() {
    let scenario = "raw_capture_agent_matrix/toggle_per_request_override_on";
    with_anthropic_cassette(
        "raw_capture_agent_matrix/toggle_per_request_override_on",
        |client| async move {
            let agent = client.agent(CLAUDE_HAIKU_4_5).max_tokens(32).build();
            let response = agent
                .prompt(TEXT_PROMPT)
                .capture_raw_response(true)
                .extended_details()
                .await
                .expect("prompt should succeed");
            assert_eq!(response.completion_calls.len(), 1);
            assert!(
                response.completion_calls[0].raw.is_some(),
                "a per-request opt-in works on an agent that never asked"
            );
        },
    )
    .await;
    assert!(
        recorded_response_body(scenario)["id"]
            .as_str()
            .is_some_and(|id| id.starts_with("msg_"))
    );
}

#[tokio::test]
async fn toggle_default_off() {
    let scenario = "raw_capture_agent_matrix/toggle_default_off";
    with_anthropic_cassette(
        "raw_capture_agent_matrix/toggle_default_off",
        |client| async move {
            let agent = client.agent(CLAUDE_HAIKU_4_5).max_tokens(32).build();
            let response = agent
                .prompt(TEXT_PROMPT)
                .extended_details()
                .await
                .expect("prompt should succeed");
            assert_eq!(response.completion_calls.len(), 1);
            assert!(
                response.completion_calls[0].raw.is_none(),
                "nobody asked, nobody pays"
            );
        },
    )
    .await;
    assert!(
        recorded_response_body(scenario)["id"]
            .as_str()
            .is_some_and(|id| id.starts_with("msg_"))
    );
}

/// None of the toggles reaches the wire: the four toggle cells recorded
/// byte-identical request bodies.
#[test]
fn toggle_request_invariant() {
    let bodies: Vec<Vec<(String, String)>> = TOGGLE_SCENARIOS
        .iter()
        .map(|scenario| crate::cassettes::recorded_interaction_bodies(ANTHROPIC_PROVIDER, scenario))
        .collect();
    for (scenario, interactions) in TOGGLE_SCENARIOS.iter().zip(&bodies) {
        assert_eq!(
            interactions.len(),
            1,
            "{scenario}: one recorded interaction"
        );
    }
    let reference = &bodies[0][0].0;
    for (scenario, interactions) in TOGGLE_SCENARIOS.iter().zip(&bodies).skip(1) {
        assert_eq!(
            &interactions[0].0, reference,
            "{scenario}: the recorded request must be byte-identical to {}",
            TOGGLE_SCENARIOS[0]
        );
    }
    assert!(
        !reference.contains("capture_raw") && !reference.contains("captureRaw"),
        "the flag is `#[serde(skip)]`; it must never serialize: {reference}"
    );
}

// Keeps `Tool` in scope for `Adder`'s definition even if a future edit stops
// naming it directly.
#[allow(dead_code)]
fn _tool_trait_in_scope() -> &'static str {
    Adder::NAME
}
