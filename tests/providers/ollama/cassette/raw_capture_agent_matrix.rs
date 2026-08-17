//! Matrix for raw provider response capture through the agent on Ollama's
//! `/api/chat` route: the hook events, `PromptResponse::completion_calls`, and
//! the streamed terminal record.
//!
//! # The feature
//!
//! Capture is always on. The provider populates `CompletionResponse::raw` /
//! `StreamFinal::raw` on every response, and the agent exposes that payload —
//! **per attempt**, never a previous attempt's — as `raw` on the
//! `CompletionResponse`, `StreamResponseFinish`, and `ModelTurnFinished` hook
//! events, on each `CompletionCall` the run records, and on the streamed
//! `StreamedAssistantContent::Final`. `raw` is `Value::Null` only on a value
//! built by hand, with no provider response behind it; `Value::Null` never
//! means "not requested".
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `hooks_observe_raw_blocking` | `agent.prompt` | `CompletionResponse` and `ModelTurnFinished` see `raw`; `eval_count`/`done_reason`/durations match the fixture body | recorded |
//! | 2 | `hooks_observe_raw_streamed` | `agent.stream_prompt` | `StreamResponseFinish` and `ModelTurnFinished` see `raw`; `eval_count`/`done_reason`/durations match the fixture's `done: true` line | recorded |
//! | 3 | `multi_turn_tool_run_records_distinct_raw_blocking` | tool run, `agent.prompt` | two `completion_calls`, two different payloads whose fingerprints equal the interactions' in order; the first carries `message.tool_calls` | recorded |
//! | 4 | `multi_turn_tool_run_records_distinct_raw_streamed` | tool run, `agent.stream_prompt` | two `CompletionCall` items, two different terminal payloads whose fingerprints equal the interactions' in order; the forwarded last `Final.raw` is the final turn's | recorded |
//!
//! Each surface fires its own response event — `CompletionResponse` on the
//! blocking surface, `StreamResponseFinish` on the streamed one — and both
//! fire the medium-neutral `ModelTurnFinished`; cells 1–2 pin exactly which
//! events fire and what each carries, so a hook observing `ModelTurnFinished`
//! alone provably sees the payload for every accepted call on both surfaces.
//!
//! # Identity on this route
//!
//! Ollama assigns no request id and no message id: nothing in a chat response
//! names the attempt. What does distinguish one attempt from the next is the
//! per-response bookkeeping — `eval_count`, `done_reason`, and the nanosecond
//! `total_duration` / `eval_duration` — which the daemon reports on every
//! blocking body and on every stream's `done: true` line, and which the
//! harness leaves untouched (only `created_at` is scrubbed, so it is never
//! compared). Each cell compares that *fingerprint* of the observed payload
//! against the fixture's, interaction by interaction, and the multi-turn cells
//! additionally require the two recorded fingerprints to differ, so "the
//! second call carries the second attempt's payload" is a real claim.
//!
//! Every recorded cell re-derives its premise from its own fixture: every
//! recorded interaction is a completed (`done: true`) turn reporting the
//! fingerprint fields, and the multi-turn cells' first interaction carries a
//! `message.tool_calls` entry naming `add`.
//!
//! Re-record with a local Ollama daemon serving `qwen3:4b`:
//! `RIG_PROVIDER_TEST_MODE=record cargo test -p rig --all-features --test ollama ollama::cassette::raw_capture_agent_matrix -- --nocapture --test-threads=1`

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use rig::agent::{
    AgentHook, HookContext, ModelTurnAction, ModelTurnFinished, MultiTurnStreamItem,
    ObservationAction, StreamResponseFinish,
};
use rig::completion::Message;
use rig::prelude::*;
use rig::streaming::StreamedAssistantContent;
use rig::tool::Tool;
use serde_json::{Value, json};

use super::super::support::with_ollama_cassette;
use crate::cassettes::recorded_interaction_bodies;
use crate::support::{Adder, TOOLS_PREAMBLE};

const OLLAMA_PROVIDER: &str = "ollama";
const MODEL: &str = "qwen3:4b";

/// A prompt whose answer is a single token keeps the recorded body small; the
/// matrix asserts on the response's metadata, never its prose.
const TEXT_PROMPT: &str = "Reply with exactly the single word: pong";
const TOOL_PROMPT: &str = "Use the add tool to add 2 and 3, then state the result.";

/// The fields that fingerprint one Ollama attempt: reported on every blocking
/// body and every stream's `done: true` line, never scrubbed by the harness.
const FINGERPRINT_FIELDS: [&str; 4] = [
    "eval_count",
    "done_reason",
    "total_duration",
    "eval_duration",
];

/// Records what each hook event carried as `raw`, per event, in fire order.
#[derive(Clone, Default)]
struct RawProbe {
    completion_response: Arc<Mutex<Vec<Value>>>,
    stream_response_finish: Arc<Mutex<Vec<Value>>>,
    model_turn_finished: Arc<Mutex<Vec<Value>>>,
}

impl RawProbe {
    fn completion_responses(&self) -> Vec<Value> {
        self.completion_response.lock().expect("probe").clone()
    }

    fn stream_finishes(&self) -> Vec<Value> {
        self.stream_response_finish.lock().expect("probe").clone()
    }

    fn model_turns(&self) -> Vec<Value> {
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
            .push(event.raw.clone());
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
            .push(event.raw.clone());
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

/// Every recorded interaction's completed record, in wire order: the blocking
/// body itself, or the stream's `done: true` NDJSON line. Asserts the premise
/// on the way: each is `done: true` and reports the fingerprint fields.
fn recorded_completed_records(scenario: &str, streamed: bool) -> Vec<Value> {
    let records: Vec<Value> = recorded_interaction_bodies(OLLAMA_PROVIDER, scenario)
        .iter()
        .map(|(_, response)| {
            let line = if streamed {
                response
                    .lines()
                    .map(str::trim)
                    .rfind(|line| !line.is_empty())
                    .unwrap_or_else(|| {
                        panic!("{scenario}: recorded stream body should not be empty")
                    })
            } else {
                response.as_str()
            };
            serde_json::from_str::<Value>(line)
                .unwrap_or_else(|err| panic!("{scenario}: recorded record should be JSON: {err}"))
        })
        .collect();
    assert!(
        !records.is_empty(),
        "{scenario}: the scenario recorded no interactions"
    );
    for (turn, record) in records.iter().enumerate() {
        assert_eq!(
            record.get("done"),
            Some(&Value::Bool(true)),
            "{scenario} turn {turn}: the recorded record must be a completed Ollama response"
        );
        for field in FINGERPRINT_FIELDS {
            assert!(
                record.get(field).is_some_and(|value| !value.is_null()),
                "{scenario} turn {turn}: the recorded record must report `{field}`, the \
                 field this matrix fingerprints an attempt by"
            );
        }
    }
    records
}

/// The fingerprint of one payload: the per-attempt bookkeeping Ollama reports.
fn fingerprint(payload: &Value) -> Vec<(&'static str, Value)> {
    FINGERPRINT_FIELDS
        .iter()
        .map(|field| (*field, payload.get(field).cloned().unwrap_or(Value::Null)))
        .collect()
}

fn fingerprints(payloads: &[Value]) -> Vec<Vec<(&'static str, Value)>> {
    payloads.iter().map(fingerprint).collect()
}

/// Whether each recorded interaction's assistant message carries a
/// `tool_calls` entry naming `add`, in wire order. Ollama streams a tool call
/// on a non-terminal line, so the streamed side scans every line.
fn recorded_add_call_turns(scenario: &str) -> Vec<bool> {
    recorded_interaction_bodies(OLLAMA_PROVIDER, scenario)
        .iter()
        .map(|(_, response)| {
            response
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .filter_map(|line| serde_json::from_str::<Value>(line).ok())
                .any(|record| {
                    record
                        .pointer("/message/tool_calls")
                        .and_then(Value::as_array)
                        .is_some_and(|calls| {
                            calls.iter().any(|call| {
                                call.pointer("/function/name")
                                    == Some(&Value::String(Adder::NAME.to_string()))
                            })
                        })
                })
        })
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

/// The premise of the multi-turn cells: the two recorded attempts are told
/// apart by their fingerprints, so matching them in order is a real claim.
fn assert_distinct_fingerprints(records: &[Value], context: &str) {
    for (i, a) in records.iter().enumerate() {
        for b in &records[i + 1..] {
            assert_ne!(
                fingerprint(a),
                fingerprint(b),
                "{context}: premise — attempts have distinct fingerprints"
            );
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
    with_ollama_cassette(
        "raw_capture_agent_matrix/hooks_observe_raw_blocking",
        move |client| async move {
            let agent = client
                .agent(MODEL)
                .max_tokens(64)
                .additional_params(json!({ "think": false }))
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
    let raw = &responses[0];
    assert!(!raw.is_null(), "CompletionResponse.raw is populated");
    assert_eq!(&turns[0], raw, "both events observe the same payload");
    // The payload is this attempt's own response: the blocking body as the
    // provider type carries it.
    let records = recorded_completed_records(scenario, false);
    assert_eq!(records.len(), 1);
    assert_eq!(fingerprints(&responses), fingerprints(&records));
    assert_eq!(raw["model"], records[0]["model"]);
    assert_eq!(raw["done"], Value::Bool(true));
    assert_eq!(
        raw.pointer("/message/content"),
        records[0].pointer("/message/content")
    );
}

#[tokio::test]
async fn hooks_observe_raw_streamed() {
    let scenario = "raw_capture_agent_matrix/hooks_observe_raw_streamed";
    let probe = RawProbe::default();
    let hook = probe.clone();
    with_ollama_cassette(
        "raw_capture_agent_matrix/hooks_observe_raw_streamed",
        move |client| async move {
            let agent = client
                .agent(MODEL)
                .max_tokens(64)
                .additional_params(json!({ "think": false }))
                .add_hook(hook)
                .build();
            let run = drain(agent.stream_prompt(Message::user(TEXT_PROMPT)).await).await;
            assert!(run.output.is_some(), "the run finished");
            assert_eq!(run.finals.len(), 1, "one text turn, one terminal record");
            assert!(
                !run.finals[0].raw.is_null(),
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
    let raw = &finishes[0];
    assert!(!raw.is_null(), "StreamResponseFinish.raw is populated");
    assert_eq!(&turns[0], raw, "both events observe the same payload");
    // The streamed payload is Ollama's *terminal* record — the `done: true`
    // line's bookkeeping, not the stream's message content.
    assert!(raw.get("eval_count").is_some() && raw.get("done_reason").is_some());
    assert!(raw.get("message").is_none());
    let records = recorded_completed_records(scenario, true);
    assert_eq!(records.len(), 1);
    assert_eq!(fingerprints(&finishes), fingerprints(&records));
    assert_eq!(raw["model"], records[0]["model"]);
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
    with_ollama_cassette(
        "raw_capture_agent_matrix/multi_turn_tool_run_records_distinct_raw_blocking",
        move |client| async move {
            let agent = client
                .agent(MODEL)
                .preamble(TOOLS_PREAMBLE)
                .additional_params(json!({ "think": false }))
                .tool(Adder)
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
            *sink.lock().expect("sink") = calls.iter().map(|call| call.raw.clone()).collect();
        },
    )
    .await;

    let raws = observed.lock().expect("sink").clone();
    assert_all_populated(&raws, scenario);
    assert_ne!(raws[0], raws[1], "two attempts, two different payloads");
    let records = recorded_completed_records(scenario, false);
    assert_eq!(records.len(), 2, "premise: two attempts were made");
    assert_distinct_fingerprints(&records, scenario);
    // Each payload is its own attempt's, in the interactions' order.
    assert_eq!(fingerprints(&raws), fingerprints(&records));
    // The first is the tool turn: it carries the wire's `message.tool_calls`.
    assert_eq!(
        recorded_add_call_turns(scenario),
        [true, false],
        "premise: an add call then a text turn"
    );
    assert!(
        raws[0]
            .pointer("/message/tool_calls/0/function/name")
            .is_some_and(|name| name == Adder::NAME),
        "the first payload carries the wire's tool_calls: {:?}",
        raws[0]
    );
    assert_eq!(
        raws[0].pointer("/message/tool_calls/0/function/arguments"),
        records[0].pointer("/message/tool_calls/0/function/arguments"),
        "raw carries the wire's tool-call arguments untouched"
    );
    assert!(
        raws[1]
            .pointer("/message/tool_calls")
            .is_none_or(|calls| calls.as_array().is_some_and(Vec::is_empty)),
        "the second payload is the text turn: {:?}",
        raws[1]
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
    let observed: Arc<Mutex<Vec<Value>>> = Default::default();
    let sink = observed.clone();
    let last_final: Arc<Mutex<Option<Value>>> = Default::default();
    let final_sink = last_final.clone();
    with_ollama_cassette(
        "raw_capture_agent_matrix/multi_turn_tool_run_records_distinct_raw_streamed",
        move |client| async move {
            let agent = client
                .agent(MODEL)
                .preamble(TOOLS_PREAMBLE)
                .additional_params(json!({ "think": false }))
                .tool(Adder)
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
    let records = recorded_completed_records(scenario, true);
    assert_eq!(records.len(), 2, "premise: two attempts were made");
    assert_distinct_fingerprints(&records, scenario);
    // Each terminal payload is its own attempt's, in the interactions' order,
    // and the first interaction is the tool turn.
    assert_eq!(fingerprints(&raws), fingerprints(&records));
    assert_eq!(
        recorded_add_call_turns(scenario),
        [true, false],
        "premise: an add call then a text turn"
    );
    // The forwarded Final is the *last* interaction's, not the first's.
    let last = last_final
        .lock()
        .expect("sink")
        .clone()
        .expect("last Final");
    assert_eq!(fingerprint(&last), fingerprint(&records[1]));
    assert_ne!(fingerprint(&last), fingerprint(&records[0]));
    // ModelTurnFinished fires for both attempts and sees each attempt's own.
    assert_eq!(probe.model_turns(), raws);
    // StreamResponseFinish fires for every turn that streamed text — always
    // the final text turn, and the tool turn too when qwen3 narrates before
    // calling — and each firing carries its own attempt's payload, the last
    // one the final turn's.
    let finishes = probe.stream_finishes();
    assert_eq!(finishes.last(), Some(&raws[1]));
    assert!(
        finishes.iter().all(|finish| raws.contains(finish))
            && finishes.len() <= raws.len()
            && (finishes.len() == 1 || finishes == raws),
        "each StreamResponseFinish carries its own attempt's payload, in order: {finishes:?}"
    );
}
