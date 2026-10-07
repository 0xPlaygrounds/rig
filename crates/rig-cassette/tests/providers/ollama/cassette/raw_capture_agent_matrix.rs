//! Matrix for raw provider response capture through the agent on Ollama's
//! OpenAI-compatible Chat Completions route: the hook events, `PromptResponse::completion_calls`, and
//! the streamed terminal record.
//!
//! # The feature
//!
//! Capture is always on. The provider populates `CompletionResponse::raw`
//! on every response, and the agent exposes that payload —
//! **per attempt**, never a previous attempt's — as `raw` on the
//! `CompletionResponse` and `ModelTurnFinished` hook events, on each
//! `CompletionCall` the run records, streamed or not. `raw` is `Value::Null` only on a value
//! built by hand, with no provider response behind it; `Value::Null` never
//! means "not requested".
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 2 | `hooks_observe_raw_streamed` | `agent.prompt(..).stream()` | `CompletionResponse` and `ModelTurnFinished` see `raw`; its usage matches the fixture's usage chunk | recorded |
//! | 3 | `multi_turn_tool_run_records_distinct_raw_blocking` | tool run, `agent.prompt` | two `completion_calls`, two different payloads whose fingerprints equal the interactions' in order; the first carries `tool_calls` | recorded |
//!
//! Both surfaces fire the same two events per accepted model turn:
//! `CompletionResponse` — after the unary call returns on the blocking
//! surface, after the whole stream is assembled on the streamed one, with
//! `HookContext::is_streaming` telling them apart — and the medium-neutral
//! `ModelTurnFinished`. Tool-only turns fire both. Cell 2 pins which events
//! fire on the streamed surface and what each carries, and cell 3 that every
//! attempt of a blocking tool run fires them.
//!
//! # Identity on this route
//!
//! Ollama assigns no request id. What distinguishes one attempt from the next
//! is its token accounting, which the daemon reports on every blocking body
//! and on every stream's usage chunk. Each cell compares that *fingerprint*
//! of the observed payload against the fixture's, interaction by
//! interaction, and the multi-turn cells additionally require the two
//! recorded fingerprints to differ, so "the second call carries the second
//! attempt's payload" is a real claim.
//!
//! Every recorded cell re-derives its premise from its own fixture: every
//! recorded interaction reports usage, and the multi-turn cells' first
//! interaction carries a `tool_calls` entry naming `add`.
//!
//! Re-record with a local Ollama daemon serving `qwen3:4b`:
//! `cargo xtask cassette record ollama/raw_capture_agent_matrix/<scenario>.yaml`.

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use rig::agent::{
    AgentHook, HookContext, ModelTurnAction, ModelTurnFinished, MultiTurnStreamItem, OutcomeAction,
    OutcomeEvent,
};
use rig::completion::Message;
use rig::message::AssistantContent;
use rig::tool::Tool;
use serde_json::Value;

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
/// body and every stream's usage chunk, never scrubbed by the harness.
const FINGERPRINT_FIELDS: [&str; 3] = [
    "/usage/prompt_tokens",
    "/usage/completion_tokens",
    "/usage/total_tokens",
];

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
    async fn on_outcome(&self, ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        let Some(response) = event.completion() else {
            return OutcomeAction::proceed();
        };
        self.completion_response
            .lock()
            .expect("probe")
            .push(ResponseSeen {
                streaming: ctx.is_streaming(),
                tool_call: response
                    .choice
                    .iter()
                    .any(|content| matches!(content, AssistantContent::ToolCall(_))),
                raw: response.raw.clone(),
            });
        OutcomeAction::proceed()
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
    output: Option<String>,
}

async fn drain(mut stream: rig::agent::StreamingResult) -> StreamedRun {
    let mut run = StreamedRun::default();
    while let Some(item) = stream.next().await {
        match item.expect("stream item should succeed") {
            MultiTurnStreamItem::CompletionCall(call) => run.completion_calls.push(call),
            MultiTurnStreamItem::FinalResponse(response) => {
                run.output = Some(response.output().to_owned());
            }
            _ => {}
        }
    }
    run
}

/// The JSON documents of one recorded response: the blocking body itself, or
/// each SSE data frame of a stream.
fn recorded_documents(response: &str) -> Vec<Value> {
    if let Ok(body) = serde_json::from_str::<Value>(response) {
        return vec![body];
    }
    response
        .lines()
        .filter_map(|line| line.trim().strip_prefix("data:"))
        .filter_map(|data| serde_json::from_str::<Value>(data.trim()).ok())
        .collect()
}

/// Every recorded interaction's usage-bearing record, in wire order: the
/// blocking body itself, or the stream's usage chunk. Asserts the premise on
/// the way: each reports the fingerprint fields.
fn recorded_completed_records(scenario: &str, streamed: bool) -> Vec<Value> {
    let records: Vec<Value> = recorded_interaction_bodies(OLLAMA_PROVIDER, scenario)
        .iter()
        .map(|(_, response)| {
            let documents = recorded_documents(response);
            assert_eq!(
                documents.len() > 1,
                streamed,
                "{scenario}: the recorded mode"
            );
            documents
                .into_iter()
                .rfind(|document| document.get("usage").is_some_and(Value::is_object))
                .unwrap_or_else(|| panic!("{scenario}: the recorded reply should report usage"))
        })
        .collect();
    assert!(
        !records.is_empty(),
        "{scenario}: the scenario recorded no interactions"
    );
    for (turn, record) in records.iter().enumerate() {
        for field in FINGERPRINT_FIELDS {
            assert!(
                record.pointer(field).is_some_and(|value| !value.is_null()),
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
        .map(|field| {
            (
                *field,
                payload.pointer(field).cloned().unwrap_or(Value::Null),
            )
        })
        .collect()
}

fn fingerprints(payloads: &[Value]) -> Vec<Vec<(&'static str, Value)>> {
    payloads.iter().map(fingerprint).collect()
}

/// Whether each recorded interaction's assistant message carries a
/// `tool_calls` entry naming `add`, in wire order. A stream carries its call
/// on a delta before the usage chunk, so the streamed side scans every frame.
fn recorded_add_call_turns(scenario: &str) -> Vec<bool> {
    recorded_interaction_bodies(OLLAMA_PROVIDER, scenario)
        .iter()
        .map(|(_, response)| {
            recorded_documents(response).into_iter().any(|record| {
                [
                    "/choices/0/message/tool_calls",
                    "/choices/0/delta/tool_calls",
                ]
                .into_iter()
                .filter_map(|pointer| record.pointer(pointer))
                .filter_map(Value::as_array)
                .any(|calls| {
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
// 2: the hook events
// ---------------------------------------------------------------------------

#[tokio::test]
async fn hooks_observe_raw_streamed() {
    let scenario = "raw_capture_agent_matrix/hooks_observe_raw_streamed";
    let probe = RawProbe::default();
    let hook = probe.clone();
    with_ollama_cassette(
        "raw_capture_agent_matrix/hooks_observe_raw_streamed",
        move |client| async move {
            let agent = rig::AgentBuilder::new(client.completion(MODEL))
                .max_tokens(64)
                .options(
                    rig::completion::GenerationOptions::default()
                        .reasoning(rig::completion::Reasoning::Off),
                )
                .add_hook(hook)
                .build();
            let run = drain(agent.prompt(Message::user(TEXT_PROMPT)).stream()).await;
            assert!(run.output.is_some(), "the run finished");
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
    // The streamed payload is the `chat.completion` the stream rebuilt: the
    // stream's accounting and its message, as a unary body states them.
    assert!(raw.get("usage").is_some_and(Value::is_object));
    assert_eq!(raw["object"], "chat.completion");
    assert!(
        raw.pointer("/choices/0/message")
            .is_some_and(Value::is_object)
    );
    let records = recorded_completed_records(scenario, true);
    assert_eq!(records.len(), 1);
    assert_eq!(fingerprints(&responses), fingerprints(&records));
    assert_eq!(raw["model"], records[0]["model"]);
}

// ---------------------------------------------------------------------------
// 3: multi-turn tool runs
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
            let agent = rig::AgentBuilder::new(client.completion(MODEL))
                .preamble(TOOLS_PREAMBLE)
                .options(
                    rig::completion::GenerationOptions::default()
                        .reasoning(rig::completion::Reasoning::Off),
                )
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
    let records = recorded_completed_records(scenario, false);
    assert_eq!(records.len(), 2, "premise: two attempts were made");
    assert_distinct_fingerprints(&records, scenario);
    // Each payload is its own attempt's, in the interactions' order.
    assert_eq!(fingerprints(&raws), fingerprints(&records));
    // The first is the tool turn: it carries the wire's `tool_calls`.
    assert_eq!(
        recorded_add_call_turns(scenario),
        [true, false],
        "premise: an add call then a text turn"
    );
    assert!(
        raws[0]
            .pointer("/choices/0/message/tool_calls/0/function/name")
            .is_some_and(|name| name == Adder::NAME),
        "the first payload carries the wire's tool_calls: {:?}",
        raws[0]
    );
    assert_eq!(
        raws[0].pointer("/choices/0/message/tool_calls/0/function/arguments"),
        records[0].pointer("/choices/0/message/tool_calls/0/function/arguments"),
        "raw carries the wire's tool-call arguments untouched"
    );
    assert!(
        raws[1]
            .pointer("/choices/0/message/tool_calls")
            .is_none_or(|calls| calls.as_array().is_some_and(Vec::is_empty)),
        "the second payload is the text turn: {:?}",
        raws[1]
    );
    // The hooks saw the same two payloads in the same order; the first
    // CompletionResponse is the tool-call turn, the second the text turn.
    assert_eq!(probe.completion_responses(), raws);
    assert_eq!(probe.response_tool_call_flags(), [true, false]);
    assert_eq!(probe.model_turns(), raws);
}
