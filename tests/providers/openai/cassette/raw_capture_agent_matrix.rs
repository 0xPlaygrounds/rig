//! Raw provider response capture through the OpenAI agent surfaces
//! (hook events, `completion_calls`, the streamed terminal).
//!
//! # What this pins
//!
//! The agent erases the model, so a caller can only reach the provider's own
//! response through the surfaces the agent exposes: the `raw` on the
//! `CompletionResponse`, `StreamResponseFinish`, and `ModelTurnFinished` hook
//! events; `CompletionCall::raw` on `PromptResponse::completion_calls`; and
//! the streamed `StreamedAssistantContent::Final` terminal record. Capture is
//! unconditional — there is no agent-, run- or request-level switch — so every
//! one of those surfaces carries the payload on every attempt, and a
//! `Value::Null` there could only mean the record was built without a
//! provider response behind it, which an agent run never does. Each surface carries the payload
//! **per attempt**: a multi-turn tool run records two different payloads, a
//! retried turn records the retried attempt's own.
//!
//! The chat route is the primary surface (each turn's payload is an
//! `openai::CompletionResponse` whose `id` is a `chatcmpl-` id); the
//! Responses route repeats the hook and multi-turn cells (payload
//! `openai::responses_api::CompletionResponse`, `resp_` ids). Per-attempt
//! identity is proven the way `response_identity.rs` proves it: each
//! recorded interaction's response id, in wire order, is the id the matching
//! `raw` carries — replay-exact, presence-checked while recording because
//! fixtures are placeholder-scrubbed.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `chat_blocking_hooks_see_raw` | chat, blocking | `CompletionResponse`/`ModelTurnFinished` hooks see `raw` = call's raw | recorded |
//! | 2 | `chat_streamed_hooks_see_raw` | chat, streamed | `StreamResponseFinish`/`ModelTurnFinished` hooks, `Final`, `CompletionCall` see raw | recorded |
//! | 3 | `chat_blocking_tool_run_records_distinct_raw` | chat, blocking, tool run | two `completion_calls`, two payloads, ids in fixture order | recorded |
//! | 4 | `chat_streamed_tool_run_records_distinct_raw` | chat, streamed, tool run | as 3, `Final` carries the final turn's raw | recorded |
//! | 5 | `chat_retried_turn_records_retried_attempt_raw` | chat, retry hook | second record carries the retried attempt's raw | recorded |
//! | 6 | `responses_blocking_hooks_see_raw` | Responses, blocking | as 1 | recorded |
//! | 7 | `responses_streamed_hooks_see_raw` | Responses, streamed | as 2 | recorded |
//! | 8 | `responses_blocking_tool_run_records_distinct_raw` | Responses, blocking, tool run | as 3 | recorded |
//! | 9 | `responses_streamed_tool_run_records_distinct_raw` | Responses, streamed, tool run | as 4 | recorded |
//!
//! Every cell is recorded; none is unit-only. Premise, re-derived from each
//! cell's fixture after the wrapper returns: the fixture holds exactly as many
//! interactions as the run made calls, each a completed turn with a response
//! id — a run whose attempts did not each get their own provider response
//! could not prove per-attempt capture.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use futures::StreamExt as _;
use rig::agent::{
    AgentBuilder, AgentHook, CompletionResponseEvent, HookContext, ModelTurnAction,
    ModelTurnFinished, MultiTurnStreamItem, ObservationAction, StreamResponseFinish,
};
use rig::completion::{Prompt, ResponseIdentity};
use rig::message::AssistantContent;
use rig::prelude::*;
use rig::providers::openai;
use rig::streaming::{StreamedAssistantContent, StreamingPrompt};
use serde_json::Value;

use super::super::support::{assert_matches_recorded_token, sse_json_frames, with_openai_cassette};
use crate::support::{Adder, TOOLS_PREAMBLE};

const PROVIDER: &str = "openai";
const MODEL: &str = openai::GPT_4_1_NANO;
const TEXT_PROMPT: &str = "Reply with exactly the single word: pong";
const TOOL_PROMPT: &str = "What is 2 + 3? Use the tool, then state the result.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Route {
    Chat,
    Responses,
}

impl Route {
    fn builder(self, client: openai::Client) -> AgentBuilder {
        match self {
            Route::Chat => client.completions_api().agent(MODEL),
            Route::Responses => client.agent(MODEL),
        }
    }

    fn id_prefix(self) -> &'static str {
        match self {
            Route::Chat => "chatcmpl",
            Route::Responses => "resp_",
        }
    }

    /// The response id of one recorded blocking body.
    fn blocking_id(self, body: &str) -> String {
        let body: Value = serde_json::from_str(body).expect("recorded body should be JSON");
        body["id"]
            .as_str()
            .unwrap_or_else(|| panic!("recorded {self:?} body must carry an id"))
            .to_owned()
    }

    /// The response id of one recorded stream: the last chat chunk's `id`, or
    /// the `response.completed` event's `response.id`.
    fn streamed_id(self, body: &str) -> String {
        let frames = sse_json_frames(body);
        let last = frames
            .last()
            .unwrap_or_else(|| panic!("recorded {self:?} stream must carry frames"));
        let id = match self {
            Route::Chat => {
                assert!(
                    last["usage"].is_object(),
                    "recorded chat stream must end on the usage-bearing terminal chunk"
                );
                last["id"].as_str()
            }
            Route::Responses => {
                assert_eq!(last["type"], "response.completed");
                last["response"]["id"].as_str()
            }
        };
        id.unwrap_or_else(|| panic!("recorded {self:?} stream terminal must carry an id"))
            .to_owned()
    }

    /// Whether one payload (a `raw` value) is a tool-call turn's.
    ///
    /// A blocking payload is the wire body, so the tool call itself is
    /// visible. A streamed payload is the terminal record: the chat terminal
    /// reports `finish_reason: tool_calls`; the Responses terminal names the
    /// assistant message (`message_id`) only when the turn produced one, and
    /// a tool-only turn produces none.
    fn raw_is_tool_turn(self, raw: &Value, streamed: bool) -> bool {
        match (self, streamed) {
            (Route::Chat, false) => raw["choices"][0]["message"]["tool_calls"]
                .as_array()
                .is_some_and(|calls| !calls.is_empty()),
            (Route::Chat, true) => raw["finish_reason"] == "tool_calls",
            (Route::Responses, false) => raw["output"]
                .as_array()
                .is_some_and(|items| items.iter().any(|item| item["type"] == "function_call")),
            (Route::Responses, true) => raw.get("message_id").is_none_or(Value::is_null),
        }
    }
}

type Seen = Arc<Mutex<Vec<(ResponseIdentity, Value)>>>;

/// Captures each event's identity and `raw` payload so a cell can compare
/// every observer surface against the run record.
#[derive(Clone, Default)]
struct RawProbe {
    completion_responses: Seen,
    stream_finishes: Seen,
    turns: Seen,
}

impl RawProbe {
    fn completion_responses(&self) -> Vec<(ResponseIdentity, Value)> {
        self.completion_responses.lock().expect("probe").clone()
    }
    fn stream_finishes(&self) -> Vec<(ResponseIdentity, Value)> {
        self.stream_finishes.lock().expect("probe").clone()
    }
    fn turns(&self) -> Vec<(ResponseIdentity, Value)> {
        self.turns.lock().expect("probe").clone()
    }
}

impl AgentHook for RawProbe {
    async fn on_completion_response(
        &self,
        _ctx: &HookContext,
        event: CompletionResponseEvent<'_>,
    ) -> ObservationAction {
        self.completion_responses
            .lock()
            .expect("probe")
            .push((event.identity.clone(), event.raw.clone()));
        ObservationAction::continue_run()
    }

    async fn on_stream_response_finish(
        &self,
        _ctx: &HookContext,
        event: StreamResponseFinish<'_>,
    ) -> ObservationAction {
        self.stream_finishes
            .lock()
            .expect("probe")
            .push((event.identity.clone(), event.raw.clone()));
        ObservationAction::continue_run()
    }

    async fn on_model_turn_finished(
        &self,
        _ctx: &HookContext,
        event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        self.turns
            .lock()
            .expect("probe")
            .push((event.identity.clone(), event.raw.clone()));
        ModelTurnAction::continue_run()
    }
}

/// Rejects the first turn that carries the `RETRY:` marker once, exactly as
/// `response_retry.rs` does, so the run makes two attempts at the same turn.
#[derive(Clone, Default)]
struct RetryAttempts(usize);

struct RetryOnceOnMarker;

impl AgentHook for RetryOnceOnMarker {
    async fn on_model_turn_finished(
        &self,
        ctx: &HookContext,
        event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        let rejected = event.content.iter().any(|content| {
            matches!(content, AssistantContent::Text(text) if text.text.contains("RETRY:"))
        });
        if !rejected {
            return ModelTurnAction::continue_run();
        }
        let attempt = ctx.scratchpad().update(|attempts: &mut RetryAttempts| {
            attempts.0 += 1;
            attempts.0
        });
        if attempt == 1 {
            ModelTurnAction::retry_with_feedback(
                "Replace the rejected response. Reply exactly `ACCEPTED`.",
            )
        } else {
            ModelTurnAction::stop("response retry limit exceeded")
        }
    }
}

/// What a cell observes from one run, moved out of the cassette closure so the
/// assertions run *after* the wrapper wrote the fixture.
#[derive(Default)]
struct RunObservation {
    calls: Vec<rig::agent::CompletionCall>,
    /// The `raw` of every streamed `StreamedAssistantContent::Final`, in order.
    finals: Vec<Value>,
    /// The `raw` of every streamed `MultiTurnStreamItem::CompletionCall`.
    stream_calls: Vec<Value>,
    output: String,
}

type Observed = Arc<Mutex<Option<RunObservation>>>;

/// The agent every hook / tool-run cell drives.
fn build_agent(
    route: Route,
    client: openai::Client,
    tools: bool,
    probe: RawProbe,
) -> (rig::agent::Agent, &'static str) {
    let builder = route.builder(client).temperature(0.0).add_hook(probe);
    if tools {
        (
            builder.preamble(TOOLS_PREAMBLE).tool(Adder).build(),
            TOOL_PROMPT,
        )
    } else {
        (builder.build(), TEXT_PROMPT)
    }
}

/// A cassette test body: boxed so the cell can build it in a helper while the
/// wrapper call — and its string-literal scenario, which the cassette safety
/// scan reads — stays in the test itself.
type Body = Box<dyn FnOnce(openai::Client) -> Pin<Box<dyn Future<Output = ()>>>>;

fn take(observed: &Observed) -> RunObservation {
    observed
        .lock()
        .expect("observation mutex")
        .take()
        .expect("test body should save its observation")
}

/// A blocking `prompt(..).extended_details()` run.
fn blocking_body(sink: Observed, route: Route, tools: bool, probe: RawProbe) -> Body {
    Box::new(move |client| {
        Box::pin(async move {
            let (agent, prompt) = build_agent(route, client, tools, probe);
            let response = agent
                .prompt(prompt)
                .max_turns(3)
                .extended_details()
                .await
                .expect("agent run should succeed");
            *sink.lock().expect("observation mutex") = Some(RunObservation {
                calls: response.completion_calls,
                output: response.output,
                ..Default::default()
            });
        })
    })
}

/// A streamed `stream_prompt(..)` run, drained to its `FinalResponse`.
fn streamed_body(sink: Observed, route: Route, tools: bool, probe: RawProbe) -> Body {
    Box::new(move |client| {
        Box::pin(async move {
            let (agent, prompt) = build_agent(route, client, tools, probe);
            let mut stream = agent.stream_prompt(prompt).max_turns(3).await;
            let mut observation = RunObservation::default();
            let mut final_response = None;
            while let Some(item) = stream.next().await {
                match item.expect("stream item should succeed") {
                    MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Final(
                        terminal,
                    )) => observation.finals.push(terminal.raw.clone()),
                    MultiTurnStreamItem::CompletionCall(call) => {
                        observation.stream_calls.push(call.raw.clone());
                    }
                    MultiTurnStreamItem::FinalResponse(response) => final_response = Some(response),
                    _ => {}
                }
            }
            let response = final_response.expect("stream should end with a FinalResponse");
            observation.calls = response.completion_calls;
            observation.output = response.output;
            *sink.lock().expect("observation mutex") = Some(observation);
        })
    })
}

/// The recorded response ids of a scenario, one per interaction, in wire
/// order — the per-attempt premise.
fn recorded_ids(scenario: &str, route: Route, streamed: bool) -> Vec<String> {
    crate::cassettes::recorded_interaction_bodies(PROVIDER, scenario)
        .iter()
        .map(|(_, body)| {
            if streamed {
                route.streamed_id(body)
            } else {
                route.blocking_id(body)
            }
        })
        .collect()
}

/// The response id a payload carries: a blocking payload is the wire body
/// (`id`), a streamed payload is the route's terminal record (`response_id`).
fn raw_id(raw: &Value) -> Option<&str> {
    raw["id"].as_str().or_else(|| raw["response_id"].as_str())
}

/// The assistant text a *blocking* chat payload carries. The captured value is
/// the response as rig's wire type parsed it, and that type models assistant
/// content as parts — so a wire string comes back as one text part.
fn chat_raw_text(raw: &Value) -> String {
    match &raw["choices"][0]["message"]["content"] {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| part["text"].as_str())
            .collect(),
        other => panic!("unexpected chat raw content shape {other}"),
    }
}

/// Every recorded call carries a payload whose id is the matching fixture
/// interaction's, in order; the payloads are pairwise distinct.
fn assert_calls_carry_recorded_raw(
    scenario: &str,
    route: Route,
    calls: &[rig::agent::CompletionCall],
    recorded: &[String],
) {
    assert_eq!(
        calls.len(),
        recorded.len(),
        "{scenario}: one recorded interaction per completion call"
    );
    for (index, (call, recorded_id)) in calls.iter().zip(recorded).enumerate() {
        let raw = &call.raw;
        assert!(
            !raw.is_null(),
            "{scenario}: completion_calls[{index}] must always carry raw"
        );
        assert!(
            raw_id(raw).is_some_and(|id| id.starts_with(route.id_prefix())),
            "{scenario}: completion_calls[{index}] raw id should be a {} id, got {:?}",
            route.id_prefix(),
            raw_id(raw)
        );
        assert_matches_recorded_token(
            raw_id(raw),
            Some(recorded_id),
            &format!("{scenario}: completion_calls[{index}] raw id vs fixture interaction {index}"),
        );
        assert_matches_recorded_token(
            call.response_id.as_deref(),
            Some(recorded_id),
            &format!("{scenario}: completion_calls[{index}] response_id vs fixture"),
        );
    }
    for (left, right) in calls.iter().zip(calls.iter().skip(1)) {
        assert_ne!(
            left.raw, right.raw,
            "{scenario}: consecutive calls carry different payloads"
        );
    }
}

// ---------------------------------------------------------------------------
// Hook surfaces (cells 1–2, 6–7)
// ---------------------------------------------------------------------------

fn assert_blocking_hooks_see_raw(
    scenario: &str,
    route: Route,
    probe: &RawProbe,
    observation: RunObservation,
) {
    let recorded = recorded_ids(scenario, route, false);
    assert_eq!(recorded.len(), 1, "{scenario}: one text turn");
    assert_calls_carry_recorded_raw(scenario, route, &observation.calls, &recorded);
    let call_raw = observation.calls[0].raw.clone();

    let responses = probe.completion_responses();
    assert_eq!(
        responses.len(),
        1,
        "{scenario}: one CompletionResponse event"
    );
    assert_eq!(
        responses[0].1, call_raw,
        "{scenario}: CompletionResponse hook sees the call's raw"
    );
    let turns = probe.turns();
    assert_eq!(turns.len(), 1, "{scenario}: one ModelTurnFinished event");
    assert_eq!(
        turns[0].1, call_raw,
        "{scenario}: ModelTurnFinished hook sees the call's raw"
    );
    assert_eq!(
        turns[0].0.response_id, observation.calls[0].response_id,
        "{scenario}: hook and record agree on the attempt"
    );
    assert!(
        probe.stream_finishes().is_empty(),
        "{scenario}: the blocking surface fires no StreamResponseFinish"
    );
}

fn assert_streamed_hooks_see_raw(
    scenario: &str,
    route: Route,
    probe: &RawProbe,
    observation: RunObservation,
) {
    let recorded = recorded_ids(scenario, route, true);
    assert_eq!(recorded.len(), 1, "{scenario}: one text turn");
    assert_calls_carry_recorded_raw(scenario, route, &observation.calls, &recorded);
    let call_raw = observation.calls[0].raw.clone();

    let finishes = probe.stream_finishes();
    assert_eq!(
        finishes.len(),
        1,
        "{scenario}: one StreamResponseFinish event"
    );
    assert_eq!(
        finishes[0].1, call_raw,
        "{scenario}: StreamResponseFinish hook sees the terminal's raw"
    );
    let turns = probe.turns();
    assert_eq!(turns.len(), 1, "{scenario}: one ModelTurnFinished event");
    assert_eq!(
        turns[0].1, call_raw,
        "{scenario}: ModelTurnFinished hook sees the terminal's raw"
    );
    assert!(
        probe.completion_responses().is_empty(),
        "{scenario}: the streamed surface fires no CompletionResponse"
    );
    assert_eq!(
        observation.finals,
        vec![call_raw.clone()],
        "{scenario}: the streamed Final carries the turn's raw"
    );
    assert_eq!(
        observation.stream_calls,
        vec![call_raw],
        "{scenario}: the streamed CompletionCall item carries the turn's raw"
    );
}

// ---------------------------------------------------------------------------
// Multi-turn tool runs (cells 3–4, 8–9)
// ---------------------------------------------------------------------------

fn assert_tool_run_records_distinct_raw(
    scenario: &str,
    route: Route,
    streamed: bool,
    probe: &RawProbe,
    observation: RunObservation,
) {
    let recorded = recorded_ids(scenario, route, streamed);
    assert!(
        recorded.len() >= 2,
        "{scenario}: a tool run makes at least two calls, got {}",
        recorded.len()
    );
    assert_calls_carry_recorded_raw(scenario, route, &observation.calls, &recorded);
    let raws: Vec<Value> = observation
        .calls
        .iter()
        .map(|call| call.raw.clone())
        .collect();
    // The payloads differ in *content*, not just id: the first turn called the
    // tool, the last answered in text.
    assert!(
        route.raw_is_tool_turn(&raws[0], streamed),
        "{scenario}: the first attempt's raw is the tool-call turn's"
    );
    let last = raws.last().expect("at least two");
    assert!(
        !route.raw_is_tool_turn(last, streamed),
        "{scenario}: the final attempt's raw is the text answer's"
    );
    assert!(
        observation.output.contains('5'),
        "{scenario}: the run answered 2 + 3, got {:?}",
        observation.output
    );
    // Every ModelTurnFinished observation sees its own attempt's payload.
    let turns = probe.turns();
    assert_eq!(
        turns.len(),
        raws.len(),
        "{scenario}: one turn event per call"
    );
    for (index, ((_, turn_raw), raw)) in turns.iter().zip(&raws).enumerate() {
        assert_eq!(
            turn_raw, raw,
            "{scenario}: turn {index} hook sees that attempt's raw"
        );
    }
    if streamed {
        assert_eq!(
            observation.stream_calls, raws,
            "{scenario}: streamed CompletionCall items carry each attempt's raw in order"
        );
        // Only turns that streamed text yield a Final; the last one is the
        // final answer's, and it carries the final turn's payload.
        assert_eq!(
            observation.finals.last(),
            Some(last),
            "{scenario}: the streamed Final carries the final turn's raw"
        );
        let finishes = probe.stream_finishes();
        assert_eq!(
            finishes.last().map(|(_, raw)| raw),
            Some(last),
            "{scenario}: the last StreamResponseFinish sees the final turn's raw"
        );
    } else {
        let responses = probe.completion_responses();
        assert_eq!(
            responses.len(),
            raws.len(),
            "{scenario}: one CompletionResponse per call"
        );
        for (index, ((_, seen), raw)) in responses.iter().zip(&raws).enumerate() {
            assert_eq!(
                seen, raw,
                "{scenario}: CompletionResponse {index} sees that attempt's raw"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Chat route
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_blocking_hooks_see_raw() {
    const SCENARIO: &str = "raw_capture_agent_matrix/chat_blocking_hooks_see_raw";
    let probe = RawProbe::default();
    let observed = Observed::default();
    with_openai_cassette(
        "raw_capture_agent_matrix/chat_blocking_hooks_see_raw",
        blocking_body(observed.clone(), Route::Chat, false, probe.clone()),
    )
    .await;
    assert_blocking_hooks_see_raw(SCENARIO, Route::Chat, &probe, take(&observed));
}

#[tokio::test]
async fn chat_streamed_hooks_see_raw() {
    const SCENARIO: &str = "raw_capture_agent_matrix/chat_streamed_hooks_see_raw";
    let probe = RawProbe::default();
    let observed = Observed::default();
    with_openai_cassette(
        "raw_capture_agent_matrix/chat_streamed_hooks_see_raw",
        streamed_body(observed.clone(), Route::Chat, false, probe.clone()),
    )
    .await;
    assert_streamed_hooks_see_raw(SCENARIO, Route::Chat, &probe, take(&observed));
}

#[tokio::test]
async fn chat_blocking_tool_run_records_distinct_raw() {
    const SCENARIO: &str = "raw_capture_agent_matrix/chat_blocking_tool_run_records_distinct_raw";
    let probe = RawProbe::default();
    let observed = Observed::default();
    with_openai_cassette(
        "raw_capture_agent_matrix/chat_blocking_tool_run_records_distinct_raw",
        blocking_body(observed.clone(), Route::Chat, true, probe.clone()),
    )
    .await;
    assert_tool_run_records_distinct_raw(SCENARIO, Route::Chat, false, &probe, take(&observed));
}

#[tokio::test]
async fn chat_streamed_tool_run_records_distinct_raw() {
    const SCENARIO: &str = "raw_capture_agent_matrix/chat_streamed_tool_run_records_distinct_raw";
    let probe = RawProbe::default();
    let observed = Observed::default();
    with_openai_cassette(
        "raw_capture_agent_matrix/chat_streamed_tool_run_records_distinct_raw",
        streamed_body(observed.clone(), Route::Chat, true, probe.clone()),
    )
    .await;
    assert_tool_run_records_distinct_raw(SCENARIO, Route::Chat, true, &probe, take(&observed));
}

/// A retried turn: the first attempt is rejected by the hook and the run
/// makes a second attempt at the same turn. `completion_calls[1]` and the
/// second `ModelTurnFinished` carry the *retried* attempt's payload — the
/// second fixture interaction's — never the rejected first one's.
#[tokio::test]
async fn chat_retried_turn_records_retried_attempt_raw() {
    const SCENARIO: &str = "raw_capture_agent_matrix/chat_retried_turn_records_retried_attempt_raw";
    let probe = RawProbe::default();
    let observed = Observed::default();
    let sink = observed.clone();
    let hook_probe = probe.clone();
    with_openai_cassette(
        "raw_capture_agent_matrix/chat_retried_turn_records_retried_attempt_raw",
        |client| async move {
            let response = client
                .completions_api()
                .agent(MODEL)
                .preamble(
                    "Follow this protocol exactly. For the initial request, reply exactly \
                 `RETRY: incomplete draft`. If the latest user message asks you to \
                 replace the rejected response, reply exactly `ACCEPTED`.",
                )
                .temperature(0.0)
                .build()
                .runner("Begin the retry-hook demonstration.")
                .max_turns(2)
                .add_hook(hook_probe)
                .add_hook(RetryOnceOnMarker)
                .run()
                .await
                .expect("the feedback retry should recover");
            *sink.lock().expect("observation mutex") = Some(RunObservation {
                calls: response.completion_calls,
                output: response.output,
                ..Default::default()
            });
        },
    )
    .await;
    let observation = take(&observed);

    // Premise: two interactions — the rejected draft, then the retried
    // attempt — each a distinct provider response with the protocol's text.
    let bodies = crate::cassettes::recorded_interaction_bodies(PROVIDER, SCENARIO);
    assert_eq!(
        bodies.len(),
        2,
        "{SCENARIO}: rejected attempt + retried attempt"
    );
    let content = |body: &str| -> String {
        let body: Value = serde_json::from_str(body).expect("recorded body should be JSON");
        body["choices"][0]["message"]["content"]
            .as_str()
            .expect("recorded chat body carries text")
            .to_owned()
    };
    assert!(
        content(&bodies[0].1).contains("RETRY:"),
        "{SCENARIO}: the first recorded attempt is the rejected draft"
    );
    assert_eq!(
        content(&bodies[1].1).trim(),
        "ACCEPTED",
        "{SCENARIO}: the second recorded attempt is the retry"
    );
    assert_eq!(observation.output.trim(), "ACCEPTED");

    let recorded = recorded_ids(SCENARIO, Route::Chat, false);
    assert_calls_carry_recorded_raw(SCENARIO, Route::Chat, &observation.calls, &recorded);
    let retried_raw = &observation.calls[1].raw;
    assert_eq!(
        chat_raw_text(retried_raw).trim(),
        "ACCEPTED",
        "{SCENARIO}: the second record carries the retried attempt's own payload"
    );
    assert!(
        chat_raw_text(&observation.calls[0].raw).contains("RETRY:"),
        "{SCENARIO}: the first record carries the rejected attempt's own payload"
    );
    let turns = probe.turns();
    assert_eq!(
        turns.len(),
        2,
        "{SCENARIO}: the retry hook saw both attempts"
    );
    assert_eq!(
        &turns[1].1, retried_raw,
        "{SCENARIO}: the retried ModelTurnFinished sees the retried attempt's raw"
    );
    assert_ne!(
        turns[0].1, turns[1].1,
        "{SCENARIO}: attempts carry different payloads"
    );
}

// ---------------------------------------------------------------------------
// Responses route
// ---------------------------------------------------------------------------

#[tokio::test]
async fn responses_blocking_hooks_see_raw() {
    const SCENARIO: &str = "raw_capture_agent_matrix/responses_blocking_hooks_see_raw";
    let probe = RawProbe::default();
    let observed = Observed::default();
    with_openai_cassette(
        "raw_capture_agent_matrix/responses_blocking_hooks_see_raw",
        blocking_body(observed.clone(), Route::Responses, false, probe.clone()),
    )
    .await;
    assert_blocking_hooks_see_raw(SCENARIO, Route::Responses, &probe, take(&observed));
}

#[tokio::test]
async fn responses_streamed_hooks_see_raw() {
    const SCENARIO: &str = "raw_capture_agent_matrix/responses_streamed_hooks_see_raw";
    let probe = RawProbe::default();
    let observed = Observed::default();
    with_openai_cassette(
        "raw_capture_agent_matrix/responses_streamed_hooks_see_raw",
        streamed_body(observed.clone(), Route::Responses, false, probe.clone()),
    )
    .await;
    assert_streamed_hooks_see_raw(SCENARIO, Route::Responses, &probe, take(&observed));
}

#[tokio::test]
async fn responses_blocking_tool_run_records_distinct_raw() {
    const SCENARIO: &str =
        "raw_capture_agent_matrix/responses_blocking_tool_run_records_distinct_raw";
    let probe = RawProbe::default();
    let observed = Observed::default();
    with_openai_cassette(
        "raw_capture_agent_matrix/responses_blocking_tool_run_records_distinct_raw",
        blocking_body(observed.clone(), Route::Responses, true, probe.clone()),
    )
    .await;
    assert_tool_run_records_distinct_raw(
        SCENARIO,
        Route::Responses,
        false,
        &probe,
        take(&observed),
    );
}

#[tokio::test]
async fn responses_streamed_tool_run_records_distinct_raw() {
    const SCENARIO: &str =
        "raw_capture_agent_matrix/responses_streamed_tool_run_records_distinct_raw";
    let probe = RawProbe::default();
    let observed = Observed::default();
    with_openai_cassette(
        "raw_capture_agent_matrix/responses_streamed_tool_run_records_distinct_raw",
        streamed_body(observed.clone(), Route::Responses, true, probe.clone()),
    )
    .await;
    assert_tool_run_records_distinct_raw(SCENARIO, Route::Responses, true, &probe, take(&observed));
}
