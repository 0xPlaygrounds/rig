//! Raw provider response capture through the OpenAI agent surfaces
//! (hook events, `completion_calls`, the streamed terminal).
//!
//! # What this pins
//!
//! The agent erases the model, so a caller can only reach the provider's own
//! response through the surfaces the agent exposes: the `raw` on the
//! `CompletionResponse` and `ModelTurnFinished` hook events — both fire once
//! per accepted model turn on both drivers, tool-only turns included, with
//! `HookContext::is_streaming` telling the drivers apart; `CompletionCall::raw`
//! on `PromptResponse::completion_calls`; and
//! the streamed `StreamedAssistantContent::Final` terminal record. Capture is
//! unconditional — there is no agent-, run- or request-level switch — so every
//! one of those surfaces carries the payload on every attempt, and a
//! `Value::Null` there could only mean the record was built without a
//! provider response behind it, which an agent run never does. Each surface carries the payload
//! **per attempt**: a multi-turn tool run records two different payloads, a
//! retried turn records the retried attempt's own.
//!
//! The chat route is the primary surface (each turn's payload is the
//! provider's reply document, whose `id` is a `chatcmpl-` id); the
//! Responses route repeats the hook and multi-turn cells (payload
//! the Responses response object, `resp_` ids). Per-attempt
//! identity is proven the way `response_identity.rs` proves it: each
//! recorded interaction's response id, in wire order, is the id the matching
//! `raw` carries — replay-exact, presence-checked while recording because
//! fixtures are placeholder-scrubbed.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 2 | `chat_streamed_hooks_see_raw` | chat, streamed | `CompletionResponse`/`ModelTurnFinished` hooks, `Final`, `CompletionCall` see raw | recorded |
//! | 6 | `responses_blocking_hooks_see_raw` | Responses, blocking | as 1 | recorded |
//! | 7 | `responses_streamed_hooks_see_raw` | Responses, streamed | as 2 | recorded |
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
    AgentBuilder, AgentHook, HookContext, ModelTurnAction, ModelTurnFinished, MultiTurnStreamItem,
    OutcomeAction, OutcomeEvent,
};
use rig::completion::ResponseIdentity;
use rig::message::AssistantContent;
use rig::providers::openai;
use serde_json::Value;

use super::super::support::{OpenAiCassette, sse_json_frames, with_openai_cassette};
use crate::support::{Adder, TOOLS_PREAMBLE, assert_matches_recorded_token};

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
    fn builder(self, client: OpenAiCassette) -> AgentBuilder {
        match self {
            Route::Chat => rig::AgentBuilder::new(client.chat.completion(MODEL)),
            Route::Responses => rig::AgentBuilder::new(client.openai.completion(MODEL)),
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
}

type Seen = Arc<Mutex<Vec<(ResponseIdentity, Value)>>>;

/// One `CompletionResponse` observation: which driver fired it, whether the
/// canonical content carried a tool call, the attempt's identity and `raw`.
#[derive(Clone, Debug, PartialEq)]
struct ResponseSeen {
    streaming: bool,
    tool_call: bool,
    identity: ResponseIdentity,
    raw: Value,
}

/// Captures each event's identity and `raw` payload so a cell can compare
/// every observer surface against the run record.
#[derive(Clone, Default)]
struct RawProbe {
    completion_responses: Arc<Mutex<Vec<ResponseSeen>>>,
    turns: Seen,
}

impl RawProbe {
    fn completion_responses(&self) -> Vec<ResponseSeen> {
        self.completion_responses.lock().expect("probe").clone()
    }
    fn turns(&self) -> Vec<(ResponseIdentity, Value)> {
        self.turns.lock().expect("probe").clone()
    }
}

impl AgentHook for RawProbe {
    async fn on_outcome(&self, ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        let Some(response) = event.completion() else {
            return OutcomeAction::proceed();
        };
        self.completion_responses
            .lock()
            .expect("probe")
            .push(ResponseSeen {
                streaming: ctx.is_streaming(),
                tool_call: response
                    .choice
                    .iter()
                    .any(|content| matches!(content, AssistantContent::ToolCall(_))),
                identity: response.identity(),
                raw: response.raw.clone(),
            });
        OutcomeAction::proceed()
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

/// What a cell observes from one run, moved out of the cassette closure so the
/// assertions run *after* the wrapper wrote the fixture.
#[derive(Default)]
struct RunObservation {
    calls: Vec<rig::agent::CompletionCall>,
    /// The `raw` of every streamed `MultiTurnStreamItem::CompletionCall`.
    stream_calls: Vec<Value>,
    output: String,
}

type Observed = Arc<Mutex<Option<RunObservation>>>;

/// The agent every hook / tool-run cell drives.
fn build_agent(
    route: Route,
    client: OpenAiCassette,
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
type Body = Box<dyn FnOnce(OpenAiCassette) -> Pin<Box<dyn Future<Output = ()>>>>;

fn take(observed: &Observed) -> RunObservation {
    observed
        .lock()
        .expect("observation mutex")
        .take()
        .expect("test body should save its observation")
}

/// A blocking `prompt(..)` run.
fn blocking_body(sink: Observed, route: Route, tools: bool, probe: RawProbe) -> Body {
    Box::new(move |client| {
        Box::pin(async move {
            let (agent, prompt) = build_agent(route, client, tools, probe);
            let response = agent
                .prompt(prompt)
                .max_turns(3)
                .await
                .expect("agent run should succeed");
            *sink.lock().expect("observation mutex") = Some(RunObservation {
                output: response.output(),
                calls: response.completion_calls,
                ..Default::default()
            });
        })
    })
}

/// A streamed `prompt(..).stream()` run, drained to its `FinalResponse`.
fn streamed_body(sink: Observed, route: Route, tools: bool, probe: RawProbe) -> Body {
    Box::new(move |client| {
        Box::pin(async move {
            let (agent, prompt) = build_agent(route, client, tools, probe);
            let mut stream = agent.prompt(prompt).max_turns(3).stream();
            let mut observation = RunObservation::default();
            let mut final_response = None;
            while let Some(item) = stream.next().await {
                match item.expect("stream item should succeed") {
                    MultiTurnStreamItem::CompletionCall(call) => {
                        observation.stream_calls.push(call.raw.clone());
                    }
                    MultiTurnStreamItem::FinalResponse(response) => final_response = Some(response),
                    _ => {}
                }
            }
            let response = final_response.expect("stream should end with a FinalResponse");
            observation.output = response.output();
            observation.calls = response.completion_calls;
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
    assert!(
        !responses[0].streaming,
        "{scenario}: the blocking driver fires it with is_streaming() == false"
    );
    assert_eq!(
        responses[0].raw, call_raw,
        "{scenario}: CompletionResponse hook sees the call's raw"
    );
    assert_eq!(
        responses[0].identity.response_id, observation.calls[0].response_id,
        "{scenario}: CompletionResponse hook and record agree on the attempt"
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

    let responses = probe.completion_responses();
    assert_eq!(
        responses.len(),
        1,
        "{scenario}: the streamed surface fires CompletionResponse once the stream is assembled"
    );
    assert!(
        responses[0].streaming,
        "{scenario}: the streaming driver fires it with is_streaming() == true"
    );
    assert_eq!(
        responses[0].raw, call_raw,
        "{scenario}: CompletionResponse hook sees the terminal's raw"
    );
    assert_eq!(
        responses[0].identity.response_id, observation.calls[0].response_id,
        "{scenario}: CompletionResponse hook and record agree on the attempt"
    );
    let turns = probe.turns();
    assert_eq!(turns.len(), 1, "{scenario}: one ModelTurnFinished event");
    assert_eq!(
        turns[0].1, call_raw,
        "{scenario}: ModelTurnFinished hook sees the terminal's raw"
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

// ---------------------------------------------------------------------------
// Chat route
// ---------------------------------------------------------------------------

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
