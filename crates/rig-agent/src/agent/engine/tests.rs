use std::collections::HashMap;

use crate::agent::{
    CompletionCallAction, CompletionCallEvent, DispatchAction, DispatchEvent, HookStack,
    InvalidToolCallAction, InvalidToolCallContext, ModelTurnAction, ModelTurnFinished,
    ObservationAction, OutcomeAction, OutcomeEvent, ReasoningDelta, TextDelta, ToolCallDelta,
};

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::SeqCst},
};

use futures::StreamExt;
use serde::Deserialize;
use serde_json::json;
use tokio::sync::{Barrier, Notify};

use crate::agent::AgentBuilder;
use crate::agent::hook::{AgentHook, HookContext, RequestPatch, StepEventKind};
use crate::agent::run::OutputMode;
use crate::agent::streaming::MultiTurnStreamItem;
use crate::completion::{Message, PromptError, Usage};
use crate::streaming::{Item, StreamEvent};
use crate::test_utils::{
    MockAddTool, MockCompletionModel, MockOperationArgs, MockScript, MockStreamEvent,
    MockSubtractTool, MockToolError, MockTurn,
};
use crate::tool::{
    Tool, ToolContext, ToolExecutionError,
    server::{ToolServer, ToolServerHandle},
};
use rig_core::driver::{Exchange, Opening};
use rig_core::message::{
    AssistantContent, ToolCall as MessageToolCall, ToolChoice, ToolFunction, UserContent,
};
use rig_core::vector_store::{
    VectorSearchIdResult, VectorSearchRequest, VectorSearchResult, VectorStoreError,
    VectorStoreIndex, request::Filter,
};
use rig_core::wasm_compat::WasmCompatSend;

/// Records the kind of every hook event (and every tool-result payload) so a
/// run() and a stream() of the same scenario can be compared.
#[derive(Clone, Default)]
struct RecordingHook {
    events: Arc<Mutex<Vec<StepEventKind>>>,
    tool_results: Arc<Mutex<Vec<String>>>,
}

impl RecordingHook {
    /// Event kinds that should be identical across streaming and
    /// non-streaming (excludes the streaming-only delta events).
    fn shared_events(&self) -> Vec<StepEventKind> {
        self.events
            .lock()
            .expect("events lock")
            .iter()
            .copied()
            .filter(|kind| {
                matches!(
                    kind,
                    StepEventKind::CompletionCall
                        | StepEventKind::CompletionDispatch
                        | StepEventKind::ToolDispatch
                        | StepEventKind::InvalidToolCall
                )
            })
            .collect()
    }

    fn tool_results(&self) -> Vec<String> {
        self.tool_results.lock().expect("results lock").clone()
    }

    /// Count of a single event kind across the whole run, including the
    /// streaming-only delta events that `shared_events` excludes.
    fn count(&self, kind: StepEventKind) -> usize {
        self.events
            .lock()
            .expect("events lock")
            .iter()
            .filter(|recorded| **recorded == kind)
            .count()
    }
}

impl RecordingHook {
    fn record(&self, kind: StepEventKind) {
        self.events.lock().expect("events lock").push(kind);
    }
}

impl AgentHook for RecordingHook {
    async fn on_completion_call(
        &self,
        _: &HookContext,
        _: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        self.record(StepEventKind::CompletionCall);
        CompletionCallAction::continue_run()
    }
    async fn on_model_turn_finished(
        &self,
        _: &HookContext,
        _: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        self.record(StepEventKind::ModelTurnFinished);
        ModelTurnAction::continue_run()
    }
    async fn on_invalid_tool_call(
        &self,
        _: &HookContext,
        _: &InvalidToolCallContext,
    ) -> Option<InvalidToolCallAction> {
        self.record(StepEventKind::InvalidToolCall);
        None
    }
    /// Records `ToolDispatch` once at the tool-call dispatch boundary.
    async fn on_dispatch(&self, _: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        if event.tool_name().is_some() {
            self.record(StepEventKind::ToolDispatch);
        }
        DispatchAction::proceed()
    }
    /// Records `CompletionDispatch` once per completed model call (the slot
    /// the completion-response observation occupies) and `ToolDispatch` once
    /// per tool outcome, capturing the model-visible result text.
    async fn on_outcome(&self, _: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        if event.completion().is_some() {
            self.record(StepEventKind::CompletionDispatch);
        } else if let Some(result) = event.tool_result() {
            self.record(StepEventKind::ToolDispatch);
            self.tool_results
                .lock()
                .expect("results lock")
                .push(result.output().render());
        }
        OutcomeAction::proceed()
    }
    async fn on_text_delta(&self, _: &HookContext, _: TextDelta<'_>) -> ObservationAction {
        self.record(StepEventKind::TextDelta);
        ObservationAction::continue_run()
    }
    async fn on_reasoning_delta(
        &self,
        _: &HookContext,
        _: ReasoningDelta<'_>,
    ) -> ObservationAction {
        self.record(StepEventKind::ReasoningDelta);
        ObservationAction::continue_run()
    }
    async fn on_tool_call_delta(&self, _: &HookContext, _: ToolCallDelta<'_>) -> ObservationAction {
        self.record(StepEventKind::ToolCallDelta);
        ObservationAction::continue_run()
    }
}

#[derive(Clone, Debug, PartialEq)]
struct CanonicalResponseSnapshot {
    prompt: Message,
    content: Vec<AssistantContent>,
    usage: Usage,
}

#[derive(Clone, Default)]
struct FinishLifecycleHook {
    snapshots: Arc<Mutex<Vec<CanonicalResponseSnapshot>>>,
    model_turns: Arc<AtomicU32>,
    stop: Arc<AtomicBool>,
    /// The prompt of the model call in flight, captured at `on_completion_call`.
    pending_prompt: Arc<Mutex<Option<Message>>>,
}

impl AgentHook for FinishLifecycleHook {
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        *self.pending_prompt.lock().expect("pending prompt") = Some(event.prompt.clone());
        CompletionCallAction::continue_run()
    }

    async fn on_outcome(&self, _ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        let Some(response) = event.completion() else {
            return OutcomeAction::proceed();
        };
        let prompt = self
            .pending_prompt
            .lock()
            .expect("pending prompt")
            .clone()
            .expect("a completion outcome follows its completion call");
        self.snapshots
            .lock()
            .expect("finish snapshots")
            .push(CanonicalResponseSnapshot {
                prompt,
                content: response.choice.clone(),
                usage: response.usage,
            });
        if self.stop.load(SeqCst) {
            OutcomeAction::stop("stop at stream EOF")
        } else {
            OutcomeAction::proceed()
        }
    }

    async fn on_model_turn_finished(
        &self,
        _ctx: &HookContext,
        _event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        self.model_turns.fetch_add(1, SeqCst);
        ModelTurnAction::continue_run()
    }
}

fn canonical_usage() -> Usage {
    Usage::new()
        .input_tokens(11)
        .output_tokens(7)
        .total_tokens(18)
}

/// One hook observation per completed model call carries the attempt's
/// full identity triple, and the run's `completion_calls` record it
/// per-attempt (mock-model unit test; the live header-capture halves are
/// cassette-tested per provider).
#[tokio::test]
async fn completion_response_hook_and_calls_carry_identity_metadata() {
    type IdentityTriple = (Option<String>, Option<String>);

    #[derive(Clone, Default)]
    struct IdentityHook {
        seen: Arc<Mutex<Vec<IdentityTriple>>>,
    }

    impl AgentHook for IdentityHook {
        async fn on_outcome(&self, _ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
            if let Some(response) = event.completion() {
                self.seen.lock().expect("identity snapshots").push((
                    response.response_id().map(str::to_owned),
                    response.provider_request_id.clone(),
                ));
            }
            OutcomeAction::proceed()
        }
    }

    let hook = IdentityHook::default();
    let response = AgentBuilder::new(MockCompletionModel::from_turns([MockTurn::text("reply")
        .with_response_id("resp_1")
        .with_provider_request_id("req_1")]))
    .add_hook(hook.clone())
    .build()
    .prompt(Message::user("prompt"))
    .run()
    .await
    .expect("blocking response");

    assert_eq!(
        *hook.seen.lock().expect("identity snapshots"),
        [(Some("resp_1".to_string()), Some("req_1".to_string()),)]
    );
    let call = &response.completion_calls[0];
    assert_eq!(call.response_id.as_deref(), Some("resp_1"));
    assert_eq!(call.provider_request_id.as_deref(), Some("req_1"));
}

/// rig#2314 error matrix: a failed attempt's error carries its *own*
/// transport id through the surfaced `PromptError`, and a run that fails
/// after a successful call never cross-attributes — the error's id and
/// the earlier success's id stay distinct.
#[tokio::test]
async fn failed_attempt_error_carries_its_own_request_id() {
    let hook = TurnIdentityHook::default();
    let error = AgentBuilder::new(MockCompletionModel::from_turns([
        MockTurn::tool_call("tc1", "add", serde_json::json!({"x": 2, "y": 3}))
            .with_provider_request_id("req-success-1"),
        MockTurn::provider_response_error(
            http::StatusCode::TOO_MANY_REQUESTS,
            r#"{"error":"rate limited"}"#,
            "req-failed-2",
        ),
    ]))
    .tool(crate::test_utils::MockAddTool)
    .add_hook(hook.clone())
    .build()
    .prompt(Message::user("add 2 and 3"))
    .max_turns(4)
    .run()
    .await
    .expect_err("the second attempt fails");

    let PromptError::Report(ref report) = error else {
        panic!("expected the failing attempt's report, got {error:?}");
    };
    assert_eq!(
        report.request_id.as_deref(),
        Some("req-failed-2"),
        "the surfaced error reports the failing attempt's id: {error:?}"
    );
    assert_eq!(
        error.provider_request_id(),
        Some("req-failed-2"),
        "the run-surface accessor reads the wire report too"
    );
    assert_eq!(
        error
            .provider_response_status()
            .map(|status| status.as_u16()),
        report.http_status,
        "and so does the status accessor"
    );
    let turns = hook.turns.lock().expect("turn identities").clone();
    assert_eq!(turns.len(), 1, "only the successful call fired the event");
    assert_eq!(
        turns[0].provider_request_id.as_deref(),
        Some("req-success-1"),
        "the success keeps its own id — no cross-attribution"
    );
}

/// Hook capturing every `ModelTurnFinished` identity plus every
/// `CompletionResponse` identity — the cross-surface "every completed
/// call" observer #2265 requires.
#[derive(Clone, Default)]
struct TurnIdentityHook {
    turns: Arc<Mutex<Vec<rig_core::completion::ResponseIdentity>>>,
    responses: Arc<Mutex<Vec<rig_core::completion::ResponseIdentity>>>,
}

impl AgentHook for TurnIdentityHook {
    async fn on_model_turn_finished(
        &self,
        _ctx: &HookContext,
        event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        self.turns
            .lock()
            .expect("turn identities")
            .push(event.identity.clone());
        ModelTurnAction::continue_run()
    }

    async fn on_outcome(&self, _ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        if let Some(response) = event.completion() {
            self.responses
                .lock()
                .expect("response identities")
                .push(response.identity());
        }
        OutcomeAction::proceed()
    }
}

fn stream_final_with_id(response_id: &str) -> MockStreamEvent {
    MockStreamEvent::FinalResponse(rig_core::operation::Finish {
        usage: Usage::default(),
        response_id: Some(response_id.into()),
        ..rig_core::operation::Finish::default()
    })
}

/// Blocking surface: a tool-only turn and the following text turn each
/// fire `ModelTurnFinished` with their *own* attempt's identity.
#[tokio::test]
async fn model_turn_finished_identity_blocking_tool_only_and_text() {
    let hook = TurnIdentityHook::default();
    let response = AgentBuilder::new(MockCompletionModel::from_turns([
        MockTurn::tool_call("tc1", "add", json!({"x": 2, "y": 3}))
            .with_provider_request_id("req-turn-1")
            .with_response_id("resp-turn-1"),
        MockTurn::text("5")
            .with_provider_request_id("req-turn-2")
            .with_response_id("resp-turn-2"),
    ]))
    .tool(crate::test_utils::MockAddTool)
    .add_hook(hook.clone())
    .build()
    .prompt(Message::user("add 2 and 3"))
    .max_turns(3)
    .run()
    .await
    .expect("blocking tool run");

    let turns = hook.turns.lock().expect("turn identities").clone();
    let request_ids: Vec<_> = turns
        .iter()
        .map(|identity| identity.provider_request_id.clone())
        .collect();
    assert_eq!(
        request_ids,
        [
            Some("req-turn-1".to_string()),
            Some("req-turn-2".to_string())
        ],
        "each attempt reports its own transport id, in order"
    );
    // The run's completion_calls agree with the hook observations.
    let call_ids: Vec<_> = response
        .completion_calls
        .iter()
        .map(|call| call.provider_request_id.clone())
        .collect();
    assert_eq!(request_ids, call_ids);
}

/// Streamed surface: a tool-only turn and the following text turn each
/// fire `CompletionResponse` and `ModelTurnFinished` with full identity —
/// so an observer of either event records every completed call. The two
/// turns report distinct per-attempt ids.
#[tokio::test]
async fn model_turn_finished_identity_streamed_tool_only_and_text() {
    let hook = TurnIdentityHook::default();
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call_name_delta("tc1", "add"),
            MockStreamEvent::tool_call_arguments_delta("tc1", "{\"x\":2,\"y\":3}"),
            MockStreamEvent::tool_call("tc1", "add", json!({"x": 2, "y": 3})),
            MockStreamEvent::RequestId("req-stream-1".to_owned()),
            stream_final_with_id("resp-stream-1"),
        ],
        vec![
            MockStreamEvent::text("5"),
            MockStreamEvent::RequestId("req-stream-2".to_owned()),
            stream_final_with_id("resp-stream-2"),
        ],
    ]);
    let mut stream = AgentBuilder::new(model)
        .tool(crate::test_utils::MockAddTool)
        .add_hook(hook.clone())
        .build()
        .prompt(Message::user("add 2 and 3"))
        .max_turns(3)
        .stream();
    while let Some(item) = stream.next().await {
        item.expect("stream item");
    }

    let turns = hook.turns.lock().expect("turn identities").clone();
    let request_ids: Vec<_> = turns
        .iter()
        .map(|identity| identity.provider_request_id.clone())
        .collect();
    assert_eq!(
        request_ids,
        [
            Some("req-stream-1".to_string()),
            Some("req-stream-2".to_string())
        ],
        "streamed tool-only and text turns each carry their own identity"
    );
    let responses = hook.responses.lock().expect("responses").clone();
    let response_ids: Vec<_> = responses
        .iter()
        .map(|identity| identity.provider_request_id.clone())
        .collect();
    assert_eq!(
        response_ids, request_ids,
        "CompletionResponse fires for the tool-only turn too, each carrying its own identity"
    );
}

/// A retried turn's `ModelTurnFinished` carries the retried attempt's own
/// identity — the first attempt's ids never leak into the second event.
#[tokio::test]
async fn retried_turn_reports_the_retried_attempts_own_identity() {
    #[derive(Clone, Default)]
    struct RetryOnceCapturingIdentity {
        seen: Arc<Mutex<Vec<Option<String>>>>,
    }

    impl AgentHook for RetryOnceCapturingIdentity {
        async fn on_model_turn_finished(
            &self,
            _ctx: &HookContext,
            event: ModelTurnFinished<'_>,
        ) -> ModelTurnAction {
            let mut seen = self.seen.lock().expect("retry identities");
            seen.push(event.identity.provider_request_id.clone());
            if seen.len() == 1 {
                ModelTurnAction::repeat()
            } else {
                ModelTurnAction::continue_run()
            }
        }
    }

    let hook = RetryOnceCapturingIdentity::default();
    AgentBuilder::new(MockCompletionModel::from_turns([
        MockTurn::text("first attempt").with_provider_request_id("req-attempt-1"),
        MockTurn::text("second attempt").with_provider_request_id("req-attempt-2"),
    ]))
    .add_hook(hook.clone())
    .build()
    .prompt(Message::user("prompt"))
    .max_turns(3)
    .run()
    .await
    .expect("retried run");

    assert_eq!(
        *hook.seen.lock().expect("retry identities"),
        [
            Some("req-attempt-1".to_string()),
            Some("req-attempt-2".to_string())
        ],
        "each attempt's event carries that attempt's id — no stale leak"
    );
}

// ---------------------------------------------------------------------
// Raw provider response capture (always on).
//
// The agent erased the model, so a caller can never reach the provider's
// `raw_completion` / `raw_stream`; the `raw` payload every response and
// stream terminal carries is the only route to it. The mock behaves like
// a real seam — a scripted payload is attached unconditionally, and a
// turn scripted without one reports `Value::Null` (nothing behind it, not
// "capture was not requested") — so these tests prove the whole route:
// the payload reaches the hook events on both surfaces, and every
// recorded call carries *its own* attempt's payload.
// ---------------------------------------------------------------------

/// The streamed `CompletionResponse` carries the same canonical fields the
/// blocking driver reports: the prompt, the assembled content, the usage
/// and the provider message id.
/// An agent builder over a bus whose default model relays `turns`
/// verbatim, items past the terminal included: a wire's driver stops at the
/// terminal, so only a relayed stream can deliver them.
fn relayed(
    turns: impl IntoIterator<Item = impl IntoIterator<Item = MockStreamEvent>>,
) -> AgentBuilder {
    let (dispatcher, registrar, mut driver) = crate::bus::Bus::channel();
    driver
        .register(
            "model:relay",
            rig_core::test_utils::MockRelay::new("relay", turns),
        )
        .expect("register");
    tokio::spawn(driver);
    AgentBuilder::over_bus(
        dispatcher,
        registrar,
        "relay",
        rig_core::effect::HandlerKey::from("model:relay"),
    )
}

/// The provider's end ends the reply: whatever a stream sends after it —
/// an error, visible content, an unmodeled item — is never read, and the
/// turn completes as the provider ended it.
#[tokio::test]
async fn frames_after_the_providers_end_are_not_read() {
    let cases = [
        ("error", MockStreamEvent::error("post-final failure")),
        ("text", MockStreamEvent::text("late text")),
        ("reasoning", MockStreamEvent::reasoning("late reasoning")),
        (
            "tool call",
            MockStreamEvent::tool_call("late", "add", json!({"x": 1, "y": 2})),
        ),
        ("unknown", MockStreamEvent::unknown(json!({"type": "late"}))),
    ];

    for (case, late) in cases {
        let hook = FinishLifecycleHook::default();
        let mut stream = relayed([vec![
            MockStreamEvent::text("canonical response"),
            MockStreamEvent::final_response(canonical_usage()),
            late,
        ]])
        .add_hook(hook.clone())
        .build()
        .prompt("canonical prompt")
        .stream();
        let mut texts = String::new();
        let mut completion_calls = 0;
        let mut response = None;
        while let Some(item) = stream.next().await {
            match item.unwrap_or_else(|error| panic!("{case}: {error:?}")) {
                MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::Text {
                    text,
                    ..
                })) => texts.push_str(&text),
                MultiTurnStreamItem::StreamAssistantItem(Item::Unknown(payload)) => {
                    panic!("{case}: a late item was read: {payload:?}")
                }
                MultiTurnStreamItem::CompletionCall(_) => completion_calls += 1,
                MultiTurnStreamItem::FinalResponse(final_response) => {
                    response = Some(final_response)
                }
                _ => {}
            }
        }

        assert_eq!(texts, "canonical response", "{case}");
        assert_eq!(completion_calls, 1, "{case}");
        assert_eq!(
            response.expect("the run completes").output(),
            "canonical response",
            "{case}"
        );
        assert_eq!(
            hook.snapshots.lock().expect("finish snapshots").len(),
            1,
            "{case}"
        );
        assert_eq!(hook.model_turns.load(SeqCst), 1, "{case}");
    }
}

fn blocking_model() -> MockCompletionModel {
    MockCompletionModel::from_turns([
        MockTurn::tool_call("tc1", "add", json!({"x": 2, "y": 3})),
        MockTurn::text("the answer is 5"),
    ])
}

/// Note the shape of turn one: the call's input streams as fragments
/// (`tc1`) *and* the wire restates it as one complete `ToolCall`. See
/// [`streamed_tool_call_items_share_one_block_id`] for the
/// correlation contract this pins.
fn streaming_model() -> MockCompletionModel {
    MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call_name_delta("tc1", "add"),
            MockStreamEvent::tool_call_arguments_delta("tc1", "{\"x\":2,\"y\":3}"),
            MockStreamEvent::tool_call("tc1", "add", json!({"x": 2, "y": 3})),
            MockStreamEvent::final_response_with_total_tokens(0),
        ],
        vec![
            MockStreamEvent::text("the answer is 5"),
            MockStreamEvent::final_response_with_total_tokens(0),
        ],
    ])
}

/// The public blocking and streaming prompt surfaces enforce the one-call
/// boundary identically after executing a tool-producing first turn.
#[tokio::test]
async fn prompt_surfaces_reject_second_tool_roundtrip_request_at_budget_one() {
    let blocking_model = blocking_model();
    let blocking_recorded = blocking_model.clone();
    let blocking_agent = AgentBuilder::new(blocking_model).tool(MockAddTool).build();
    let blocking_err = blocking_agent
        .prompt("add 2 and 3")
        .max_turns(1)
        .await
        .expect_err("blocking prompt should reject request two");
    assert!(matches!(
        blocking_err,
        PromptError::MaxTurns { max_turns: 1, .. }
    ));
    assert_eq!(blocking_recorded.request_count(), 1);

    let streaming_model = streaming_model();
    let streaming_recorded = streaming_model.clone();
    let streaming_agent = AgentBuilder::new(streaming_model).tool(MockAddTool).build();
    let mut stream = streaming_agent.prompt("add 2 and 3").max_turns(1).stream();
    let mut streaming_err = None;
    while let Some(item) = stream.next().await {
        if let Err(err) = item {
            streaming_err = Some(err);
            break;
        }
    }
    match streaming_err {
        Some(err) => assert!(matches!(err, PromptError::MaxTurns { max_turns: 1, .. })),
        other => panic!("expected streaming max-turns error, got {other:?}"),
    }
    assert_eq!(streaming_recorded.request_count(), 1);
}

/// Structured tool-execution results reach the tool `OutcomeEvent` as machine
/// metadata (error/refusal state plus result context), on both the blocking and streaming paths,
/// so hooks can steer on a classified failure without parsing the result
/// string.
mod structured_tool_results {
    use std::sync::{Arc, Mutex};

    use futures::StreamExt;
    use serde_json::json;

    use crate::agent::{
        AgentBuilder, AgentHook, DispatchAction, DispatchEvent, HookContext, HookStack,
        OutcomeAction, OutcomeEvent,
    };
    use crate::test_utils::{
        MockAddTool, MockCompletionModel, MockDeniedTool, MockFailingTool, MockHandledFailureTool,
        MockMetadataTool, MockRequestId, MockStreamEvent, MockTurn,
    };
    use crate::tool::{ToolErrorKind, ToolResult};

    /// Records, for every `ToolResult` event, a compact outcome label and the
    /// model-visible result string — the machine metadata a policy reads.
    #[derive(Clone, Default)]
    struct OutcomeHook {
        outcomes: Arc<Mutex<Vec<String>>>,
        results: Arc<Mutex<Vec<String>>>,
    }

    impl OutcomeHook {
        fn outcomes(&self) -> Vec<String> {
            self.outcomes.lock().expect("outcomes").clone()
        }

        fn results(&self) -> Vec<String> {
            self.results.lock().expect("results").clone()
        }
    }

    /// A compact string label for an outcome, e.g. `error:timeout`.
    fn outcome_label(result: &ToolResult) -> String {
        if result.is_skipped() {
            "skipped".to_string()
        } else if result.is_refused() {
            "denied".to_string()
        } else if let Some(error) = result.error() {
            format!("error:{}", error.kind().as_str())
        } else {
            "success".to_string()
        }
    }

    impl AgentHook for OutcomeHook {
        async fn on_outcome(&self, _ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
            if let Some(result) = event.tool_result() {
                self.outcomes
                    .lock()
                    .expect("outcomes")
                    .push(outcome_label(result));
                self.results
                    .lock()
                    .expect("results")
                    .push(result.output().render());
            }
            OutcomeAction::proceed()
        }
    }

    /// A blocking model that calls `tool` once, then answers.
    fn model_one_tool_then_text(tool: &str) -> MockCompletionModel {
        MockCompletionModel::from_turns([
            MockTurn::tool_call("tc1", tool, json!({})),
            MockTurn::text("done"),
        ])
    }

    /// A streaming model that calls `tool` once, then answers.
    fn stream_model_one_tool_then_text(tool: &str) -> MockCompletionModel {
        MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call_name_delta("tc1", tool),
                MockStreamEvent::tool_call_arguments_delta("tc1", "{}"),
                MockStreamEvent::tool_call("tc1", tool, json!({})),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
            vec![
                MockStreamEvent::text("done"),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
        ])
    }

    // (1) A `Timeout` failure reaches the tool `OutcomeEvent` as structured
    // metadata (not just a string), with the model-visible feedback intact.
    #[tokio::test]
    async fn timeout_failure_surfaces_structured_outcome() {
        let hook = OutcomeHook::default();
        AgentBuilder::new(model_one_tool_then_text("flaky_tool"))
            .tool(MockFailingTool::new(ToolErrorKind::Timeout))
            .add_hook(hook.clone())
            .build()
            .prompt("go")
            .max_turns(3)
            .run()
            .await
            .expect("run should succeed; a tool timeout is model-visible feedback, not fatal");

        assert_eq!(hook.outcomes(), vec!["error:timeout".to_string()]);
        // (4) The model still receives useful text for the handled failure.
        assert_eq!(hook.results(), vec!["mock tool call failed".to_string()]);
    }

    // (2) A hook counts timeout failures in the run scratchpad and terminates
    // the run after a threshold — the motivating use case.
    #[tokio::test]
    async fn hook_terminates_after_repeated_timeouts() {
        #[derive(Clone, Default)]
        struct TimeoutCount(usize);

        struct TimeoutTerminator;
        impl AgentHook for TimeoutTerminator {
            async fn on_outcome(
                &self,
                ctx: &HookContext,
                event: OutcomeEvent<'_>,
            ) -> OutcomeAction {
                if let Some(result) = event.tool_result()
                    && result.is_error_kind(ToolErrorKind::Timeout)
                {
                    let count = ctx.scratchpad().update(|c: &mut TimeoutCount| {
                        c.0 += 1;
                        c.0
                    });
                    if count >= 2 {
                        return OutcomeAction::stop("aborting after repeated tool timeouts");
                    }
                }
                OutcomeAction::proceed()
            }
        }

        let observer = OutcomeHook::default();
        let err = AgentBuilder::new(MockCompletionModel::from_turns([
            MockTurn::tool_call("tc1", "flaky_tool", json!({})),
            MockTurn::tool_call("tc2", "flaky_tool", json!({})),
            MockTurn::text("unreachable"),
        ]))
        .tool(MockFailingTool::new(ToolErrorKind::Timeout))
        // Observer first so it records both timeouts before the terminator fires.
        .add_hook(observer.clone())
        .add_hook(TimeoutTerminator)
        .build()
        .prompt("go")
        .max_turns(5)
        .run()
        .await
        .expect_err("the run must terminate after two timeouts");

        assert!(
            err.to_string()
                .contains("aborting after repeated tool timeouts"),
            "unexpected error: {err}"
        );
        assert_eq!(
            observer.outcomes(),
            vec!["error:timeout".to_string(), "error:timeout".to_string()],
            "both timeout outcomes must be observed before termination"
        );
    }

    // (3) A not-found (404) failure surfaces as structured `NotFound` metadata
    // but does not terminate the run by default — the model may try another path.
    #[tokio::test]
    async fn not_found_outcome_is_structured_and_non_fatal() {
        let hook = OutcomeHook::default();
        let status: Arc<Mutex<Option<u16>>> = Arc::new(Mutex::new(None));

        struct StatusProbe(Arc<Mutex<Option<u16>>>);
        impl AgentHook for StatusProbe {
            async fn on_outcome(
                &self,
                _ctx: &HookContext,
                event: OutcomeEvent<'_>,
            ) -> OutcomeAction {
                if let Some(error) = event.tool_result().and_then(|result| result.error()) {
                    *self.0.lock().expect("status") = error.http_status();
                }
                OutcomeAction::proceed()
            }
        }

        AgentBuilder::new(model_one_tool_then_text("flaky_tool"))
            .tool(MockFailingTool::new(ToolErrorKind::NotFound))
            .add_hook(hook.clone())
            .add_hook(StatusProbe(status.clone()))
            .build()
            .prompt("go")
            .max_turns(3)
            .run()
            .await
            .expect("a 404 must not terminate the run by default");

        assert_eq!(hook.outcomes(), vec!["error:not_found".to_string()]);
        assert_eq!(
            *status.lock().expect("status"),
            Some(404),
            "the structured failure must carry the HTTP status"
        );
    }

    // (4) A tool that returns a handled failure via ordinary `Result` shows the
    // model useful output while the outcome is a classified error.
    #[tokio::test]
    async fn handled_failure_delivers_model_output_and_error_outcome() {
        let hook = OutcomeHook::default();
        AgentBuilder::new(model_one_tool_then_text("lookup"))
            .tool(MockHandledFailureTool)
            .add_hook(hook.clone())
            .build()
            .prompt("go")
            .max_turns(3)
            .run()
            .await
            .expect("a handled failure is not fatal");

        assert_eq!(hook.outcomes(), vec!["error:not_found".to_string()]);
        assert_eq!(
            hook.results(),
            vec!["no record found for id 42; try a different id".to_string()],
            "the tool's model-visible output must survive alongside the error outcome"
        );
    }

    // (7) `DispatchAction::skip` on the tool-call dispatch produces a structured `Skipped`
    // outcome that the result hook observes.
    #[tokio::test]
    async fn flow_skip_produces_skipped_outcome() {
        struct SkipHook;
        impl AgentHook for SkipHook {
            async fn on_dispatch(
                &self,
                _ctx: &HookContext,
                event: DispatchEvent<'_>,
            ) -> DispatchAction {
                if event.tool_name().is_some() {
                    DispatchAction::skip("not executed (denied by policy); do not retry")
                } else {
                    DispatchAction::proceed()
                }
            }
        }

        let observer = OutcomeHook::default();
        AgentBuilder::new(model_one_tool_then_text("flaky_tool"))
            .tool(MockFailingTool::new(ToolErrorKind::Timeout))
            .add_hook(SkipHook)
            .add_hook(observer.clone())
            .build()
            .prompt("go")
            .max_turns(3)
            .run()
            .await
            .expect("run should succeed after skipping the tool");

        assert_eq!(observer.outcomes(), vec!["skipped".to_string()]);
        assert_eq!(
            observer.results(),
            vec!["not executed (denied by policy); do not retry".to_string()]
        );
    }

    // A *tool-authored* refusal surfaces as a `Denied`
    // outcome — distinct from a hook `DispatchAction::skip`, which is `Skipped`. This
    // pins the documented `Skipped` vs `Denied` split: `Denied` comes only
    // from the tool, never from a hook skip.
    #[tokio::test]
    async fn tool_authored_denial_produces_denied_outcome() {
        let hook = OutcomeHook::default();
        AgentBuilder::new(model_one_tool_then_text("guarded"))
            .tool(MockDeniedTool)
            .add_hook(hook.clone())
            .build()
            .prompt("go")
            .max_turns(3)
            .run()
            .await
            .expect("a tool-authored denial is not fatal");

        assert_eq!(hook.outcomes(), vec!["denied".to_string()]);
        assert_eq!(
            hook.results(),
            vec!["access to this resource is not permitted".to_string()],
            "the model still receives the tool's denial message"
        );
    }

    #[tokio::test]
    async fn permission_denied_failure_is_not_a_tool_refusal() {
        let hook = OutcomeHook::default();
        AgentBuilder::new(model_one_tool_then_text("flaky_tool"))
            .tool(MockFailingTool::new(ToolErrorKind::PermissionDenied))
            .add_hook(hook.clone())
            .build()
            .prompt("go")
            .max_turns(3)
            .run()
            .await
            .expect("a permission failure is model-visible feedback, not fatal");

        assert_eq!(hook.outcomes(), vec!["error:permission_denied".to_string()]);
        assert_eq!(hook.results(), vec!["mock tool call failed".to_string()]);
    }

    // A `DispatchAction::Patch` hook followed by a skip hook: the tool must not run,
    // the `ToolResult` reports the *rewritten* args (not the model's
    // original), and the outcome is `Skipped` — the rewrite (e.g. a
    // redaction) is not lost when a later hook short-circuits. Verified on
    // both the blocking and streaming surfaces.
    #[tokio::test]
    async fn rewrite_args_then_skip_reports_rewritten_args() {
        // Rewrites the tool args, replacing whatever the model emitted.
        struct RewriteHook;
        impl AgentHook for RewriteHook {
            async fn on_dispatch(
                &self,
                _ctx: &HookContext,
                event: DispatchEvent<'_>,
            ) -> DispatchAction {
                DispatchAction::rewrite_tool_args(event.kind, json!({ "x": 41, "y": 1 }))
            }
        }
        // Skips *after* the rewrite (registered second).
        struct SkipHook;
        impl AgentHook for SkipHook {
            async fn on_dispatch(
                &self,
                _ctx: &HookContext,
                event: DispatchEvent<'_>,
            ) -> DispatchAction {
                if event.tool_name().is_some() {
                    DispatchAction::skip("denied after rewrite")
                } else {
                    DispatchAction::proceed()
                }
            }
        }
        // Records the args + outcome seen on the `ToolResult` event.
        #[derive(Clone, Default)]
        struct ArgsProbe {
            args: Arc<Mutex<Option<String>>>,
            outcome: Arc<Mutex<Option<String>>>,
        }
        impl AgentHook for ArgsProbe {
            async fn on_outcome(
                &self,
                _ctx: &HookContext,
                event: OutcomeEvent<'_>,
            ) -> OutcomeAction {
                if let (Some(args), Some(result)) = (event.tool_args(), event.tool_result()) {
                    *self.args.lock().expect("args") = Some(args.to_string());
                    *self.outcome.lock().expect("outcome") = Some(outcome_label(result));
                }
                OutcomeAction::proceed()
            }
        }

        async fn run_surface(streaming: bool) -> (String, String) {
            let probe = ArgsProbe::default();
            // The tool must never execute; `MockAddTool` would produce a
            // `Success` outcome with result "42" if it (wrongly) ran.
            if streaming {
                let mut stream = AgentBuilder::new(stream_model_one_tool_then_text("add"))
                    .tool(MockAddTool)
                    .add_hook(RewriteHook)
                    .add_hook(SkipHook)
                    .add_hook(probe.clone())
                    .build()
                    .prompt("go")
                    .max_turns(3)
                    .stream();
                while let Some(item) = stream.next().await {
                    if let Err(err) = item {
                        panic!("stream item errored: {err}");
                    }
                }
            } else {
                AgentBuilder::new(model_one_tool_then_text("add"))
                    .tool(MockAddTool)
                    .add_hook(RewriteHook)
                    .add_hook(SkipHook)
                    .add_hook(probe.clone())
                    .build()
                    .prompt("go")
                    .max_turns(3)
                    .run()
                    .await
                    .expect("run should succeed after skipping the tool");
            }
            let args = probe.args.lock().expect("args").clone().expect("args seen");
            let outcome = probe
                .outcome
                .lock()
                .expect("outcome")
                .clone()
                .expect("outcome seen");
            (args, outcome)
        }

        for streaming in [false, true] {
            let (args, outcome) = run_surface(streaming).await;
            assert_eq!(
                outcome, "skipped",
                "the skipped tool must produce a Skipped outcome (streaming={streaming})"
            );
            let parsed: serde_json::Value =
                serde_json::from_str(&args).expect("ToolResult args are valid JSON");
            assert_eq!(
                parsed,
                json!({ "x": 41, "y": 1 }),
                "the skipped ToolResult must report the rewritten args, not the model's \
                     original {{}} (streaming={streaming}); got {args}"
            );
        }
    }

    // End-to-end nesting: a *nested* `HookStack` that rewrites args then skips
    // must still report the rewritten args on the skipped `ToolResult` — the
    // inner rewrite is not lost behind the inner skip when the stack is added
    // as a single composed hook. Guards the nested-composition fix.
    #[tokio::test]
    async fn nested_hook_stack_rewrite_then_skip_reports_rewritten_args() {
        struct RewriteHook;
        impl AgentHook for RewriteHook {
            async fn on_dispatch(
                &self,
                _ctx: &HookContext,
                event: DispatchEvent<'_>,
            ) -> DispatchAction {
                DispatchAction::rewrite_tool_args(event.kind, json!({ "x": 41, "y": 1 }))
            }
        }
        struct SkipHook;
        impl AgentHook for SkipHook {
            async fn on_dispatch(
                &self,
                _ctx: &HookContext,
                event: DispatchEvent<'_>,
            ) -> DispatchAction {
                if event.tool_name().is_some() {
                    DispatchAction::skip("denied after nested rewrite")
                } else {
                    DispatchAction::proceed()
                }
            }
        }
        #[derive(Clone, Default)]
        struct ArgsProbe {
            args: Arc<Mutex<Option<String>>>,
            outcome: Arc<Mutex<Option<String>>>,
        }
        impl AgentHook for ArgsProbe {
            async fn on_outcome(
                &self,
                _ctx: &HookContext,
                event: OutcomeEvent<'_>,
            ) -> OutcomeAction {
                if let (Some(args), Some(result)) = (event.tool_args(), event.tool_result()) {
                    *self.args.lock().expect("args") = Some(args.to_string());
                    *self.outcome.lock().expect("outcome") = Some(outcome_label(result));
                }
                OutcomeAction::proceed()
            }
        }

        // The rewrite + skip live inside a *nested* stack added as one hook.
        fn nested_stack() -> HookStack {
            let mut nested = HookStack::new();
            nested.push(RewriteHook);
            nested.push(SkipHook);
            nested
        }

        // Verified on both surfaces: run_single_tool (shared) drives the same
        // nested resolution, so blocking and streaming must agree.
        for streaming in [false, true] {
            let probe = ArgsProbe::default();
            if streaming {
                let mut stream = AgentBuilder::new(stream_model_one_tool_then_text("add"))
                    .tool(MockAddTool)
                    .add_hook(nested_stack())
                    .add_hook(probe.clone())
                    .build()
                    .prompt("go")
                    .max_turns(3)
                    .stream();
                while let Some(item) = stream.next().await {
                    if let Err(err) = item {
                        panic!("stream item errored: {err}");
                    }
                }
            } else {
                AgentBuilder::new(model_one_tool_then_text("add"))
                    .tool(MockAddTool)
                    .add_hook(nested_stack())
                    .add_hook(probe.clone())
                    .build()
                    .prompt("go")
                    .max_turns(3)
                    .run()
                    .await
                    .expect("run should succeed after the nested stack skips the tool");
            }

            assert_eq!(
                probe.outcome.lock().expect("outcome").clone(),
                Some("skipped".to_string()),
                "streaming={streaming}"
            );
            let args = probe.args.lock().expect("args").clone().expect("args seen");
            let parsed: serde_json::Value = serde_json::from_str(&args).expect("valid JSON args");
            assert_eq!(
                parsed,
                json!({ "x": 41, "y": 1 }),
                "the nested stack's rewrite must survive its skip and reach the ToolResult \
                     (streaming={streaming}); got {args}"
            );
        }
    }

    // (8) Invalid JSON arguments are classified as a structured `InvalidArgs`
    // failure rather than surfacing as an opaque string.
    #[tokio::test]
    async fn invalid_args_are_classified_as_invalid_args() {
        let hook = OutcomeHook::default();
        AgentBuilder::new(MockCompletionModel::from_turns([
            // `add` needs integers; a string is a hard parse failure.
            MockTurn::tool_call("tc1", "add", json!({ "x": "not-a-number", "y": 1 })),
            MockTurn::text("done"),
        ]))
        .tool(MockAddTool)
        .add_hook(hook.clone())
        .build()
        .prompt("go")
        .max_turns(3)
        .run()
        .await
        .expect("an invalid-args failure is model-visible feedback, not fatal");

        assert_eq!(hook.outcomes(), vec!["error:invalid_args".to_string()]);
    }

    // Result metadata a tool attaches reaches the hook but never appears in the
    // model-visible output on either execution surface.
    #[tokio::test]
    async fn success_result_metadata_reaches_hook_but_not_model() {
        struct MetadataProbe {
            seen: Arc<Mutex<Option<String>>>,
            model_output: Arc<Mutex<Option<String>>>,
        }
        impl AgentHook for MetadataProbe {
            async fn on_outcome(
                &self,
                _ctx: &HookContext,
                event: OutcomeEvent<'_>,
            ) -> OutcomeAction {
                if let (Some(result), Some(tool_context)) =
                    (event.tool_result(), event.tool_context())
                {
                    *self.seen.lock().expect("seen") = tool_context
                        .result::<MockRequestId>()
                        .expect("request id decodes")
                        .map(|id| id.0);
                    *self.model_output.lock().expect("model_output") =
                        Some(result.output().render());
                }
                OutcomeAction::proceed()
            }
        }

        async fn run_surface(streaming: bool) -> (Option<String>, String) {
            let seen: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
            let model_output: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
            let probe = MetadataProbe {
                seen: seen.clone(),
                model_output: model_output.clone(),
            };

            if streaming {
                let mut stream = AgentBuilder::new(stream_model_one_tool_then_text("with_meta"))
                    .tool(MockMetadataTool)
                    .add_hook(probe)
                    .build()
                    .prompt("go")
                    .max_turns(3)
                    .stream();
                while let Some(item) = stream.next().await {
                    if let Err(error) = item {
                        panic!("stream item errored: {error}");
                    }
                }
            } else {
                AgentBuilder::new(model_one_tool_then_text("with_meta"))
                    .tool(MockMetadataTool)
                    .add_hook(probe)
                    .build()
                    .prompt("go")
                    .max_turns(3)
                    .run()
                    .await
                    .expect("run should succeed");
            }

            let seen_value = seen.lock().expect("seen").clone();
            let output = model_output
                .lock()
                .expect("model_output")
                .clone()
                .expect("output");
            (seen_value, output)
        }

        for streaming in [false, true] {
            let (seen, output) = run_surface(streaming).await;
            assert_eq!(
                seen,
                Some("req-7".to_string()),
                "the tool's result metadata must reach the hook (streaming={streaming})"
            );
            assert_eq!(output, "done");
            assert!(
                !output.contains("req-7"),
                "result metadata must never leak into model output (streaming={streaming})"
            );
        }
    }

    // (6) An `OutcomeAction::rewrite_tool_result` hook redacts the model-visible text, but a later
    // policy hook still sees the tool's *raw* structured outcome — a rewrite
    // changes only what the model sees, not the classification.
    #[tokio::test]
    async fn rewrite_result_does_not_mask_the_structured_outcome() {
        struct Redact;
        impl AgentHook for Redact {
            async fn on_outcome(
                &self,
                _ctx: &HookContext,
                event: OutcomeEvent<'_>,
            ) -> OutcomeAction {
                OutcomeAction::rewrite_tool_result(&event, "[REDACTED]")
            }
        }

        let observer = OutcomeHook::default();
        AgentBuilder::new(model_one_tool_then_text("flaky_tool"))
            .tool(MockFailingTool::new(ToolErrorKind::NotFound))
            // Observer AFTER the redactor: it still sees the true outcome, and
            // the chained (redacted) model-visible result.
            .add_hook(Redact)
            .add_hook(observer.clone())
            .build()
            .prompt("go")
            .max_turns(3)
            .run()
            .await
            .expect("run should succeed");

        assert_eq!(observer.outcomes(), vec!["error:not_found".to_string()]);
        assert_eq!(observer.results(), vec!["[REDACTED]".to_string()]);
    }

    // (9) The blocking and streaming surfaces observe identical structured
    // outcomes for the same scenario.
    #[tokio::test]
    async fn streaming_and_blocking_outcomes_match() {
        let blocking = OutcomeHook::default();
        AgentBuilder::new(model_one_tool_then_text("flaky_tool"))
            .tool(MockFailingTool::new(ToolErrorKind::Timeout))
            .add_hook(blocking.clone())
            .build()
            .prompt("go")
            .max_turns(3)
            .run()
            .await
            .expect("blocking run should succeed");

        let streaming = OutcomeHook::default();
        let mut stream = AgentBuilder::new(stream_model_one_tool_then_text("flaky_tool"))
            .tool(MockFailingTool::new(ToolErrorKind::Timeout))
            .add_hook(streaming.clone())
            .build()
            .prompt("go")
            .max_turns(3)
            .stream();
        while let Some(item) = stream.next().await {
            if let Err(err) = item {
                panic!("stream item errored: {err}");
            }
        }

        assert_eq!(blocking.outcomes(), vec!["error:timeout".to_string()]);
        assert_eq!(blocking.outcomes(), streaming.outcomes());
        assert_eq!(blocking.results(), streaming.results());
    }

    // (10) With two tools in one turn at `concurrency > 1`, both structured
    // outcomes are observed and the persisted tool results keep call order.
    #[tokio::test]
    async fn concurrent_tools_preserve_order_and_both_outcomes() {
        use rig_core::message::{
            AssistantContent, ToolCall as MessageToolCall, ToolFunction, UserContent,
        };

        let turn = MockTurn::from_contents([
            AssistantContent::ToolCall(MessageToolCall::from_wire(
                "tc_add",
                ToolFunction::new(
                    rig_core::message::ToolName::new("add".to_string()).expect("tool name"),
                    json!({ "x": 2, "y": 3 }),
                ),
            )),
            AssistantContent::ToolCall(MessageToolCall::from_wire(
                "tc_flaky",
                ToolFunction::new(
                    rig_core::message::ToolName::new("flaky_tool".to_string()).expect("tool name"),
                    json!({}),
                ),
            )),
        ]);

        let observer = OutcomeHook::default();
        let response = AgentBuilder::new(MockCompletionModel::from_turns([
            turn,
            MockTurn::text("done"),
        ]))
        .tool(MockAddTool)
        .tool(MockFailingTool::new(ToolErrorKind::Timeout))
        .add_hook(observer.clone())
        .build()
        .prompt("go")
        .max_turns(3)
        .tool_concurrency(2)
        .run()
        .await
        .expect("run should succeed");

        // Hook order may interleave under concurrency, so compare as a set.
        let mut outcomes = observer.outcomes();
        outcomes.sort();
        assert_eq!(
            outcomes,
            vec!["error:timeout".to_string(), "success".to_string()]
        );

        // The persisted tool results must keep tool-call order regardless of
        // completion timing: `add` (tc_add) before `flaky_tool` (tc_flaky).
        let messages = response.messages;
        let tool_result_ids: Vec<String> = messages
            .iter()
            .flat_map(|message| match message {
                crate::completion::Message::User { content } => content
                    .iter()
                    .filter_map(|c| match c {
                        UserContent::ToolResult(result) => Some(
                            result
                                .call
                                .provider()
                                .map(|provider| provider.as_str())
                                .expect("explicit provider ID")
                                .to_owned(),
                        ),
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
                _ => Vec::new(),
            })
            .collect();
        assert_eq!(
            tool_result_ids,
            vec!["tc_add".to_string(), "tc_flaky".to_string()],
            "tool results must be persisted in call order"
        );
    }
}

/// Safety net for the streaming/non-streaming unification: pins the blocking
/// driver's span topology (span name, `invoke_agent` creation, the
/// `follows_from` chain, and `created_agent_span`-gated run-level usage) so a
/// later refactor onto a shared engine cannot silently drift it. The
/// streaming side is already pinned by `assert_stream_usage_recorded_on_chat_spans`.
mod span_safety_net {

    use serde_json::Value;

    use crate::agent::{AgentBuilder, HookContext, OutcomeAction, OutcomeEvent};
    use crate::completion::Usage;
    use crate::test_utils::{MockAddTool, MockCompletionModel, MockTurn, TraceCapture};
    use crate::tool::{ToolContext, ToolExecutionError};

    fn usage(input: u64, output: u64) -> Usage {
        Usage::new().input_tokens(input).output_tokens(output)
    }

    /// Two-turn tool scenario: the blocking driver emits chat -> execute_tool
    /// -> chat, exercising the `follows_from` chain.
    fn tool_then_text_model() -> MockCompletionModel {
        MockCompletionModel::from_turns([
            MockTurn::tool_call("tc1", "add", serde_json::json!({"x": 2, "y": 3}))
                .with_usage(usage(7, 11)),
            MockTurn::text("the answer is 5").with_usage(usage(13, 17)),
        ])
    }

    /// Register the blocking driver's span callsites against the scoped
    /// subscriber before asserting, mirroring the streaming usage test's
    /// interest-cache warm-up (a foreign thread without our subscriber can
    /// otherwise cache `Interest::never` for these callsites).
    async fn warm_blocking_callsites() {
        let agent = AgentBuilder::new(tool_then_text_model())
            .record_content_telemetry(true)
            .tool(MockAddTool)
            .build();
        let _ = agent.prompt("add 2 and 3").max_turns(3).run().await;
    }

    // --- Tool-result rewrites preserve raw policy data and redact telemetry ---

    /// A tool that returns a raw marker; a rewrite hook replaces the
    /// effective model and telemetry presentation.
    struct RawOutputTool;
    impl crate::tool::Tool for RawOutputTool {
        const NAME: &'static str = "raw_output";
        type Error = rig::tool::ToolExecutionError;
        type Args = serde_json::Value;
        type Output = String;
        fn description(&self) -> String {
            "returns a raw output marker".to_string()
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({ "type": "object", "properties": {} })
        }
        async fn call(
            &self,
            _context: &mut ToolContext,
            _args: Self::Args,
        ) -> Result<Self::Output, ToolExecutionError> {
            Ok("RAW_EXECUTION_OUTPUT_42".to_string())
        }
    }

    /// Redacts every tool result before the model sees it.
    struct RedactResultHook;
    impl crate::agent::AgentHook for RedactResultHook {
        async fn on_outcome(&self, _ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
            OutcomeAction::rewrite_tool_result(&event, "[REDACTED]")
        }
    }

    /// Stops the run after observing a completed tool result.
    struct StopOnResultHook;
    impl crate::agent::AgentHook for StopOnResultHook {
        async fn on_outcome(&self, _ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
            if event.tool_result().is_some() {
                OutcomeAction::stop("stop after raw result")
            } else {
                OutcomeAction::proceed()
            }
        }
    }

    /// A `ToolResult` rewrite applies to both model presentation and
    /// telemetry so redaction hooks cannot leak the raw output through spans.
    #[tokio::test]
    async fn tool_result_rewrite_redacts_span_output() {
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
        let capture = TraceCapture::default();
        let _default = tracing::subscriber::set_default(capture.subscriber());

        // Warm the `execute_tool` result callsite under this subscriber, then
        // reset — mirroring the usage tests' interest-cache warm-up.
        warm_blocking_callsites().await;
        tracing::callsite::rebuild_interest_cache();
        capture.clear();

        let model = MockCompletionModel::from_turns([
            MockTurn::tool_call("tc1", "raw_output", serde_json::json!({})),
            MockTurn::text("ok"),
        ]);
        let response = AgentBuilder::new(model)
            .record_content_telemetry(true)
            .tool(RawOutputTool)
            .add_hook(RedactResultHook)
            .build()
            .prompt("go")
            .max_turns(3)
            .run()
            .await
            .expect("run should succeed");
        assert_eq!(response.output(), "ok");

        let captured = capture.values_of("gen_ai.tool.call.result");
        let captured: Vec<&str> = captured.iter().filter_map(Value::as_str).collect();
        assert!(
            captured.iter().any(|v| v.contains("[REDACTED]")),
            "the rewritten presentation must reach telemetry; captured: {captured:?}"
        );
        assert!(
            !captured
                .iter()
                .any(|v| v.contains("RAW_EXECUTION_OUTPUT_42")),
            "the raw tool output must not leak through telemetry; captured: {captured:?}"
        );
    }

    /// Stopping from the result hook retains outcome metadata but omits
    /// potentially sensitive result content from telemetry.
    #[tokio::test]
    async fn tool_result_stop_omits_span_output() {
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
        let capture = TraceCapture::default();
        let _default = tracing::subscriber::set_default(capture.subscriber());

        warm_blocking_callsites().await;
        tracing::callsite::rebuild_interest_cache();
        capture.clear();

        let result = AgentBuilder::new(MockCompletionModel::from_turns([MockTurn::tool_call(
            "tc1",
            "raw_output",
            serde_json::json!({}),
        )]))
        .tool(RawOutputTool)
        .add_hook(StopOnResultHook)
        .build()
        .prompt("go")
        .max_turns(2)
        .run()
        .await;
        assert!(result.is_err(), "the result hook should stop the run");

        let captured = capture.values_of("gen_ai.tool.call.result");
        let captured: Vec<&str> = captured.iter().filter_map(Value::as_str).collect();
        assert!(
            !captured
                .iter()
                .any(|value| value.contains("RAW_EXECUTION_OUTPUT_42")),
            "a Stop must not leak raw execution telemetry; captured: {captured:?}"
        );
    }
}

fn tool_call_content(id: &str, args: serde_json::Value) -> AssistantContent {
    // `from_wire` mirrors the provider boundary (and the streamed mock's
    // conversion): the wire id becomes both the durable id and the
    // provider correlator, keeping blocking/streaming parity exact.
    AssistantContent::ToolCall(MessageToolCall::from_wire(
        id,
        ToolFunction::new(
            rig_core::message::ToolName::new("add".to_string()).expect("tool name"),
            args,
        ),
    ))
}

/// Whether any tool result in `messages` carries `expected` as verbatim text.
/// Used to pin a skip reason's actual value (a reason dropped or altered on
/// both drivers would still satisfy a blocking == streaming equality check).
fn tool_result_text_in_history(messages: &[Message], expected: &str) -> bool {
    messages.iter().any(|message| {
        matches!(
            message,
            Message::User { content }
                if content.iter().any(|item| matches!(
                    item,
                    UserContent::ToolResult(result)
                        if result.content.iter().any(|c| matches!(
                            c,
                            rig_core::message::ToolResultContent::Text(text)
                                if text.text == expected
                        ))
                ))
        )
    })
}

/// Whether any tool result in `messages` carries the exact structured JSON value.
fn tool_result_json_in_history(messages: &[Message], expected: &serde_json::Value) -> bool {
    messages.iter().any(|message| {
        matches!(
            message,
            Message::User { content }
                if content.iter().any(|item| matches!(
                    item,
                    UserContent::ToolResult(result)
                        if result.content.iter().any(|content| matches!(
                            content,
                            rig_core::message::ToolResultContent::Json { value }
                                if value == expected
                        ))
                ))
        )
    })
}

/// A tool whose first-*called* invocation completes *after* the second, so
/// `buffer_unordered` yields the results in completion order — yet the
/// persisted history stays in call order because each result is written into
/// its original call-index slot. The first call (in poll/call order) waits on
/// a gate the second call releases.
#[derive(Clone)]
struct OutOfOrderTool {
    gate: Arc<tokio::sync::Notify>,
    order: Arc<AtomicU32>,
}

impl Tool for OutOfOrderTool {
    const NAME: &'static str = "add";
    type Error = MockToolError;
    type Args = MockOperationArgs;
    type Output = i32;

    fn description(&self) -> String {
        MockAddTool.description()
    }

    fn parameters(&self) -> serde_json::Value {
        MockAddTool.parameters()
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let nth = self.order.fetch_add(1, SeqCst);
        if nth == 0 {
            // First call: cannot finish until a later call releases us.
            self.gate.notified().await;
        } else {
            // Later call: finishes immediately and releases the first.
            self.gate.notify_one();
        }
        Ok(nth as i32)
    }
}

/// `run()` must persist tool results in tool-call (emission) order even when
/// tools complete out of order under concurrency — it runs them with
/// `buffer_unordered` but reindexes each result into its original call-index
/// slot. (This is what keeps its message history identical to the sequential
/// streaming driver.)
#[tokio::test]
async fn run_preserves_tool_call_order_under_out_of_order_completion() {
    let model = MockCompletionModel::from_turns([
        MockTurn::from_contents([
            tool_call_content("tc1", json!({"x": 1, "y": 0})),
            tool_call_content("tc2", json!({"x": 2, "y": 0})),
        ]),
        MockTurn::text("done"),
    ]);
    let response = AgentBuilder::new(model)
        .tool(OutOfOrderTool {
            gate: Arc::new(tokio::sync::Notify::new()),
            order: Arc::new(AtomicU32::new(0)),
        })
        .build()
        .prompt("go")
        .max_turns(3)
        .tool_concurrency(4)
        .run()
        .await
        .expect("run should succeed");

    let messages = response.messages;
    let result_ids: Vec<String> = messages
        .iter()
        .flat_map(|message| match message {
            Message::User { content } => content
                .iter()
                .filter_map(|item| match item {
                    UserContent::ToolResult(result) => Some(
                        result
                            .call
                            .provider()
                            .map(|provider| provider.as_str())
                            .expect("explicit provider ID")
                            .to_owned(),
                    ),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        })
        .collect();
    // Call order (tc1 then tc2), even though tc2 finished first.
    assert_eq!(result_ids, vec!["tc1".to_string(), "tc2".to_string()]);
}

/// An `add` tool whose `x == 1` call cannot finish until another call has
/// run, so under concurrency the first call settles last.
#[derive(Clone)]
struct FirstCallSettlesLastTool {
    gate: Arc<Notify>,
}

#[derive(Deserialize)]
struct AddArgs {
    x: i64,
    y: i64,
}

impl Tool for FirstCallSettlesLastTool {
    const NAME: &'static str = "add";
    type Error = MockToolError;
    type Args = AddArgs;
    type Output = i64;

    fn description(&self) -> String {
        MockAddTool.description()
    }

    fn parameters(&self) -> serde_json::Value {
        MockAddTool.parameters()
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        if args.x == 1 {
            self.gate.notified().await;
        } else {
            self.gate.notify_one();
        }
        Ok(args.x + args.y)
    }
}

/// A host-driven run may hold a turn whose calls share a provider ID (the
/// run keeps such calls answerable, one result per occurrence). Resumed into
/// the engine at `tool_concurrency(2)`, the second call settles first, yet
/// each call must keep its own result: `add(1,1)` answers 2 and `add(2,2)`
/// answers 4, in call order.
#[tokio::test]
async fn duplicate_call_ids_keep_their_own_results_when_settling_out_of_order() {
    use crate::run::{AgentRun, AgentRunStep, ModelTurn, TurnPolicy};

    let agent = AgentBuilder::new(MockCompletionModel::from_turns([MockTurn::text("done")]))
        .tool(FirstCallSettlesLastTool {
            gate: Arc::new(Notify::new()),
        })
        .build();
    let mut run = AgentRun::from_spec(&agent.run_spec(), Message::user("go"), None).max_turns(2);
    let Ok(AgentRunStep::CallModel { turn, .. }) = run.next_step() else {
        panic!("a fresh run calls the model");
    };
    run.advertise_tools(
        turn,
        vec![rig_core::completion::ToolDefinition {
            name: rig_core::message::ToolName::new("add").expect("tool name"),
            description: MockAddTool.description(),
            parameters: MockAddTool.parameters(),
        }],
    );
    let policy = TurnPolicy::new(["add".to_string()].into(), None, None).expect("policy");
    run.model_response(ModelTurn::new(
        rig_core::message::AssistantMessage::default(),
        vec![
            tool_call_content("dup", json!({"x": 1, "y": 1})),
            tool_call_content("dup", json!({"x": 2, "y": 2})),
        ],
        Usage::default(),
        policy,
        json!({}),
    ))
    .expect("the tool turn is accepted");
    assert!(matches!(run.next_step(), Ok(AgentRunStep::CallTools { calls }) if calls.len() == 2));

    let response = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        agent.resume(run).tool_concurrency(2).run(),
    )
    .await
    .expect("the tools run concurrently")
    .expect("the resumed run completes");

    let answers: Vec<String> = response
        .messages()
        .iter()
        .flat_map(|message| match message {
            Message::User { content } => content.clone(),
            _ => Vec::new(),
        })
        .filter_map(|item| match item {
            UserContent::ToolResult(result) => Some(
                result
                    .content
                    .iter()
                    .map(|content| match content {
                        rig_core::message::ToolResultContent::Json { value } => value.to_string(),
                        other => format!("{other:?}"),
                    })
                    .collect::<String>(),
            ),
            _ => None,
        })
        .collect();
    assert_eq!(answers, ["2", "4"], "results stay with their calls");
}

/// Drive a stream to completion, panicking on any stream error, and return
/// its final response.
async fn drive_to_final_response(
    mut stream: crate::agent::streaming::StreamingResult,
) -> crate::agent::PromptResponse {
    let mut final_response = None;
    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::FinalResponse(resp) =
            item.unwrap_or_else(|err| panic!("stream item errored: {err}"))
        {
            final_response = Some(resp);
        }
    }
    final_response.expect("stream should yield a final response")
}

/// Tool-result ids, in history order, across a run's message history.
fn tool_result_ids(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .flat_map(|message| match message {
            Message::User { content } => content
                .iter()
                .filter_map(|item| match item {
                    UserContent::ToolResult(result) => Some(
                        result
                            .call
                            .provider()
                            .map(|provider| provider.as_str())
                            .expect("explicit provider ID")
                            .to_owned(),
                    ),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        })
        .collect()
}

/// The streaming driver under concurrency persists tool results in **call
/// order** even when tools complete out of order. `OutOfOrderTool`'s
/// first-called invocation only finishes once the second runs, so this also
/// proves the tools run concurrently: sequential execution would deadlock on
/// the first call.
#[tokio::test]
async fn stream_preserves_history_order_under_out_of_order_completion() {
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("tc1", "add", json!({"x": 1, "y": 0})),
            MockStreamEvent::tool_call("tc2", "add", json!({"x": 2, "y": 0})),
            MockStreamEvent::final_response_with_total_tokens(0),
        ],
        vec![
            MockStreamEvent::text("done"),
            MockStreamEvent::final_response_with_total_tokens(0),
        ],
    ]);
    let stream = AgentBuilder::new(model)
        .tool(OutOfOrderTool {
            gate: Arc::new(tokio::sync::Notify::new()),
            order: Arc::new(AtomicU32::new(0)),
        })
        .build()
        .prompt("go")
        .max_turns(3)
        .tool_concurrency(4)
        .stream();
    // Timeout so a regression to sequential execution fails cleanly instead
    // of hanging (the first call only completes once the second runs).
    let final_response = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        drive_to_final_response(stream),
    )
    .await
    .expect("streamed tools must run concurrently, not deadlock on the first call");

    let messages = final_response.messages().to_vec();
    // History stays in call order (tc1 then tc2), even though tc2 finished first.
    assert_eq!(
        tool_result_ids(&messages),
        vec!["tc1".to_string(), "tc2".to_string()]
    );
}

/// Under concurrency the streaming driver surfaces tool results **atomically
/// after the whole batch settles**, in **call order** — not as each tool
/// completes. The second call completes first (via the gate), yet its result
/// is still surfaced second, matching persisted history order.
#[tokio::test]
async fn stream_emits_tool_results_in_call_order_after_batch_settles_under_concurrency() {
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("tc1", "add", json!({"x": 1, "y": 0})),
            MockStreamEvent::tool_call("tc2", "add", json!({"x": 2, "y": 0})),
            MockStreamEvent::final_response_with_total_tokens(0),
        ],
        vec![
            MockStreamEvent::text("done"),
            MockStreamEvent::final_response_with_total_tokens(0),
        ],
    ]);
    let mut stream = AgentBuilder::new(model)
        .tool(OutOfOrderTool {
            gate: Arc::new(tokio::sync::Notify::new()),
            order: Arc::new(AtomicU32::new(0)),
        })
        .build()
        .prompt("go")
        .max_turns(3)
        .tool_concurrency(4)
        .stream();

    let mut streamed_result_ids = Vec::new();
    let mut final_response = None;
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while let Some(item) = stream.next().await {
            match item.unwrap_or_else(|err| panic!("stream item errored: {err}")) {
                MultiTurnStreamItem::ToolResult { tool_result, .. } => streamed_result_ids.push(
                    tool_result
                        .call
                        .provider()
                        .map(|provider| provider.as_str())
                        .expect("explicit provider ID")
                        .to_owned(),
                ),
                MultiTurnStreamItem::FinalResponse(resp) => final_response = Some(resp),
                _ => {}
            }
        }
    })
    .await
    .expect("streamed tools must run concurrently, not deadlock on the first call");

    // Call order, even though tc2 completed first — results are surfaced only
    // after the whole batch settles.
    assert_eq!(
        streamed_result_ids,
        vec!["tc1".to_string(), "tc2".to_string()]
    );
    let final_response = final_response.expect("stream should yield a final response");
    assert_eq!(
        tool_result_ids(final_response.messages()),
        vec!["tc1".to_string(), "tc2".to_string()]
    );
}

/// A the event-specific stop action from the `ToolCall` event with a reason keyed by the
/// call's `x` arg, forcing the `x == 2` call (tc2) to terminate *before* the
/// `x == 1` call (tc1): tc2 opens the gate after terminating, tc1 awaits it
/// first. So completion order (tc2) differs from call order (tc1).
struct OrderedTerminateHook {
    gate: Arc<tokio::sync::Notify>,
}

impl AgentHook for OrderedTerminateHook {
    async fn on_dispatch(&self, _ctx: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        if let Some(args) = event.tool_args() {
            let x = serde_json::from_str::<serde_json::Value>(args)
                .ok()
                .and_then(|v| v.get("x").and_then(serde_json::Value::as_i64));
            match x {
                Some(2) => {
                    self.gate.notify_one();
                    return DispatchAction::stop("terminated-by-tc2".to_string());
                }
                Some(1) => {
                    self.gate.notified().await;
                    return DispatchAction::stop("terminated-by-tc1".to_string());
                }
                _ => {}
            }
        }
        DispatchAction::proceed()
    }
}

fn two_terminating_tools_blocking_model() -> MockCompletionModel {
    MockCompletionModel::from_turns([
        MockTurn::from_contents([
            tool_call_content("tc1", json!({"x": 1, "y": 1})),
            tool_call_content("tc2", json!({"x": 2, "y": 2})),
        ]),
        MockTurn::text("unreachable"),
    ])
}

fn two_terminating_tools_streaming_model() -> MockCompletionModel {
    MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("tc1", "add", json!({"x": 1, "y": 1})),
            MockStreamEvent::tool_call("tc2", "add", json!({"x": 2, "y": 2})),
            MockStreamEvent::final_response_with_total_tokens(0),
        ],
        vec![
            MockStreamEvent::text("unreachable"),
            MockStreamEvent::final_response_with_total_tokens(0),
        ],
    ])
}

/// When two tool calls in one turn both terminate the run under
/// `tool_concurrency > 1`, run() and stream() surface the **same** reason —
/// the first-called tool's (call order), not whichever finished first. tc2
/// terminates before tc1, so a completion-order pick would surface tc2's
/// reason and the two drivers would disagree.
#[tokio::test]
async fn concurrent_simultaneous_tool_terminations_pick_call_order_on_both_drivers() {
    let run_err = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        AgentBuilder::new(two_terminating_tools_blocking_model())
            .tool(MockAddTool)
            .build()
            .prompt("go")
            .max_turns(3)
            .tool_concurrency(2)
            .add_hook(OrderedTerminateHook {
                gate: Arc::new(tokio::sync::Notify::new()),
            })
            .run(),
    )
    .await
    .expect("blocking run must not hang")
    .expect_err("the run must terminate");

    let mut stream = AgentBuilder::new(two_terminating_tools_streaming_model())
        .tool(MockAddTool)
        .build()
        .prompt("go")
        .max_turns(3)
        .tool_concurrency(2)
        .add_hook(OrderedTerminateHook {
            gate: Arc::new(tokio::sync::Notify::new()),
        })
        .stream();

    let stream_err = tokio::time::timeout(std::time::Duration::from_secs(5), async move {
        while let Some(item) = stream.next().await {
            if let Err(err) = item {
                return Some(err);
            }
        }
        None
    })
    .await
    .expect("streamed run must not hang")
    .expect("the stream must surface a terminate error");

    let run_msg = run_err.to_string();
    let stream_msg = stream_err.to_string();
    assert!(
        run_msg.contains("terminated-by-tc1"),
        "blocking run should surface the first-called tool's reason, got: {run_msg}"
    );
    assert!(
        stream_msg.contains("terminated-by-tc1"),
        "stream should surface the first-called tool's reason, got: {stream_msg}"
    );
    assert!(
        !run_msg.contains("terminated-by-tc2") && !stream_msg.contains("terminated-by-tc2"),
        "neither driver should surface the later-completing tool's reason"
    );
}

/// Terminates the run from the `ToolCall` event of the first tool only
/// (`x == 1`), letting any later tool through.
struct TerminateOnFirstToolHook;
impl AgentHook for TerminateOnFirstToolHook {
    async fn on_dispatch(&self, _ctx: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        if event
            .tool_args()
            .and_then(|args| serde_json::from_str::<serde_json::Value>(args).ok())
            .and_then(|v| v.get("x").and_then(serde_json::Value::as_i64))
            == Some(1)
        {
            return DispatchAction::stop("stop".to_string());
        }
        DispatchAction::proceed()
    }
}

/// Fail-fast, lock-step across surfaces: on a multi-tool turn whose first
/// tool's hook terminates the run, the SEQUENTIAL default (`tool_concurrency`
/// == 1) surfaces the terminate immediately and does **not** start the
/// remaining sibling tools — so tool B's side effect never runs. The
/// terminating tool's own body never runs either (its `ToolCall` hook fired
/// first), so `calls == 0` on both drivers, which share the tool driver.
#[tokio::test]
async fn default_concurrency_terminate_skips_remaining_tools_on_both_drivers() {
    let blocking_calls = Arc::new(AtomicU32::new(0));
    AgentBuilder::new(two_terminating_tools_blocking_model())
        .tool(CountingAddTool {
            calls: blocking_calls.clone(),
        })
        .build()
        .prompt("go")
        .max_turns(3)
        .add_hook(TerminateOnFirstToolHook)
        .run()
        .await
        .expect_err("the run terminates");
    assert_eq!(
        blocking_calls.load(SeqCst),
        0,
        "fail-fast: blocking run() must not start the second tool after the first terminates"
    );

    let streaming_calls = Arc::new(AtomicU32::new(0));
    let mut stream = AgentBuilder::new(two_terminating_tools_streaming_model())
        .tool(CountingAddTool {
            calls: streaming_calls.clone(),
        })
        .build()
        .prompt("go")
        .max_turns(3)
        .add_hook(TerminateOnFirstToolHook)
        .stream();
    let mut saw_error = false;
    while let Some(item) = stream.next().await {
        if let Err(err) = item {
            saw_error = true;
            assert!(
                err.to_string().contains("stop"),
                "stream() should surface the terminate reason, got: {err}"
            );
            break;
        }
    }
    assert!(saw_error, "stream() must surface the terminate error");
    assert_eq!(
        streaming_calls.load(SeqCst),
        0,
        "fail-fast: stream() must not start the second tool after the first terminates"
    );
}

/// Fail-fast keeps what settled: a stop on the second call cancels with the
/// first call's real result committed and the stopped call closed by `close_pending`,
/// so the cancelled history is canonical.
#[tokio::test]
async fn a_stop_mid_batch_cancels_with_the_settled_result_and_the_rest_closed() {
    struct StopSecondToolHook;
    impl AgentHook for StopSecondToolHook {
        async fn on_dispatch(
            &self,
            _ctx: &HookContext,
            event: DispatchEvent<'_>,
        ) -> DispatchAction {
            if event
                .tool_args()
                .and_then(|args| serde_json::from_str::<serde_json::Value>(args).ok())
                .and_then(|v| v.get("x").and_then(serde_json::Value::as_i64))
                == Some(2)
            {
                return DispatchAction::stop("stop".to_string());
            }
            DispatchAction::proceed()
        }
    }

    let calls = Arc::new(AtomicU32::new(0));
    let err = AgentBuilder::new(two_terminating_tools_blocking_model())
        .tool(CountingAddTool {
            calls: calls.clone(),
        })
        .build()
        .prompt("go")
        .max_turns(3)
        .add_hook(StopSecondToolHook)
        .run()
        .await
        .expect_err("the run terminates");
    assert_eq!(calls.load(SeqCst), 1, "the first call ran before the stop");

    let PromptError::Cancelled { chat_history, .. } = err else {
        panic!("a hook stop cancels the run, got {err:?}");
    };
    assert_eq!(
        rig_core::transcript::validate_canonical(&chat_history),
        Ok(())
    );
    let Some(Message::User { content }) = chat_history.last() else {
        panic!("the batch must be closed: {chat_history:?}");
    };
    let [
        UserContent::ToolResult(settled),
        UserContent::ToolResult(aborted),
    ] = content.as_slice()
    else {
        panic!("one result per call, in call order: {content:?}");
    };
    assert_eq!(settled.call.to_string(), "tc1");
    assert!(!settled.is_error, "tc1 ran: {settled:?}");
    assert_eq!(aborted.call.to_string(), "tc2");
    assert!(aborted.is_error);
    let AssistantContent::ToolCall(tc2) = tool_call_content("tc2", json!({"x": 2, "y": 2})) else {
        panic!("tool_call_content builds a tool call");
    };
    assert_eq!(
        Message::User {
            content: content[1..].to_vec()
        },
        rig_core::transcript::close_pending([&tc2]),
        "the stopped call is closed as transcript closes it"
    );
}

/// A dispatch hook `DispatchAction::skip` surfaces the skip result as a `ToolResult`
/// (the model sees it, and it is committed to history) but produces **no**
/// `ToolExecutionCommitted` — nothing actually ran.
#[tokio::test]
async fn stream_hook_skip_surfaces_result_without_execution_commit() {
    struct SkipHook;
    impl AgentHook for SkipHook {
        async fn on_dispatch(
            &self,
            _ctx: &HookContext,
            event: DispatchEvent<'_>,
        ) -> DispatchAction {
            if event.tool_name().is_some() {
                DispatchAction::skip("blocked by policy")
            } else {
                DispatchAction::proceed()
            }
        }
    }

    let calls = Arc::new(AtomicU32::new(0));
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("tc1", "add", json!({"x": 1, "y": 2})),
            MockStreamEvent::final_response_with_total_tokens(0),
        ],
        vec![
            MockStreamEvent::text("done"),
            MockStreamEvent::final_response_with_total_tokens(0),
        ],
    ]);
    let stream = AgentBuilder::new(model)
        .tool(CountingAddTool {
            calls: calls.clone(),
        })
        .add_hook(SkipHook)
        .build()
        .prompt("go")
        .max_turns(3)
        .stream();

    let mut exec_commits = 0;
    let mut results = 0;
    let mut final_response = None;
    let mut stream = stream;
    while let Some(item) = stream.next().await {
        match item.unwrap_or_else(|err| panic!("stream item errored: {err}")) {
            MultiTurnStreamItem::ToolExecutionCommitted { .. } => exec_commits += 1,
            MultiTurnStreamItem::ToolResult { .. } => {
                results += 1;
            }
            MultiTurnStreamItem::FinalResponse(resp) => final_response = Some(resp),
            _ => {}
        }
    }

    assert_eq!(calls.load(SeqCst), 0, "a skipped tool's body never runs");
    assert_eq!(
        exec_commits, 0,
        "a hook-skipped tool produces no execution-commit"
    );
    assert_eq!(
        results, 1,
        "the skip result is still surfaced to the consumer"
    );
    let final_response = final_response.expect("stream should yield a final response");
    // The skip result is committed to history (the model sees the reason).
    let history = final_response.messages();
    assert!(
        history.iter().any(|m| serde_json::to_string(m)
            .map(|s| s.contains("blocked by policy"))
            .unwrap_or(false)),
        "the skip result is committed to history"
    );
}

/// `ToolChoice::Required` + a hook whose `active_tools([])` advertises no tools
/// is a **local** error: the run fails before any provider round-trip.
#[tokio::test]
async fn required_with_empty_active_tools_errors_locally_without_provider_call() {
    struct EmptyActiveToolsHook;
    impl AgentHook for EmptyActiveToolsHook {
        async fn on_completion_call(
            &self,
            _ctx: &HookContext,
            event: CompletionCallEvent<'_>,
        ) -> CompletionCallAction {
            if let CompletionCallEvent { .. } = event {
                CompletionCallAction::patch(RequestPatch::new().active_tools(Vec::<String>::new()))
            } else {
                CompletionCallAction::continue_run()
            }
        }
    }

    let model = MockCompletionModel::from_turns([MockTurn::text("unreachable")]);
    let probe = model.clone();
    let err = AgentBuilder::new(model)
        .tool(MockAddTool)
        .tool_choice(ToolChoice::Required)
        .add_hook(EmptyActiveToolsHook)
        .build()
        .prompt("go")
        .run()
        .await
        .expect_err("Required with an empty active_tools filter must fail locally");

    assert!(
        probe.requests().is_empty(),
        "the request must fail locally, with no provider round-trip"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("Required"),
        "error should mention Required: {msg}"
    );
    assert!(
        msg.contains("active_tools"),
        "error should name active_tools: {msg}"
    );
}

/// `ToolChoice::Specific` naming a tool that a hook's `active_tools` filtered
/// out is a **local** error naming the filter, before any provider round-trip.
#[tokio::test]
async fn specific_naming_filtered_out_tool_errors_locally_without_provider_call() {
    struct FilterToAddHook;
    impl AgentHook for FilterToAddHook {
        async fn on_completion_call(
            &self,
            _ctx: &HookContext,
            event: CompletionCallEvent<'_>,
        ) -> CompletionCallAction {
            if let CompletionCallEvent { .. } = event {
                CompletionCallAction::patch(RequestPatch::new().active_tools(["add"]))
            } else {
                CompletionCallAction::continue_run()
            }
        }
    }

    let model = MockCompletionModel::from_turns([MockTurn::text("unreachable")]);
    let probe = model.clone();
    let err = AgentBuilder::new(model)
        .tool(MockAddTool)
        .tool(MockSubtractTool)
        .tool_choice(ToolChoice::Specific {
            function_names: vec![rig_core::message::ToolName::new("subtract").expect("tool name")],
        })
        .add_hook(FilterToAddHook)
        .build()
        .prompt("go")
        .run()
        .await
        .expect_err("Specific naming a filtered-out tool must fail locally");

    assert!(
        probe.requests().is_empty(),
        "the request must fail locally, with no provider round-trip"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("subtract"),
        "error should name the missing tool: {msg}"
    );
    assert!(
        msg.contains("active_tools"),
        "error should name active_tools: {msg}"
    );
}

/// A tool that counts how many times it executes.
#[derive(Clone)]
struct CountingAddTool {
    calls: Arc<AtomicU32>,
}
impl Tool for CountingAddTool {
    const NAME: &'static str = "add";
    type Error = MockToolError;
    type Args = MockOperationArgs;
    type Output = i32;
    fn description(&self) -> String {
        MockAddTool.description()
    }
    fn parameters(&self) -> serde_json::Value {
        MockAddTool.parameters()
    }
    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.calls.fetch_add(1, SeqCst);
        MockAddTool.call(_context, args).await
    }
}

/// The shared driver events a hook can terminate the run from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TerminatePoint {
    CompletionCall,
    ToolDispatch,
    ToolOutcome,
}

/// Terminates the run when it sees a chosen event kind, observing every other
/// event as `Continue`.
struct TerminateOn(TerminatePoint);

impl AgentHook for TerminateOn {
    async fn on_completion_call(
        &self,
        _: &HookContext,
        _: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        if self.0 == TerminatePoint::CompletionCall {
            CompletionCallAction::stop("stop here")
        } else {
            CompletionCallAction::continue_run()
        }
    }
    async fn on_dispatch(&self, _: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        if self.0 == TerminatePoint::ToolDispatch && event.tool_name().is_some() {
            DispatchAction::stop("stop here")
        } else {
            DispatchAction::proceed()
        }
    }
    async fn on_outcome(&self, _: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        if self.0 == TerminatePoint::ToolOutcome && event.tool_result().is_some() {
            OutcomeAction::stop("stop here")
        } else {
            OutcomeAction::proceed()
        }
    }
}

/// Renames an invalid tool call to a known tool; observes everything else.
struct RepairInvalidToHook(&'static str);

impl AgentHook for RepairInvalidToHook {
    async fn on_invalid_tool_call(
        &self,
        _ctx: &HookContext,
        event: &InvalidToolCallContext,
    ) -> Option<InvalidToolCallAction> {
        Some(if let _ = event {
            InvalidToolCallAction::repair(self.0)
        } else {
            InvalidToolCallAction::fail()
        })
    }
}

/// An invalid tool call repaired by a hook recovers identically under run()
/// and stream(): the renamed tool executes and both drivers reach the same
/// output, tool-result content, and final message history.
#[tokio::test]
async fn invalid_tool_call_repair_parity_across_run_and_stream() {
    let blocking_model = MockCompletionModel::from_turns([
        MockTurn::tool_call("tc1", "default_api", json!({"x": 2, "y": 3})),
        MockTurn::text("the answer is 5"),
    ]);
    let blocking_hook = RecordingHook::default();
    let blocking = AgentBuilder::new(blocking_model)
        .tool(MockAddTool)
        .build()
        .prompt("add 2 and 3")
        .max_turns(3)
        .add_hook(blocking_hook.clone())
        .add_hook(RepairInvalidToHook("add"))
        .run()
        .await
        .expect("blocking run should recover via repair");

    // Emit the invalid call as a single complete tool call (mirroring the
    // blocking model). A provider stream carries one tool call via one
    // mechanism — deltas *or* a complete call — so this is the apples-to-
    // apples comparison; mixing both would trip the assembler's two
    // independent invalid-detection sites and fire the event twice.
    let streaming_model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("tc1", "default_api", json!({"x": 2, "y": 3})),
            MockStreamEvent::final_response_with_total_tokens(0),
        ],
        vec![
            MockStreamEvent::text("the answer is 5"),
            MockStreamEvent::final_response_with_total_tokens(0),
        ],
    ]);
    let streaming_hook = RecordingHook::default();
    let mut stream = AgentBuilder::new(streaming_model)
        .tool(MockAddTool)
        .build()
        .prompt("add 2 and 3")
        .max_turns(3)
        .add_hook(streaming_hook.clone())
        .add_hook(RepairInvalidToHook("add"))
        .stream();
    let mut final_response = None;
    while let Some(item) = stream.next().await {
        if let Ok(MultiTurnStreamItem::FinalResponse(resp)) =
            item.map_err(|err| panic!("stream item errored: {err}"))
        {
            final_response = Some(resp);
        }
    }
    let final_response = final_response.expect("stream should recover and yield a final response");

    // Same recovered output.
    assert_eq!(blocking.output(), "the answer is 5");
    assert_eq!(final_response.output(), blocking.output());

    // Both drivers reported the invalid tool call to the hook, then executed
    // the repaired tool, so the shared event sequences match.
    assert_eq!(
        blocking_hook.shared_events(),
        streaming_hook.shared_events()
    );
    assert!(
        blocking_hook
            .shared_events()
            .contains(&StepEventKind::InvalidToolCall),
        "the hook must observe the invalid tool call"
    );
    assert_eq!(blocking_hook.tool_results(), streaming_hook.tool_results());
    assert_eq!(blocking_hook.tool_results(), vec!["5".to_string()]);

    // Same final message history.
    let blocking_messages = blocking.messages;
    let streaming_messages = final_response.messages().to_vec();
    assert_eq!(
        serde_json::to_value(&blocking_messages).expect("serialize blocking"),
        serde_json::to_value(&streaming_messages).expect("serialize streaming"),
    );
}

// ----------------------------------------------------------------------
// Single-source-of-truth parity harness
// ----------------------------------------------------------------------
//
// `run()` and `stream()` are two implementations of one agent loop; testing
// they agree on the same input is *differential testing*, with each driver
// acting as the other's oracle. The hazard such tests have (and that bit the
// invalid-tool-repair test above) is *fixture drift*: when the blocking
// `MockTurn` list and the streaming `MockStreamEvent` list are hand-written
// separately, they can silently encode different model behavior, so a
// passing test proves nothing.
//
// The fix — the single-source-of-truth / data-driven principle, embodied by
// pydantic-ai's `TestModel` (one scripted response replayed as a stream) and
// litellm's `stream_chunk_builder` (reassemble the stream, compare to the
// whole) — is to derive *both* encodings from one canonical `ScriptedTurn`
// list. The two drivers are then provably fed identical model behavior and
// can be asserted equal on the medium-independent projection (final output,
// message history, tool-result content, shared hook-event sequence).

/// One tool call inside a scripted turn.
#[derive(Clone)]
struct ScriptedToolCall {
    id: &'static str,
    name: &'static str,
    args: serde_json::Value,
}

/// One scripted model turn, described once and rendered into both a blocking
/// `MockTurn` and a streaming `Vec<MockStreamEvent>`.
#[derive(Clone)]
enum ScriptedTurn {
    /// A final text answer.
    Text(&'static str),
    /// One or more tool calls emitted in a single turn.
    ToolCalls(Vec<ScriptedToolCall>),
}

/// How a tool call is rendered onto the wire for the streaming driver. Both
/// shapes must yield the *same* canonical turn ("chunked-input invariance",
/// the `tokio-util` `LengthDelimitedCodec` lesson): the assembled message
/// history and tool results may not depend on whether a provider sends a
/// complete tool call or streams it as deltas.
#[derive(Clone, Copy)]
enum StreamShape {
    /// One complete tool-call event per call (mirrors the blocking turn).
    Complete,
    /// Name + argument deltas followed by the complete call, additionally
    /// exercising the delta-hook path and the assembler's delta buffering.
    Chunked,
}

impl ScriptedTurn {
    fn as_blocking_turn(&self) -> MockTurn {
        match self {
            ScriptedTurn::Text(text) => MockTurn::text(*text),
            ScriptedTurn::ToolCalls(calls) => {
                // `from_wire` matches the streamed rendering's provider
                // boundary so both encodings yield identical calls.
                MockTurn::from_contents(calls.iter().map(|call| {
                    AssistantContent::ToolCall(MessageToolCall::from_wire(
                        call.id,
                        ToolFunction::new(
                            rig_core::message::ToolName::new(call.name.to_string())
                                .expect("tool name"),
                            call.args.clone(),
                        ),
                    ))
                }))
            }
        }
    }

    fn as_stream_events(&self, shape: StreamShape) -> Vec<MockStreamEvent> {
        let mut events = Vec::new();
        match self {
            ScriptedTurn::Text(text) => events.push(MockStreamEvent::text(*text)),
            ScriptedTurn::ToolCalls(calls) => {
                for call in calls {
                    if let StreamShape::Chunked = shape {
                        // The canonical args still come from the complete
                        // event below, so this exercises the delta path
                        // without changing the turn.
                        let args = serde_json::to_string(&call.args)
                            .expect("scripted args serialize to json");
                        events.push(MockStreamEvent::tool_call_name_delta(call.id, call.name));
                        events.push(MockStreamEvent::tool_call_arguments_delta(call.id, &args));
                    }
                    events.push(MockStreamEvent::tool_call(
                        call.id,
                        call.name,
                        call.args.clone(),
                    ));
                }
            }
        }
        events.push(MockStreamEvent::final_response_with_total_tokens(0));
        events
    }
}

/// The medium-independent projection of a run that both drivers must agree
/// on.
struct ParityOutcome {
    output: String,
    messages: Vec<Message>,
    shared_events: Vec<StepEventKind>,
    tool_results: Vec<String>,
}

async fn run_blocking_scenario(prompt: &'static str, turns: &[ScriptedTurn]) -> ParityOutcome {
    let model = MockCompletionModel::from_turns(turns.iter().map(ScriptedTurn::as_blocking_turn));
    let hook = RecordingHook::default();
    let response = AgentBuilder::new(model)
        .tool(MockAddTool)
        .build()
        .prompt(prompt)
        .max_turns(8)
        .add_hook(hook.clone())
        .run()
        .await
        .expect("blocking scenario should succeed");
    ParityOutcome {
        output: response.output(),
        messages: response.messages,
        shared_events: hook.shared_events(),
        tool_results: hook.tool_results(),
    }
}

async fn run_streaming_scenario(
    prompt: &'static str,
    turns: &[ScriptedTurn],
    shape: StreamShape,
) -> ParityOutcome {
    let model = MockCompletionModel::from_stream_turns(
        turns.iter().map(|turn| turn.as_stream_events(shape)),
    );
    let hook = RecordingHook::default();
    let mut stream = AgentBuilder::new(model)
        .tool(MockAddTool)
        .build()
        .prompt(prompt)
        .max_turns(8)
        .add_hook(hook.clone())
        .stream();
    let mut final_response = None;
    while let Some(item) = stream.next().await {
        if let Ok(MultiTurnStreamItem::FinalResponse(resp)) =
            item.map_err(|err| panic!("stream item errored: {err}"))
        {
            final_response = Some(resp);
        }
    }
    let final_response = final_response.expect("streaming scenario should yield a final response");
    ParityOutcome {
        output: final_response.output().to_string(),
        messages: final_response.messages().to_vec(),
        shared_events: hook.shared_events(),
        tool_results: hook.tool_results(),
    }
}

fn assert_outcomes_match(blocking: &ParityOutcome, streaming: &ParityOutcome, label: &str) {
    assert_eq!(
        blocking.output, streaming.output,
        "{label}: final output diverged"
    );
    assert_eq!(
        blocking.shared_events, streaming.shared_events,
        "{label}: hook event sequence diverged"
    );
    assert_eq!(
        blocking.tool_results, streaming.tool_results,
        "{label}: tool-result content diverged"
    );
    assert_eq!(
        serde_json::to_value(&blocking.messages).expect("serialize blocking"),
        serde_json::to_value(&streaming.messages).expect("serialize streaming"),
        "{label}: message history diverged"
    );
}

/// Drive one canonical scenario through `run()` and through `stream()` in
/// both wire shapes, asserting the medium-independent projection is
/// identical every way. Because both stream shapes are compared against the
/// same blocking outcome, they are also transitively equal to each other.
async fn assert_run_stream_parity(prompt: &'static str, turns: &[ScriptedTurn]) {
    let blocking = run_blocking_scenario(prompt, turns).await;
    for (shape, label) in [
        (StreamShape::Complete, "complete-stream"),
        (StreamShape::Chunked, "chunked-stream"),
    ] {
        let streaming = run_streaming_scenario(prompt, turns, shape).await;
        assert_outcomes_match(&blocking, &streaming, label);
    }
}

fn add_call(id: &'static str, x: i64, y: i64) -> ScriptedToolCall {
    ScriptedToolCall {
        id,
        name: "add",
        args: json!({ "x": x, "y": y }),
    }
}

#[tokio::test]
async fn parity_text_only_run() {
    assert_run_stream_parity("just say hi", &[ScriptedTurn::Text("hi there")]).await;
}

#[tokio::test]
async fn parity_single_tool_then_text() {
    assert_run_stream_parity(
        "add 2 and 3",
        &[
            ScriptedTurn::ToolCalls(vec![add_call("tc1", 2, 3)]),
            ScriptedTurn::Text("the answer is 5"),
        ],
    )
    .await;
}

#[tokio::test]
async fn parity_multiple_tools_in_one_turn() {
    assert_run_stream_parity(
        "add two pairs",
        &[
            ScriptedTurn::ToolCalls(vec![add_call("tc1", 2, 3), add_call("tc2", 10, 20)]),
            ScriptedTurn::Text("done"),
        ],
    )
    .await;
}

#[tokio::test]
async fn parity_multi_turn_sequential_tools() {
    assert_run_stream_parity(
        "chain two additions",
        &[
            ScriptedTurn::ToolCalls(vec![add_call("tc1", 1, 1)]),
            ScriptedTurn::ToolCalls(vec![add_call("tc2", 2, 2)]),
            ScriptedTurn::Text("chained"),
        ],
    )
    .await;
}

/// Skips an invalid tool call (synthetic result, no execution); observes
/// everything else.
struct SkipInvalidHook(&'static str);

impl AgentHook for SkipInvalidHook {
    async fn on_invalid_tool_call(
        &self,
        _ctx: &HookContext,
        event: &InvalidToolCallContext,
    ) -> Option<InvalidToolCallAction> {
        Some(if let _ = event {
            InvalidToolCallAction::skip(self.0)
        } else {
            InvalidToolCallAction::fail()
        })
    }
}

/// An invalid tool call *skipped* by a hook recovers identically under
/// `run()` and `stream()`: the synthetic skip result enters the history
/// verbatim (it is never re-parsed as tool output) and both drivers reach
/// the same output and message history. Complements the repair-parity test.
#[tokio::test]
async fn invalid_tool_call_skip_parity_across_run_and_stream() {
    let blocking_model = MockCompletionModel::from_turns([
        MockTurn::tool_call("tc1", "default_api", json!({"x": 2, "y": 3})),
        MockTurn::text("acknowledged"),
    ]);
    let blocking_hook = RecordingHook::default();
    let blocking = AgentBuilder::new(blocking_model)
        .tool(MockAddTool)
        .build()
        .prompt("do the thing")
        .max_turns(3)
        .add_hook(blocking_hook.clone())
        .add_hook(SkipInvalidHook("tool not permitted"))
        .run()
        .await
        .expect("blocking run should recover via skip");

    // Single complete tool call (mirrors the blocking model; see the
    // repair-parity test for why deltas are not mixed in here).
    let streaming_model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("tc1", "default_api", json!({"x": 2, "y": 3})),
            MockStreamEvent::final_response_with_total_tokens(0),
        ],
        vec![
            MockStreamEvent::text("acknowledged"),
            MockStreamEvent::final_response_with_total_tokens(0),
        ],
    ]);
    let streaming_hook = RecordingHook::default();
    let mut stream = AgentBuilder::new(streaming_model)
        .tool(MockAddTool)
        .build()
        .prompt("do the thing")
        .max_turns(3)
        .add_hook(streaming_hook.clone())
        .add_hook(SkipInvalidHook("tool not permitted"))
        .stream();
    let mut final_response = None;
    while let Some(item) = stream.next().await {
        if let Ok(MultiTurnStreamItem::FinalResponse(resp)) =
            item.map_err(|err| panic!("stream item errored: {err}"))
        {
            final_response = Some(resp);
        }
    }
    let final_response = final_response.expect("stream should recover and yield a final response");

    assert_eq!(blocking.output(), "acknowledged");
    assert_eq!(final_response.output(), blocking.output());
    assert_eq!(
        blocking_hook.shared_events(),
        streaming_hook.shared_events()
    );
    assert!(
        blocking_hook
            .shared_events()
            .contains(&StepEventKind::InvalidToolCall),
        "the hook must observe the invalid tool call"
    );

    // A streamed turn cut short at its invalid call replays canonically;
    // the parity is in the content.
    let blocking_messages = canonical_history(&blocking.messages);
    let streaming_messages = canonical_history(final_response.messages());
    assert_eq!(
        serde_json::to_value(&blocking_messages).expect("serialize blocking"),
        serde_json::to_value(&streaming_messages).expect("serialize streaming"),
    );
    // Pin the actual reason, not just blocking == streaming (see the valid-tool
    // skip test): a reason dropped or altered on BOTH paths would still pass.
    assert!(
        tool_result_text_in_history(&blocking_messages, "tool not permitted"),
        "the verbatim invalid-tool skip reason must be the tool result content"
    );
}

/// Skips a *valid* tool call before execution; observes everything else.
struct SkipToolCallHook(&'static str);

impl AgentHook for SkipToolCallHook {
    async fn on_dispatch(&self, _ctx: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        if event.tool_name().is_some() {
            DispatchAction::skip(self.0)
        } else {
            DispatchAction::proceed()
        }
    }
}

/// A hook that skips a *valid* tool call (`DispatchAction::skip` on the tool dispatch, the
/// honored-action path — distinct from skipping an *invalid* call) recovers
/// identically under `run()` and `stream()`: the synthetic skip result enters
/// the history verbatim without executing the tool, and both drivers reach the
/// same output, tool-result content and message history.
#[tokio::test]
async fn valid_tool_call_skip_parity_across_run_and_stream() {
    let turns = [
        ScriptedTurn::ToolCalls(vec![add_call("tc1", 2, 3)]),
        ScriptedTurn::Text("acknowledged"),
    ];

    let blocking_model =
        MockCompletionModel::from_turns(turns.iter().map(ScriptedTurn::as_blocking_turn));
    let blocking_hook = RecordingHook::default();
    let blocking = AgentBuilder::new(blocking_model)
        .tool(MockAddTool)
        .build()
        .prompt("add 2 and 3")
        .max_turns(3)
        .add_hook(blocking_hook.clone())
        .add_hook(SkipToolCallHook("skipped by policy"))
        .run()
        .await
        .expect("blocking run should succeed with a skipped tool call");

    let streaming_model = MockCompletionModel::from_stream_turns(
        turns
            .iter()
            .map(|turn| turn.as_stream_events(StreamShape::Complete)),
    );
    let streaming_hook = RecordingHook::default();
    let mut stream = AgentBuilder::new(streaming_model)
        .tool(MockAddTool)
        .build()
        .prompt("add 2 and 3")
        .max_turns(3)
        .add_hook(streaming_hook.clone())
        .add_hook(SkipToolCallHook("skipped by policy"))
        .stream();
    let mut final_response = None;
    while let Some(item) = stream.next().await {
        if let Ok(MultiTurnStreamItem::FinalResponse(resp)) =
            item.map_err(|err| panic!("stream item errored: {err}"))
        {
            final_response = Some(resp);
        }
    }
    let final_response = final_response.expect("stream should yield a final response");

    assert_eq!(blocking.output(), "acknowledged");
    assert_eq!(final_response.output(), blocking.output());
    assert_eq!(
        blocking_hook.shared_events(),
        streaming_hook.shared_events()
    );
    // A skipped valid tool call fires the `ToolResult` hook carrying a
    // structured `Skipped` outcome (the redesign surfaces the skip to result
    // hooks), so both drivers record the verbatim skip reason as the result.
    assert_eq!(blocking_hook.tool_results(), streaming_hook.tool_results());
    assert_eq!(
        blocking_hook.tool_results(),
        vec!["skipped by policy".to_string()],
        "a skipped tool fires a ToolResult hook with the verbatim skip reason"
    );

    let blocking_messages = blocking.messages;
    let streaming_messages = final_response.messages().to_vec();
    assert_eq!(
        serde_json::to_value(&blocking_messages).expect("serialize blocking"),
        serde_json::to_value(&streaming_messages).expect("serialize streaming"),
    );
    // Pin the actual reason, not just blocking == streaming: a reason dropped
    // or altered on BOTH paths would still satisfy the equality above.
    assert!(
        tool_result_text_in_history(&blocking_messages, "skipped by policy"),
        "the verbatim skip reason must be the tool result content in the history"
    );
}

/// A hook that rewrites a valid tool call's arguments (`DispatchAction::Patch` on
/// `ToolCall`) so the tool executes with the replacement instead of what the
/// model emitted.
struct RewriteToolArgsHook(serde_json::Value);

impl AgentHook for RewriteToolArgsHook {
    async fn on_dispatch(&self, _ctx: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        DispatchAction::rewrite_tool_args(event.kind, self.0.clone())
    }
}

// Distinct inert tools stand in for a destructive target and a safe target.
struct RenameBoundaryTool<const SAFE: bool>(Arc<AtomicU32>);

impl<const SAFE: bool> Tool for RenameBoundaryTool<SAFE> {
    const NAME: &'static str = if SAFE {
        "safe_target"
    } else {
        "original_target"
    };
    type Error = ToolExecutionError;
    type Args = serde_json::Value;
    type Output = String;

    fn description(&self) -> String {
        "Count dispatches to this target".into()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({"type": "object"})
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        _args: Self::Args,
    ) -> Result<String, Self::Error> {
        self.0.fetch_add(1, SeqCst);
        Ok(Self::NAME.into())
    }
}

struct RenameToolTargetHook;

impl AgentHook for RenameToolTargetHook {
    async fn on_dispatch(&self, _ctx: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        match event.kind {
            rig_core::effect::EffectKind::ToolCall { args, .. } => {
                DispatchAction::Patch(rig_core::effect::EffectKind::ToolCall {
                    name: RenameBoundaryTool::<true>::NAME.into(),
                    args: args.clone(),
                })
            }
            _ => DispatchAction::Proceed,
        }
    }
}

#[derive(Clone, Default)]
struct ToolTargetPolicySpy(Arc<AtomicU32>);
impl AgentHook for ToolTargetPolicySpy {
    async fn on_dispatch(&self, _: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        if event.tool_name().is_some() {
            self.0.fetch_add(1, SeqCst);
        }
        DispatchAction::Proceed
    }
}

async fn check_tool_target_patch_is_refused(streaming: bool) {
    let original = Arc::new(AtomicU32::new(0));
    let safe = Arc::new(AtomicU32::new(0));
    let turns = [
        ScriptedTurn::ToolCalls(vec![ScriptedToolCall {
            id: "rename-call",
            name: RenameBoundaryTool::<false>::NAME,
            args: json!({}),
        }]),
        ScriptedTurn::Text("done"),
    ];
    let model = if streaming {
        MockCompletionModel::from_stream_turns(
            turns
                .iter()
                .map(|turn| turn.as_stream_events(StreamShape::Complete)),
        )
    } else {
        MockCompletionModel::from_turns(turns.iter().map(ScriptedTurn::as_blocking_turn))
    };
    let observed = RecordingHook::default();
    let policy = ToolTargetPolicySpy::default();
    let runner = AgentBuilder::new(model)
        .tool(RenameBoundaryTool::<false>(original.clone()))
        .tool(RenameBoundaryTool::<true>(safe.clone()))
        .build()
        .prompt("perform the requested operation")
        .max_turns(3)
        .add_hook(RenameToolTargetHook)
        .add_hook(policy.clone())
        .add_hook(observed.clone());
    if streaming {
        let mut stream = runner.stream();
        let mut finished = false;
        while let Some(item) = stream.next().await {
            if let MultiTurnStreamItem::FinalResponse(response) =
                item.expect("stream remains usable")
            {
                assert_eq!(response.output(), "done");
                finished = true;
            }
        }
        assert!(finished);
    } else {
        assert_eq!(
            runner.run().await.expect("run remains usable").output(),
            "done"
        );
    }
    assert_eq!(
        original.load(SeqCst),
        0,
        "a renamed call must not execute the original tool"
    );
    assert_eq!(
        safe.load(SeqCst),
        0,
        "dispatch patches cannot authorize a new target"
    );
    assert_eq!(
        policy.0.load(SeqCst),
        0,
        "later policy must not observe an invalid target patch"
    );
    assert!(
        observed
            .tool_results()
            .iter()
            .any(|result| result.contains("target")),
        "the model must receive the refusal reason"
    );
}

#[tokio::test]
async fn tool_target_patch_is_refused_on_run() {
    check_tool_target_patch_is_refused(false).await;
}

#[tokio::test]
async fn tool_target_patch_is_refused_on_stream() {
    check_tool_target_patch_is_refused(true).await;
}

#[derive(serde::Deserialize)]
struct FirstGenerationArgs {
    old: String,
}

struct FirstGenerationTool(Arc<AtomicU32>);

impl Tool for FirstGenerationTool {
    const NAME: &'static str = "generation_pinned";
    type Error = rig::tool::ToolExecutionError;
    type Args = FirstGenerationArgs;
    type Output = String;

    fn description(&self) -> String {
        "first generation schema".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {"old": {"type": "string"}},
            "required": ["old"]
        })
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, ToolExecutionError> {
        self.0.fetch_add(1, SeqCst);
        Ok(format!("first:{}", args.old))
    }
}

/// Pauses the first provider call after its request has been built. Tests
/// replace the live registry while that request is in flight, then let the
/// script return a call that is valid only for the advertised generation.
#[derive(Clone)]
struct PausingScript {
    inner: rig_core::test_utils::MockRuntime,
    request_started: Arc<Notify>,
    release_response: Arc<Notify>,
    requests: Arc<AtomicU32>,
}

impl PausingScript {
    /// `inner`'s script behind the pause, as a model.
    fn model(
        inner: MockCompletionModel,
    ) -> (rig_core::Model<MockScript, Self>, Arc<Notify>, Arc<Notify>) {
        let request_started = Arc::new(Notify::new());
        let release_response = Arc::new(Notify::new());
        let script = Self {
            inner: inner.transport,
            request_started: request_started.clone(),
            release_response: release_response.clone(),
            requests: Arc::new(AtomicU32::new(0)),
        };
        (
            rig_core::Model::new(inner.wire, script),
            request_started,
            release_response,
        )
    }

    async fn inspect_and_pause(&self, request: &crate::completion::CompletionRequest) {
        let request_index = self.requests.fetch_add(1, SeqCst);
        let definition = request
            .tools
            .iter()
            .find(|definition| definition.name == FirstGenerationTool::NAME)
            .expect("generation tool must be advertised");
        if request_index == 0 {
            assert_eq!(definition.description, "first generation schema");
            self.request_started.notify_one();
            self.release_response.notified().await;
        } else {
            assert_eq!(definition.description, "second generation schema");
        }
    }
}

impl rig_core::driver::Transport<MockScript> for PausingScript {
    fn send(
        &self,
        payload: crate::completion::CompletionRequest,
        exchange: Exchange,
    ) -> Opening<rig_core::test_utils::MockFrame> {
        let this = self.clone();
        Opening::new(async move {
            this.inspect_and_pause(&payload).await;
            rig_core::driver::Transport::<MockScript>::send(&this.inner, payload, exchange).await
        })
    }
}

#[test]
fn one_hook_instance_attaches_to_distinct_completion_models() {
    #[derive(Clone)]
    struct ProviderIndependentHook;

    impl AgentHook for ProviderIndependentHook {}

    let hook = ProviderIndependentHook;
    let _mock_agent = AgentBuilder::new(MockCompletionModel::from_turns([]))
        .add_hook(hook.clone())
        .build();
    let (other_model, _, _) = PausingScript::model(MockCompletionModel::from_turns([]));
    let _other_agent = AgentBuilder::new(other_model).add_hook(hook).build();
}

/// A hook that rewrites a *valid* tool call's arguments (`DispatchAction::Patch`
/// on `ToolCall`) is honored identically under `run()` and `stream()`: the
/// tool executes with the replacement, so both drivers observe the same
/// rewritten tool result and reach the same output, tool-result content and
/// message history. Both drivers share `run_single_tool`, so they stay in
/// lock-step.
#[tokio::test]
async fn valid_tool_call_rewrite_args_parity_across_run_and_stream() {
    // The model asks to add 2 + 3; the hook rewrites the arguments to 2 + 40,
    // so the tool returns 42 rather than 5.
    let turns = [
        ScriptedTurn::ToolCalls(vec![add_call("tc1", 2, 3)]),
        ScriptedTurn::Text("acknowledged"),
    ];
    let replacement = json!({"x": 2, "y": 40});

    let blocking_model =
        MockCompletionModel::from_turns(turns.iter().map(ScriptedTurn::as_blocking_turn));
    let blocking_hook = RecordingHook::default();
    let blocking = AgentBuilder::new(blocking_model)
        .tool(MockAddTool)
        .build()
        .prompt("add 2 and 3")
        .max_turns(3)
        .add_hook(blocking_hook.clone())
        .add_hook(RewriteToolArgsHook(replacement.clone()))
        .run()
        .await
        .expect("blocking run should succeed with rewritten tool arguments");

    let streaming_model = MockCompletionModel::from_stream_turns(
        turns
            .iter()
            .map(|turn| turn.as_stream_events(StreamShape::Complete)),
    );
    let streaming_hook = RecordingHook::default();
    let mut stream = AgentBuilder::new(streaming_model)
        .tool(MockAddTool)
        .build()
        .prompt("add 2 and 3")
        .max_turns(3)
        .add_hook(streaming_hook.clone())
        .add_hook(RewriteToolArgsHook(replacement))
        .stream();
    let mut final_response = None;
    while let Some(item) = stream.next().await {
        if let Ok(MultiTurnStreamItem::FinalResponse(resp)) =
            item.map_err(|err| panic!("stream item errored: {err}"))
        {
            final_response = Some(resp);
        }
    }
    let final_response = final_response.expect("stream should yield a final response");

    // The tool ran with the rewritten arguments (2 + 40 = 42), not the
    // model's emitted 2 + 3 = 5 — on both drivers.
    assert_eq!(blocking_hook.tool_results(), vec!["42".to_string()]);
    assert_eq!(blocking.output(), "acknowledged");
    assert_eq!(final_response.output(), blocking.output());
    assert_eq!(
        blocking_hook.shared_events(),
        streaming_hook.shared_events()
    );
    assert_eq!(blocking_hook.tool_results(), streaming_hook.tool_results());
}

/// A hook that rewrites a tool's result (`OutcomeAction::rewrite_tool_result` on
/// the tool outcome) so the model sees the replacement instead of the tool's
/// actual output.
struct RewriteToolResultHook(&'static str);

impl AgentHook for RewriteToolResultHook {
    async fn on_outcome(&self, _ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        OutcomeAction::rewrite_tool_result(&event, self.0)
    }
}

/// A hook that rewrites a tool's result (`OutcomeAction::rewrite_tool_result` on
/// the tool outcome) is honored identically under `run()` and `stream()`: the
/// model-visible history carries the replacement while the `ToolResult` event
/// still observed the tool's actual output, and both drivers reach the same
/// output and history. Both share `run_single_tool`, so they stay in
/// lock-step.
#[tokio::test]
async fn valid_tool_result_rewrite_parity_across_run_and_stream() {
    // The tool computes 2 + 3 = 5; the hook replaces what the model sees with
    // "redacted-result".
    let turns = [
        ScriptedTurn::ToolCalls(vec![add_call("tc1", 2, 3)]),
        ScriptedTurn::Text("acknowledged"),
    ];

    let blocking_model =
        MockCompletionModel::from_turns(turns.iter().map(ScriptedTurn::as_blocking_turn));
    let blocking_hook = RecordingHook::default();
    let blocking = AgentBuilder::new(blocking_model)
        .tool(MockAddTool)
        .build()
        .prompt("add 2 and 3")
        .max_turns(3)
        .add_hook(blocking_hook.clone())
        .add_hook(RewriteToolResultHook("redacted-result"))
        .run()
        .await
        .expect("blocking run should succeed with a rewritten tool result");

    let streaming_model = MockCompletionModel::from_stream_turns(
        turns
            .iter()
            .map(|turn| turn.as_stream_events(StreamShape::Complete)),
    );
    let streaming_hook = RecordingHook::default();
    let mut stream = AgentBuilder::new(streaming_model)
        .tool(MockAddTool)
        .build()
        .prompt("add 2 and 3")
        .max_turns(3)
        .add_hook(streaming_hook.clone())
        .add_hook(RewriteToolResultHook("redacted-result"))
        .stream();
    let mut final_response = None;
    while let Some(item) = stream.next().await {
        if let Ok(MultiTurnStreamItem::FinalResponse(resp)) =
            item.map_err(|err| panic!("stream item errored: {err}"))
        {
            final_response = Some(resp);
        }
    }
    let final_response = final_response.expect("stream should yield a final response");

    assert_eq!(blocking.output(), "acknowledged");
    assert_eq!(final_response.output(), blocking.output());

    // The ToolResult event observes the tool's ACTUAL output (5) on both
    // drivers — the replacement is applied after the event fires.
    assert_eq!(blocking_hook.tool_results(), vec!["5".to_string()]);
    assert_eq!(blocking_hook.tool_results(), streaming_hook.tool_results());

    // The model-visible history carries the REWRITTEN result, not "5", and is
    // byte-identical across drivers.
    let blocking_messages = blocking.messages;
    let streaming_messages = final_response.messages().to_vec();
    assert_eq!(
        serde_json::to_value(&blocking_messages).expect("serialize blocking"),
        serde_json::to_value(&streaming_messages).expect("serialize streaming"),
    );
    assert!(
        tool_result_text_in_history(&blocking_messages, "redacted-result"),
        "the model-visible tool result must be the hook's replacement"
    );
    assert!(
        !tool_result_text_in_history(&blocking_messages, "5"),
        "the tool's original output must not reach the model after a rewrite"
    );
}

// --- Hook system v2: extra_context, history view, ModelTurnFinished, chained rewrites ---

fn hook_doc(id: &str, text: &str) -> crate::completion::Document {
    crate::completion::Document {
        id: id.to_string(),
        text: text.to_string(),
        additional_props: Default::default(),
    }
}

/// Injects one extra context document on every completion call.
struct ExtraContextHook {
    id: &'static str,
    text: &'static str,
}

impl AgentHook for ExtraContextHook {
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        if let CompletionCallEvent { .. } = event {
            CompletionCallAction::patch(RequestPatch::new().context(hook_doc(self.id, self.text)))
        } else {
            CompletionCallAction::continue_run()
        }
    }
}

#[derive(Clone)]
struct RecordingContextIndex {
    id: &'static str,
    queries: Arc<Mutex<Vec<(String, u64)>>>,
}

impl VectorStoreIndex for RecordingContextIndex {
    type Filter = Filter<serde_json::Value>;

    async fn top_n<T: for<'a> Deserialize<'a> + WasmCompatSend>(
        &self,
        req: VectorSearchRequest,
    ) -> Result<Vec<VectorSearchResult<T>>, VectorStoreError> {
        self.queries
            .lock()
            .expect("context query recorder lock")
            .push((req.query().to_string(), req.samples()));
        let value = serde_json::from_value(json!({ "source": self.id }))?;
        Ok(vec![VectorSearchResult {
            score: 1.0,
            id: self.id.to_string(),
            document: value,
        }])
    }

    async fn top_n_ids(
        &self,
        _req: VectorSearchRequest,
    ) -> Result<Vec<VectorSearchIdResult>, VectorStoreError> {
        Ok(vec![VectorSearchIdResult {
            score: 1.0,
            id: self.id.to_string(),
        }])
    }
}

struct FailingContextIndex;

impl VectorStoreIndex for FailingContextIndex {
    type Filter = Filter<serde_json::Value>;

    async fn top_n<T: for<'a> Deserialize<'a> + WasmCompatSend>(
        &self,
        _req: VectorSearchRequest,
    ) -> Result<Vec<VectorSearchResult<T>>, VectorStoreError> {
        Err(VectorStoreError::datastore(std::io::Error::other(
            "context index unavailable",
        )))
    }

    async fn top_n_ids(
        &self,
        _req: VectorSearchRequest,
    ) -> Result<Vec<VectorSearchIdResult>, VectorStoreError> {
        Err(VectorStoreError::datastore(std::io::Error::other(
            "context index unavailable",
        )))
    }
}

fn one_text_stream_turn(text: &'static str) -> Vec<MockStreamEvent> {
    vec![
        MockStreamEvent::text(text),
        MockStreamEvent::final_response_with_total_tokens(0),
    ]
}

use rig_core::test_utils::sent_documents;

/// A single hook's `extra_context` document appears in the completion request,
/// after the agent's static context, on both `run()` and `stream()`.
#[tokio::test]
async fn extra_context_appears_after_static_context_on_both_surfaces() {
    fn assert_docs(req: &crate::completion::CompletionRequest) {
        let documents = sent_documents(req);
        let ids: Vec<&str> = documents.iter().map(|(id, _)| id.as_str()).collect();
        let static_pos = ids
            .iter()
            .position(|id| id.starts_with("static_doc"))
            .expect("static context document present");
        let extra_pos = ids
            .iter()
            .position(|id| *id == "hook-doc")
            .expect("hook extra_context document present");
        assert!(
            static_pos < extra_pos,
            "static context precedes hook extras: {ids:?}"
        );
        assert!(
            documents.iter().any(|(_, text)| text == "injected"),
            "the hook document's text is present"
        );
    }

    let blocking_model = MockCompletionModel::from_turns([MockTurn::text("done")]);
    let blocking_probe = blocking_model.clone();
    AgentBuilder::new(blocking_model)
        .context("static context text")
        .add_hook(ExtraContextHook {
            id: "hook-doc",
            text: "injected",
        })
        .build()
        .prompt("go")
        .run()
        .await
        .expect("blocking run should succeed");
    assert_docs(blocking_probe.requests().first().expect("one request"));

    let streaming_model = MockCompletionModel::from_stream_turns([one_text_stream_turn("done")]);
    let streaming_probe = streaming_model.clone();
    let mut stream = AgentBuilder::new(streaming_model)
        .context("static context text")
        .add_hook(ExtraContextHook {
            id: "hook-doc",
            text: "injected",
        })
        .build()
        .prompt("go")
        .stream();
    while let Some(item) = stream.next().await {
        let _ = item.map_err(|err| panic!("stream item errored: {err}"));
    }
    assert_docs(streaming_probe.requests().first().expect("one request"));
}

#[tokio::test]
async fn dynamic_context_preserves_query_selection_formatting_and_order_on_both_surfaces() {
    fn assert_documents(request: &crate::completion::CompletionRequest) {
        let sent = sent_documents(request);
        let documents = sent
            .iter()
            .map(|(id, text)| (id.as_str(), text.as_str()))
            .collect::<Vec<_>>();
        assert_eq!(
            documents,
            vec![
                ("static_doc_0", "static context"),
                ("blocking", "{\n  \"source\": \"blocking\"\n}"),
            ]
        );
    }

    let blocking_queries = Arc::new(Mutex::new(Vec::new()));
    let blocking_model = MockCompletionModel::from_turns([MockTurn::text("done")]);
    let blocking_probe = blocking_model.clone();
    AgentBuilder::new(blocking_model)
        .context("static context")
        .dynamic_context(
            2,
            RecordingContextIndex {
                id: "blocking",
                queries: blocking_queries.clone(),
            },
        )
        .build()
        .prompt("current blocking query")
        .history(vec![Message::user("ignored history query")])
        .run()
        .await
        .expect("blocking dynamic-context run should succeed");
    assert_eq!(
        *blocking_queries.lock().expect("blocking queries"),
        vec![("current blocking query".to_string(), 2)]
    );
    assert_documents(blocking_probe.requests().first().expect("one request"));

    let streaming_queries = Arc::new(Mutex::new(Vec::new()));
    let streaming_model = MockCompletionModel::from_stream_turns([one_text_stream_turn("done")]);
    let streaming_probe = streaming_model.clone();
    let mut stream = AgentBuilder::new(streaming_model)
        .dynamic_context(
            3,
            RecordingContextIndex {
                id: "streaming",
                queries: streaming_queries.clone(),
            },
        )
        .build()
        .prompt(Message::User {
            content: vec![UserContent::image_url(
                "https://example.com/prompt.png",
                None,
                None,
            )],
        })
        .history(vec![
            Message::user("older history query"),
            Message::user("latest history query"),
        ])
        .stream();
    while let Some(item) = stream.next().await {
        item.expect("streaming dynamic-context run should succeed");
    }
    assert_eq!(
        *streaming_queries.lock().expect("streaming queries"),
        vec![("latest history query".to_string(), 3)]
    );
    let streaming_requests = streaming_probe.requests();
    let request = streaming_requests.first().expect("one request");
    assert_eq!(
        sent_documents(request),
        [(
            "streaming".to_owned(),
            "{\n  \"source\": \"streaming\"\n}".to_owned()
        )]
    );
}

#[tokio::test]
async fn dynamic_context_and_application_hooks_follow_registration_order() {
    let queries = Arc::new(Mutex::new(Vec::new()));
    let model = MockCompletionModel::from_turns([MockTurn::text("done")]);
    let probe = model.clone();
    AgentBuilder::new(model)
        .context("static")
        .add_hook(ExtraContextHook {
            id: "before",
            text: "before dynamic context",
        })
        .dynamic_context(
            1,
            RecordingContextIndex {
                id: "first",
                queries: queries.clone(),
            },
        )
        .add_hook(ExtraContextHook {
            id: "between",
            text: "between dynamic contexts",
        })
        .dynamic_context(
            2,
            RecordingContextIndex {
                id: "second",
                queries: queries.clone(),
            },
        )
        .add_hook(ExtraContextHook {
            id: "after",
            text: "after dynamic context",
        })
        .build()
        .prompt("query")
        .run()
        .await
        .expect("run should succeed");

    assert_eq!(
        sent_documents(&probe.requests()[0])
            .into_iter()
            .map(|(id, _)| id)
            .collect::<Vec<_>>(),
        vec![
            "static_doc_0",
            "before",
            "first",
            "between",
            "second",
            "after",
        ]
    );
    assert_eq!(
        *queries.lock().expect("context queries"),
        vec![("query".to_string(), 1), ("query".to_string(), 2)]
    );

    let skipped_queries = Arc::new(Mutex::new(Vec::new()));
    let error = AgentBuilder::new(MockCompletionModel::from_turns([MockTurn::text("unused")]))
        .add_hook(TerminateOn(TerminatePoint::CompletionCall))
        .dynamic_context(
            1,
            RecordingContextIndex {
                id: "skipped",
                queries: skipped_queries.clone(),
            },
        )
        .build()
        .prompt("query")
        .run()
        .await
        .expect_err("an earlier stop hook should terminate before retrieval");
    assert!(matches!(error, PromptError::Cancelled { .. }));
    assert!(skipped_queries.lock().expect("skipped queries").is_empty());
}

#[tokio::test]
async fn dynamic_context_retrieval_failure_stops_before_provider_io_on_both_surfaces() {
    let blocking_model = MockCompletionModel::from_turns([MockTurn::text("unused")]);
    let blocking_probe = blocking_model.clone();
    let error = AgentBuilder::new(blocking_model)
        .dynamic_context(1, FailingContextIndex)
        .build()
        .prompt("retrieve this")
        .run()
        .await
        .expect_err("failed retrieval should stop the run");
    assert!(matches!(
        error,
        PromptError::Cancelled { reason, .. }
            if reason.contains("context index unavailable")
    ));
    assert_eq!(blocking_probe.request_count(), 0);

    let streaming_model = MockCompletionModel::from_stream_turns([one_text_stream_turn("unused")]);
    let streaming_probe = streaming_model.clone();
    let mut stream = AgentBuilder::new(streaming_model)
        .dynamic_context(1, FailingContextIndex)
        .build()
        .prompt("retrieve this")
        .stream();
    let error = stream
        .next()
        .await
        .expect("stream should report retrieval failure")
        .expect_err("failed retrieval should stop the stream");
    assert!(matches!(
        error,
        PromptError::Cancelled { reason, .. } if reason.contains("context index unavailable")
    ));
    assert_eq!(streaming_probe.request_count(), 0);
}

/// A hook that overrides `history` changes the messages sent to the provider
/// for the turn without touching the persisted transcript, on both surfaces.
#[tokio::test]
async fn history_patch_changes_sent_messages_not_transcript_on_both_surfaces() {
    const SENTINEL: &str = "COMPACTED-HISTORY-SENTINEL";

    struct HistoryOverrideHook;
    impl AgentHook for HistoryOverrideHook {
        async fn on_completion_call(
            &self,
            _ctx: &HookContext,
            event: CompletionCallEvent<'_>,
        ) -> CompletionCallAction {
            if let CompletionCallEvent { .. } = event {
                CompletionCallAction::patch(RequestPatch::new().history([Message::user(SENTINEL)]))
            } else {
                CompletionCallAction::continue_run()
            }
        }
    }

    fn request_has_sentinel(req: &crate::completion::CompletionRequest) -> bool {
        req.chat_history.iter().any(|m| match m {
            Message::User { content } => content
                .iter()
                .any(|c| matches!(c, UserContent::Text(text) if text.text.contains(SENTINEL))),
            _ => false,
        })
    }

    fn messages_have_sentinel(messages: &[Message]) -> bool {
        messages.iter().any(|m| match m {
            Message::User { content } => content
                .iter()
                .any(|c| matches!(c, UserContent::Text(text) if text.text.contains(SENTINEL))),
            _ => false,
        })
    }

    let blocking_model = MockCompletionModel::from_turns([MockTurn::text("done")]);
    let blocking_probe = blocking_model.clone();
    let blocking = AgentBuilder::new(blocking_model)
        .add_hook(HistoryOverrideHook)
        .build()
        .prompt("real prompt")
        .run()
        .await
        .expect("blocking run should succeed");
    assert!(
        request_has_sentinel(blocking_probe.requests().first().expect("one request")),
        "the overridden history reaches the provider"
    );
    assert!(
        !messages_have_sentinel(&blocking.messages),
        "the persisted transcript is untouched by the per-turn history override"
    );

    let streaming_model = MockCompletionModel::from_stream_turns([one_text_stream_turn("done")]);
    let streaming_probe = streaming_model.clone();
    let stream = AgentBuilder::new(streaming_model)
        .add_hook(HistoryOverrideHook)
        .build()
        .prompt("real prompt")
        .stream();
    let final_response = drive_to_final_response(stream).await;
    assert!(
        request_has_sentinel(streaming_probe.requests().first().expect("one request")),
        "the overridden history reaches the provider on the streaming surface too"
    );
    assert!(
        !messages_have_sentinel(final_response.messages()),
        "the persisted transcript is untouched by the per-turn history override on \
             the streaming surface too"
    );
}

/// `DispatchAction::Patch` and `OutcomeAction::Replace` chain across hooks: a later hook observes
/// (and further rewrites) the value produced by earlier hooks.
#[tokio::test]
async fn chained_rewrites_compose_across_hooks() {
    /// Sets one key of the tool arguments, preserving the rest.
    struct SetArg {
        key: &'static str,
        value: i64,
    }
    impl AgentHook for SetArg {
        async fn on_dispatch(
            &self,
            _ctx: &HookContext,
            event: DispatchEvent<'_>,
        ) -> DispatchAction {
            if let Some(args) = event.tool_args() {
                let mut parsed: serde_json::Value =
                    serde_json::from_str(args).unwrap_or_else(|_| json!({}));
                parsed[self.key] = json!(self.value);
                DispatchAction::rewrite_tool_args(event.kind, parsed)
            } else {
                DispatchAction::proceed()
            }
        }
    }

    /// Wraps the tool result in `label(...)`.
    struct WrapResult(&'static str);
    impl AgentHook for WrapResult {
        async fn on_outcome(&self, _ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
            if let Some(result) = event.tool_result() {
                OutcomeAction::rewrite_tool_result(
                    &event,
                    format!("{}({})", self.0, result.output().render()),
                )
            } else {
                OutcomeAction::proceed()
            }
        }
    }

    // The model asks add(2, 3). SetArg{y:40} then SetArg{x:100} chain, so the
    // tool runs with (100, 40) = 140 — proving arg rewrites compose. Then
    // WrapResult "A" and "B" chain, and a trailing recorder observes the fully
    // chained result "B(A(140))".
    let recorder = RecordingHook::default();
    let blocking = AgentBuilder::new(blocking_model())
        .tool(MockAddTool)
        .add_hook(SetArg {
            key: "y",
            value: 40,
        })
        .add_hook(SetArg {
            key: "x",
            value: 100,
        })
        .add_hook(WrapResult("A"))
        .add_hook(WrapResult("B"))
        .add_hook(recorder.clone())
        .build()
        .prompt("add 2 and 3")
        .max_turns(3)
        .run()
        .await
        .expect("blocking run should succeed");
    assert_eq!(blocking.output(), "the answer is 5");
    assert_eq!(
        recorder.tool_results(),
        vec!["B(A(140))".to_string()],
        "arg rewrites compose (100+40=140) and result rewrites nest B(A(...))"
    );

    // Same on the streaming surface.
    let stream_recorder = RecordingHook::default();
    let mut stream = AgentBuilder::new(streaming_model())
        .tool(MockAddTool)
        .add_hook(SetArg {
            key: "y",
            value: 40,
        })
        .add_hook(SetArg {
            key: "x",
            value: 100,
        })
        .add_hook(WrapResult("A"))
        .add_hook(WrapResult("B"))
        .add_hook(stream_recorder.clone())
        .build()
        .prompt("add 2 and 3")
        .max_turns(3)
        .stream();
    while let Some(item) = stream.next().await {
        let _ = item.map_err(|err| panic!("stream item errored: {err}"));
    }
    assert_eq!(
        stream_recorder.tool_results(),
        vec!["B(A(140))".to_string()],
        "chained rewrites compose identically on the streaming surface"
    );
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
#[allow(dead_code)]
struct Answer {
    answer: String,
}

/// A real tool whose name equals the default synthetic output-tool name
/// (`final_result`). Used to prove a per-turn `active_tools` filter cannot
/// make the picked output-tool name collide with it.
struct FinalResultTool;

impl Tool for FinalResultTool {
    const NAME: &'static str = "final_result";
    type Error = MockToolError;
    type Args = serde_json::Value;
    type Output = String;

    fn description(&self) -> String {
        "A real tool sharing the default output-tool name".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({ "type": "object", "properties": {} })
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok("real final_result output".to_string())
    }
}

/// Registers a real `final_result` tool after the first model turn, once the
/// run has already reserved that name for structured output. An optional
/// second-turn patch lets tests exercise filtering and tool-choice changes
/// without changing the collision source.
#[derive(Clone)]
struct RegisterLateFinalResultTool {
    handle: ToolServerHandle,
    second_turn_patch: Option<RequestPatch>,
}

impl AgentHook for RegisterLateFinalResultTool {
    async fn on_model_turn_finished(
        &self,
        ctx: &HookContext,
        _event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        if ctx.turn() == 1 {
            self.handle.add_tool(FinalResultTool);
        }

        ModelTurnAction::continue_run()
    }

    async fn on_completion_call(
        &self,
        ctx: &HookContext,
        _event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        if ctx.turn() == 2
            && let Some(patch) = &self.second_turn_patch
        {
            return CompletionCallAction::patch(patch.clone());
        }

        CompletionCallAction::continue_run()
    }
}

fn assert_structured_output_collision_error(message: &str) {
    assert!(
        message.contains("final_result"),
        "error should name the conflicting tool: {message}"
    );
    assert!(
        message.contains("structured-output") && message.contains("reserved"),
        "error should explain the structured-output reservation: {message}"
    );
    assert!(
        message.contains("rename or remove"),
        "error should provide an actionable resolution: {message}"
    );
}

/// A late colliding tool is harmless while `active_tools` filters it out,
/// but the run must fail as soon as the non-sticky filter lifts and the real
/// tool becomes effective again.
#[tokio::test]
async fn late_output_tool_collision_is_checked_after_active_tools_filtering() {
    let handle = ToolServer::new().tool(MockAddTool).run();
    let model = MockCompletionModel::from_turns([
        MockTurn::tool_call("add-1", "add", json!({ "x": 1, "y": 2 })),
        MockTurn::tool_call("add-2", "add", json!({ "x": 3, "y": 4 })),
        MockTurn::tool_call(
            "shadowed",
            "final_result",
            json!({ "answer": "wrongly finalized" }),
        ),
    ]);
    let probe = model.clone();
    let err = AgentBuilder::new(model)
        .tool_server_handle(handle.clone())
        .output_schema::<Answer>()
        .output_mode(OutputMode::Tool)
        .add_hook(RegisterLateFinalResultTool {
            handle,
            second_turn_patch: Some(RequestPatch::new().active_tools(["add"])),
        })
        .build()
        .prompt("go")
        .max_turns(4)
        .run()
        .await
        .expect_err("the exposed third-turn collision should fail locally");

    assert_eq!(
        probe.request_count(),
        2,
        "the filtered second turn may run, but the exposed third turn may not"
    );
    let requests = probe.requests();
    let second_turn_names = requests[1]
        .tools
        .iter()
        .map(|tool| tool.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(second_turn_names.len(), 2);
    for expected in ["add", "final_result"] {
        assert_eq!(
            second_turn_names
                .iter()
                .filter(|name| **name == expected)
                .count(),
            1,
            "the second request should advertise `{expected}` exactly once: \
                 {second_turn_names:?}"
        );
    }
    assert_structured_output_collision_error(&err.to_string());
}

/// Narrows the advertised tools to `add` for the turn, filtering out the real
/// `final_result` tool.
struct ActiveToolsAddOnly;

impl AgentHook for ActiveToolsAddOnly {
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        if let CompletionCallEvent { .. } = event {
            CompletionCallAction::patch(RequestPatch::new().active_tools(["add"]))
        } else {
            CompletionCallAction::continue_run()
        }
    }
}

/// Regression guard: a per-turn `active_tools` allow-list that filters out a
/// real tool whose name equals the default synthetic output-tool name must not
/// let the picked output-tool name collide with that (filtered) real tool. The
/// name is pinned for the whole run, so picking it against the FULL advertised
/// set — not just this turn's narrowed executable set — keeps it collision-safe
/// once the filter lifts on a later turn. With the bug, the output tool would
/// be named `final_result` (picked against the narrowed `{add}`), colliding
/// with the real `final_result` whenever the filter is gone.
#[tokio::test]
async fn active_tools_filter_does_not_let_output_tool_collide_with_a_filtered_real_tool() {
    // The model finalizes by calling the (correctly-picked) output tool, so a
    // run on the fixed code completes cleanly in a single turn. Asserting the
    // run succeeds also exercises finalization: the model's call to
    // `final_result_1` must be intercepted as the output tool, so this fails if
    // the picked name and the intercept name ever drift apart.
    let model = MockCompletionModel::from_turns([MockTurn::tool_call(
        "out1",
        "final_result_1",
        json!({ "answer": "done" }),
    )]);
    let probe = model.clone();
    let response = AgentBuilder::new(model)
        .tool(MockAddTool)
        .tool(FinalResultTool)
        .output_schema::<Answer>()
        .output_mode(OutputMode::Tool)
        .add_hook(ActiveToolsAddOnly)
        .build()
        .prompt("go")
        .max_turns(2)
        .run()
        .await
        .expect("run should finalize via the picked output tool `final_result_1`");
    assert!(
        response.output().contains("done"),
        "the intercepted output-tool call should produce the structured result, \
             got {:?}",
        response.output()
    );

    let requests = probe.requests();
    assert!(
        !requests.is_empty(),
        "the first model request should be captured"
    );
    let tool_names: Vec<&str> = requests[0].tools.iter().map(|t| t.name.as_str()).collect();
    assert!(
        tool_names.contains(&"add"),
        "active_tools keeps `add` advertised, saw {tool_names:?}"
    );
    assert!(
        tool_names.contains(&"final_result_1"),
        "the synthetic output tool must avoid the filtered real `final_result` name, \
             saw {tool_names:?}"
    );
    assert!(
        !tool_names.contains(&"final_result"),
        "the real `final_result` is filtered out and the output tool must not reuse \
             its name, saw {tool_names:?}"
    );
}

/// A structured-output Tool-mode output-tool call finalizes the run directly, so
/// on the streaming surface it is **not** re-emitted as a complete
/// `MultiTurnStreamItem::ToolCall` item (it bypasses
/// `drive_tool_calls`); its structured result is surfaced in the final `PromptResponse`.
/// Guards the narrowed `StreamAssistantItem` contract.
#[tokio::test]
async fn output_tool_finalization_emits_no_complete_tool_call_stream_item() {
    let mut stream = AgentBuilder::new(MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::tool_call("out1", "final_result", json!({ "answer": "done" })),
        MockStreamEvent::final_response_with_total_tokens(0),
    ]]))
    .output_schema::<Answer>()
    .output_mode(OutputMode::Tool)
    .build()
    .prompt("go")
    .max_turns(2)
    .stream();

    let mut saw_complete_output_tool_call = false;
    let mut final_has_output = false;
    while let Some(item) = stream.next().await {
        match item.expect("stream item") {
            MultiTurnStreamItem::ToolCall { tool_call, .. }
                if tool_call.function.name == "final_result" =>
            {
                saw_complete_output_tool_call = true;
            }
            MultiTurnStreamItem::FinalResponse(res) => {
                final_has_output = res.output().contains("done");
            }
            _ => {}
        }
    }
    assert!(
        !saw_complete_output_tool_call,
        "the output-tool call finalizes the run, so no complete \
             StreamAssistantItem::ToolCall item must be emitted for it"
    );
    assert!(
        final_has_output,
        "the structured output must be surfaced via the FinalResponse"
    );
}

// -----------------------------------------------------------------------
// Human-in-the-loop (HITL): one hook gates each tool call behind a human
// decision, mapping approve/deny/edit/abort onto the event-specific actions
// (cont / skip / rewrite_args / terminate). The runnable interactive
// version lives in `examples/agent_with_human_in_the_loop`.
// -----------------------------------------------------------------------

/// A human reviewer's decision for a pending tool call.
enum Decision {
    /// Run the tool as the model requested.
    Approve,
    /// Don't run the tool; feed `reason` back to the model as the result.
    Deny(&'static str),
    /// Run the tool with these arguments instead of the model's.
    Edit(serde_json::Value),
    /// Abort the whole run with this reason.
    Abort(&'static str),
}

/// Simulates a human reviewer by popping a scripted decision per `ToolCall`
/// and mapping it to the matching event-specific action. A real reviewer would `.await`
/// interactive input here (the hook is async) rather than read a queue.
#[derive(Clone)]
struct HumanApprovalHook {
    decisions: Arc<Mutex<std::collections::VecDeque<Decision>>>,
    reviewed: Arc<Mutex<Vec<String>>>,
}

impl HumanApprovalHook {
    fn new(decisions: impl IntoIterator<Item = Decision>) -> Self {
        Self {
            decisions: Arc::new(Mutex::new(decisions.into_iter().collect())),
            reviewed: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// `"name(args)"` for each call presented for review, in order.
    fn reviewed(&self) -> Vec<String> {
        self.reviewed.lock().unwrap().clone()
    }
}

impl AgentHook for HumanApprovalHook {
    async fn on_dispatch(&self, _ctx: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        let (Some(tool_name), Some(args)) = (event.tool_name(), event.tool_args()) else {
            return DispatchAction::proceed();
        };
        self.reviewed
            .lock()
            .unwrap()
            .push(format!("{tool_name}({args})"));
        let decision = self.decisions.lock().unwrap().pop_front();
        match decision {
            Some(Decision::Approve) => DispatchAction::proceed(),
            Some(Decision::Deny(reason)) => DispatchAction::skip(reason),
            Some(Decision::Edit(args)) => DispatchAction::rewrite_tool_args(event.kind, args),
            Some(Decision::Abort(reason)) => DispatchAction::stop(reason),
            // Fail closed if the script is exhausted (it shouldn't be) — deny
            // rather than silently approve, matching the example's contract.
            None => DispatchAction::skip("denied: no scripted decision (fail-closed)"),
        }
    }
}

/// A HITL hook that approves the first tool call, denies the second, and
/// edits the third's arguments behaves identically under `run()` and
/// `stream()`: approved/edited tools execute (and the edit takes effect),
/// the denied tool runs nothing while its reason reaches the model, and the
/// model-visible history is identical across drivers (compared structurally).
#[tokio::test]
async fn human_in_the_loop_approve_deny_edit_parity_across_run_and_stream() {
    // One turn issues three tool calls; the reviewer decides each differently.
    let turns = [
        ScriptedTurn::ToolCalls(vec![
            add_call("tc1", 2, 3),   // approve -> runs, 2 + 3 = 5
            add_call("tc2", 10, 20), // deny    -> skipped; model sees the reason
            add_call("tc3", 1, 1),   // edit    -> runs 1 + 100 = 101, not 1 + 1 = 2
        ]),
        ScriptedTurn::Text("done"),
    ];
    let denial = "denied by reviewer: amount too large";
    let decisions = || {
        vec![
            Decision::Approve,
            Decision::Deny(denial),
            Decision::Edit(json!({"x": 1, "y": 100})),
        ]
    };

    let blocking_model =
        MockCompletionModel::from_turns(turns.iter().map(ScriptedTurn::as_blocking_turn));
    let blocking_recorder = RecordingHook::default();
    let blocking_approver = HumanApprovalHook::new(decisions());
    let blocking = AgentBuilder::new(blocking_model)
        .tool(MockAddTool)
        .build()
        .prompt("carry out the plan")
        .max_turns(3)
        .add_hook(blocking_recorder.clone())
        .add_hook(blocking_approver.clone())
        .run()
        .await
        .expect("blocking HITL run should succeed");

    let streaming_model = MockCompletionModel::from_stream_turns(
        turns
            .iter()
            .map(|turn| turn.as_stream_events(StreamShape::Complete)),
    );
    let streaming_recorder = RecordingHook::default();
    let streaming_approver = HumanApprovalHook::new(decisions());
    let mut stream = AgentBuilder::new(streaming_model)
        .tool(MockAddTool)
        .build()
        .prompt("carry out the plan")
        .max_turns(3)
        .add_hook(streaming_recorder.clone())
        .add_hook(streaming_approver.clone())
        .stream();
    let mut final_response = None;
    while let Some(item) = stream.next().await {
        if let Ok(MultiTurnStreamItem::FinalResponse(resp)) =
            item.map_err(|err| panic!("stream item errored: {err}"))
        {
            final_response = Some(resp);
        }
    }
    let final_response = final_response.expect("stream should yield a final response");

    // Approved (5) and edited (101) tools executed, in call order; the denied
    // call executed nothing but now fires a ToolResult carrying its verbatim
    // denial reason (structured `Skipped` outcome) — identically on both
    // drivers.
    assert_eq!(
        blocking_recorder.tool_results(),
        vec![
            "5".to_string(),
            "denied by reviewer: amount too large".to_string(),
            "101".to_string()
        ]
    );
    assert_eq!(
        blocking_recorder.tool_results(),
        streaming_recorder.tool_results()
    );

    // The denied call (10 + 20) never executed, so its result 30 is absent —
    // the denial reason stands in its place, ruling out deny being silently
    // treated as approve.
    assert!(
        !blocking_recorder.tool_results().contains(&"30".to_string()),
        "the denied call must not have executed"
    );

    // The reviewer was consulted for all three calls, in order, identically per
    // driver — pinning each decision to its call (approve=2+3, deny=10+20,
    // edit=the third).
    let reviewed = blocking_approver.reviewed();
    assert_eq!(reviewed.len(), 3);
    assert_eq!(reviewed, streaming_approver.reviewed());
    assert!(
        reviewed[0].contains('2') && reviewed[0].contains('3'),
        "first reviewed call should be add(2, 3): {reviewed:?}"
    );
    assert!(
        reviewed[1].contains("10") && reviewed[1].contains("20"),
        "the denied (second) call should be add(10, 20): {reviewed:?}"
    );

    assert_eq!(blocking.output(), "done");
    assert_eq!(final_response.output(), blocking.output());
    assert_eq!(
        blocking_recorder.shared_events(),
        streaming_recorder.shared_events()
    );

    // Model-visible history is identical across drivers (compared structurally
    // as serde_json::Value) and carries the denial reason and the edited result
    // 101 (not the model's 1 + 1 = 2).
    let blocking_messages = blocking.messages;
    let streaming_messages = final_response.messages().to_vec();
    assert_eq!(
        serde_json::to_value(&blocking_messages).expect("serialize blocking"),
        serde_json::to_value(&streaming_messages).expect("serialize streaming"),
    );
    assert!(
        tool_result_text_in_history(&blocking_messages, denial),
        "the denial reason must be the denied call's tool result in the history"
    );
    assert!(
        tool_result_json_in_history(&blocking_messages, &json!(101)),
        "the edited call must have executed with the rewritten arguments"
    );
}

/// A HITL hook that aborts a tool call (`Decision::Abort` -> `DispatchAction::stop`)
/// stops the run and surfaces the reason as a `Cancelled` error — on both
/// the blocking and streaming drivers.
#[tokio::test]
async fn human_in_the_loop_abort_terminates_the_run() {
    let turns = [
        ScriptedTurn::ToolCalls(vec![add_call("tc1", 2, 3)]),
        ScriptedTurn::Text("unreachable"),
    ];
    const ABORT_REASON: &str = "aborted by the human reviewer";

    // Blocking driver: the run resolves to a Cancelled error.
    let blocking_model =
        MockCompletionModel::from_turns(turns.iter().map(ScriptedTurn::as_blocking_turn));
    let err = AgentBuilder::new(blocking_model)
        .tool(MockAddTool)
        .build()
        .prompt("do the sensitive thing")
        .max_turns(3)
        .add_hook(HumanApprovalHook::new([Decision::Abort(ABORT_REASON)]))
        .run()
        .await
        .expect_err("an aborted tool call should terminate the blocking run");
    assert!(
        format!("{err}").contains(ABORT_REASON),
        "the abort reason should surface in the blocking error, got: {err}"
    );

    // Streaming driver: the stream yields an error carrying the same reason and
    // never reaches the "unreachable" final text.
    let streaming_model = MockCompletionModel::from_stream_turns(
        turns
            .iter()
            .map(|turn| turn.as_stream_events(StreamShape::Complete)),
    );
    let mut stream = AgentBuilder::new(streaming_model)
        .tool(MockAddTool)
        .build()
        .prompt("do the sensitive thing")
        .max_turns(3)
        .add_hook(HumanApprovalHook::new([Decision::Abort(ABORT_REASON)]))
        .stream();
    let mut stream_error = None;
    while let Some(item) = stream.next().await {
        match item {
            Err(err) => stream_error = Some(format!("{err}")),
            Ok(MultiTurnStreamItem::FinalResponse(resp)) => {
                panic!("aborted stream must not finalize, got: {}", resp.output())
            }
            Ok(_) => {}
        }
    }
    let stream_error = stream_error.expect("an aborted tool call should error the stream");
    assert!(
        stream_error.contains(ABORT_REASON),
        "the abort reason should surface in the streaming error, got: {stream_error}"
    );
}

/// A non-interactive *policy* HITL hook: auto-approve an allow-list, deny
/// everything else (fail-closed), and cache each decision so a repeated tool
/// is not re-evaluated ("sticky", like the OpenAI Agents SDK's
/// `always_approve`). Backs `examples/agent_with_approval_policy`.
#[derive(Clone)]
struct PolicyHook {
    auto_approve: std::collections::HashSet<&'static str>,
    /// Tool names the policy actually evaluated (cache misses), in order.
    evaluated: Arc<Mutex<Vec<String>>>,
    /// Sticky cache of prior decisions, keyed by tool name.
    cache: Arc<Mutex<std::collections::HashMap<String, bool>>>,
}

impl PolicyHook {
    fn new(auto_approve: impl IntoIterator<Item = &'static str>) -> Self {
        Self {
            auto_approve: auto_approve.into_iter().collect(),
            evaluated: Arc::new(Mutex::new(Vec::new())),
            cache: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }

    fn evaluated(&self) -> Vec<String> {
        self.evaluated.lock().unwrap().clone()
    }
}

impl AgentHook for PolicyHook {
    async fn on_dispatch(&self, _ctx: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        let Some(tool_name) = event.tool_name() else {
            return DispatchAction::proceed();
        };
        let cached = self.cache.lock().unwrap().get(tool_name).copied();
        let approved = match cached {
            Some(decision) => decision, // sticky: reuse without re-evaluating
            None => {
                self.evaluated.lock().unwrap().push(tool_name.to_string());
                let decision = self.auto_approve.contains(tool_name);
                self.cache
                    .lock()
                    .unwrap()
                    .insert(tool_name.to_string(), decision);
                decision
            }
        };
        if approved {
            DispatchAction::proceed()
        } else {
            DispatchAction::skip(format!("denied by policy: `{tool_name}` not allowed"))
        }
    }
}

/// The policy hook auto-approves `add` and denies `subtract`, and its decision
/// is sticky: a second `add` call reuses the cached approval instead of being
/// re-evaluated. The denied call never runs and its reason reaches the model.
#[tokio::test]
async fn approval_policy_allow_list_with_sticky_decisions() {
    // One turn issues three calls: add, subtract (denied), add again (sticky).
    let turns = [
        ScriptedTurn::ToolCalls(vec![
            add_call("c1", 2, 3),
            ScriptedToolCall {
                id: "c2",
                name: "subtract",
                args: json!({ "x": 10, "y": 4 }),
            },
            add_call("c3", 2, 3),
        ]),
        ScriptedTurn::Text("done"),
    ];

    let model = MockCompletionModel::from_turns(turns.iter().map(ScriptedTurn::as_blocking_turn));
    let recorder = RecordingHook::default();
    let policy = PolicyHook::new(["add"]);
    let out = AgentBuilder::new(model)
        .tool(MockAddTool)
        .tool(MockSubtractTool)
        .build()
        .prompt("go")
        .max_turns(3)
        .add_hook(recorder.clone())
        .add_hook(policy.clone())
        .run()
        .await
        .expect("policy run should succeed");

    assert_eq!(out.output(), "done");
    // `add` ran twice (auto-approved, then sticky-reused); `subtract` was denied
    // and executed nothing, but its denial reason now surfaces as a ToolResult
    // (structured `Skipped` outcome) between the two `add` results.
    assert_eq!(
        recorder.tool_results(),
        vec![
            "5".to_string(),
            "denied by policy: `subtract` not allowed".to_string(),
            "5".to_string()
        ]
    );
    // The policy evaluated each distinct tool once; the second `add` reused the
    // cached decision rather than being re-evaluated.
    assert_eq!(
        policy.evaluated(),
        vec!["add".to_string(), "subtract".to_string()]
    );
    let messages = out.messages;
    assert!(
        tool_result_text_in_history(&messages, "denied by policy: `subtract` not allowed"),
        "the policy denial reason must reach the model as the subtract tool result"
    );
}

static NEXT_RESPONSE_RETRY_HOOK_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Default)]
struct ResponseRetryAttempts(HashMap<u64, usize>);

#[derive(Clone)]
enum TestRetryMode {
    Repeat,
    Feedback(&'static str),
}

/// A policy-owned retry budget. The framework only enforces `max_turns`;
/// this hook stores its narrower limit in the run-scoped scratchpad.
#[derive(Clone)]
struct BoundedResponseRetry {
    id: u64,
    rejected_text: &'static str,
    max_retries: usize,
    mode: TestRetryMode,
}

impl BoundedResponseRetry {
    fn new(rejected_text: &'static str, max_retries: usize, mode: TestRetryMode) -> Self {
        Self {
            id: NEXT_RESPONSE_RETRY_HOOK_ID.fetch_add(1, SeqCst),
            rejected_text,
            max_retries,
            mode,
        }
    }
}

impl AgentHook for BoundedResponseRetry {
    async fn on_model_turn_finished(
        &self,
        ctx: &HookContext,
        event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        let has_rejected_text = event.content.iter().any(
            |content| matches!(content, AssistantContent::Text(text) if text.text == self.rejected_text),
        );
        // A hook watching for the empty response has to recognise both
        // spellings of it. Blocking turns carry an explicit empty text part;
        // a stream that produced nothing now carries no parts at all, where
        // it used to be padded with a fabricated empty-text part that made
        // the two look alike.
        let rejected =
            has_rejected_text || (self.rejected_text.is_empty() && event.content.is_empty());
        if !rejected {
            return ModelTurnAction::continue_run();
        }

        let attempt = ctx
            .scratchpad()
            .update::<ResponseRetryAttempts, _>(|attempts| {
                let attempt = attempts.0.entry(self.id).or_default();
                *attempt += 1;
                *attempt
            });
        if attempt > self.max_retries {
            return ModelTurnAction::stop(format!(
                "response retry limit ({}) exceeded",
                self.max_retries
            ));
        }

        match self.mode {
            TestRetryMode::Repeat => ModelTurnAction::repeat(),
            TestRetryMode::Feedback(feedback) => ModelTurnAction::retry_with_feedback(feedback),
        }
    }
}

fn retry_usage(input_tokens: u64, output_tokens: u64) -> Usage {
    Usage::new()
        .input_tokens(input_tokens)
        .output_tokens(output_tokens)
        .total_tokens(input_tokens + output_tokens)
}

// ---------------------------------------------------------------------
// rig#2184: portable model-turn termination metadata.
//
// A hook must be able to tell *why* a turn stopped and *what cap* that
// exact attempt ran under, without naming a provider or touching a raw
// response type, and must see the same thing on both surfaces.
// ---------------------------------------------------------------------

#[tokio::test]
async fn blocking_empty_feedback_retry_omits_empty_assistant_history() {
    let first_usage = retry_usage(5, 1);
    let second_usage = retry_usage(7, 2);
    let model = MockCompletionModel::from_turns([
        MockTurn::text("").with_usage(first_usage),
        MockTurn::text("accepted").with_usage(second_usage),
    ]);
    let response = AgentBuilder::new(model.clone())
        .add_hook(BoundedResponseRetry::new(
            "",
            1,
            TestRetryMode::Feedback("provide an answer"),
        ))
        .build()
        .prompt("question")
        .max_turns(2)
        .run()
        .await
        .expect("feedback retry should recover from an empty turn");

    assert_eq!(response.output(), "accepted");
    assert_eq!(response.usage, first_usage + second_usage);
    assert_eq!(response.completion_calls.len(), 2);
    assert_eq!(
        response.messages,
        vec![
            Message::user("question"),
            Message::user("provide an answer"),
            mock_reply("accepted"),
        ]
    );
    assert_eq!(
        model.requests()[1].chat_history.clone(),
        vec![Message::User {
            content: vec![
                UserContent::text("question"),
                UserContent::text("provide an answer")
            ],
        }],
        "the retry request holds no empty assistant message, and the user messages it separated become one"
    );
}

struct AlwaysRepeatModelTurn;

impl AgentHook for AlwaysRepeatModelTurn {
    async fn on_model_turn_finished(
        &self,
        _ctx: &HookContext,
        _event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        ModelTurnAction::repeat()
    }
}

#[tokio::test]
async fn model_turn_retry_rejects_tool_turn_before_tool_hooks_or_execution() {
    let recorder = RecordingHook::default();
    let executions = Arc::new(AtomicU32::new(0));
    let err = AgentBuilder::new(MockCompletionModel::from_turns([MockTurn::tool_call(
        "tc1",
        "add",
        json!({"x": 1, "y": 2}),
    )]))
    .tool(CountingAddTool {
        calls: executions.clone(),
    })
    .add_hook(recorder.clone())
    .add_hook(AlwaysRepeatModelTurn)
    .build()
    .prompt("add")
    .max_turns(2)
    .run()
    .await
    .expect_err("tool-bearing retry must fail closed");

    let PromptError::Cancelled {
        chat_history,
        reason,
    } = err
    else {
        panic!("tool-bearing retry should return Cancelled");
    };
    assert!(reason.contains("tool-bearing model turns"));
    assert!(reason.contains("tool-call hooks"));
    assert_eq!(chat_history.into_vec(), vec![Message::user("add")]);
    assert_eq!(recorder.count(StepEventKind::ToolDispatch), 0);
    assert_eq!(executions.load(SeqCst), 0);
}

#[tokio::test]
async fn streaming_model_turn_retry_rejects_tool_turn_without_committed_execution() {
    let recorder = RecordingHook::default();
    let executions = Arc::new(AtomicU32::new(0));
    let mut stream = AgentBuilder::new(MockCompletionModel::from_stream_turns([[
        MockStreamEvent::tool_call_name_delta("tc1", "add"),
        MockStreamEvent::tool_call_arguments_delta("tc1", r#"{"x":1,"y":2}"#),
        MockStreamEvent::tool_call("tc1", "add", json!({"x": 1, "y": 2})),
        MockStreamEvent::final_response_with_default_usage(),
    ]]))
    .tool(CountingAddTool {
        calls: executions.clone(),
    })
    .add_hook(recorder.clone())
    .add_hook(AlwaysRepeatModelTurn)
    .build()
    .prompt("add")
    .max_turns(2)
    .stream();

    let mut execution_commits = 0;
    let mut tool_results = 0;
    let mut completion_calls = 0;
    let mut agent_finals = 0;
    let mut retry_markers = 0;
    let mut error = None;
    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::ToolExecutionCommitted { .. }) => execution_commits += 1,
            Ok(MultiTurnStreamItem::ToolResult { .. }) => tool_results += 1,
            Ok(MultiTurnStreamItem::CompletionCall(_)) => completion_calls += 1,
            Ok(MultiTurnStreamItem::FinalResponse(_)) => agent_finals += 1,
            Ok(MultiTurnStreamItem::ModelTurnRetried { .. }) => retry_markers += 1,
            Ok(_) => {}
            Err(err) => error = Some(err),
        }
    }

    let Some(error) = error else {
        panic!("tool-bearing streaming retry should return Cancelled");
    };
    let PromptError::Cancelled {
        chat_history,
        reason,
    } = error
    else {
        panic!("tool-bearing streaming retry should return Cancelled");
    };
    assert!(reason.contains("tool-bearing model turns"));
    assert!(reason.contains("tool-call hooks"));
    assert_eq!(*chat_history, [Message::user("add")]);
    assert_eq!(execution_commits, 0);
    assert_eq!(tool_results, 0);
    // The attempt's call is recorded when its reply ends, before the hook
    // rejects the turn.
    assert_eq!(completion_calls, 1);
    assert_eq!(agent_finals, 0);
    assert_eq!(retry_markers, 0);
    assert_eq!(recorder.count(StepEventKind::ToolDispatch), 0);
    assert_eq!(executions.load(SeqCst), 0);
}

#[derive(Clone)]
struct BarrierResponseRetry {
    inner: BoundedResponseRetry,
    barrier: Arc<Barrier>,
}

impl AgentHook for BarrierResponseRetry {
    async fn on_model_turn_finished(
        &self,
        ctx: &HookContext,
        event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        let rejected = event.content.iter().any(
            |content| matches!(content, AssistantContent::Text(text) if text.text == "rejected"),
        );
        if rejected {
            self.barrier.wait().await;
        }
        self.inner.on_model_turn_finished(ctx, event).await
    }
}

#[tokio::test]
async fn concurrent_runs_of_same_agent_have_independent_retry_budgets() {
    let hook = BarrierResponseRetry {
        inner: BoundedResponseRetry::new("rejected", 1, TestRetryMode::Repeat),
        barrier: Arc::new(Barrier::new(2)),
    };
    let agent = AgentBuilder::new(MockCompletionModel::from_turns([
        MockTurn::text("rejected"),
        MockTurn::text("rejected"),
        MockTurn::text("accepted one"),
        MockTurn::text("accepted two"),
    ]))
    .add_hook(hook)
    .build();

    let first = agent.prompt("first").max_turns(2).run();
    let second = agent.prompt("second").max_turns(2).run();
    let (first, second) = tokio::join!(first, second);
    let first = first.expect("first run");
    let second = second.expect("second run");

    let outputs = std::collections::HashSet::from([first.output(), second.output()]);
    assert_eq!(
        outputs,
        std::collections::HashSet::from(["accepted one".to_string(), "accepted two".to_string(),])
    );
    assert_eq!(first.completion_calls.len(), 2);
    assert_eq!(second.completion_calls.len(), 2);
}

mod run_lifecycle {
    use super::*;
    use crate::agent::hook::{RunSettled, RunStart, RunStartAction, SettledOutcome};

    /// Records run-start firings, the prompt each completion call carried,
    /// and every settle outcome — enough to pin the whole run lifecycle.
    #[derive(Clone, Default)]
    struct LifecycleProbe {
        starts: Arc<AtomicU32>,
        seen_prompts: Arc<Mutex<Vec<String>>>,
        settles: Arc<Mutex<Vec<String>>>,
        rewrite_on_start: bool,
        stop_on_start: bool,
    }

    impl AgentHook for LifecycleProbe {
        async fn on_run_start(&self, _ctx: &HookContext, event: RunStart<'_>) -> RunStartAction {
            self.starts.fetch_add(1, SeqCst);
            if self.stop_on_start {
                return RunStartAction::stop("vetoed at start");
            }
            if self.rewrite_on_start {
                let current = event.prompt.rag_text().unwrap_or_default();
                return RunStartAction::rewrite(Message::user(format!("{current} (rewritten)")));
            }
            RunStartAction::Continue
        }

        async fn on_completion_call(
            &self,
            _ctx: &HookContext,
            event: CompletionCallEvent<'_>,
        ) -> CompletionCallAction {
            self.seen_prompts
                .lock()
                .expect("prompts")
                .push(event.prompt.rag_text().unwrap_or_default());
            CompletionCallAction::Continue
        }

        async fn on_run_settled(&self, _ctx: &HookContext, event: RunSettled<'_>) {
            let outcome = match event.outcome {
                SettledOutcome::Response(_) => "ok".to_string(),
                SettledOutcome::Error(reason) => format!("err:{reason}"),
            };
            self.settles.lock().expect("settles").push(outcome);
        }
    }

    fn probe(rewrite_on_start: bool, stop_on_start: bool) -> LifecycleProbe {
        LifecycleProbe {
            rewrite_on_start,
            stop_on_start,
            ..LifecycleProbe::default()
        }
    }

    #[tokio::test]
    async fn blocking_run_fires_start_once_rewrites_prompt_and_settles_ok() {
        let hook = probe(true, false);
        let model = MockCompletionModel::from_turns([MockTurn::text("done")]);
        let response = AgentBuilder::new(model)
            .add_hook(hook.clone())
            .build()
            .prompt("hi")
            .await
            .expect("prompt succeeds");
        assert_eq!(response.output(), "done");

        assert_eq!(hook.starts.load(SeqCst), 1);
        assert_eq!(
            hook.seen_prompts.lock().expect("prompts").as_slice(),
            ["hi (rewritten)".to_string()]
        );
        assert_eq!(hook.settles.lock().expect("settles").as_slice(), ["ok"]);
    }

    #[tokio::test]
    async fn run_start_stop_terminates_before_any_provider_call_and_settles_err() {
        let hook = probe(false, true);
        let model = MockCompletionModel::from_stream_turns([vec![
            MockStreamEvent::text("never sent"),
            MockStreamEvent::final_response_with_total_tokens(0),
        ]]);
        let mut stream = AgentBuilder::new(model)
            .add_hook(hook.clone())
            .build()
            .prompt(Message::user("hi"))
            .stream();
        let first = stream.next().await.expect("terminal item");
        assert!(matches!(
            first,
            Err(ref err)
                if matches!(err, PromptError::Cancelled { .. })
        ));
        assert!(stream.next().await.is_none());

        assert_eq!(hook.starts.load(SeqCst), 1);
        // No completion call was ever issued.
        assert!(hook.seen_prompts.lock().expect("prompts").is_empty());
        let settles = hook.settles.lock().expect("settles").clone();
        assert_eq!(settles.len(), 1, "settled exactly once");
        assert!(
            settles[0].starts_with("err:"),
            "stop settles as an error outcome: {settles:?}"
        );
    }

    #[tokio::test]
    async fn error_termination_settles_exactly_once() {
        let hook = probe(false, false);
        // A tool-calling turn with a one-call budget: the run errors after
        // the turn instead of finishing, exercising the error settle path.
        let model = MockCompletionModel::from_stream_turns([vec![
            MockStreamEvent::tool_call_name_delta("tc1", "add"),
            MockStreamEvent::tool_call_arguments_delta("tc1", "{\"x\":2,\"y\":3}"),
            MockStreamEvent::tool_call("tc1", "add", json!({"x": 2, "y": 3})),
            MockStreamEvent::final_response_with_total_tokens(0),
        ]]);
        let mut stream = AgentBuilder::new(model)
            .tool(crate::test_utils::MockAddTool)
            .add_hook(hook.clone())
            .build()
            .prompt(Message::user("add 2 and 3"))
            .max_turns(1)
            .stream();
        let mut saw_error = false;
        while let Some(item) = stream.next().await {
            if item.is_err() {
                saw_error = true;
            }
        }
        assert!(saw_error, "the exhausted budget surfaces as a stream error");

        assert_eq!(hook.starts.load(SeqCst), 1);
        let settles = hook.settles.lock().expect("settles").clone();
        assert_eq!(settles.len(), 1, "settled exactly once: {settles:?}");
        assert!(settles[0].starts_with("err:"), "error outcome: {settles:?}");
    }
}

#[tokio::test]
async fn outcome_stop_is_terminal_through_nested_hooks_on_both_surfaces() {
    struct StopTool;
    impl AgentHook for StopTool {
        async fn on_outcome(&self, _: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
            if event.tool_name().is_some() {
                OutcomeAction::stop("terminal policy")
            } else {
                OutcomeAction::Proceed
            }
        }
    }
    struct Revive(Arc<AtomicU32>);
    impl AgentHook for Revive {
        async fn on_outcome(&self, _: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
            if event.tool_name().is_none() {
                return OutcomeAction::Proceed;
            }
            self.0.fetch_add(1, SeqCst);
            OutcomeAction::Replace(Ok(rig_core::effect::Outcome::ToolResult {
                result: crate::tool::ToolResult::success(crate::tool::ToolOutput::text("revived")),
            }))
        }
    }
    for streaming in [false, true] {
        for nested in [false, true] {
            let later = Arc::new(AtomicU32::new(0));
            let mut inner = HookStack::with(StopTool);
            inner.push(Revive(later.clone()));
            let mut stack = if nested {
                HookStack::with(inner)
            } else {
                inner
            };
            stack.push(Revive(later.clone()));
            let model = if streaming {
                MockCompletionModel::from_stream_turns([
                    vec![
                        MockStreamEvent::tool_call("tc1", "add", json!({"x": 1, "y": 2})),
                        MockStreamEvent::final_response_with_total_tokens(0),
                    ],
                    vec![
                        MockStreamEvent::text("done"),
                        MockStreamEvent::final_response_with_total_tokens(0),
                    ],
                ])
            } else {
                MockCompletionModel::from_turns([
                    MockTurn::tool_call("tc1", "add", json!({"x": 1, "y": 2})),
                    MockTurn::text("done"),
                ])
            };
            let agent = AgentBuilder::new(model)
                .tool(MockAddTool)
                .add_hook(stack)
                .build();
            let error = if streaming {
                let mut stream = agent.prompt("go").max_turns(3).stream();
                let mut error = None;
                while let Some(item) = stream.next().await {
                    match item {
                        Err(err) => error = Some(err.to_string()),
                        Ok(
                            MultiTurnStreamItem::ToolExecutionCommitted { .. }
                            | MultiTurnStreamItem::FinalResponse(_),
                        ) => panic!("stopped outcome must not commit"),
                        _ => {}
                    }
                }
                error.expect("stream must stop")
            } else {
                agent
                    .prompt("go")
                    .max_turns(3)
                    .run()
                    .await
                    .expect_err("run must stop")
                    .to_string()
            };
            assert!(error.contains("terminal policy"), "{error}");
            assert_eq!(later.load(SeqCst), 0, "later hooks must not undo stop");
        }
    }
}

/// The assistant message a scripted mock reply of `text` folds into.
fn mock_reply(text: &str) -> Message {
    Message::Assistant(
        rig_core::message::AssistantMessage::new(vec![AssistantContent::text(text)])
            .with_origin(rig_core::message::Origin::new(
                rig_core::test_utils::MOCK_API,
                rig_core::test_utils::MOCK_PROVIDER,
                rig_core::test_utils::MOCK_MODEL,
            ))
            .with_stop(rig_core::message::StopReason::Stop),
    )
}

/// `messages` with every assistant turn's origin, stop and provider items
/// left out.
fn canonical_history(messages: &[Message]) -> Vec<Message> {
    messages
        .iter()
        .map(|message| match message {
            Message::Assistant(turn) => {
                Message::Assistant(rig_core::message::AssistantMessage::new(
                    turn.content
                        .iter()
                        .map(AssistantContent::canonical)
                        .collect(),
                ))
            }
            other => other.clone(),
        })
        .collect()
}

// ---------------------------------------------------------------------
// Completion dispatch pairing.
//
// Every completion `on_dispatch` id is closed by exactly one `on_outcome`
// carrying the same id, on every terminal path: a provider error, a
// stream error, a hook stop mid-stream, and a rejected (retried) attempt.
// These are mock-model unit tests rather than cassette tests: the property
// is the engine's hook bookkeeping, which no provider wire shape affects.
// ---------------------------------------------------------------------

/// One completion hook event: the dispatch id, and for an outcome whether
/// it was an error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CompletionHookEvent {
    Dispatch(rig_core::effect::EffectId),
    Outcome {
        id: rig_core::effect::EffectId,
        is_err: bool,
    },
}

/// One completion hook event in full: the dispatch or outcome, its id and
/// turn, and an outcome's error kind and message.
#[derive(Clone, Debug)]
struct CompletionHookDetail {
    outcome: bool,
    id: rig_core::effect::EffectId,
    turn: usize,
    error: Option<(rig_core::error::ErrorKind, String)>,
}

/// Records completion `on_dispatch` and `on_outcome` events in order.
#[derive(Clone, Default)]
struct CompletionPairingHook {
    log: Arc<Mutex<Vec<CompletionHookEvent>>>,
    details: Arc<Mutex<Vec<CompletionHookDetail>>>,
    stop_on_text_delta: bool,
}

impl CompletionPairingHook {
    fn stopping_on_text_delta() -> Self {
        Self {
            stop_on_text_delta: true,
            ..Self::default()
        }
    }

    fn push(&self, event: CompletionHookEvent, detail: CompletionHookDetail) {
        if let Ok(mut log) = self.log.lock() {
            log.push(event);
        }
        if let Ok(mut details) = self.details.lock() {
            details.push(detail);
        }
    }

    fn details(&self) -> anyhow::Result<Vec<CompletionHookDetail>> {
        self.details
            .lock()
            .map(|details| details.clone())
            .map_err(|_| anyhow::anyhow!("pairing details poisoned"))
    }

    fn events(&self) -> anyhow::Result<Vec<CompletionHookEvent>> {
        self.log
            .lock()
            .map(|log| log.clone())
            .map_err(|_| anyhow::anyhow!("pairing log poisoned"))
    }

    /// Every dispatched id has exactly one outcome after it, and every
    /// outcome follows a dispatch of the same id. Returns the outcomes'
    /// error flags in dispatch order.
    fn paired_outcomes(&self) -> anyhow::Result<Vec<bool>> {
        let events = self.events()?;
        let mut flags = Vec::new();
        for (index, event) in events.iter().enumerate() {
            match *event {
                CompletionHookEvent::Dispatch(id) => {
                    let closing: Vec<bool> = events[index..]
                        .iter()
                        .filter_map(|later| match *later {
                            CompletionHookEvent::Outcome { id: closed, is_err } if closed == id => {
                                Some(is_err)
                            }
                            _ => None,
                        })
                        .collect();
                    anyhow::ensure!(
                        closing.len() == 1,
                        "completion dispatch {id:?} closed by {} outcomes, want 1: {events:?}",
                        closing.len()
                    );
                    flags.extend(closing);
                }
                CompletionHookEvent::Outcome { id, .. } => {
                    anyhow::ensure!(
                        events[..index].contains(&CompletionHookEvent::Dispatch(id)),
                        "outcome for {id:?} has no dispatch before it: {events:?}"
                    );
                }
            }
        }
        Ok(flags)
    }
}

impl AgentHook for CompletionPairingHook {
    async fn on_dispatch(&self, _: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        if matches!(event.kind, rig_core::effect::EffectKind::Completion { .. }) {
            let detail = CompletionHookDetail {
                outcome: false,
                id: event.id,
                turn: event.turn,
                error: None,
            };
            self.push(CompletionHookEvent::Dispatch(event.id), detail);
        }
        DispatchAction::proceed()
    }

    async fn on_outcome(&self, _: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        if matches!(event.kind, rig_core::effect::EffectKind::Completion { .. }) {
            let detail = CompletionHookDetail {
                outcome: true,
                id: event.id,
                turn: event.turn,
                error: event
                    .outcome
                    .as_ref()
                    .err()
                    .map(|report| (report.kind, report.message.clone())),
            };
            let is_err = event.outcome.is_err();
            self.push(
                CompletionHookEvent::Outcome {
                    id: event.id,
                    is_err,
                },
                detail,
            );
        }
        OutcomeAction::proceed()
    }

    async fn on_text_delta(&self, _: &HookContext, _: TextDelta<'_>) -> ObservationAction {
        if self.stop_on_text_delta {
            ObservationAction::stop("stop mid-stream")
        } else {
            ObservationAction::continue_run()
        }
    }
}

/// Drains a stream, returning whether any item was an error.
async fn drain_stream<S>(mut stream: S) -> bool
where
    S: futures::Stream<Item = Result<MultiTurnStreamItem, PromptError>> + Unpin,
{
    let mut errored = false;
    while let Some(item) = stream.next().await {
        errored |= item.is_err();
    }
    errored
}

/// A unary provider error (a 429) after the completion was dispatched is
/// reported to `on_outcome` as an error for that dispatch id.
#[tokio::test]
async fn unary_provider_error_closes_its_completion_dispatch() -> anyhow::Result<()> {
    let hook = CompletionPairingHook::default();
    let result = AgentBuilder::new(MockCompletionModel::from_turns([
        MockTurn::provider_response_error(
            http::StatusCode::TOO_MANY_REQUESTS,
            r#"{"error":"rate limited"}"#,
            "req-failed",
        ),
    ]))
    .add_hook(hook.clone())
    .build()
    .prompt(Message::user("hi"))
    .run()
    .await;
    anyhow::ensure!(result.is_err(), "the provider error fails the run");

    let outcomes = hook.paired_outcomes()?;
    anyhow::ensure!(
        outcomes == [true],
        "one dispatch closed by one error outcome, got {outcomes:?}"
    );
    Ok(())
}

/// A stream that fails mid-reply is reported to `on_outcome` as an error
/// for that dispatch id.
#[tokio::test]
async fn streamed_error_closes_its_completion_dispatch() -> anyhow::Result<()> {
    let hook = CompletionPairingHook::default();
    let stream = AgentBuilder::new(MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::text("partial"),
        MockStreamEvent::error("connection reset"),
    ]]))
    .add_hook(hook.clone())
    .build()
    .prompt("hi")
    .stream();
    anyhow::ensure!(drain_stream(stream).await, "the stream error fails the run");

    let outcomes = hook.paired_outcomes()?;
    anyhow::ensure!(
        outcomes == [true],
        "one dispatch closed by one error outcome, got {outcomes:?}"
    );
    Ok(())
}

/// A hook that stops the run on a text delta still sees the dispatched
/// completion closed by an outcome.
#[tokio::test]
async fn streamed_delta_stop_closes_its_completion_dispatch() -> anyhow::Result<()> {
    let hook = CompletionPairingHook::stopping_on_text_delta();
    let stream = AgentBuilder::new(MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::text("partial"),
        MockStreamEvent::text(" more"),
        MockStreamEvent::final_response_with_total_tokens(0),
    ]]))
    .add_hook(hook.clone())
    .build()
    .prompt("hi")
    .stream();
    anyhow::ensure!(drain_stream(stream).await, "the hook stop ends the run");

    let outcomes = hook.paired_outcomes()?;
    anyhow::ensure!(
        outcomes.len() == 1,
        "one dispatch closed by one outcome, got {outcomes:?}"
    );
    Ok(())
}

/// An attempt rejected for an invalid tool call and retried was still
/// dispatched (and billed), so its id is closed by an outcome on both media.
#[tokio::test]
async fn retried_attempt_closes_its_completion_dispatch() -> anyhow::Result<()> {
    struct RetryInvalid;
    impl AgentHook for RetryInvalid {
        async fn on_invalid_tool_call(
            &self,
            _: &HookContext,
            _: &InvalidToolCallContext,
        ) -> Option<InvalidToolCallAction> {
            Some(InvalidToolCallAction::retry("no such tool"))
        }
    }

    let blocking_hook = CompletionPairingHook::default();
    let blocking = AgentBuilder::new(MockCompletionModel::from_turns([
        MockTurn::tool_call("tc1", "default_api", json!({"x": 2, "y": 3})),
        MockTurn::text("done"),
    ]))
    .tool(MockAddTool)
    .add_hook(blocking_hook.clone())
    .add_hook(RetryInvalid)
    .build()
    .prompt("do the thing")
    .max_turns(3)
    .max_invalid_tool_call_retries(1)
    .run()
    .await;
    anyhow::ensure!(blocking.is_ok(), "the retry recovers: {blocking:?}");
    let outcomes = blocking_hook.paired_outcomes()?;
    anyhow::ensure!(
        outcomes == [false, false],
        "run: both attempts closed by an outcome, got {outcomes:?}"
    );

    let streaming_hook = CompletionPairingHook::default();
    let stream = AgentBuilder::new(MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("tc1", "default_api", json!({"x": 2, "y": 3})),
            MockStreamEvent::final_response_with_total_tokens(0),
        ],
        vec![
            MockStreamEvent::text("done"),
            MockStreamEvent::final_response_with_total_tokens(0),
        ],
    ]))
    .tool(MockAddTool)
    .add_hook(streaming_hook.clone())
    .add_hook(RetryInvalid)
    .build()
    .prompt("do the thing")
    .max_turns(3)
    .max_invalid_tool_call_retries(1)
    .stream();
    anyhow::ensure!(!drain_stream(stream).await, "the streamed retry recovers");
    let outcomes = streaming_hook.paired_outcomes()?;
    anyhow::ensure!(
        outcomes == [false, false],
        "stream: both attempts closed by an outcome, got {outcomes:?}"
    );
    Ok(())
}

/// A turn whose invalid tool call a hook repaired is recovered: it fires no
/// accepted-turn hooks, but its dispatch is still closed by an outcome.
#[tokio::test]
async fn recovered_turn_closes_its_completion_dispatch() -> anyhow::Result<()> {
    let blocking_hook = CompletionPairingHook::default();
    let blocking = AgentBuilder::new(MockCompletionModel::from_turns([
        MockTurn::tool_call("tc1", "default_api", json!({"x": 2, "y": 3})),
        MockTurn::text("the answer is 5"),
    ]))
    .tool(MockAddTool)
    .add_hook(blocking_hook.clone())
    .add_hook(RepairInvalidToHook("add"))
    .build()
    .prompt("add 2 and 3")
    .max_turns(3)
    .run()
    .await;
    anyhow::ensure!(blocking.is_ok(), "the repair recovers: {blocking:?}");
    let outcomes = blocking_hook.paired_outcomes()?;
    anyhow::ensure!(
        outcomes == [false, false],
        "run: the recovered and the answer turn closed, got {outcomes:?}"
    );

    let streaming_hook = CompletionPairingHook::default();
    let stream = AgentBuilder::new(MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("tc1", "default_api", json!({"x": 2, "y": 3})),
            MockStreamEvent::final_response_with_total_tokens(0),
        ],
        vec![
            MockStreamEvent::text("the answer is 5"),
            MockStreamEvent::final_response_with_total_tokens(0),
        ],
    ]))
    .tool(MockAddTool)
    .add_hook(streaming_hook.clone())
    .add_hook(RepairInvalidToHook("add"))
    .build()
    .prompt("add 2 and 3")
    .max_turns(3)
    .stream();
    anyhow::ensure!(!drain_stream(stream).await, "the streamed repair recovers");
    let outcomes = streaming_hook.paired_outcomes()?;
    anyhow::ensure!(
        outcomes == [false, false],
        "stream: the recovered and the answer turn closed, got {outcomes:?}"
    );
    Ok(())
}

/// A denied completion never reaches the bus, so it has no outcome, as for
/// any other denied effect.
#[tokio::test]
async fn denied_completion_has_no_outcome() -> anyhow::Result<()> {
    struct DenyCompletion;
    impl AgentHook for DenyCompletion {
        async fn on_dispatch(&self, _: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
            match event.kind {
                rig_core::effect::EffectKind::Completion { .. } => DispatchAction::deny(
                    rig_core::error::ErrorReport::new(rig_core::error::ErrorKind::Denied, "no"),
                ),
                _ => DispatchAction::proceed(),
            }
        }
    }

    let hook = CompletionPairingHook::default();
    let result = AgentBuilder::new(MockCompletionModel::from_turns([MockTurn::text("unused")]))
        .add_hook(hook.clone())
        .add_hook(DenyCompletion)
        .build()
        .prompt(Message::user("hi"))
        .run()
        .await;
    anyhow::ensure!(result.is_err(), "the denial fails the run");
    let events = hook.events()?;
    anyhow::ensure!(
        matches!(events.as_slice(), [CompletionHookEvent::Dispatch(_)]),
        "a dispatch with no outcome, got {events:?}"
    );
    Ok(())
}

/// The outcome that closes a rejected attempt is observe-only: a hook's
/// replacement of it is ignored and the retried run still succeeds.
#[tokio::test]
async fn unsettled_close_ignores_a_replacement() -> anyhow::Result<()> {
    #[derive(Clone, Default)]
    struct ReplaceFirstOutcome(Arc<AtomicU32>);
    impl AgentHook for ReplaceFirstOutcome {
        async fn on_invalid_tool_call(
            &self,
            _: &HookContext,
            _: &InvalidToolCallContext,
        ) -> Option<InvalidToolCallAction> {
            Some(InvalidToolCallAction::retry("no such tool"))
        }
        async fn on_outcome(&self, _: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
            let completion = matches!(event.kind, rig_core::effect::EffectKind::Completion { .. });
            if completion && self.0.fetch_add(1, SeqCst) == 0 {
                return OutcomeAction::Replace(Err(rig_core::error::ErrorReport::new(
                    rig_core::error::ErrorKind::Other,
                    "replaced",
                )));
            }
            OutcomeAction::proceed()
        }
    }

    let hook = ReplaceFirstOutcome::default();
    let result = AgentBuilder::new(MockCompletionModel::from_turns([
        MockTurn::tool_call("tc1", "default_api", json!({"x": 2, "y": 3})),
        MockTurn::text("done"),
    ]))
    .tool(MockAddTool)
    .add_hook(hook.clone())
    .build()
    .prompt("do the thing")
    .max_turns(3)
    .max_invalid_tool_call_retries(1)
    .run()
    .await;
    anyhow::ensure!(result.is_ok(), "the replacement is ignored: {result:?}");
    anyhow::ensure!(hook.0.load(SeqCst) == 2, "both attempts closed");
    Ok(())
}

/// A streamed turn abandoned for an invalid tool call (the hook skips the
/// call) still finished its reply, so its dispatch closes with one `Ok`
/// outcome carrying that reply.
#[tokio::test]
async fn streamed_abandoned_turn_closes_its_completion_dispatch() -> anyhow::Result<()> {
    struct SkipInvalid;
    impl AgentHook for SkipInvalid {
        async fn on_invalid_tool_call(
            &self,
            _: &HookContext,
            _: &InvalidToolCallContext,
        ) -> Option<InvalidToolCallAction> {
            Some(InvalidToolCallAction::skip("no such tool"))
        }
    }

    let hook = CompletionPairingHook::default();
    let stream = AgentBuilder::new(MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("tc1", "default_api", json!({"x": 2, "y": 3})),
            MockStreamEvent::final_response_with_total_tokens(0),
        ],
        vec![
            MockStreamEvent::text("done"),
            MockStreamEvent::final_response_with_total_tokens(0),
        ],
    ]))
    .tool(MockAddTool)
    .add_hook(hook.clone())
    .add_hook(SkipInvalid)
    .build()
    .prompt("do the thing")
    .max_turns(3)
    .stream();
    anyhow::ensure!(!drain_stream(stream).await, "the skipped call recovers");
    let outcomes = hook.paired_outcomes()?;
    anyhow::ensure!(
        outcomes == [false, false],
        "the abandoned and the answer turn each closed once with Ok, got {outcomes:?}"
    );
    Ok(())
}

/// An unknown tool under the default `Fail` policy fails the run on both
/// media, and the attempt's dispatch is closed by exactly one outcome. The
/// unary answer arrived whole, so it closes `Ok` with the provider's
/// response; the streamed reply is cut off at the rejected call, before the
/// provider ended it, so it closes `Err` with the run's failure.
#[tokio::test]
async fn unknown_tool_under_fail_policy_closes_its_completion_dispatch() -> anyhow::Result<()> {
    let blocking_hook = CompletionPairingHook::default();
    let blocking = AgentBuilder::new(MockCompletionModel::from_turns([MockTurn::tool_call(
        "tc1",
        "default_api",
        json!({"x": 2, "y": 3}),
    )]))
    .tool(MockAddTool)
    .add_hook(blocking_hook.clone())
    .build()
    .prompt("do the thing")
    .max_turns(3)
    .run()
    .await;
    anyhow::ensure!(blocking.is_err(), "the unknown tool fails the run");
    let outcomes = blocking_hook.paired_outcomes()?;
    anyhow::ensure!(
        outcomes == [false],
        "run: one Ok outcome with the provider's answer, got {outcomes:?}"
    );

    let streaming_hook = CompletionPairingHook::default();
    let stream = AgentBuilder::new(MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::tool_call("tc1", "default_api", json!({"x": 2, "y": 3})),
        MockStreamEvent::final_response_with_total_tokens(0),
    ]]))
    .tool(MockAddTool)
    .add_hook(streaming_hook.clone())
    .build()
    .prompt("do the thing")
    .max_turns(3)
    .stream();
    anyhow::ensure!(
        drain_stream(stream).await,
        "the unknown tool fails the stream"
    );
    let outcomes = streaming_hook.paired_outcomes()?;
    anyhow::ensure!(
        outcomes == [true],
        "stream: one Err outcome, got {outcomes:?}"
    );
    Ok(())
}

/// A hook that stops the run in `on_model_turn_finished` does so after the
/// accepted turn settled: the dispatch is closed once, by the settled
/// outcome, and the engine's close on the way out adds nothing.
#[tokio::test]
async fn model_turn_stop_closes_its_completion_dispatch_once() -> anyhow::Result<()> {
    struct StopOnTurn;
    impl AgentHook for StopOnTurn {
        async fn on_model_turn_finished(
            &self,
            _: &HookContext,
            _: ModelTurnFinished<'_>,
        ) -> ModelTurnAction {
            ModelTurnAction::stop("enough")
        }
    }

    let blocking_hook = CompletionPairingHook::default();
    let blocking = AgentBuilder::new(MockCompletionModel::from_turns([MockTurn::text("hello")]))
        .add_hook(blocking_hook.clone())
        .add_hook(StopOnTurn)
        .build()
        .prompt("hi")
        .run()
        .await;
    anyhow::ensure!(blocking.is_err(), "the stop ends the run");
    let outcomes = blocking_hook.paired_outcomes()?;
    anyhow::ensure!(
        outcomes == [false],
        "run: one settled outcome, got {outcomes:?}"
    );

    let streaming_hook = CompletionPairingHook::default();
    let stream = AgentBuilder::new(MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::text("hello"),
        MockStreamEvent::final_response_with_total_tokens(0),
    ]]))
    .add_hook(streaming_hook.clone())
    .add_hook(StopOnTurn)
    .build()
    .prompt("hi")
    .stream();
    anyhow::ensure!(drain_stream(stream).await, "the stop ends the stream");
    let outcomes = streaming_hook.paired_outcomes()?;
    anyhow::ensure!(
        outcomes == [false],
        "stream: one settled outcome, got {outcomes:?}"
    );
    Ok(())
}

/// A completion handler that answers every dispatch with a memory outcome.
struct WrongOutcomeHandler;

impl rig_core::serve::Serve for WrongOutcomeHandler {
    type Family = rig_core::effect::family::Completion;

    fn descriptor(&self) -> rig_core::effect::HandlerDescriptor {
        rig_core::effect::HandlerDescriptor {
            key: rig_core::effect::model_key("wrong"),
            family: rig_core::effect::FamilyDescriptor::Completion {
                model: "wrong".into(),
                capabilities: rig_core::completion::ProviderCapabilities::default(),
            },
            layers: Vec::new(),
        }
    }

    async fn serve(
        &self,
        _: rig_core::effect::EffectKind,
        _: rig_core::serve::Dispatch,
    ) -> rig_core::serve::Reply {
        rig_core::serve::Reply::Outcome(Ok(rig_core::effect::Outcome::Memory(
            rig_core::effect::MemoryOutcome::Appended,
        )))
    }
}

/// A unary handler that answers a completion with another family's outcome
/// fails the run, and the dispatch closes with the `Internal` report that
/// names the wrong outcome.
#[tokio::test]
async fn wrong_outcome_closes_its_completion_dispatch_with_internal() -> anyhow::Result<()> {
    let (dispatcher, registrar, mut driver) = crate::bus::Bus::channel();
    driver
        .register("model:wrong", WrongOutcomeHandler)
        .map_err(|report| anyhow::anyhow!("register: {report}"))?;
    tokio::spawn(driver);
    let hook = CompletionPairingHook::default();
    let result = AgentBuilder::over_bus(
        dispatcher,
        registrar,
        "wrong",
        rig_core::effect::HandlerKey::from("model:wrong"),
    )
    .add_hook(hook.clone())
    .build()
    .prompt(Message::user("hi"))
    .run()
    .await;
    anyhow::ensure!(result.is_err(), "the wrong outcome fails the run");
    anyhow::ensure!(hook.paired_outcomes()? == [true], "one Err outcome");
    let details = hook.details()?;
    let closed = details.iter().find(|detail| detail.outcome);
    let Some((kind, message)) = closed.and_then(|detail| detail.error.clone()) else {
        anyhow::bail!("no error outcome: {details:?}");
    };
    anyhow::ensure!(
        kind == rig_core::error::ErrorKind::Internal && message.contains("memory"),
        "the Internal wrong-outcome report, got {kind:?}: {message}"
    );
    Ok(())
}

/// An attempt closed unsettled reports the turn it was dispatched in: the
/// outcome's `turn` is its dispatch's `turn`, here on the second turn.
#[tokio::test]
async fn unsettled_close_reports_the_dispatch_turn() -> anyhow::Result<()> {
    let hook = CompletionPairingHook::default();
    let result = AgentBuilder::new(MockCompletionModel::from_turns([
        MockTurn::tool_call("tc1", "add", json!({"x": 2, "y": 3})),
        MockTurn::provider_response_error(
            http::StatusCode::TOO_MANY_REQUESTS,
            r#"{"error":"rate limited"}"#,
            "req-failed",
        ),
    ]))
    .tool(MockAddTool)
    .add_hook(hook.clone())
    .build()
    .prompt(Message::user("add 2 and 3"))
    .max_turns(3)
    .run()
    .await;
    anyhow::ensure!(result.is_err(), "the provider error fails the run");
    anyhow::ensure!(
        hook.paired_outcomes()? == [false, true],
        "both turns closed"
    );
    let details = hook.details()?;
    let failed = details
        .iter()
        .find(|detail| detail.outcome && detail.error.is_some());
    let Some(failed) = failed else {
        anyhow::bail!("no failed outcome: {details:?}");
    };
    let dispatched = details
        .iter()
        .find(|detail| !detail.outcome && detail.id == failed.id);
    let Some(dispatched) = dispatched else {
        anyhow::bail!("no dispatch for the failed outcome: {details:?}");
    };
    anyhow::ensure!(
        failed.turn == dispatched.turn && failed.turn > 0,
        "outcome turn {} matches dispatch turn {}",
        failed.turn,
        dispatched.turn
    );
    Ok(())
}

/// Dispatch ids are minted, and dispatch and outcome events built, only in
/// the dispatch scope module, the hook stack (which re-wraps events) and the
/// bus. A source scan, so a lint rather than a type-level guarantee: it
/// walks every non-test file under `src`.
#[test]
fn no_dispatch_events_are_built_outside_the_scope() -> anyhow::Result<()> {
    const ALLOWED: [&str; 3] = ["agent/engine/dispatch.rs", "agent/hook.rs", "bus/"];
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut pending = vec![root.clone()];
    let mut scanned = 0;
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let path = entry?.path();
            if path.is_dir() {
                pending.push(path);
                continue;
            }
            let relative = path
                .strip_prefix(&root)?
                .to_string_lossy()
                .replace('\\', "/");
            let name = path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned());
            let is_test =
                name.is_some_and(|name| name == "tests.rs" || name.ends_with("_tests.rs"));
            let allowed = ALLOWED.iter().any(|prefix| relative.starts_with(prefix));
            if !relative.ends_with(".rs") || is_test || allowed {
                continue;
            }
            scanned += 1;
            let source = std::fs::read_to_string(&path)?;
            for needle in ["DispatchEvent {", "OutcomeEvent {", "mint_id("] {
                anyhow::ensure!(!source.contains(needle), "{relative} contains `{needle}`");
            }
        }
    }
    anyhow::ensure!(
        scanned > 10,
        "the scan found the source tree ({scanned} files)"
    );
    Ok(())
}

/// A host that drives the step protocol, answers one call of a batch and
/// persists the run hands off to the engine: `resume` runs only the
/// unanswered call and commits both results in call order.
#[tokio::test]
async fn resume_of_a_part_answered_batch_runs_only_the_unanswered_call() {
    use crate::run::{AgentRun, AgentRunStep, ModelTurn, TurnPolicy};

    let calls = Arc::new(AtomicU32::new(0));
    let agent = AgentBuilder::new(MockCompletionModel::from_turns([MockTurn::text("done")]))
        .tool(CountingAddTool {
            calls: calls.clone(),
        })
        .build();
    let mut run = AgentRun::from_spec(&agent.run_spec(), Message::user("go"), None).max_turns(2);
    let Ok(AgentRunStep::CallModel { turn, .. }) = run.next_step() else {
        panic!("a fresh run calls the model");
    };
    // The host records what it offered, as the engine does, so the resumed
    // batch binds to this process's tools.
    run.advertise_tools(
        turn,
        vec![rig_core::completion::ToolDefinition {
            name: rig_core::message::ToolName::new("add").expect("tool name"),
            description: MockAddTool.description(),
            parameters: MockAddTool.parameters(),
        }],
    );
    let policy = TurnPolicy::new(["add".to_string()].into(), None, None).expect("policy");
    run.model_response(ModelTurn::new(
        rig_core::message::AssistantMessage::default(),
        vec![
            tool_call_content("tc1", json!({"x": 1, "y": 1})),
            tool_call_content("tc2", json!({"x": 2, "y": 2})),
        ],
        Usage::default(),
        policy,
        json!({}),
    ))
    .expect("the tool turn is accepted");
    let Ok(AgentRunStep::CallTools { calls: mut pending }) = run.next_step() else {
        panic!("the tool turn asks for its calls");
    };
    let Some(crate::run::PendingToolCall::Execute(first)) = pending.drain(..).next() else {
        panic!("tc1 is executable");
    };
    let host =
        rig_core::tool::ToolResult::success(rig_core::tool::ToolOutput::text("from the host"));
    let host_result =
        rig_core::transcript::tool_result_output(first.id().clone(), first.name().clone(), &host);
    run.answer(first.answer(host))
        .expect("the host answers tc1");

    let json = serde_json::to_string(&run).expect("the run serializes");
    let restored: AgentRun = serde_json::from_str(&json).expect("the run deserializes");
    let response = agent
        .resume(restored)
        .await
        .expect("the resumed run completes");

    assert_eq!(response.output(), "done");
    assert_eq!(calls.load(SeqCst), 1, "only tc2 runs after the handoff");
    let results = response
        .messages()
        .iter()
        .find_map(|message| match message {
            Message::User { content }
                if content
                    .iter()
                    .any(|item| matches!(item, UserContent::ToolResult(_))) =>
            {
                Some(content.clone())
            }
            _ => None,
        })
        .expect("the batch is committed");
    let [first, UserContent::ToolResult(second)] = results.as_slice() else {
        panic!("one result per call: {results:?}");
    };
    assert_eq!(*first, host_result, "tc1 keeps the host's answer, first");
    assert_eq!(second.call.to_string(), "tc2");
    assert!(!second.is_error, "tc2 ran: {second:?}");
}
