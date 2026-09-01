use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use serde_json::json;

use super::*;
use crate::agent::{CompletionResponseEvent, HookContext, ModelTurnAction, ObservationAction};
use crate::test_utils::{MockCompletionModel, MockTurn};
use rig_core::message::{AssistantContent, ToolCall, ToolFunction};
use rig_core::vector_store::{
    VectorSearchRequest, VectorStoreError, VectorStoreIndex, request::Filter,
};
use serde::Deserialize;

#[derive(Debug, PartialEq, Deserialize, Serialize, JsonSchema)]
struct Person {
    name: String,
}

fn usage(total_tokens: u64) -> Usage {
    Usage {
        total_tokens,
        ..Usage::new()
    }
}

fn extractor(model: MockCompletionModel, retries: u64) -> Extractor<Person> {
    ExtractorBuilder::new(model).retries(retries).build()
}

fn submit_turn(name: &str) -> MockTurn {
    MockTurn::tool_call("id1", SUBMIT_TOOL_NAME, json!({ "name": name }))
}

fn tool_call(id: &str, name: &str, arguments: serde_json::Value) -> AssistantContent {
    AssistantContent::ToolCall(ToolCall::from_wire(
        id,
        ToolFunction::new(name.to_string(), arguments),
    ))
}

#[derive(Clone, Default)]
struct LifecycleCounts {
    completion_calls: Arc<AtomicUsize>,
    completion_responses: Arc<AtomicUsize>,
    model_turns: Arc<AtomicUsize>,
    invalid_tool_calls: Arc<AtomicUsize>,
}

impl AgentHook for LifecycleCounts {
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        _event: crate::agent::CompletionCallEvent<'_>,
    ) -> crate::agent::CompletionCallAction {
        self.completion_calls.fetch_add(1, Ordering::SeqCst);
        crate::agent::CompletionCallAction::Continue
    }

    async fn on_completion_response(
        &self,
        _ctx: &HookContext,
        _event: CompletionResponseEvent<'_>,
    ) -> ObservationAction {
        self.completion_responses.fetch_add(1, Ordering::SeqCst);
        ObservationAction::Continue
    }

    async fn on_model_turn_finished(
        &self,
        _ctx: &HookContext,
        _event: crate::agent::ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        self.model_turns.fetch_add(1, Ordering::SeqCst);
        ModelTurnAction::Continue
    }

    async fn on_invalid_tool_call(
        &self,
        _ctx: &HookContext,
        _event: &crate::agent::InvalidToolCallContext,
    ) -> Option<crate::agent::InvalidToolCallAction> {
        self.invalid_tool_calls.fetch_add(1, Ordering::SeqCst);
        None
    }
}

type ExtractorResponseSnapshot = (Message, Vec<AssistantContent>, Usage, Option<String>);

#[derive(Clone, Default)]
struct ExtractorResponseCapture {
    snapshot: Arc<Mutex<Option<ExtractorResponseSnapshot>>>,
}

impl AgentHook for ExtractorResponseCapture {
    async fn on_completion_response(
        &self,
        _ctx: &HookContext,
        event: CompletionResponseEvent<'_>,
    ) -> ObservationAction {
        *self.snapshot.lock().expect("extractor response snapshot") = Some((
            event.prompt.clone(),
            event.content.clone(),
            event.usage,
            event.message_id.map(str::to_owned),
        ));
        ObservationAction::continue_run()
    }
}

struct StopBeforeCompletion;

impl AgentHook for StopBeforeCompletion {
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        _event: crate::agent::CompletionCallEvent<'_>,
    ) -> crate::agent::CompletionCallAction {
        crate::agent::CompletionCallAction::stop("extractor stopped")
    }
}

struct ExtractorContextIndex {
    queries: Arc<Mutex<Vec<(String, u64)>>>,
}

impl VectorStoreIndex for ExtractorContextIndex {
    type Filter = Filter<serde_json::Value>;

    async fn top_n<T: DeserializeOwned + WasmCompatSend>(
        &self,
        req: VectorSearchRequest,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        self.queries
            .lock()
            .expect("extractor query recorder")
            .push((req.query().to_string(), req.samples()));
        let value = serde_json::from_value(json!({ "question": "retrieved" }))?;
        Ok(vec![(1.0, "extractor-context".to_string(), value)])
    }

    async fn top_n_ids(
        &self,
        _req: VectorSearchRequest,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        Ok(vec![(1.0, "extractor-context".to_string())])
    }
}

#[derive(Clone, Copy)]
enum StopFirstBilledResponseAt {
    CompletionResponse,
    ModelTurnFinished,
}

#[derive(Clone)]
struct StopFirstBilledResponse {
    phase: StopFirstBilledResponseAt,
    calls: Arc<AtomicUsize>,
}

impl AgentHook for StopFirstBilledResponse {
    async fn on_completion_response(
        &self,
        _ctx: &HookContext,
        _event: CompletionResponseEvent<'_>,
    ) -> ObservationAction {
        if matches!(self.phase, StopFirstBilledResponseAt::CompletionResponse)
            && self.calls.fetch_add(1, Ordering::SeqCst) == 0
        {
            ObservationAction::stop("stop first billed response")
        } else {
            ObservationAction::continue_run()
        }
    }

    async fn on_model_turn_finished(
        &self,
        _ctx: &HookContext,
        _event: crate::agent::ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        if matches!(self.phase, StopFirstBilledResponseAt::ModelTurnFinished)
            && self.calls.fetch_add(1, Ordering::SeqCst) == 0
        {
            ModelTurnAction::stop("stop first billed model turn")
        } else {
            ModelTurnAction::continue_run()
        }
    }
}

struct StopOnInvalidToolCall;

impl AgentHook for StopOnInvalidToolCall {
    async fn on_invalid_tool_call(
        &self,
        _ctx: &HookContext,
        _event: &crate::agent::InvalidToolCallContext,
    ) -> Option<crate::agent::InvalidToolCallAction> {
        Some(crate::agent::InvalidToolCallAction::stop(
            "unexpected extractor tool call",
        ))
    }
}

struct RepairUnexpectedAsSubmit;

impl AgentHook for RepairUnexpectedAsSubmit {
    async fn on_invalid_tool_call(
        &self,
        _ctx: &HookContext,
        _event: &crate::agent::InvalidToolCallContext,
    ) -> Option<crate::agent::InvalidToolCallAction> {
        Some(crate::agent::InvalidToolCallAction::repair(
            SUBMIT_TOOL_NAME,
        ))
    }
}

struct SkipUnexpected;

impl AgentHook for SkipUnexpected {
    async fn on_invalid_tool_call(
        &self,
        _ctx: &HookContext,
        _event: &crate::agent::InvalidToolCallContext,
    ) -> Option<crate::agent::InvalidToolCallAction> {
        Some(crate::agent::InvalidToolCallAction::skip(
            "ignored by extractor hook",
        ))
    }
}

#[tokio::test]
async fn extractor_runs_through_full_response_lifecycle() {
    let model = MockCompletionModel::new([submit_turn("John")]);
    let counts = LifecycleCounts::default();
    let response = ExtractorBuilder::<Person>::new(model.clone())
        .add_hook(counts.clone())
        .build()
        .extract("John")
        .await
        .expect("extraction should succeed");

    assert_eq!(response.name, "John");
    assert_eq!(model.request_count(), 1);
    assert_eq!(counts.completion_calls.load(Ordering::SeqCst), 1);
    assert_eq!(counts.completion_responses.load(Ordering::SeqCst), 1);
    assert_eq!(counts.model_turns.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn extractor_hook_receives_canonical_response_fields() {
    let capture = ExtractorResponseCapture::default();
    let expected_usage = usage(23);
    let response = ExtractorBuilder::<Person>::new(MockCompletionModel::new([submit_turn("John")
        .with_usage(expected_usage)
        .with_message_id("extractor-message")]))
    .add_hook(capture.clone())
    .build()
    .extract("John")
    .await
    .expect("extraction should succeed");
    assert_eq!(response.name, "John");

    let (prompt, content, observed_usage, message_id) = capture
        .snapshot
        .lock()
        .expect("extractor response snapshot")
        .clone()
        .expect("extractor response hook should fire");
    assert_eq!(prompt, Message::user("John"));
    assert_eq!(observed_usage, expected_usage);
    assert_eq!(message_id.as_deref(), Some("extractor-message"));
    assert!(matches!(
        content.as_slice(),
        [AssistantContent::ToolCall(tool_call)]
            if tool_call.function.name == SUBMIT_TOOL_NAME
                && tool_call.function.arguments == json!({"name": "John"})
    ));
}

#[tokio::test]
async fn extractor_dynamic_context_uses_the_agent_hook_lifecycle() {
    let model = MockCompletionModel::new([submit_turn("John")]);
    let probe = model.clone();
    let queries = Arc::new(Mutex::new(Vec::new()));
    let response = ExtractorBuilder::<Person>::new(model)
        .dynamic_context(
            2,
            ExtractorContextIndex {
                queries: queries.clone(),
            },
        )
        .build()
        .extract("John")
        .await
        .expect("extraction should succeed");

    assert_eq!(response.name, "John");
    assert_eq!(
        *queries.lock().expect("extractor queries"),
        vec![("John".to_string(), 2)]
    );
    let requests = probe.requests();
    let request = requests.first().expect("one extractor request");
    assert!(
        request
            .documents
            .iter()
            .any(|document| document.id == "extractor-context"
                && document.text == "{\n  \"question\": \"retrieved\"\n}")
    );
}

#[tokio::test]
async fn extractor_completion_call_stop_prevents_provider_io() {
    let model = MockCompletionModel::new([submit_turn("John")]);
    let error = ExtractorBuilder::<Person>::new(model.clone())
        .add_hook(StopBeforeCompletion)
        .build()
        .extract("John")
        .await
        .expect_err("terminating hook should cancel extraction");

    assert!(matches!(
        error,
        ExtractionError::PromptError(PromptError::PromptCancelled { reason, .. })
            if reason == "extractor stopped"
    ));
    assert_eq!(model.request_count(), 0);
}

#[tokio::test]
async fn usage_accumulates_across_failed_attempts() {
    let model = MockCompletionModel::new([
        MockTurn::text("no submit call").with_usage(usage(10)),
        submit_turn("John").with_usage(usage(5)),
    ]);

    let response = extractor(model, 1)
        .extract_with_usage("John")
        .await
        .expect("second attempt should succeed");

    assert_eq!(
        response.data,
        Person {
            name: "John".to_string()
        }
    );
    assert_eq!(response.usage.total_tokens, 15);
}

async fn assert_billed_hook_termination_usage(phase: StopFirstBilledResponseAt) {
    let model = MockCompletionModel::new([
        submit_turn("ignored").with_usage(usage(10)),
        submit_turn("John").with_usage(usage(5)),
    ]);
    let response = ExtractorBuilder::<Person>::new(model)
        .retries(1)
        .add_hook(StopFirstBilledResponse {
            phase,
            calls: Arc::new(AtomicUsize::new(0)),
        })
        .build()
        .extract_with_usage("John")
        .await
        .expect("second attempt should succeed");

    assert_eq!(response.data.name, "John");
    assert_eq!(response.usage.total_tokens, 15);
}

#[tokio::test]
async fn completion_response_hook_termination_preserves_billed_usage() {
    assert_billed_hook_termination_usage(StopFirstBilledResponseAt::CompletionResponse).await;
}

#[tokio::test]
async fn model_turn_finished_hook_termination_preserves_billed_usage() {
    assert_billed_hook_termination_usage(StopFirstBilledResponseAt::ModelTurnFinished).await;
}

#[tokio::test]
async fn unexpected_tool_call_preserves_usage_and_retries() {
    let model = MockCompletionModel::new([
        MockTurn::tool_call("unknown", "unexpected", json!({})).with_usage(usage(10)),
        submit_turn("John").with_usage(usage(5)),
    ]);

    let response = extractor(model, 1)
        .extract_with_usage("John")
        .await
        .expect("second attempt should succeed");

    assert_eq!(response.data.name, "John");
    assert_eq!(response.usage.total_tokens, 15);
}

#[tokio::test]
async fn unexpected_tool_call_runs_hooks_before_extractor_fallback() {
    let model = MockCompletionModel::new([
        MockTurn::tool_call("unknown", "unexpected", json!({})).with_usage(usage(10)),
        submit_turn("John").with_usage(usage(5)),
    ]);
    let counts = LifecycleCounts::default();

    let response = ExtractorBuilder::<Person>::new(model)
        .retries(1)
        .add_hook(counts.clone())
        .build()
        .extract_with_usage("John")
        .await
        .expect("deferred invalid call should use extractor fallback");

    assert_eq!(response.data.name, "John");
    assert_eq!(response.usage.total_tokens, 15);
    assert_eq!(counts.invalid_tool_calls.load(Ordering::SeqCst), 1);
    assert_eq!(counts.completion_responses.load(Ordering::SeqCst), 2);
    assert_eq!(counts.model_turns.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn unexpected_tool_call_hook_can_stop_extraction() {
    let model = MockCompletionModel::new([MockTurn::tool_call("unknown", "unexpected", json!({}))]);

    let error = ExtractorBuilder::<Person>::new(model)
        .add_hook(StopOnInvalidToolCall)
        .build()
        .extract("John")
        .await
        .expect_err("invalid-tool hook should retain control");

    assert!(matches!(
        error,
        ExtractionError::PromptError(PromptError::PromptCancelled { reason, .. })
            if reason == "unexpected extractor tool call"
    ));
}

#[tokio::test]
async fn unexpected_tool_call_hook_can_repair_to_submit() {
    let model = MockCompletionModel::new([MockTurn::tool_call(
        "unknown",
        "unexpected",
        json!({ "name": "John" }),
    )]);

    let response = ExtractorBuilder::<Person>::new(model)
        .add_hook(RepairUnexpectedAsSubmit)
        .build()
        .extract("John")
        .await
        .expect("repaired output-tool call should finalize extraction");

    assert_eq!(response.name, "John");
}

#[tokio::test]
async fn skip_hook_preserves_valid_submit_sibling() {
    let turn = MockTurn::from_contents([
        tool_call("unknown", "unexpected", json!({})),
        tool_call("submit", SUBMIT_TOOL_NAME, json!({ "name": "John" })),
    ]);
    let model = MockCompletionModel::new([turn]);

    let response = ExtractorBuilder::<Person>::new(model)
        .add_hook(SkipUnexpected)
        .build()
        .extract("John")
        .await
        .expect("skipping an invalid sibling should preserve submit");

    assert_eq!(response.name, "John");
}

#[tokio::test]
async fn submit_call_wins_over_unexpected_sibling_call() {
    let turn = MockTurn::from_contents([
        tool_call("unknown", "unexpected", json!({})),
        tool_call("submit", SUBMIT_TOOL_NAME, json!({ "name": "John" })),
    ])
    .with_usage(usage(7));
    let model = MockCompletionModel::new([turn]);

    let response = extractor(model, 0)
        .extract_with_usage("John")
        .await
        .expect("submit should remain authoritative");

    assert_eq!(response.data.name, "John");
    assert_eq!(response.usage.total_tokens, 7);
}

#[tokio::test]
async fn submit_call_wins_before_unexpected_sibling_call() {
    let turn = MockTurn::from_contents([
        tool_call("submit", SUBMIT_TOOL_NAME, json!({ "name": "John" })),
        tool_call("unknown", "unexpected", json!({})),
    ]);

    let response = extractor(MockCompletionModel::new([turn]), 0)
        .extract("John")
        .await
        .expect("an earlier submit should remain authoritative");

    assert_eq!(response.name, "John");
}

#[tokio::test]
async fn multiple_unexpected_calls_surrounding_submit_are_ignored() {
    let turn = MockTurn::from_contents([
        tool_call("unknown-before", "unexpected_before", json!({})),
        tool_call("submit", SUBMIT_TOOL_NAME, json!({ "name": "John" })),
        tool_call("unknown-after", "unexpected_after", json!({})),
    ]);

    let response = extractor(MockCompletionModel::new([turn]), 0)
        .extract("John")
        .await
        .expect("unexpected siblings should not displace submit");

    assert_eq!(response.name, "John");
}

#[tokio::test]
async fn transport_errors_contribute_no_usage() {
    let model = MockCompletionModel::new([
        MockTurn::error("boom"),
        submit_turn("John").with_usage(usage(5)),
    ]);

    let response = extractor(model, 1)
        .extract_with_usage("John")
        .await
        .expect("second attempt should succeed");

    assert_eq!(response.usage.total_tokens, 5);
}

#[tokio::test]
async fn single_successful_attempt_reports_its_own_usage() {
    let model = MockCompletionModel::new([submit_turn("John").with_usage(usage(7))]);

    let response = extractor(model, 0)
        .extract_with_usage("John")
        .await
        .expect("extraction should succeed");

    assert_eq!(response.usage.total_tokens, 7);
}

#[tokio::test]
async fn exhausted_retries_return_last_error() {
    let model = MockCompletionModel::new([MockTurn::text("no submit call").with_usage(usage(10))]);

    let err = extractor(model, 0)
        .extract("John")
        .await
        .expect_err("extraction should fail");

    assert!(matches!(err, ExtractionError::NoData));
}

#[tokio::test]
async fn exhausted_retries_return_error_from_final_attempt() {
    let model = MockCompletionModel::new([MockTurn::error("first"), MockTurn::error("second")]);

    let err = extractor(model, 1)
        .extract("John")
        .await
        .expect_err("extraction should fail");

    assert!(matches!(
        err,
        ExtractionError::CompletionError(CompletionError::ProviderError(message))
            if message == "second"
    ));
}
