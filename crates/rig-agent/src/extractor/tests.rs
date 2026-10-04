use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use serde_json::json;

use super::*;
use crate::agent::{HookContext, ModelTurnAction, OutcomeAction, OutcomeEvent};
use crate::completion::{PromptError, StructuredOutputError, Usage};
use crate::test_utils::{MockCompletionModel, MockTurn};
use rig_core::message::{AssistantContent, ToolCall, ToolFunction};
use rig_core::vector_store::{
    VectorSearchIdResult, VectorSearchRequest, VectorSearchResult, VectorStoreError,
    VectorStoreIndex, request::Filter,
};
use serde::Deserialize;

#[derive(Debug, PartialEq, Deserialize, Serialize, JsonSchema)]
struct Person {
    name: String,
}

fn usage(total_tokens: u64) -> Usage {
    Usage {
        total_tokens: Some(total_tokens),
        ..Usage::default()
    }
}

fn extractor(model: MockCompletionModel, retries: usize) -> Extractor<Person> {
    ExtractorBuilder::new(model).retries(retries).build()
}

fn submit_turn(name: &str) -> MockTurn {
    MockTurn::tool_call("id1", SUBMIT_TOOL_NAME, json!({ "name": name }))
        .with_response_id("extractor-message")
}

fn tool_call(id: &str, name: &str, arguments: serde_json::Value) -> AssistantContent {
    AssistantContent::ToolCall(ToolCall::from_wire(
        id,
        ToolFunction::new(
            rig_core::message::ToolName::new(name.to_string()).expect("tool name"),
            arguments,
        ),
    ))
}

struct ExtractorContextIndex {
    queries: Arc<Mutex<Vec<(String, u64)>>>,
}

impl VectorStoreIndex for ExtractorContextIndex {
    type Filter = Filter<serde_json::Value>;

    async fn top_n<T: DeserializeOwned + WasmCompatSend>(
        &self,
        req: VectorSearchRequest,
    ) -> Result<Vec<VectorSearchResult<T>>, VectorStoreError> {
        self.queries
            .lock()
            .expect("extractor query recorder")
            .push((req.query().to_string(), req.samples()));
        let value = serde_json::from_value(json!({ "question": "retrieved" }))?;
        Ok(vec![VectorSearchResult {
            score: 1.0,
            id: "extractor-context".to_string(),
            document: value,
        }])
    }

    async fn top_n_ids(
        &self,
        _req: VectorSearchRequest,
    ) -> Result<Vec<VectorSearchIdResult>, VectorStoreError> {
        Ok(vec![VectorSearchIdResult {
            score: 1.0,
            id: "extractor-context".to_string(),
        }])
    }
}

#[derive(Clone, Copy)]
enum StopFirstBilledResponseAt {
    CompletionOutcome,
    ModelTurnFinished,
}

#[derive(Clone)]
struct StopFirstBilledResponse {
    phase: StopFirstBilledResponseAt,
    calls: Arc<AtomicUsize>,
}

impl AgentHook for StopFirstBilledResponse {
    async fn on_outcome(&self, _ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        if event.completion().is_some()
            && matches!(self.phase, StopFirstBilledResponseAt::CompletionOutcome)
            && self.calls.fetch_add(1, Ordering::SeqCst) == 0
        {
            OutcomeAction::stop("stop first billed response")
        } else {
            OutcomeAction::proceed()
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
async fn extractor_dynamic_context_uses_the_agent_hook_lifecycle() {
    let model = MockCompletionModel::from_turns([submit_turn("John")]);
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

    assert_eq!(response.output.name, "John");
    assert_eq!(
        *queries.lock().expect("extractor queries"),
        vec![("John".to_string(), 2)]
    );
    let requests = probe.requests();
    let request = requests.first().expect("one extractor request");
    assert!(
        rig_core::test_utils::sent_documents(request)
            .iter()
            .any(|(id, text)| id == "extractor-context"
                && text == "{\n  \"question\": \"retrieved\"\n}")
    );
}

#[tokio::test]
async fn usage_accumulates_across_failed_attempts() {
    let model = MockCompletionModel::from_turns([
        MockTurn::text("no submit call").with_usage(usage(10)),
        submit_turn("John").with_usage(usage(5)),
    ]);

    let response = extractor(model, 1)
        .extract("John")
        .await
        .expect("second attempt should succeed");

    assert_eq!(
        response.output,
        Person {
            name: "John".to_string()
        }
    );
    assert_eq!(response.usage.total_tokens, Some(15));
}

async fn assert_billed_hook_termination_usage(phase: StopFirstBilledResponseAt) {
    let model = MockCompletionModel::from_turns([
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
        .extract("John")
        .await
        .expect("second attempt should succeed");

    assert_eq!(response.output.name, "John");
    assert_eq!(response.usage.total_tokens, Some(15));
}

#[tokio::test]
async fn completion_outcome_hook_termination_preserves_billed_usage() {
    assert_billed_hook_termination_usage(StopFirstBilledResponseAt::CompletionOutcome).await;
}

#[tokio::test]
async fn model_turn_finished_hook_termination_preserves_billed_usage() {
    assert_billed_hook_termination_usage(StopFirstBilledResponseAt::ModelTurnFinished).await;
}

#[tokio::test]
async fn unexpected_tool_call_preserves_usage_and_retries() {
    let model = MockCompletionModel::from_turns([
        MockTurn::tool_call("unknown", "unexpected", json!({})).with_usage(usage(10)),
        submit_turn("John").with_usage(usage(5)),
    ]);

    let response = extractor(model, 1)
        .extract("John")
        .await
        .expect("second attempt should succeed");

    assert_eq!(response.output.name, "John");
    assert_eq!(response.usage.total_tokens, Some(15));
}

#[tokio::test]
async fn skip_hook_preserves_valid_submit_sibling() {
    let turn = MockTurn::from_contents([
        tool_call("unknown", "unexpected", json!({})),
        tool_call("submit", SUBMIT_TOOL_NAME, json!({ "name": "John" })),
    ]);
    let model = MockCompletionModel::from_turns([turn]);

    let response = ExtractorBuilder::<Person>::new(model)
        .add_hook(SkipUnexpected)
        .build()
        .extract("John")
        .await
        .expect("skipping an invalid sibling should preserve submit");

    assert_eq!(response.output.name, "John");
}

#[tokio::test]
async fn exhausted_retries_return_last_error() {
    let model =
        MockCompletionModel::from_turns([MockTurn::text("no submit call").with_usage(usage(10))]);

    let err = extractor(model, 0)
        .extract("John")
        .await
        .expect_err("extraction should fail");

    assert!(matches!(err, StructuredOutputError::EmptyResponse));
}

#[tokio::test]
async fn exhausted_retries_return_error_from_final_attempt() {
    let model =
        MockCompletionModel::from_turns([MockTurn::error("first"), MockTurn::error("second")]);

    let err = extractor(model, 1)
        .extract("John")
        .await
        .expect_err("extraction should fail");

    assert!(matches!(
        err,
        StructuredOutputError::Prompt(err)
            if matches!(
                err,
                PromptError::Report(ref report)
                    if report.kind == rig_core::error::ErrorKind::Provider
                        && report.message.ends_with("second")
            )
    ));
}
