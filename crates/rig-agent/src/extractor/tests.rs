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
    Usage::new().total_tokens(total_tokens)
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

async fn assert_billed_hook_termination_usage(case: &str, phase: StopFirstBilledResponseAt) {
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
        .unwrap_or_else(|error| panic!("{case}: second attempt should succeed: {error}"));

    assert_eq!(response.output.name, "John", "{case}");
    assert_eq!(response.usage.total_tokens, Some(15), "{case}");
}

#[tokio::test]
async fn hook_termination_preserves_billed_usage_at_either_phase() {
    for (case, phase) in [
        (
            "completion outcome",
            StopFirstBilledResponseAt::CompletionOutcome,
        ),
        (
            "model turn finished",
            StopFirstBilledResponseAt::ModelTurnFinished,
        ),
    ] {
        assert_billed_hook_termination_usage(case, phase).await;
    }
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

/// A seed index that always finds the node with `id`.
struct FixedSeed(&'static str);

impl VectorStoreIndex for FixedSeed {
    type Filter = Filter<serde_json::Value>;

    async fn top_n<T: DeserializeOwned + WasmCompatSend>(
        &self,
        _req: VectorSearchRequest,
    ) -> Result<Vec<VectorSearchResult<T>>, VectorStoreError> {
        Ok(vec![VectorSearchResult {
            score: 1.0,
            id: "row0".to_string(),
            document: serde_json::from_value(json!({ "id": self.0 }))?,
        }])
    }

    async fn top_n_ids(
        &self,
        _req: VectorSearchRequest,
    ) -> Result<Vec<VectorSearchIdResult>, VectorStoreError> {
        Ok(Vec::new())
    }
}

#[tokio::test]
async fn extracted_knowledge_graph_serves_as_an_agents_dynamic_context() {
    use rig_core::graph_store::{
        GraphIndex, GraphStore, KnowledgeGraph, in_memory::InMemoryGraphStore,
    };

    let extraction = MockCompletionModel::from_turns([MockTurn::tool_call(
        "id1",
        SUBMIT_TOOL_NAME,
        json!({
            "entities": [
                { "name": "Lumen cell", "kind": "product", "description": "A battery." },
                { "name": "Mira Okonkwo", "kind": "person", "description": "Its inventor." },
            ],
            "relations": [
                { "source": "Mira Okonkwo", "relation": "invented", "target": "Lumen cell", "description": "" },
                { "source": "Mira Okonkwo", "relation": "works at", "target": "Halvard Dynamics", "description": "" },
            ],
        }),
    )]);
    let extracted = ExtractorBuilder::knowledge_graph(extraction)
        .build()
        .extract("Mira Okonkwo, who works at Halvard Dynamics, invented the Lumen cell.")
        .await
        .expect("extraction should succeed")
        .output;
    let store = InMemoryGraphStore::default();
    store
        .insert_graph(KnowledgeGraph::from(extracted))
        .await
        .expect("insert");

    let model = MockCompletionModel::from_turns([MockTurn::text("Halvard Dynamics")]);
    let probe = model.clone();
    AgentBuilder::new(model)
        .dynamic_context(1, GraphIndex::new(FixedSeed("lumen cell"), store))
        .build()
        .prompt("Where does the inventor of the Lumen cell work?")
        .await
        .expect("prompt should succeed");

    let requests = probe.requests();
    let documents = rig_core::test_utils::sent_documents(requests.first().expect("one request"));
    let (id, text) = documents.first().expect("one graph document");
    assert_eq!(id, "lumen cell");
    let graph: KnowledgeGraph = serde_json::from_str(text).expect("graph json");
    assert_eq!(
        graph
            .edges
            .iter()
            .map(|edge| (
                edge.source.as_str(),
                edge.relation.as_str(),
                edge.target.as_str()
            ))
            .collect::<Vec<_>>(),
        [
            ("mira okonkwo", "invented", "lumen cell"),
            ("mira okonkwo", "works at", "halvard dynamics"),
        ]
    );
}
