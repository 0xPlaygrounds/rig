use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use futures::StreamExt;
use serde_json::json;

use crate::{
    agent::{AgentBuilder, AgentHook, HookContext, OutcomeAction, OutcomeEvent},
    completion::{CompletionModel, Document},
    test_utils::{MockCompletionModel, MockStreamEvent, MockTurn},
    tool::{Tool, ToolContext, ToolErrorKind, ToolExecutionError},
};
use rig_core::message::ToolChoice;

struct MetadataFailingTool;

#[derive(serde::Serialize, serde::Deserialize)]
struct SnapshotValue {
    value: usize,
}

impl rig_core::tool::ContextValue for SnapshotValue {
    const KEY: &'static str = "test.snapshot_value";
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SnapshotResult(usize);

impl rig_core::tool::ContextValue for SnapshotResult {
    const KEY: &'static str = "test.snapshot_result";
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ResultMetadata(String);

impl rig_core::tool::ContextValue for ResultMetadata {
    const KEY: &'static str = "test.result_metadata";
}

#[derive(Clone, Default)]
struct SnapshotMutatingTool(Arc<Mutex<Vec<usize>>>);

impl Tool for SnapshotMutatingTool {
    const NAME: &'static str = "snapshot_mutator";
    type Error = rig::tool::ToolExecutionError;
    type Args = serde_json::Value;
    type Output = String;

    fn description(&self) -> String {
        "Mutates its per-dispatch context snapshot".into()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({"type": "object", "properties": {}})
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, ToolExecutionError> {
        let initial = context.require::<SnapshotValue>()?.value;
        self.0.lock().expect("observed values").push(initial);
        let updated = initial + 1;
        context.insert(SnapshotValue { value: updated })?;
        context.insert_result(SnapshotResult(updated))?;
        Ok(updated.to_string())
    }
}

#[derive(Clone, Default)]
struct SnapshotResults(Arc<Mutex<Vec<usize>>>);

impl AgentHook for SnapshotResults {
    async fn on_outcome(&self, _ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        let Some(context) = event.tool_context() else {
            return OutcomeAction::proceed();
        };
        self.0.lock().expect("result values").push(
            context
                .require_result::<SnapshotResult>()
                .expect("per-dispatch result metadata")
                .0,
        );
        OutcomeAction::proceed()
    }
}

impl Tool for MetadataFailingTool {
    const NAME: &'static str = "flaky_tool";
    type Error = rig::tool::ToolExecutionError;
    type Args = serde_json::Value;
    type Output = String;

    fn description(&self) -> String {
        "Fails after attaching result metadata".into()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({"type": "object", "properties": {}})
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, ToolExecutionError> {
        context.insert_result(ResultMetadata("shared-result-metadata".to_string()))?;
        Err(ToolExecutionError::timeout("raw timeout failure"))
    }
}

#[derive(Clone, Default)]
struct Results(Arc<Mutex<Vec<(ToolErrorKind, String, String)>>>);

impl AgentHook for Results {
    async fn on_outcome(&self, _ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        let Some(result) = event.tool_result() else {
            return OutcomeAction::proceed();
        };
        if let Some(error) = result.error() {
            self.0.lock().expect("results").push((
                error.kind(),
                result.output().render(),
                event
                    .tool_context()
                    .expect("tool outcome carries its context")
                    .result::<ResultMetadata>()
                    .expect("tool result metadata decodes")
                    .expect("tool result metadata")
                    .0,
            ));
        }
        OutcomeAction::rewrite_tool_result(&event, "rewritten for model")
    }
}

#[test]
fn agent_exposes_read_only_name_and_description() {
    let named = AgentBuilder::new(MockCompletionModel::text("done"))
        .name("researcher")
        .description("Finds evidence")
        .build();
    assert_eq!(named.name(), Some("researcher"));
    assert_eq!(named.description(), Some("Finds evidence"));

    let unnamed = AgentBuilder::new(MockCompletionModel::text("done")).build();
    assert_eq!(unnamed.name(), None);
    assert_eq!(unnamed.description(), None);
}

#[tokio::test]
async fn runner_applies_per_run_request_overrides() {
    let model = MockCompletionModel::text("done");
    AgentBuilder::new(model.clone())
        .preamble("baseline preamble")
        .context("baseline document")
        .temperature(0.1)
        .max_tokens(10)
        .additional_params(json!({"baseline": true}))
        .build()
        .runner("go")
        .preamble("run preamble")
        .document(Document {
            id: "run-one".into(),
            text: "first run document".into(),
            additional_props: Default::default(),
        })
        .documents([Document {
            id: "run-two".into(),
            text: "second run document".into(),
            additional_props: Default::default(),
        }])
        .temperature(0.7)
        .max_tokens(42)
        .replace_additional_params(json!({"override": true}))
        .tool_choice(ToolChoice::None)
        .run()
        .await
        .expect("runner request should succeed");

    let requests = model.requests();
    let request = requests.first().expect("one request");
    assert!(request.chat_history.iter().any(
        |message| matches!(message, crate::completion::Message::System { content } if content == "run preamble")
    ));
    assert!(
        request
            .documents
            .iter()
            .any(|document| document.text == "baseline document")
    );
    assert!(
        request
            .documents
            .iter()
            .any(|document| document.id == "run-one")
    );
    assert!(
        request
            .documents
            .iter()
            .any(|document| document.id == "run-two")
    );
    assert_eq!(request.temperature, Some(0.7));
    assert_eq!(request.max_tokens, Some(42));
    assert_eq!(request.additional_params, Some(json!({"override": true})));
    assert_eq!(request.tool_choice, Some(ToolChoice::None));
}

#[tokio::test]
async fn runner_can_merge_additional_params_into_the_baseline() {
    let model = MockCompletionModel::text("done");
    AgentBuilder::new(model.clone())
        .additional_params(json!({"baseline": true, "winner": "baseline"}))
        .build()
        .runner("go")
        .merge_additional_params(
            json!({"override": true, "winner": "runner"})
                .as_object()
                .expect("object")
                .clone(),
        )
        .run()
        .await
        .expect("runner request should succeed");

    assert_eq!(
        model
            .requests()
            .first()
            .expect("one request")
            .additional_params,
        Some(json!({"baseline": true, "override": true, "winner": "runner"}))
    );
}

#[tokio::test]
async fn runner_can_replace_additional_params_wholesale() {
    let model = MockCompletionModel::text("done");
    AgentBuilder::new(model.clone())
        .additional_params(json!({"baseline": true}))
        .build()
        .runner("go")
        .replace_additional_params(json!({"replacement": true}))
        .run()
        .await
        .expect("runner request should succeed");

    let requests = model.requests();
    let request = requests.first().expect("one request");
    assert_eq!(
        request.additional_params,
        Some(json!({"replacement": true}))
    );
}

#[tokio::test]
async fn runner_can_clear_configured_request_defaults() {
    let model = MockCompletionModel::text("done");
    AgentBuilder::new(model.clone())
        .preamble("baseline")
        .temperature(0.1)
        .max_tokens(10)
        .additional_params(json!({"baseline": true}))
        .tool_choice(ToolChoice::Required)
        .build()
        .runner("go")
        .without_preamble()
        .without_temperature()
        .without_max_tokens()
        .without_additional_params()
        .without_tool_choice()
        .run()
        .await
        .expect("runner request should succeed");

    let requests = model.requests();
    let request = requests.first().expect("one request");
    assert!(
        !request
            .chat_history
            .iter()
            .any(|message| matches!(message, crate::completion::Message::System { .. }))
    );
    assert_eq!(request.temperature, None);
    assert_eq!(request.max_tokens, None);
    assert_eq!(request.additional_params, None);
    assert_eq!(request.tool_choice, None);
}

#[tokio::test]
async fn direct_completion_model_requests_are_intentionally_hook_free() {
    #[derive(Clone)]
    struct CountCompletionCalls(Arc<AtomicUsize>);

    impl AgentHook for CountCompletionCalls {
        async fn on_completion_call(
            &self,
            _ctx: &HookContext,
            _event: crate::agent::CompletionCallEvent<'_>,
        ) -> crate::agent::CompletionCallAction {
            self.0.fetch_add(1, Ordering::SeqCst);
            crate::agent::CompletionCallAction::Continue
        }
    }

    let model = MockCompletionModel::text("raw response");
    let calls = Arc::new(AtomicUsize::new(0));
    let _agent = AgentBuilder::new(model.clone())
        .add_hook(CountCompletionCalls(calls.clone()))
        .build();

    model
        .completion_request("raw request")
        .send()
        .await
        .expect("direct model request should succeed");

    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(model.request_count(), 1);
}

#[tokio::test]
async fn blocking_and_streaming_preserve_raw_failure_while_rewriting_presentation() {
    let blocking = Results::default();
    let blocking_model = MockCompletionModel::from_turns([
        MockTurn::tool_call("tc1", "flaky_tool", json!({})),
        MockTurn::text("done"),
    ]);
    AgentBuilder::new(blocking_model.clone())
        .tool(MetadataFailingTool)
        .add_hook(blocking.clone())
        .build()
        .runner("go")
        .max_turns(3)
        .run()
        .await
        .expect("blocking run");

    let streaming = Results::default();
    let streaming_model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call_name_delta("tc1", "flaky_tool"),
            MockStreamEvent::tool_call_arguments_delta("tc1", "{}"),
            MockStreamEvent::tool_call("tc1", "flaky_tool", json!({})),
            MockStreamEvent::final_response_with_total_tokens(0),
        ],
        vec![
            MockStreamEvent::text("done"),
            MockStreamEvent::final_response_with_total_tokens(0),
        ],
    ]);
    let mut stream = AgentBuilder::new(streaming_model.clone())
        .tool(MetadataFailingTool)
        .add_hook(streaming.clone())
        .build()
        .runner("go")
        .max_turns(3)
        .stream()
        .await;
    while let Some(item) = stream.next().await {
        item.expect("stream item");
    }

    assert_eq!(*blocking.0.lock().unwrap(), *streaming.0.lock().unwrap());
    assert_eq!(
        *blocking.0.lock().unwrap(),
        vec![(
            ToolErrorKind::Timeout,
            "raw timeout failure".into(),
            "shared-result-metadata".into()
        )]
    );

    let blocking_history = serde_json::to_value(
        &blocking_model
            .requests()
            .get(1)
            .expect("second blocking request")
            .chat_history,
    )
    .unwrap();
    let streaming_history = serde_json::to_value(
        &streaming_model
            .requests()
            .get(1)
            .expect("second streaming request")
            .chat_history,
    )
    .unwrap();
    assert_eq!(blocking_history, streaming_history);
    let history = blocking_history.to_string();
    assert!(history.contains("rewritten for model"));
    assert!(!history.contains("raw timeout failure"));
}

#[tokio::test]
async fn agent_dispatch_snapshot_isolates_tool_mutations() {
    let mut context = ToolContext::new();
    context.insert(SnapshotValue { value: 0 }).unwrap();
    let tool = SnapshotMutatingTool::default();
    let results = SnapshotResults::default();

    AgentBuilder::new(MockCompletionModel::from_turns([
        MockTurn::tool_call("tc1", SnapshotMutatingTool::NAME, json!({})),
        MockTurn::tool_call("tc2", SnapshotMutatingTool::NAME, json!({})),
        MockTurn::text("done"),
    ]))
    .tool(tool.clone())
    .add_hook(results.clone())
    .build()
    .runner("go")
    .tool_context(context)
    .max_turns(4)
    .run()
    .await
    .expect("agent run");

    assert_eq!(*tool.0.lock().expect("observed values"), vec![0, 0]);
    assert_eq!(*results.0.lock().expect("result values"), vec![1, 1]);
}
