use std::sync::{Arc, Mutex};

use futures::StreamExt;
use serde_json::json;

use crate::{
    agent::{AgentBuilder, AgentHook, HookContext, OutcomeAction, OutcomeEvent},
    completion::Document,
    test_utils::{MockCompletionModel, MockStreamEvent, MockTurn},
    tool::{Tool, ToolContext, ToolErrorKind, ToolExecutionError},
};
use rig_core::message::ToolChoice;

struct MetadataFailingTool;

#[derive(serde::Serialize, serde::Deserialize)]
struct ResultMetadata(String);

impl rig_core::tool::ContextValue for ResultMetadata {
    const KEY: &'static str = "test.result_metadata";
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
        .prompt("go")
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
    let documents = rig_core::test_utils::sent_documents(request);
    assert!(
        documents
            .iter()
            .any(|(_, text)| text == "baseline document")
    );
    assert!(documents.iter().any(|(id, _)| id == "run-one"));
    assert!(documents.iter().any(|(id, _)| id == "run-two"));
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
        .prompt("go")
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
async fn the_agents_options_reach_the_request_and_a_runs_options_overlay_them() {
    use rig_core::completion::{CacheRetention, Effort, GenerationOptions, Reasoning};

    let model = MockCompletionModel::from_turns([MockTurn::text("one"), MockTurn::text("two")]);
    let agent = AgentBuilder::new(model.clone())
        .options(GenerationOptions::default().reasoning(Effort::High).seed(7))
        .build();
    agent
        .prompt("go")
        .run()
        .await
        .expect("the agent's request succeeds");
    agent
        .prompt("again")
        .options(
            GenerationOptions::default()
                .cache(CacheRetention::Long)
                .seed(9),
        )
        .run()
        .await
        .expect("the run's request succeeds");

    let requests = model.requests();
    let [first, second] = requests.as_slice() else {
        panic!("two requests: {requests:?}");
    };
    assert_eq!(
        first.options,
        GenerationOptions::default().reasoning(Effort::High).seed(7)
    );
    assert_eq!(
        second.options.reasoning,
        Some(Reasoning::Effort(Effort::High))
    );
    assert_eq!(second.options.cache, Some(CacheRetention::Long));
    assert_eq!(second.options.seed, Some(9));
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
        .prompt("go")
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
        .prompt("go")
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
        .prompt("go")
        .max_turns(3)
        .stream();
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

/// A runner's content-telemetry opt-in overrides the agent's, for that run
/// only.
#[test]
fn a_runner_overrides_content_telemetry_for_its_run() {
    let agent = AgentBuilder::new(MockCompletionModel::text("ok")).build();
    let runner = agent.prompt("go").record_content_telemetry(true);
    assert!(runner.config.record_telemetry_content);
    assert!(!agent.config.record_telemetry_content);
    assert!(
        !runner
            .record_content_telemetry(false)
            .config
            .record_telemetry_content
    );
}
