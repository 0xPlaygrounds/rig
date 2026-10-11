use std::sync::{Arc, Mutex};

use opentelemetry::Value;
use opentelemetry::trace::Status;
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::trace::{SpanData, SpanExporter};
use rig_cassette::journal::MemoryStore;
use rig_core::catalog::{Catalog, Connector};
use rig_core::completion::Usage;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use super::*;
use rig_harness_test_support::{app_on, calls, connect, reply, run_until};
use rig_subagents::{SubagentsPlugin, TASK};

/// The spans exported, kept in memory.
#[derive(Debug, Clone, Default)]
struct Memory(Arc<Mutex<Vec<SpanData>>>);

impl SpanExporter for Memory {
    async fn export(&self, batch: Vec<SpanData>) -> OTelSdkResult {
        if let Ok(mut spans) = self.0.lock() {
            spans.extend(batch);
        }
        Ok(())
    }
}

#[derive(Deserialize, JsonSchema)]
struct ReadArgs {
    /// The file.
    #[allow(dead_code)]
    path: String,
}

/// A `read` tool that answers every call with the same text.
struct Read;

impl PortableTool for Read {
    const NAME: &'static str = "read";
    type Args = ReadArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Read a file.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<ReadArgs>()
    }

    async fn call(&self, _args: ReadArgs) -> Result<String, ToolExecutionError> {
        Ok("hello".to_owned())
    }
}

/// Spec 01's denial, reduced: every `read` call is refused before it runs.
fn deny(called: On<ToolCalled<ReadArgs>>, mut commands: Commands) {
    let denied = ToolExecutionError::refused("The user denied this command");
    commands
        .entity(called.call)
        .insert(ToolOutput(ToolResult::failed(denied)));
}

/// How many turns ended.
#[derive(Resource, Default)]
struct Ended(usize);

/// A reply that ends with this usage.
fn usage() -> MockStreamEvent {
    MockStreamEvent::final_response(
        Usage::new()
            .input_tokens(120)
            .output_tokens(8)
            .cached_input_tokens(100),
    )
}

/// Runs one turn, "read a.txt", answered by a `read` call then text, with
/// `read` answered or denied, and returns the spans exported.
fn one_turn(denied: bool) -> Vec<SpanData> {
    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    let memory = Memory::default();
    app.insert_resource(Tracing::new(memory.clone(), "memory", true))
        .add_plugins((AgentPlugin, JournalPlugin, OtelTracingPlugin))
        .init_resource::<Ended>()
        .add_observer(|_: On<TurnEnded>, mut ended: ResMut<Ended>| ended.0 += 1);
    match denied {
        true => app.add_open_tool("read", "Read a file.", ToolOptions::default(), deny),
        false => app.add_tool(Read),
    };
    app.finish();
    app.update();
    let mut read = calls("c1", "read", json!({ "path": "a.txt" }));
    read.pop();
    read.push(usage());
    let model = MockCompletionModel::from_stream_turns([read, reply("It says hello.")]);
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next();
    let (Some(agent), Some(connection)) = (agent, connect(&model)) else {
        return Vec::new();
    };
    world.entity_mut(agent).insert(connection);
    world.trigger(Deliver::new(agent, "read a.txt", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| world.resource::<Ended>().0 > 0);
    app.world().resource::<Tracing>().flush().ok();
    memory
        .0
        .lock()
        .map(|spans| spans.clone())
        .unwrap_or_default()
}

fn named<'a>(spans: &'a [SpanData], name: &str) -> Vec<&'a SpanData> {
    spans.iter().filter(|span| span.name == name).collect()
}

fn attribute<'a>(span: &'a SpanData, key: &str) -> Option<&'a Value> {
    span.attributes
        .iter()
        .find(|attribute| attribute.key.as_str() == key)
        .map(|attribute| &attribute.value)
}

#[test]
fn a_turn_is_one_trace_of_its_model_and_tool_calls() {
    let spans = one_turn(false);
    let turns = named(&spans, "turn");
    let mut models = named(&spans, "model.request");
    models.sort_by_key(|span| span.start_time);
    let tools = named(&spans, "tool.call");
    let found = (turns.len(), models.len(), tools.len());
    assert_eq!(found, (1, 2, 1), "{spans:#?}");
    let (Some(turn), Some(first), Some(second), Some(tool)) =
        (turns.first(), models.first(), models.get(1), tools.first())
    else {
        return;
    };
    let trace = turn.span_context.trace_id();
    for child in [first, second, tool] {
        assert_eq!(child.parent_span_id, turn.span_context.span_id());
        assert_eq!(child.span_context.trace_id(), trace);
    }
    assert_eq!(attribute(tool, "gen_ai.tool.name"), Some(&"read".into()));
    assert_eq!(
        attribute(tool, "gen_ai.tool.call.result"),
        Some(&"hello".into())
    );
    assert_eq!(tool.status, Status::Ok);
    assert!(first.end_time <= tool.start_time && tool.end_time <= second.start_time);
    assert_eq!(
        attribute(first, "gen_ai.usage.input_tokens"),
        Some(&120.into())
    );
    assert_eq!(
        attribute(first, "gen_ai.usage.output_tokens"),
        Some(&8.into())
    );
    assert_eq!(
        attribute(first, "gen_ai.usage.cache_read.input_tokens"),
        Some(&100.into())
    );
    assert_eq!(
        attribute(turn, "rig.turn.outcome"),
        Some(&"answered".into())
    );
    assert_eq!(attribute(turn, "rig.user_message.length"), Some(&10.into()));
}

#[test]
fn a_denied_tool_call_ends_its_span_as_denied() {
    let spans = one_turn(true);
    let tools = named(&spans, "tool.call");
    assert_eq!(tools.len(), 1, "{spans:#?}");
    let Some(tool) = tools.first() else {
        return;
    };
    assert_eq!(tool.status, Status::error("denied"));
    assert_eq!(attribute(tool, "rig.tool.status"), Some(&"denied".into()));
    assert_eq!(named(&spans, "model.request").len(), 2);
}

#[test]
fn a_subagent_turn_is_a_child_of_the_task_call_that_started_it() {
    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    let memory = Memory::default();
    app.insert_resource(Tracing::new(memory.clone(), "memory", false))
        .insert_resource(Models(Connector::new(Catalog::default())))
        .add_plugins((
            AgentPlugin,
            JournalPlugin,
            SubagentsPlugin,
            OtelTracingPlugin,
        ))
        .init_resource::<Ended>()
        .add_observer(|_: On<TurnEnded>, mut ended: ResMut<Ended>| ended.0 += 1);
    // Parent and subagent share the model; after the `task` call every
    // reply is text, so which of them gets which does not matter.
    let task = json!({ "description": "Look", "prompt": "Look around." });
    let model = MockCompletionModel::from_stream_turns([
        calls("t1", TASK, task),
        reply("Started."),
        reply("Nothing here."),
        reply("Done."),
    ]);
    let Some(connection) = connect(&model) else {
        return;
    };
    let choice = ModelChoice(connection.spec.reference());
    let parent = app.world_mut().spawn((Agent, connection, choice)).id();
    app.finish();
    app.update();
    app.world_mut()
        .trigger(Deliver::new(parent, "look around", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| world.resource::<Ended>().0 >= 2);
    app.world().resource::<Tracing>().flush().ok();
    let spans = memory
        .0
        .lock()
        .map(|spans| spans.clone())
        .unwrap_or_default();
    let tasks: Vec<_> = named(&spans, "tool.call")
        .into_iter()
        .filter(|span| attribute(span, "gen_ai.tool.name") == Some(&TASK.into()))
        .collect();
    let subturns: Vec<_> = named(&spans, "turn")
        .into_iter()
        .filter(|span| attribute(span, "rig.agent.subagent") == Some(&true.into()))
        .collect();
    assert_eq!((tasks.len(), subturns.len()), (1, 1), "{spans:#?}");
    let (Some(task), Some(subturn)) = (tasks.first(), subturns.first()) else {
        return;
    };
    assert_eq!(subturn.parent_span_id, task.span_context.span_id());
    assert_eq!(
        subturn.span_context.trace_id(),
        task.span_context.trace_id()
    );
}
