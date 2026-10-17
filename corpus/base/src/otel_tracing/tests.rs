use std::sync::{Arc, Mutex};

use opentelemetry::Value;
use opentelemetry::trace::Status;
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::trace::{SpanData, SpanExporter};
use rig_core::catalog::{Catalog, Connector};
use rig_core::completion::Usage;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::MemoryStore;
use schemars::JsonSchema;
use serde_json::json;

use super::*;
use rig_basics::subagents::{SubagentsPlugin, TASK};
use rig_harness_test_support::{Ended, app_on, calls, connect, log_ended, reply, run_until};

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

impl Memory {
    /// What was exported once `tracing` flushed.
    fn flushed(&self, tracing: &Tracing) -> Vec<SpanData> {
        tracing.flush().ok();
        self.0.lock().map(|spans| spans.clone()).unwrap_or_default()
    }
}

#[derive(Deserialize, JsonSchema)]
struct ReadArgs {
    /// The file.
    path: String,
}

/// A `read` of a project whose only file, `a.txt`, says `hello`.
fn read(called: On<ToolCalled<ReadArgs>>, mut commands: Commands) {
    let text = if called.args.path == "a.txt" {
        "hello"
    } else {
        ""
    };
    let read = Outcome::from(ToolResult::success(text.into()));
    commands.entity(called.call).insert_if_new(read);
}

/// Spec 01's denial, reduced to its gate: every `read` call is refused
/// before it runs.
fn deny(calls: Query<(Entity, &ToolRequest), ReadyToStart>, mut commands: Commands) {
    for (call, request) in &calls {
        if request.call.function.name.as_str() == "read" {
            let denied = Outcome::refused("The user denied this command");
            commands.entity(call).insert_if_new(denied);
        }
    }
}

/// An app tracing into memory on a session with no catalog model.
fn traced_app(store: &MemoryStore) -> (App, std::sync::mpsc::Receiver<()>, Memory) {
    let (mut app, wakes) = app_on(store);
    let memory = Memory::default();
    app.insert_resource(Tracing::new(memory.clone(), "memory", true))
        .insert_resource(Models(Connector::new(Catalog::default())))
        .add_plugins((AgentPlugin, SessionLogPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    (app, wakes, memory)
}

/// Runs one turn, "read a.txt", answered by a `read` call then text, with
/// `read` answered or denied, and returns the spans exported.
fn one_turn(denied: bool) -> Vec<SpanData> {
    let store = MemoryStore::default();
    let (mut app, wakes, memory) = traced_app(&store);
    app.add_plugins(OtelTracingPlugin).add_open_tool(
        "read",
        "Read a file.",
        ToolOptions::default(),
        read,
    );
    if denied {
        app.add_systems(AgentPass, deny.in_set(Stage::Gate));
    }
    let mut read = calls("c1", "read", json!({ "path": "a.txt" }));
    read.pop();
    let usage = Usage::new()
        .input_tokens(120)
        .output_tokens(8)
        .cached_input_tokens(100);
    read.push(MockStreamEvent::final_response(usage));
    let model = MockCompletionModel::from_stream_turns([read, reply("It says hello.")]);
    let Some(connection) = connect(&model) else {
        return Vec::new();
    };
    let agent = app.world_mut().spawn((Agent, connection)).id();
    app.finish();
    app.update();
    app.world_mut()
        .trigger(Deliver::new(agent, "read a.txt", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    memory.flushed(app.world().resource::<Tracing>())
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
    let tokens = [
        ("gen_ai.usage.input_tokens", 120),
        ("gen_ai.usage.output_tokens", 8),
        ("gen_ai.usage.cache_read.input_tokens", 100),
    ];
    for (key, value) in tokens {
        assert_eq!(attribute(first, key), Some(&value.into()), "{key}");
    }
    // The first reply only calls a tool; both are timed to their first item.
    for model in [first, second] {
        let waited = attribute(model, "gen_ai.response.time_to_first_chunk_ms");
        assert!(waited.is_some(), "{model:#?}");
    }
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
    let (mut app, wakes, memory) = traced_app(&store);
    app.add_plugins((SubagentsPlugin, OtelTracingPlugin));
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
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() >= 2
    });
    let spans = memory.flushed(app.world().resource::<Tracing>());
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
