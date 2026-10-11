use std::path::PathBuf;
use std::sync::mpsc::Receiver;

use rig_cassette::journal::MemoryStore;
use rig_core::test_utils::{MockCompletionModel, MockError, MockStreamEvent};
use rig_harness_test_support::{app_on, calls, connect, reply, run_until};
use rig_telemetry::{Spending, UsagePlugin};
use schemars::JsonSchema;

use super::*;

/// Lines the scripted `shell` prints.
const LINES: usize = 2000;

/// A `shell` that prints [`LINES`] numbered lines, whatever it is asked.
struct Shell;

#[derive(Deserialize, JsonSchema)]
struct Args {
    /// The command.
    #[allow(dead_code)]
    command: String,
}

impl PortableTool for Shell {
    const NAME: &'static str = "shell";
    type Args = Args;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Run a command.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<Args>()
    }

    async fn call(&self, _args: Args) -> Result<String, ToolExecutionError> {
        Ok((1..=LINES).map(|n| format!("line {n}\n")).collect())
    }
}

/// How each turn ended.
#[derive(Resource, Default)]
struct Ended(usize);

/// One turn of an agent answered by `main`, whose outputs `cheap`
/// summarizes above 100 tokens into `dir`; the app and the agent's entity.
fn run(
    main: &MockCompletionModel,
    cheap: &MockCompletionModel,
    dir: PathBuf,
) -> Option<(App, Entity)> {
    let store = MemoryStore::default();
    let (mut app, wakes): (App, Receiver<()>) = app_on(&store);
    let config = SummarizeOutputs {
        threshold_tokens: 100,
        dir,
        ..SummarizeOutputs::default()
    };
    let cheap = connect(cheap).ok_or_else(|| "the catalog lists the model".to_owned());
    app.set_error_handler(bevy_ecs::error::warn)
        .add_plugins((
            AgentPlugin,
            JournalPlugin,
            SummarizeToolOutputPlugin,
            UsagePlugin,
        ))
        .insert_resource(config)
        .insert_resource(CheapModel(cheap))
        .add_tool(Shell)
        .init_resource::<Ended>()
        .add_observer(|_: On<TurnEnded>, mut ended: ResMut<Ended>| ended.0 += 1);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next()?;
    world.entity_mut(agent).insert(connect(main)?);
    world.trigger(Deliver::new(agent, "Run the tests.", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| world.resource::<Ended>().0 > 0);
    Some((app, agent))
}

/// A main model that runs `shell` once, then answers.
fn main_model() -> MockCompletionModel {
    MockCompletionModel::from_stream_turns([
        calls(
            "call-1",
            "shell",
            serde_json::json!({ "command": "seq 2000" }),
        ),
        reply("done"),
    ])
}

/// The tool result the main model's second request sent, as JSON.
fn second_request(main: &MockCompletionModel) -> String {
    let requests = main.requests();
    let second = requests.get(1);
    second
        .and_then(|request| serde_json::to_string(request).ok())
        .unwrap_or_default()
}

/// A fresh directory for a test's saved outputs, under this crate.
fn fresh(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test-outputs")
        .join(name);
    std::fs::remove_dir_all(&dir).ok();
    dir
}

/// Removes `dir`, and its parent once empty.
fn clean(dir: &Path) {
    std::fs::remove_dir_all(dir).ok();
    if let Some(parent) = dir.parent() {
        std::fs::remove_dir(parent).ok();
    }
}

#[test]
fn a_big_output_is_saved_and_summarized_by_the_cheap_model() {
    let dir = fresh("summarized");
    let (main, cheap) = (
        main_model(),
        MockCompletionModel::from_stream_turns([reply("SUMMARY: 2000 lines, no errors")]),
    );
    let ran = run(&main, &cheap, dir.clone());
    assert!(ran.is_some(), "an agent on the scripted model");
    let Some((app, agent)) = ran else {
        return;
    };
    let saved = dir.join("call-1.txt");
    let sent = second_request(&main);
    assert!(sent.contains("SUMMARY: 2000 lines, no errors"), "{sent}");
    assert!(sent.contains(&saved.display().to_string()), "{sent}");
    assert!(!sent.contains("line 1000"), "the output is not sent whole");
    let whole = std::fs::read_to_string(&saved).unwrap_or_default();
    assert_eq!(whole.lines().count(), LINES);
    assert_eq!(cheap.request_count(), 1);
    // The summary counts in the agent's usage, and on its own as the
    // summarizer's.
    let spent = app
        .world()
        .get::<Spending>(agent)
        .map(|spent| spent.0.calls);
    let summarizer = app.world().get::<SummarizerSpending>(agent);
    assert_eq!(spent, Some(3));
    assert_eq!(summarizer.map(|spent| spent.0.calls), Some(1));
    clean(&dir);
}

#[test]
fn a_failed_summary_sends_the_head_and_tail() {
    let dir = fresh("failed");
    let failure = vec![MockStreamEvent::Error(MockError::provider("down"))];
    let (main, cheap) = (
        main_model(),
        MockCompletionModel::from_stream_turns([failure]),
    );
    assert!(run(&main, &cheap, dir.clone()).is_some());
    let sent = second_request(&main);
    assert!(sent.contains("line 1\\n"), "the head: {sent}");
    assert!(sent.contains("line 2000"), "the tail: {sent}");
    assert!(!sent.contains("line 1000\\n"), "the middle is cut: {sent}");
    assert!(sent.contains("the summarizer failed"), "{sent}");
    assert!(sent.contains(&dir.join("call-1.txt").display().to_string()));
    clean(&dir);
}
