use rig_basics::coding_tools::ShellTool;
use rig_basics::telemetry::usage::{AgentUsage, UsagePlugin};
use rig_core::test_utils::{MockCompletionModel, MockError, MockStreamEvent};
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{
    Ended, app_on, calls, connect, first_agent, log_ended, reply, run_until, scripted_models,
};

use super::*;

/// One turn of an agent on `main`, with the real `shell`, in a project at
/// `root`, whose outputs over 100 tokens `cheap` summarizes; the app and
/// the agent's entity.
fn run(
    main: &MockCompletionModel,
    cheap: &MockCompletionModel,
    root: &Path,
) -> Option<(App, Entity)> {
    run_on(
        main,
        scripted_models([("corpus-summarizer/cheap", cheap)]),
        root,
    )
}

/// [`run`], with the cheap model among `models`.
fn run_on(main: &MockCompletionModel, models: Models, root: &Path) -> Option<(App, Entity)> {
    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    let settings =
        "[summarize_outputs]\nthreshold_tokens = 100\nmodel = \"corpus-summarizer/cheap\"";
    app.insert_resource(Settings::from_toml(settings))
        .insert_resource(ProjectRoot(root.to_owned()))
        .insert_resource(models)
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            ShellTool,
            SummarizeToolOutputPlugin,
            UsagePlugin,
        ))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = first_agent(world)?;
    world.entity_mut(agent).insert(connect(main)?);
    world.trigger(Deliver::new(agent, "Run the tests.", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    Some((app, agent))
}

/// A main model that runs `command` in `shell` once, then answers.
fn main_model(command: &str) -> MockCompletionModel {
    MockCompletionModel::from_stream_turns([
        calls("call-1", "shell", serde_json::json!({ "command": command })),
        reply("done"),
    ])
}

/// The main model's second request, as JSON.
fn second_request(main: &MockCompletionModel) -> String {
    let requests = main.requests();
    let second = requests.get(1);
    second
        .and_then(|request| serde_json::to_string(request).ok())
        .unwrap_or_default()
}

/// A fresh project directory for a test, under this crate's target-free
/// test folder; removed by [`clean`].
fn fresh(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test-projects")
        .join(name);
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).ok();
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
    let root = fresh("summarized");
    let cheap = MockCompletionModel::from_stream_turns([reply("SUMMARY: 2000 lines, no errors")]);
    let main = main_model("seq 2000");
    let ran = run(&main, &cheap, &root);
    assert!(ran.is_some(), "an agent on the scripted model");
    let Some((app, agent)) = ran else {
        return;
    };
    let sent = second_request(&main);
    assert!(sent.contains("SUMMARY: 2000 lines, no errors"), "{sent}");
    assert!(sent.contains(".rig/outputs/call-1.txt"), "{sent}");
    assert!(!sent.contains("\\n1000\\n"), "the output is not sent whole");
    let whole = std::fs::read_to_string(root.join(".rig/outputs/call-1.txt")).unwrap_or_default();
    let numbers = whole.lines().filter(|line| line.parse::<u32>().is_ok());
    assert_eq!(numbers.count(), 2000, "{whole}");
    assert_eq!(cheap.request_count(), 1);
    // The summary counts in the agent's usage, as the summarizer's.
    let spent = app.world().get::<AgentUsage>(agent);
    let summarizer = spent.and_then(|spent| spent.purposes.get("summarizer"));
    assert_eq!(summarizer.map(|spent| spent.calls), Some(1));
    clean(&root);
}

/// The head survives: `shell` no longer keeps only its last 2000 lines.
#[test]
fn a_failed_summary_sends_the_head_and_tail() {
    let root = fresh("failed");
    let failure = vec![MockStreamEvent::Error(MockError::provider("down"))];
    let cheap = MockCompletionModel::from_stream_turns([failure]);
    let main = main_model("seq 3000");
    assert!(run(&main, &cheap, &root).is_some());
    let sent = second_request(&main);
    assert!(sent.contains("\"1\\n2\\n3\\n"), "the head: {sent}");
    assert!(sent.contains("\\n3000\\n"), "the tail: {sent}");
    assert!(!sent.contains("\\n1500\\n"), "the middle is cut: {sent}");
    assert!(sent.contains("the summarizer failed"), "{sent}");
    assert!(sent.contains(".rig/outputs/call-1.txt"), "{sent}");
    clean(&root);
}

/// The cheap model is sent what its input limit takes, not its window.
#[test]
fn an_output_past_the_cheap_models_input_limit_is_cut_for_it() {
    let root = fresh("limited");
    let cheap = MockCompletionModel::from_stream_turns([reply("SUMMARY: 5000 lines")]);
    let mut models = scripted_models([("corpus-summarizer/cheap", &cheap)]);
    let Ok(vendor) =
        rig_harness::rig_core::providers::registry::ProviderId::declare("corpus-summarizer")
    else {
        return;
    };
    let spec = rig_harness::rig_core::catalog::ModelSpec::new(vendor, "cheap")
        .with_tools(true)
        .with_context_window(1_000_000)
        .with_max_input_tokens(2_000);
    models.0.catalog_mut().insert(spec);
    let main = main_model("seq 5000");
    assert!(run_on(&main, models, &root).is_some());
    let asked = cheap.requests();
    let asked = asked
        .first()
        .and_then(|request| serde_json::to_string(request).ok());
    let asked = asked.unwrap_or_default();
    assert!(asked.contains("cut to fit the summarizer"), "{asked}");
    assert!(!asked.contains("\\n2500\\n"), "the middle is cut: {asked}");
    clean(&root);
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(SummarizeToolOutputPlugin);
}
