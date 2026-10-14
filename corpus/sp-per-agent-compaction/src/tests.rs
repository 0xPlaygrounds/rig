use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;

use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::commands::RunCommand;
use rig_harness::harness_protocol::Home;
use rig_harness::watch::WatchPlugin;
use rig_harness_test_support::{Collected, MemoryStore, app_on, calls, collect, reply, run_until};
use rig_subagents::SubagentsPlugin;
use serde_json::json;

use super::*;

/// The window of every scripted model.
const WINDOW: u32 = 4_000;

/// `main` summarized by `sum` at 80%, explorers dropping tool output at
/// `explorer_percent`, reviewers never compacted.
fn rules(explorer_percent: u8) -> String {
    format!(
        r#"
[[rule]]
name = "main"
agent = "main"
strategy = "summary"
trigger_percent = 80
model = "test/sum"
keep_recent_turns = 1

[[rule]]
name = "explorer"
type = "explorer"
strategy = "drop-tool-output"
trigger_percent = {explorer_percent}
keep_recent_turns = 1
stub = "[dropped]"

[[rule]]
name = "reviewer"
type = "reviewer"
strategy = "never"
"#
    )
}

/// A reply calling `tool` that reports `tokens` of context.
fn calls_at(id: &str, tool: &str, args: serde_json::Value, tokens: u64) -> Vec<MockStreamEvent> {
    vec![
        MockStreamEvent::tool_call(id, tool, args),
        MockStreamEvent::final_response_with_total_tokens(tokens),
    ]
}

/// A reply of `text` that reports `tokens` of context.
fn reply_at(text: &str, tokens: u64) -> Vec<MockStreamEvent> {
    vec![
        MockStreamEvent::text(text),
        MockStreamEvent::final_response_with_total_tokens(tokens),
    ]
}

/// The catalog with `served` as `test/<name>`, each with a 4,000-token
/// window.
fn models(served: &[(&str, &MockCompletionModel)]) -> Models {
    let mut models = rig_harness_test_support::scripted_models(served.iter().copied());
    let catalog = models.0.catalog_mut();
    let ours: Vec<_> = catalog
        .iter()
        .filter(|spec| spec.reference().starts_with("test/"))
        .cloned()
        .collect();
    for spec in ours {
        catalog.insert(spec.with_context_window(WINDOW));
    }
    models
}

/// The kernel, journal, watches, subagents and this plugin on `store`,
/// `RIG_HOME` at `home`, a `read` tool that answers with a large file, and
/// the rules loaded.
fn start(store: &MemoryStore, home: &Path, models: Models) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    app.insert_resource(RigHome(Home::at(home)))
        .insert_resource(ProjectRoot(home.join("project")))
        .insert_resource(models)
        .add_plugins((AgentPlugin, JournalPlugin, WatchPlugin, SubagentsPlugin))
        .add_plugins(PerAgentCompactionPlugin);
    collect::<Notice>(&mut app);
    let read = |called: On<ToolCalled<serde_json::Value>>,
                mut calls: Query<&mut Decision, ToolCalls>| {
        let path = called.args["path"].as_str().unwrap_or_default().to_owned();
        let text = format!("CONTENT OF {path}: {}", "line ".repeat(60));
        if let Ok(mut call) = calls.get_mut(called.call) {
            call.decide(Decider::plugin("test"), ToolResult::success(text.into()));
        }
    };
    app.add_open_tool("read", "Read a file.", ToolOptions::default(), read);
    app.finish();
    app.update();
    run_until(&mut app, &wakes, |world| {
        world.resource::<CompactionRules>().0.len() == 3
    });
    (app, wakes)
}

/// The agent nothing spawned, on `test/main`.
fn main_agent(app: &mut App) -> Entity {
    let world = app.world_mut();
    let found = world
        .query_filtered::<Entity, (With<Agent>, Without<SpawnedBy>)>()
        .iter(world)
        .next();
    let agent = found.unwrap_or_else(|| world.spawn(Agent).id());
    let connection = world.resource::<Models>().connect("test/main");
    if let Ok(connection) = connection {
        let choice = ModelChoice(connection.spec.reference());
        world.entity_mut(agent).insert((connection, choice));
    }
    agent
}

/// The subagent titled `title`.
fn subagent(world: &mut World, title: &str) -> Option<Entity> {
    let mut agents = world.query::<(Entity, &AgentTitle)>();
    let found = agents.iter(world).find(|(_, it)| it.0 == title);
    found.map(|(agent, _)| agent)
}

/// Runs frames until `model` got `count` requests and no turn runs.
fn until(app: &mut App, wakes: &Receiver<()>, model: &MockCompletionModel, count: usize) {
    run_until(app, wakes, |world| {
        let idle = world
            .try_query_filtered::<(), With<ActiveTurn>>()
            .is_none_or(|mut busy| busy.iter(world).next().is_none());
        model.requests().len() >= count && idle
    });
    app.update();
}

/// The request `model` got `at`, as JSON.
fn request(model: &MockCompletionModel, at: usize) -> String {
    let requests = model.requests();
    serde_json::to_string(&requests.get(at)).unwrap_or_default()
}

/// The messages of `agent`, as JSON.
fn messages(world: &World, agent: Entity) -> String {
    let conversation = world.get::<Conversation>(agent);
    let messages = conversation.map(|conversation| conversation.messages().to_vec());
    serde_json::to_string(&messages).unwrap_or_default()
}

/// Says `text` to `agent` as the user.
fn say(app: &mut App, agent: Entity, text: &str) {
    let message = Deliver::new(agent, text, DeliveryMode::Queue);
    app.world_mut().trigger(message);
}

#[test]
#[expect(clippy::too_many_lines, reason = "one session, start to resume")]
fn each_agent_is_compacted_by_its_own_rule() {
    let home: PathBuf = std::env::current_exe()
        .unwrap_or_default()
        .with_file_name("per-agent-compaction");
    std::fs::remove_dir_all(&home).ok();
    std::fs::create_dir_all(&home).ok();
    std::fs::write(home.join(FILE), rules(60)).ok();
    let task = |id: &str, title: &str, kind: &str| {
        let model = format!("test/{kind}");
        let args =
            json!({ "description": title, "prompt": "Do it.", "model": model, "type": kind });
        calls(id, "task", args)
    };
    let main = MockCompletionModel::from_stream_turns([
        task("t1", "explore", "explorer"),
        reply("Waiting for the explorer."),
        reply("The explorer reported."),
        task("t2", "review", "reviewer"),
        reply("Waiting for the reviewer."),
        reply_at("The reviewer failed.", 3_500),
        reply("Going on."),
    ]);
    let explorer = MockCompletionModel::from_stream_turns([
        calls_at("a", "read", json!({ "path": "a.rs" }), 1_000),
        calls_at("b", "read", json!({ "path": "b.rs" }), 1_500),
        calls_at("c", "read", json!({ "path": "c.rs" }), 2_500),
        reply_at("Explored.", 2_800),
        reply("Explored more."),
    ]);
    let reviewer = MockCompletionModel::from_stream_turns([calls_at(
        "d",
        "read",
        json!({ "path": "d.rs" }),
        4_500,
    )]);
    let sum = MockCompletionModel::from_stream_turns([reply("SUMMARY-OF-MAIN")]);
    let served = [
        ("test/main", &main),
        ("test/explorer", &explorer),
        ("test/reviewer", &reviewer),
        ("test/sum", &sum),
    ];
    let store = MemoryStore::default();
    let (mut app, wakes) = start(&store, &home, models(&served));
    let agent = main_agent(&mut app);

    // The explorer's fourth request, past 60%, has its first two results
    // dropped, every call kept; no model summarizes, main is untouched.
    say(&mut app, agent, "Audit the repo.");
    until(&mut app, &wakes, &main, 3);
    let fourth = request(&explorer, 3);
    assert_eq!(fourth.matches("[dropped]").count(), 2, "{fourth}");
    assert!(fourth.contains("CONTENT OF c.rs"), "{fourth}");
    for path in ["a.rs", "b.rs", "c.rs"] {
        // The calls stay: only their arguments name the dropped files.
        assert!(fourth.contains(path), "{path}: {fourth}");
    }
    assert!(!fourth.contains("CONTENT OF a.rs"), "{fourth}");
    assert!(sum.requests().is_empty());
    assert!(!messages(app.world(), agent).contains("[dropped]"));

    // The reviewer would overflow: it is not compacted, and main gets its
    // failure as its report.
    say(&mut app, agent, "Review it.");
    until(&mut app, &wakes, &main, 6);
    let reviewer_agent = subagent(app.world_mut(), "review");
    let reviewed = reviewer_agent
        .map(|it| messages(app.world(), it))
        .unwrap_or_default();
    assert!(
        !reviewed.contains("[dropped]") && !reviewed.contains("SUMMARY"),
        "{reviewed}"
    );
    assert_eq!(reviewer.requests().len(), 1);
    let reported = request(&main, 5);
    assert!(reported.contains("context limit"), "{reported}");

    // Main past 80%: `sum` summarizes once; main's request has the summary
    // and the last turn; the explorer is untouched.
    let explorer_agent = subagent(app.world_mut(), "explore").unwrap_or(Entity::PLACEHOLDER);
    let explored = messages(app.world(), explorer_agent);
    say(&mut app, agent, "Go on.");
    until(&mut app, &wakes, &main, 7);
    assert_eq!(sum.requests().len(), 1);
    let summarized = request(&main, 6);
    assert!(summarized.contains("SUMMARY-OF-MAIN"), "{summarized}");
    assert!(summarized.contains("The reviewer failed."), "{summarized}");
    assert!(
        !summarized.contains("Waiting for the explorer."),
        "{summarized}"
    );
    assert_eq!(messages(app.world(), explorer_agent), explored);

    // `/compaction` lists the three agents with their rules.
    app.world_mut()
        .trigger(RunCommand::new(agent, "compaction"));
    app.update();
    let notices = &app.world().resource::<Collected<Notice>>().0;
    let listed = notices
        .iter()
        .rev()
        .find(|notice| notice.text.starts_with("Compaction by agent"))
        .map(|notice| notice.text.clone())
        .unwrap_or_default();
    for line in [
        "main: rule `main` (summary at 80%)",
        "explore: rule `explorer` (drop-tool-output at 60%)",
        "review: rule `reviewer` (never at 100%)",
        "last: summary by `main`",
        "last: drop-tool-output by `explorer`",
    ] {
        assert!(listed.contains(line), "{line}: {listed}");
    }

    // At 90%, the explorer's next request at 70% keeps its last result.
    std::fs::write(home.join(FILE), rules(90)).ok();
    run_until(&mut app, &wakes, |world| {
        let rules = world.resource::<CompactionRules>();
        rules
            .0
            .get(1)
            .is_some_and(|rule| rule.trigger_percent == 90)
    });
    say(&mut app, explorer_agent, "Read more.");
    until(&mut app, &wakes, &explorer, 5);
    let fifth = request(&explorer, 4);
    assert!(fifth.contains("CONTENT OF c.rs"), "{fifth}");
    assert!(!fifth.contains("SUMMARY-OF-MAIN"), "{fifth}");
    drop(app);

    // A restart: main's request still holds the summary, and the record.
    let main = MockCompletionModel::from_stream_turns([reply("Resumed.")]);
    let (mut app, wakes) = start(&store, &home, models(&[("test/main", &main)]));
    let agent = main_agent(&mut app);
    let record = app
        .world()
        .get::<Conversation>(agent)
        .and_then(CompactionRecord::newest);
    assert_eq!(
        record.map(|record| (record.strategy, record.rule)),
        Some((Strategy::Summary, "main".to_owned()))
    );
    say(&mut app, agent, "Where were we?");
    until(&mut app, &wakes, &main, 1);
    let resumed = request(&main, 0);
    assert!(resumed.contains("SUMMARY-OF-MAIN"), "{resumed}");
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(PerAgentCompactionPlugin);
}
