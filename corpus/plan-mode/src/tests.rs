use std::path::PathBuf;

use rig_cassette::journal::MemoryStore;
use rig_coding_tools::{ReadTool, WriteTool};
use rig_core::catalog::{Catalog, Connector};
use rig_core::completion::{CompletionRequest, Message};
use rig_core::test_utils::MockCompletionModel;
use serde_json::json;

use super::*;
use rig_harness_test_support::{app_on, calls, connect, reply, run_until};

/// The pickers asked for, by title and labels.
#[derive(Resource, Default)]
struct Picks(Vec<Vec<String>>);

fn record_picks(pickers: Query<&Picker, Added<Picker>>, mut seen: ResMut<Picks>) {
    for picker in &pickers {
        seen.0
            .push(picker.items.iter().map(|item| item.label.clone()).collect());
    }
}

/// A directory of its own for this test, beside the test binary, so
/// nothing lands in the source tree.
fn scratch() -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    let dir = exe
        .parent()
        .map(|deps| deps.join("corpus-plan-mode-test"))
        .unwrap_or_default();
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).ok();
    dir
}

fn tool_names(request: Option<&CompletionRequest>) -> Vec<String> {
    let tools = request.map(|request| request.tools.iter());
    let names = tools.into_iter().flatten();
    names.map(|tool| tool.name.as_str().to_owned()).collect()
}

/// The text items of a request's last message.
fn prompt(request: Option<&CompletionRequest>) -> String {
    match request.and_then(|request| request.chat_history.last()) {
        Some(Message::User { content }) => format!("{content:?}"),
        _ => String::new(),
    }
}

#[test]
fn a_plan_is_made_without_changes_then_executed_with_the_tools_back() {
    let dir = scratch();
    let target = dir.join("x.txt");
    let path = target.display().to_string();
    let write = json!({ "path": path, "content": "hello" });
    let model = MockCompletionModel::from_stream_turns([
        calls("w1", "write", write.clone()),
        reply("Plan: 1. create x.txt"),
        calls("w2", "write", write),
        reply("done"),
    ]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(Models(Connector::new(Catalog::default())))
        .insert_resource(PlanFile(dir.join(".rig").join("plan.md")))
        .add_plugins((
            AgentPlugin,
            JournalPlugin,
            ReadTool,
            WriteTool,
            PlanModePlugin,
        ))
        .init_resource::<Picks>()
        .add_systems(Update, record_picks);
    let mut agent = app.world_mut().spawn(Agent);
    if let Some(connection) = connect(&model) {
        let choice = ModelChoice(connection.spec.reference());
        agent.insert((connection, choice));
    }
    let agent = agent.id();
    app.finish();
    app.update();

    let world = app.world_mut();
    world.trigger(RunCommand {
        entity: agent,
        line: "plan".to_owned(),
    });
    world.trigger(Deliver::new(agent, "Create x.txt.", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| {
        world
            .get::<PlanMode>(agent)
            .is_some_and(|mode| mode.plan.is_some())
            && !world.resource::<Picks>().0.is_empty()
    });

    let requests = model.requests();
    let first = tool_names(requests.first());
    assert!(first.contains(&"read".to_owned()), "{first:?}");
    assert!(!first.contains(&"write".to_owned()), "{first:?}");
    let told = |at: usize| format!("{:?}", requests.get(at).map(|r| &r.chat_history));
    assert!(told(0).contains("Plan mode: you may read"));
    let refused = format!("{:?}", requests.get(1).map(|r| &r.chat_history));
    assert!(refused.contains("is_error: true"), "{refused}");
    assert!(refused.contains(REFUSED), "{refused}");
    assert!(!target.exists());
    let plan = std::fs::read_to_string(dir.join(".rig").join("plan.md")).unwrap_or_default();
    assert_eq!(plan, "Plan: 1. create x.txt");
    let labels = ["Execute", "Keep planning", "Discard"].map(str::to_owned);
    assert_eq!(app.world().resource::<Picks>().0, vec![labels.to_vec()]);

    app.world_mut().trigger(RunCommand {
        entity: agent,
        line: "plan execute".to_owned(),
    });
    run_until(&mut app, &wakes, |world| {
        model.request_count() == 4 && !world.entity(agent).contains::<ActiveTurn>()
    });

    let requests = model.requests();
    let third = tool_names(requests.get(2));
    assert!(third.contains(&"write".to_owned()), "{third:?}");
    let told = format!("{:?}", requests.get(2).map(|r| &r.chat_history));
    assert!(!told.contains("Plan mode: you may read"));
    let asked = prompt(requests.get(2));
    assert!(
        asked.contains("text: \"Execute this plan: Plan: 1. create x.txt\""),
        "{asked}"
    );
    assert!(target.exists());
    let world = app.world();
    assert!(!world.entity(agent).contains::<PlanMode>());
    assert!(matches!(
        world.get::<ToolAccess>(agent),
        Some(ToolAccess::All)
    ));
    std::fs::remove_dir_all(&dir).ok();
}
