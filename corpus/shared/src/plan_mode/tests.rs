use std::path::PathBuf;

use rig_basics::coding_tools::{ReadTool, WriteTool};
use rig_core::catalog::{Catalog, Connector};
use rig_core::completion::{CompletionRequest, Message};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::MemoryStore;
use serde_json::json;

use super::*;
use rig_harness_test_support::{
    AnswerAsks, Answers, ToolOutputs, answer_open, app_on, calls, connect, log_tool_outputs, reply,
    run_until,
};

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
    match request
        .and_then(|request| request.chat_history.last())
        .map(|message| &**message)
    {
        Some(Message::User { content }) => format!("{content:?}"),
        _ => String::new(),
    }
}

#[test]
fn a_plan_is_made_without_changes_then_executed_with_the_tools_back() {
    let dir = scratch();
    let target = dir.join("x.txt");
    let write = json!({ "path": target.display().to_string(), "content": "hello" });
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("w1", "write", write.clone()),
            MockStreamEvent::tool_call("s1", "schedule", json!({})),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        reply("Plan: 1. create x.txt"),
        calls("w2", "write", write),
        reply("done"),
    ]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(Models(Connector::new(Catalog::default())))
        .insert_resource(ProjectRoot(dir.clone()))
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs)
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            ReadTool,
            WriteTool,
            AnswerAsks,
            PlanModePlugin,
        ));
    // Changes nothing the reply's other calls see, but acts: a stand-in
    // for cron-agent's `schedule`.
    let scheduled = |called: On<ToolCalled<serde_json::Value>>, mut commands: Commands| {
        let done = Outcome::from(ToolResult::success("scheduled".into()));
        commands.entity(called.call).insert_if_new(done);
    };
    let independent = ToolOptions {
        footprint: Some(Touches::Independent),
    };
    app.add_open_tool("schedule", "Run a prompt later.", independent, scheduled);
    let mut agent = app.world_mut().spawn(Agent);
    if let Some(connection) = connect(&model) {
        let choice = ModelChoice(connection.spec.reference());
        agent.insert((connection, choice));
    }
    let agent = agent.id();
    app.finish();
    app.update();

    let world = app.world_mut();
    world.trigger(RunCommand::new(agent, "plan".to_owned()));
    world.trigger(Deliver::new(agent, "Create x.txt.", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Answers>().asked.is_empty()
    });

    let requests = model.requests();
    let first = tool_names(requests.first());
    assert!(first.contains(&"read".to_owned()), "{first:?}");
    assert!(!first.contains(&"write".to_owned()), "{first:?}");
    assert!(!first.contains(&"schedule".to_owned()), "{first:?}");
    let told = |at: usize| format!("{:?}", requests.get(at).map(|r| &r.chat_history));
    assert!(told(0).contains("Plan mode: you may read"));
    let refused = app.world().resource::<ToolOutputs>().0.get("w1").cloned();
    assert!(refused.unwrap_or_default().contains(REFUSED));
    let refused = app.world().resource::<ToolOutputs>().0.get("s1").cloned();
    assert!(refused.unwrap_or_default().contains(REFUSED));
    assert!(!target.exists());
    let plan = std::fs::read_to_string(dir.join(PLAN_FILE)).unwrap_or_default();
    assert_eq!(plan, "Plan: 1. create x.txt");

    // The user picks Execute.
    let execute = Answer::choice("execute");
    assert!(answer_open(app.world_mut(), "The plan", &execute));
    run_until(&mut app, &wakes, |world| {
        model.request_count() == 4 && !world.entity(agent).contains::<ActiveTurn>()
    });

    let requests = model.requests();
    let third = tool_names(requests.get(2));
    assert!(third.contains(&"write".to_owned()), "{third:?}");
    assert!(third.contains(&"schedule".to_owned()), "{third:?}");
    let told = |at: usize| format!("{:?}", requests.get(at).map(|r| &r.chat_history));
    assert!(!told(2).contains("Plan mode: you may read"));
    let asked = prompt(requests.get(2));
    assert!(
        asked.contains("text: \"Execute this plan: Plan: 1. create x.txt\""),
        "{asked}"
    );
    assert!(target.exists());
    assert!(!app.world().entity(agent).contains::<PlanMode>());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(PlanModePlugin);
}
