use std::path::PathBuf;
use std::sync::mpsc::Receiver;

use rig_basics::coding_tools::ShellTool;
use rig_core::completion::{CompletionRequest, Message};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::MemoryStore;
use serde_json::json;

use super::*;
use rig_harness_test_support::{
    AnswerAsks, Answers, Ended, app_on, connect, first_agent, log_ended, reply, run_until,
};

/// The kernel with the real `shell` tool and this plugin, a front that
/// answers with `answers` when given, and its first agent, answered by
/// `model`.
fn start(
    store: &MemoryStore,
    model: &MockCompletionModel,
    answers: Option<Answers>,
) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(store);
    if let Some(answers) = answers {
        app.add_plugins(AnswerAsks).insert_resource(answers);
    }
    app.add_plugins((
        AgentPlugin,
        SessionLogPlugin,
        ShellTool,
        ConfirmDestructivePlugin,
    ))
    .init_resource::<Ended>()
    .add_observer(log_ended);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    if let Some(connection) = connect(model) {
        app.world_mut().entity_mut(agent).insert(connection);
    }
    (app, wakes, agent)
}

/// A directory with a `build/` inside, under the test binary's target
/// directory.
fn workspace(name: &str) -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    let dir = exe.with_file_name(format!("confirm-destructive-{name}-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("build")).ok();
    dir
}

/// One reply removing `dir/build/` and echoing `ok`, then a final answer.
fn model(dir: &std::path::Path) -> MockCompletionModel {
    let remove = format!("rm -rf {}/build/", dir.display());
    MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("c1", "shell", json!({ "command": remove })),
            MockStreamEvent::tool_call("c2", "shell", json!({ "command": "echo ok" })),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        reply("Done."),
    ])
}

/// The tool results a request sends, in order, as JSON text.
fn results(request: Option<&CompletionRequest>) -> Vec<String> {
    let messages = request.map(|request| request.chat_history.iter());
    messages
        .into_iter()
        .flatten()
        .filter_map(|message| match &**message {
            Message::User { content } => Some(content.iter()),
            _ => None,
        })
        .flatten()
        .filter_map(|content| match content {
            UserContent::ToolResult(result) => serde_json::to_string(result).ok(),
            _ => None,
        })
        .collect()
}

#[test]
fn a_command_denied_with_a_reason_does_not_run_and_its_sibling_does() {
    let dir = workspace("deny");
    let model = model(&dir);
    // The user answers the question with a line of text: a denial with
    // that reason.
    let answers = Answers::new([Answer::text("we need it")]);
    let (mut app, wakes, agent) = start(&MemoryStore::default(), &model, Some(answers));
    app.world_mut().trigger(Deliver::new(
        agent,
        "clean the build directory",
        DeliveryMode::Queue,
    ));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });

    let asked = &app.world().resource::<Answers>().asked;
    assert_eq!(asked.len(), 1, "{asked:?}");
    assert!(
        asked.iter().all(|title| title.starts_with("Allow `rm -rf")),
        "{asked:?}"
    );
    assert!(dir.join("build").is_dir(), "build/ was removed");
    let requests = model.requests();
    let sent = results(requests.get(1));
    assert_eq!(sent.len(), 2, "{sent:?}");
    let denied = sent
        .first()
        .is_some_and(|denied| denied.contains(&format!("{DENIED}: we need it")));
    assert!(denied, "{sent:?}");
    assert!(
        sent.get(1)
            .is_some_and(|ran| ran.contains(r"ok\n[exit code 0]")),
        "{sent:?}"
    );
    let ended = &app.world().resource::<Ended>().0;
    assert!(
        matches!(ended.as_slice(), [TurnOutcome::Answered(_)]),
        "{ended:?}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn with_nobody_at_the_terminal_a_destructive_command_is_denied_at_once() {
    let dir = workspace("headless");
    let model = model(&dir);
    let (mut app, wakes, agent) = start(&MemoryStore::default(), &model, None);
    app.world_mut().trigger(Deliver::new(
        agent,
        "clean the build directory",
        DeliveryMode::Queue,
    ));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });

    let mut asks = app.world_mut().query::<&Ask>();
    assert_eq!(asks.iter(app.world()).count(), 0);
    assert!(dir.join("build").is_dir(), "build/ was removed");
    let requests = model.requests();
    let sent = results(requests.get(1));
    assert!(
        sent.first().is_some_and(|denied| denied.contains(HEADLESS)),
        "{sent:?}"
    );
    assert!(
        sent.get(1)
            .is_some_and(|ran| ran.contains(r"ok\n[exit code 0]")),
        "{sent:?}"
    );
    let ended = &app.world().resource::<Ended>().0;
    assert!(
        matches!(ended.as_slice(), [TurnOutcome::Answered(_)]),
        "{ended:?}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(ConfirmDestructivePlugin);
}
