use std::path::PathBuf;
use std::sync::mpsc::Receiver;

use rig_cassette::journal::MemoryStore;
use rig_coding_tools::ShellTool;
use rig_core::completion::{CompletionRequest, Message};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::harness_protocol::Invocation;
use serde_json::json;

use super::*;
use rig_harness_test_support::{Ended, app_on, connect, log_ended, reply, run_until};

/// The pickers the plugin asked the view to show.
#[derive(Resource, Default)]
struct Asked(Vec<PickRequest>);

fn collect_picks(mut picks: MessageReader<PickRequest>, mut asked: ResMut<Asked>) {
    asked.0.extend(picks.read().cloned());
}

/// The kernel with the real `shell` tool and this plugin, someone at the
/// terminal when `interactive`, and its first agent, answered by `model`.
fn start(
    store: &MemoryStore,
    model: &MockCompletionModel,
    interactive: bool,
) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(store);
    if interactive {
        app.insert_resource(Invoked {
            args: Invocation::default(),
            terminal: true,
        });
    }
    app.insert_resource(DestructivePatterns::default())
        .add_plugins((
            AgentPlugin,
            JournalPlugin,
            ShellTool,
            ConfirmDestructivePlugin,
        ))
        .init_resource::<Asked>()
        .init_resource::<Ended>()
        .add_systems(Update, collect_picks)
        .add_observer(log_ended);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next()
        .unwrap_or(Entity::PLACEHOLDER);
    if let Some(connection) = connect(model) {
        world.entity_mut(agent).insert(connection);
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
        .filter_map(|message| match message {
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
fn a_denied_command_does_not_run_while_its_harmless_sibling_does() {
    let dir = workspace("deny");
    let model = model(&dir);
    let (mut app, wakes, agent) = start(&MemoryStore::default(), &model, true);
    app.world_mut().trigger(Deliver::new(
        agent,
        "clean the build directory",
        DeliveryMode::Queue,
    ));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Asked>().0.is_empty()
    });

    // The user is asked about the `rm`, and picks "Deny" as the view would.
    let asked = app.world().resource::<Asked>().0.clone();
    assert_eq!(asked.len(), 1, "{asked:?}");
    let picker = asked.first();
    assert!(picker.is_some_and(|picker| picker.title.starts_with("Allow `rm -rf")));
    let deny = picker
        .and_then(|picker| picker.items.iter().find(|item| item.label == "Deny"))
        .map(|item| item.command.clone())
        .unwrap_or_default();
    app.world_mut().trigger(RunCommand {
        entity: agent,
        line: deny,
    });
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });

    assert!(dir.join("build").is_dir(), "build/ was removed");
    let requests = model.requests();
    let sent = results(requests.get(1));
    assert_eq!(sent.len(), 2, "{sent:?}");
    let denied = sent.first().is_some_and(|denied| denied.contains(DENIED));
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
    let (mut app, wakes, agent) = start(&MemoryStore::default(), &model, false);
    app.world_mut().trigger(Deliver::new(
        agent,
        "clean the build directory",
        DeliveryMode::Queue,
    ));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });

    assert!(app.world().resource::<Asked>().0.is_empty());
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
