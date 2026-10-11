use std::sync::mpsc::Receiver;

use rig_cassette::journal::MemoryStore;
use rig_core::completion::Message;
use rig_core::message::ToolResultContent;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::Invocation;
use rig_tui::PickRequest;
use serde_json::json;

use super::*;
use rig_harness_test_support::{app_on, calls, connect, reply, run_until};

/// The pickers the plugin asked the view to show.
#[derive(Resource, Default)]
struct Picks(Vec<PickRequest>);

/// How many turns ended.
#[derive(Resource, Default)]
struct Ended(usize);

fn collect_picks(mut picks: MessageReader<PickRequest>, mut seen: ResMut<Picks>) {
    seen.0.extend(picks.read().cloned());
}

/// The kernel and this plugin, someone at the terminal when
/// `interactive`, and its first agent, answered by `model` asking `DB?`
/// and then answering.
fn start(model: &MockCompletionModel, interactive: bool) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    if interactive {
        app.insert_resource(Invoked {
            args: Invocation::default(),
            terminal: true,
        });
    }
    app.add_plugins((AgentPlugin, JournalPlugin, AskUserPlugin))
        .init_resource::<Picks>()
        .init_resource::<Ended>()
        .add_systems(Update, collect_picks)
        .add_observer(|_: On<TurnEnded>, mut ended: ResMut<Ended>| ended.0 += 1);
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
    world.trigger(Deliver::new(agent, "Set up storage.", DeliveryMode::Steer));
    (app, wakes, agent)
}

fn model() -> MockCompletionModel {
    MockCompletionModel::from_stream_turns([
        calls(
            "1",
            "ask_user",
            json!({ "question": "DB?", "choices": ["pg", "sqlite"] }),
        ),
        reply("Using it."),
    ])
}

/// The text of the tool result the second request ends with.
fn answered(model: &MockCompletionModel) -> Option<String> {
    let requests = model.requests();
    let Some(Message::User { content }) = requests.get(1)?.chat_history.last() else {
        return None;
    };
    content.iter().find_map(|part| match part {
        UserContent::ToolResult(result) => result.content.iter().find_map(|part| match part {
            ToolResultContent::Text(text) => Some(text.text.clone()),
            _ => None,
        }),
        _ => None,
    })
}

#[test]
fn the_turn_waits_for_the_answer_the_picker_sends() {
    let model = model();
    let (mut app, wakes, agent) = start(&model, true);
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Picks>().0.is_empty()
    });
    for _ in 0..5 {
        app.update();
    }
    assert_eq!(model.requests().len(), 1, "the turn waits for the user");
    let picks = &app.world().resource::<Picks>().0;
    assert_eq!(picks.len(), 1, "one picker: {picks:?}");
    let Some(pick) = picks.first() else {
        return;
    };
    assert_eq!(pick.title, "DB?");
    let labels: Vec<&str> = pick.items.iter().map(|item| item.label.as_str()).collect();
    assert_eq!(labels, ["pg", "sqlite"]);
    // What the picker sends when "sqlite" is chosen.
    let line = pick
        .items
        .iter()
        .find(|item| item.label == "sqlite")
        .map(|item| item.command.clone())
        .unwrap_or_default();
    app.world_mut().trigger(RunCommand {
        entity: agent,
        line,
    });
    run_until(&mut app, &wakes, |world| world.resource::<Ended>().0 == 1);
    assert_eq!(answered(&model).as_deref(), Some("sqlite"));
}

#[test]
fn without_a_user_the_tool_answers_at_once() {
    let model = model();
    let (mut app, wakes, _) = start(&model, false);
    run_until(&mut app, &wakes, |world| world.resource::<Ended>().0 == 1);
    assert!(app.world().resource::<Picks>().0.is_empty());
    assert_eq!(answered(&model).as_deref(), Some(NO_USER));
}
