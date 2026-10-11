use std::fs;
use std::path::PathBuf;
use std::sync::mpsc::Receiver;

use rig_core::test_utils::MockCompletionModel;
use rig_core::transcript::final_answer;
use rig_harness::harness_protocol::env;
use rig_harness_test_support::{connect, reply, run_until};

use super::*;

/// A plugin's saved state: the turns the agent ended.
#[derive(Component, Reflect, Default, Debug, PartialEq, Eq)]
#[reflect(Component, Saved, Default)]
struct Turns(u32);

fn count(ended: On<TurnEnded>, mut turns: Query<&mut Turns>) {
    if let Ok(mut turns) = turns.get_mut(ended.entity) {
        turns.0 += 1;
    }
}

/// The kernel and this plugin on the session in `dir`, after its first
/// frame restored it.
fn start(dir: &SessionDir) -> (App, Receiver<()>) {
    let (sender, wakes) = std::sync::mpsc::channel();
    let mut app = App::new();
    app.add_plugins(TaskPoolPlugin::default())
        .insert_resource(Wake::new(move || {
            sender.send(()).ok();
        }))
        .insert_resource(SessionPaths(dir.clone()))
        .insert_resource(SessionStore::new(JsonlDirStore::new(dir.path())))
        .set_error_handler(bevy_ecs::error::warn)
        .add_plugins((AgentPlugin, JournalPlugin, ForkSessionPlugin))
        .register_required_components::<Agent, Turns>()
        .add_observer(count);
    app.finish();
    app.update();
    (app, wakes)
}

fn first_agent(app: &mut App) -> Option<Entity> {
    let world = app.world_mut();
    world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next()
}

/// What each message of the first agent says, and its saved count.
fn state(app: &mut App) -> (Vec<String>, Option<u32>) {
    let world = app.world_mut();
    let Some((conversation, turns)) = world.query::<(&Conversation, &Turns)>().iter(world).next()
    else {
        return (Vec::new(), None);
    };
    let texts = conversation
        .messages()
        .iter()
        .map(|message| match message {
            Message::User { .. } => typed_text(message),
            Message::Assistant(_) => final_answer(message).unwrap_or_default(),
            Message::System { .. } => String::new(),
        })
        .collect();
    (texts, Some(turns.0))
}

fn typed_text(message: &Message) -> String {
    match message {
        Message::User { content } => content
            .iter()
            .filter_map(|item| match item {
                UserContent::Text(text) => Some(text.text.clone()),
                _ => None,
            })
            .collect(),
        _ => String::new(),
    }
}

/// A `RIG_HOME` of the test's own beside its binary, emptied.
fn home() -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    let root = exe
        .parent()
        .unwrap_or(Path::new("."))
        .join("fork-session-home");
    fs::remove_dir_all(&root).ok();
    // SAFETY: this crate has one test, and nothing else reads the
    // environment while it is set.
    unsafe { std::env::set_var(env::HOME, &root) };
    root
}

#[test]
fn a_branch_before_turn_two_keeps_turn_one_and_its_state() {
    home();
    let id = "1000-1".parse::<SessionId>();
    assert!(id.is_ok(), "a session id");
    let Ok(id) = id else {
        return;
    };
    let original = Home::from_env().session(&id);
    assert!(fs::create_dir_all(original.path()).is_ok());
    let (mut app, wakes) = start(&original);
    let model = MockCompletionModel::from_stream_turns([reply("r1"), reply("r2"), reply("r3")]);
    let agent = first_agent(&mut app);
    assert!(agent.is_some(), "the first agent");
    let Some(agent) = agent else {
        return;
    };
    if let Some(connection) = connect(&model) {
        app.world_mut().entity_mut(agent).insert(connection);
    }
    for (turn, text) in ["one", "two", "three"].into_iter().enumerate() {
        app.world_mut()
            .trigger(Deliver::new(agent, text, DeliveryMode::Queue));
        let done = u32::try_from(turn + 1).unwrap_or_default();
        run_until(&mut app, &wakes, |world| {
            world
                .get::<Turns>(agent)
                .is_some_and(|turns| turns.0 == done)
        });
    }
    app.update();
    assert_eq!(state(&mut app).1, Some(3));

    // `/branch` opens a picker of the user's messages; choose "two".
    app.world_mut()
        .trigger(Deliver::new(agent, "/branch", DeliveryMode::Queue));
    app.world_mut().flush();
    let picks = app.world().resource::<Messages<PickRequest>>();
    let choice = picks
        .get_cursor()
        .read(picks)
        .flat_map(|pick| pick.items.clone())
        .find(|item| item.label == "two")
        .map(|item| item.command);
    assert_eq!(choice.as_deref(), Some("branch 2"));
    app.world_mut().trigger(RunCommand {
        entity: agent,
        line: choice.unwrap_or_default(),
    });
    app.update();
    drop(app);

    let branched: Vec<SessionId> = fs::read_dir(Home::from_env().sessions())
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok()?.file_name().to_str()?.parse().ok())
        .filter(|session| *session != id)
        .collect();
    assert_eq!(branched.len(), 1, "one new session: {branched:?}");
    let Some(branched) = branched.first() else {
        return;
    };
    let branched = Home::from_env().session(branched);
    let family: Option<Family> = read_json(&branched.path().join(FAMILY));
    assert_eq!(
        family.and_then(|family| family.parent).as_deref(),
        Some("1000-1")
    );

    // The branch holds turn one, the count as of then, and "two" waits in
    // the input.
    let (mut app, _) = start(&branched);
    assert_eq!(
        state(&mut app),
        (vec!["one".to_owned(), "r1".to_owned()], Some(1))
    );
    let recalled = app.world().resource::<Messages<Recalled>>();
    let input: Vec<String> = recalled
        .get_cursor()
        .read(recalled)
        .map(|recalled| recalled.text.clone())
        .collect();
    assert_eq!(input, ["two"]);
    drop(app);

    // The original, reloaded from its log, still has all three turns.
    let (mut app, _) = start(&original);
    let (messages, turns) = state(&mut app);
    assert_eq!(messages, ["one", "r1", "two", "r2", "three", "r3"]);
    assert_eq!(turns, Some(3));
}
