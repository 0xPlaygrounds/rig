use std::fs;
use std::path::PathBuf;
use std::sync::mpsc::Receiver;

use rig_cassette::journal::JsonlDirStore;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::{SessionDir, SessionId};
use rig_harness::host::session::SessionPlugin;
use rig_harness::rig_ecs::journal::{JournalPlugin, SessionStore};
use rig_harness::settings::Settings;
use rig_harness_test_support::{connect, reply, run_until};

use super::*;

/// A plugin's saved state: the turns the agent ended.
#[derive(Component, Reflect, Default, Debug, PartialEq, Eq)]
#[reflect(Component, Saved, Default)]
struct Turns(u32);

fn count(
    settled: On<Add<Settled>>,
    turns: Query<&ForAgent, With<TurnOutcome>>,
    mut counts: Query<&mut Turns>,
) {
    if let Ok(ForAgent(agent)) = turns.get(settled.entity)
        && let Ok(mut count) = counts.get_mut(*agent)
    {
        count.0 += 1;
    }
}

/// The harness's session service, the kernel and this plugin on the
/// session in `dir` of `home`, after its first frame restored it.
fn start(home: &Home, dir: &SessionDir) -> (App, Receiver<()>) {
    let (sender, wakes) = std::sync::mpsc::channel();
    let mut app = App::new();
    app.add_plugins(TaskPoolPlugin::default())
        .insert_resource(Wake::new(move || {
            sender.send(()).ok();
        }))
        .insert_resource(Settings::default())
        .insert_resource(RigHome(home.clone()))
        .insert_resource(SessionPaths(dir.clone()))
        .insert_resource(SessionStore::new(JsonlDirStore::new(dir.path())))
        .init_resource::<TuiInput>()
        .add_plugins((AgentPlugin, JournalPlugin, SessionPlugin, ForkSessionPlugin))
        .register_required_components::<Agent, Turns>()
        .add_observer(count);
    // The session service sends panics to its log; the test shows them.
    drop(std::panic::take_hook());
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
            Message::User { content } => content
                .iter()
                .filter_map(|item| match item {
                    UserContent::Text(text) => Some(text.text.clone()),
                    _ => None,
                })
                .collect(),
            message => final_answer(message).unwrap_or_default(),
        })
        .collect();
    (texts, Some(turns.0))
}

/// A `RIG_HOME` of the test's own beside its binary, emptied.
fn home() -> Home {
    let exe = std::env::current_exe().unwrap_or_default();
    let root: PathBuf = exe.with_file_name("fork-session-home");
    fs::remove_dir_all(&root).ok();
    Home::at(root)
}

#[test]
fn a_branch_before_turn_two_keeps_turn_one_and_its_state() {
    let home = home();
    let id: SessionId = "1000-1".parse().unwrap_or_else(|_| SessionId::generate());
    let original = home.session(&id);
    assert!(fs::create_dir_all(original.path()).is_ok());
    let (mut app, wakes) = start(&home, &original);
    let model = MockCompletionModel::from_stream_turns([reply("r1"), reply("r2"), reply("r3")]);
    let (agent, connection) = (first_agent(&mut app), connect(&model));
    assert!(
        agent.is_some() && connection.is_some(),
        "an agent and a model"
    );
    let (Some(agent), Some(connection)) = (agent, connection) else {
        return;
    };
    app.world_mut().entity_mut(agent).insert(connection);
    for (done, text) in (1..).zip(["one", "two", "three"]) {
        let typed = Deliver::new(agent, text, DeliveryMode::Queue).with_origin(Origin::user());
        app.world_mut().trigger(typed);
        run_until(&mut app, &wakes, |world| {
            world
                .get::<Turns>(agent)
                .is_some_and(|turns| turns.0 == done)
        });
    }
    app.update();
    assert_eq!(state(&mut app).1, Some(3));

    // `/branch` opens a picker of the user's messages; choose "two".
    let line = Deliver::new(agent, "/branch", DeliveryMode::Queue).with_origin(Origin::user());
    app.world_mut().trigger(line);
    run_until(&mut app, &wakes, |world| {
        world
            .iter_entities()
            .any(|entity| entity.contains::<Picker>())
    });
    let world = app.world_mut();
    let picker = world
        .query::<(Entity, &Picker)>()
        .iter(world)
        .next()
        .map(|(entity, picker)| {
            let labels: Vec<&str> = picker
                .items
                .iter()
                .map(|item| item.label.as_str())
                .collect();
            (
                entity,
                labels.iter().position(|label| *label == "two"),
                labels.len(),
            )
        });
    assert!(
        matches!(picker, Some((_, Some(_), 3))),
        "a picker of the three messages: {picker:?}"
    );
    let Some((picker, Some(two), _)) = picker else {
        return;
    };
    app.world_mut().trigger(Picked {
        entity: picker,
        chosen: Some(Chosen::Item(two)),
    });
    // The fork is written at the frame's end, its family the next frame.
    let branched = || {
        fs::read_dir(home.sessions())
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok()?.file_name().to_str()?.parse::<SessionId>().ok())
            .find(|session| *session != id)
            .map(|session| home.session(&session))
            .filter(|dir| dir.path().join(FAMILY).exists())
    };
    run_until(&mut app, &wakes, |_| branched().is_some());
    drop(app);
    let branched = branched();
    assert!(branched.is_some(), "a branched session");
    let Some(branched) = branched else {
        return;
    };
    let family = read_json::<Family>(&branched.path().join(FAMILY));
    assert_eq!(
        family.and_then(|family| family.parent).as_deref(),
        Some(id.as_str())
    );

    // The branch holds turn one and the count as of then; "two" waits in
    // the input.
    let (mut app, _) = start(&home, &branched);
    assert_eq!(
        state(&mut app),
        (vec!["one".to_owned(), "r1".to_owned()], Some(1))
    );
    assert_eq!(app.world().resource::<TuiInput>().text(), "two");
    drop(app);

    // The original, reloaded from its log, still has all three turns.
    let (mut app, _) = start(&home, &original);
    let (messages, turns) = state(&mut app);
    assert_eq!(messages, ["one", "r1", "two", "r2", "three", "r3"]);
    assert_eq!(turns, Some(3));
    drop(app);
    fs::remove_dir_all(home.root()).ok();
}
