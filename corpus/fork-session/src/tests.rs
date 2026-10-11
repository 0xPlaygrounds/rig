use std::fs;
use std::sync::mpsc::Receiver;

use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::SessionId;
use rig_harness::rig_ecs::journal::JournalPlugin;
use rig_harness_test_support::{app_in, connect, first_agent, reply, run_until, scratch_home};

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

/// The kernel, the harness's session service and this plugin on session
/// `id` of `home`, after its first frame restored it.
fn start(home: &Home, id: &SessionId) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_in(home, id);
    app.init_resource::<TuiInput>()
        .add_plugins((AgentPlugin, JournalPlugin, ForkSessionPlugin))
        .register_required_components::<Agent, Turns>()
        .add_observer(count);
    app.finish();
    app.update();
    (app, wakes)
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

#[test]
fn a_branch_before_turn_two_keeps_turn_one_and_its_state() {
    let home = scratch_home("fork-session-home");
    let id: SessionId = "1000-1".parse().unwrap_or_else(|_| SessionId::generate());
    let (mut app, wakes) = start(&home, &id);
    let model = MockCompletionModel::from_stream_turns([reply("r1"), reply("r2"), reply("r3")]);
    let (agent, connection) = (first_agent(app.world_mut()), connect(&model));
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
    // The fork is written at the frame's end, its seed the next frame.
    let branched = || {
        fs::read_dir(home.sessions())
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok()?.file_name().to_str()?.parse::<SessionId>().ok())
            .find(|session| home.session(session).path().join(SEED).exists())
    };
    run_until(&mut app, &wakes, |_| branched().is_some());
    drop(app);
    let branched = branched();
    assert!(branched.is_some(), "a branched session");
    let Some(branched) = branched else {
        return;
    };
    let parent = ForkedFrom::of(&home.session(&branched));
    assert_eq!(
        parent.map(|parent| (parent.session, parent.text)),
        Some((id.clone(), Some("two".to_owned())))
    );

    // The branch holds turn one and the count as of then; "two" waits in
    // the input.
    let (mut app, _) = start(&home, &branched);
    assert_eq!(
        state(&mut app),
        (vec!["one".to_owned(), "r1".to_owned()], Some(1))
    );
    assert_eq!(app.world().resource::<TuiInput>().text(), "two");
    assert!(
        !home.session(&branched).path().join(SEED).exists(),
        "seeded once"
    );
    drop(app);

    // The original, reloaded from its log, still has all three turns.
    let (mut app, _) = start(&home, &id);
    let (messages, turns) = state(&mut app);
    assert_eq!(messages, ["one", "r1", "two", "r2", "three", "r3"]);
    assert_eq!(turns, Some(3));
    drop(app);
    fs::remove_dir_all(home.root()).ok();
}
