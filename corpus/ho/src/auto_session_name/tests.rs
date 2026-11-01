use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::SessionId;
use rig_harness_test_support::{
    app_in, first_agent, reply, run_until, scratch_home, scripted_models,
};

use super::*;

/// Whether a naming call settled, kept by [`note_named`].
#[derive(Resource, Default)]
struct NamingSettled(bool);

fn note_named(
    settled: On<Add<Settled>>,
    calls: Query<(), With<NamingCall>>,
    mut seen: ResMut<NamingSettled>,
) {
    if calls.contains(settled.entity) {
        seen.0 = true;
    }
}

/// An app on session `id` of `home` with the plugin, its agent on
/// `model`, and the agent.
fn app(
    home: &Home,
    id: &SessionId,
    model: &MockCompletionModel,
) -> (App, std::sync::mpsc::Receiver<()>, Entity) {
    let (mut app, wakes) = app_in(home, id);
    app.insert_resource(scripted_models([("test/model", model)]))
        .add_plugins((AgentPlugin, SessionLogPlugin, AutoSessionNamePlugin))
        .init_resource::<NamingSettled>()
        .add_observer(note_named);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let model = "test/model".to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model,
        kind: ChoiceKind::ForThisRun,
    });
    (app, wakes, agent)
}

fn name(world: &World) -> Option<String> {
    world.resource::<SessionTitle>().name.clone()
}

#[test]
fn the_first_exchange_names_the_session_outside_the_conversation() {
    let model =
        MockCompletionModel::from_stream_turns([reply("Done."), reply("\"Fix Parser Bug.\"")]);
    let home = scratch_home("ho-auto-session-name-named");
    let id = SessionId::generate();
    let (mut app, wakes, agent) = app(&home, &id, &model);
    app.world_mut().trigger(Deliver::new(
        agent,
        "fix the parser bug",
        DeliveryMode::Steer,
    ));
    run_until(&mut app, &wakes, |world| name(world).is_some());

    assert_eq!(name(app.world()).as_deref(), Some("Fix Parser Bug"));
    // One turn call and one naming call.
    assert_eq!(model.request_count(), 2);
    let conversation = app.world().get::<Conversation>(agent);
    let messages = conversation
        .map(|conversation| conversation.messages())
        .unwrap_or_default();
    assert_eq!(messages.len(), 2, "{messages:?}");
    assert_eq!(
        messages
            .last()
            .and_then(|message| message.final_answer())
            .as_deref(),
        Some("Done.")
    );
}

/// The hold a test keeps on the naming call, as the user types.
#[derive(Component)]
struct UserTyping;

/// Holds the naming call once, before it starts.
fn hold_naming(
    calls: Query<Entity, (With<NamingCall>, ReadyToStart)>,
    mut held: Local<bool>,
    mut commands: Commands,
) {
    for call in &calls {
        if !*held {
            *held = true;
            commands.spawn((UserTyping, HoldOf(call)));
        }
    }
}

#[test]
fn a_name_the_user_sets_while_naming_wins_and_is_kept() {
    let model = MockCompletionModel::from_stream_turns([reply("Done."), reply("Fix Parser Bug.")]);
    let home = scratch_home("ho-auto-session-name-mine");
    let id = SessionId::generate();
    let (mut app, wakes, agent) = app(&home, &id, &model);
    app.add_systems(AgentPass, hold_naming.in_set(Stage::Gate));
    let message = Deliver::new(agent, "fix the parser bug", DeliveryMode::Steer);
    app.world_mut().trigger(message);
    let held = |world: &World| {
        world
            .try_query_filtered::<(), With<UserTyping>>()
            .is_some_and(|mut holds| holds.iter(world).next().is_some())
    };
    run_until(&mut app, &wakes, held);
    assert!(held(app.world()));

    let name_it = Deliver::new(agent, "/name mine", DeliveryMode::Steer);
    app.world_mut().trigger(name_it);
    run_until(&mut app, &wakes, |world| name(world).is_some());
    assert_eq!(name(app.world()).as_deref(), Some("mine"));
    let mut holds = app.world_mut().query_filtered::<Entity, With<UserTyping>>();
    let holds: Vec<Entity> = holds.iter(app.world()).collect();
    for hold in holds {
        app.world_mut().despawn(hold);
    }
    run_until(&mut app, &wakes, |world| {
        world.resource::<NamingSettled>().0
    });
    assert!(app.world().resource::<NamingSettled>().0);
    assert_eq!(model.request_count(), 2);
    assert_eq!(name(app.world()).as_deref(), Some("mine"));
    app.update();
    drop(app);

    let (restarted, _, _) = self::app(&home, &id, &model);
    assert_eq!(name(restarted.world()).as_deref(), Some("mine"));
}
