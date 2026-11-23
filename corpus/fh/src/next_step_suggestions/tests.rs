use std::sync::mpsc::Receiver;

use rig_core::effect::Disposition;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::SessionId;
use rig_harness::pick::PickPlugin;
use rig_harness_test_support::{
    Collected, Ended, app_in, collect, first_agent, log_ended, reply, run_until, scratch_home,
    scripted_models,
};

use super::*;

/// How many suggesting calls were asked for.
#[derive(Resource, Default)]
struct Asked(usize);

fn count_asks(_: On<Add<SuggestCall>>, mut asked: ResMut<Asked>) {
    asked.0 += 1;
}

/// A hold of the test's, as a slow model would keep a call.
#[derive(Component)]
struct Slow;

/// Holds the first effect `F` selects before it starts.
fn hold_first<F: bevy_ecs::query::QueryFilter>(
    calls: Query<Entity, (F, ReadyToStart)>,
    mut held: Local<bool>,
    mut commands: Commands,
) {
    if let (false, Some(call)) = (*held, calls.iter().next()) {
        *held = true;
        commands.spawn((Slow, HoldOf(call)));
    }
}

/// The plugin with the agent on `test/agent` and suggestions from
/// `test/suggester`, as the terminal view's run has it, and the agent.
fn start(
    name: &str,
    agent_model: &MockCompletionModel,
    suggester: &MockCompletionModel,
) -> (App, Receiver<()>, Entity) {
    let home = scratch_home(name);
    let (mut app, wakes) = app_in(&home, &SessionId::generate());
    let models = [("test/agent", agent_model), ("test/suggester", suggester)];
    app.insert_resource(Settings::from_toml("[suggest]\nmodel = \"test/suggester\""))
        .insert_resource(scripted_models(models))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            PickPlugin,
            NextStepSuggestionsPlugin,
        ))
        .insert_resource(ChipsShown)
        .init_resource::<Asked>()
        .add_observer(count_asks)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<EffectRecorded>(&mut app);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let model = "test/agent".to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model,
        kind: ChoiceKind::ForThisRun,
    });
    (app, wakes, agent)
}

fn suggested(world: &World, agent: Entity) -> Option<Vec<String>> {
    world.get::<Suggestions>(agent).map(|it| it.0.clone())
}

/// What the user typed into the agent's conversation, in order.
fn typed_by_user(world: &World, agent: Entity) -> Vec<String> {
    let Some(conversation) = world.get::<Conversation>(agent) else {
        return Vec::new();
    };
    (0..conversation.messages().len())
        .filter_map(|at| typed(conversation, at))
        .collect()
}

/// The enabled bindings labelled `label`.
fn bindings(app: &mut App, label: &str) -> Vec<Entity> {
    let world = app.world_mut();
    let mut bindings = world.query::<(Entity, &KeyBinding)>();
    bindings
        .iter(world)
        .filter(|(_, binding)| binding.label == label)
        .map(|(entity, _)| entity)
        .collect()
}

/// Presses the key bound as `label` with `agent` shown.
fn press(app: &mut App, agent: Entity, label: &str) {
    let found = bindings(app, label);
    assert_eq!(found.len(), 1, "{label}");
    for entity in found {
        app.world_mut().trigger(KeyPressed {
            entity,
            agent: Some(agent),
            released: false,
        });
    }
}

#[test]
fn alt_2_sends_the_second_suggestion_as_the_users_next_message() {
    let agent_model =
        MockCompletionModel::from_stream_turns([reply("Fixed the parser."), reply("Done.")]);
    let suggester = MockCompletionModel::from_stream_turns([
        reply("1. run the full test suite\n2. update CHANGELOG\n3. add a regression test"),
        reply("commit it"),
    ]);
    let (mut app, wakes, agent) = start("fh-suggest-alt-2", &agent_model, &suggester);
    assert_eq!(
        bindings(&mut app, "Send suggestion 2"),
        [],
        "no keys without chips"
    );
    let typed = Deliver::new(agent, "fix the parser", DeliveryMode::Steer);
    app.world_mut().trigger(typed);
    run_until(&mut app, &wakes, |world| suggested(world, agent).is_some());

    let shown = suggested(app.world(), agent).unwrap_or_default();
    assert_eq!(
        chips(&shown).to_string(),
        "[1] run the full test suite  [2] update CHANGELOG  [3] add a regression test   \
         Alt+0 hides"
    );
    app.update();
    let mut panels = app
        .world_mut()
        .query_filtered::<&TuiPanel, With<ChipsPanel>>();
    let placed: Vec<Placement> = panels
        .iter(app.world())
        .map(|panel| panel.placement)
        .collect();
    assert_eq!(placed, [Placement::BelowInput(Constraint::Length(1))]);
    // A question beside the work keeps its own Alt+N keys while it waits.
    let ask = Ask::yes_no("Run it?").inline();
    let question = app.world_mut().spawn((ask, ForAgent(agent))).id();
    app.update();
    assert_eq!(bindings(&mut app, "Send suggestion 2"), []);
    app.world_mut().despawn(question);
    app.update();
    let asked = format!("{:?}", suggester.requests());
    assert!(asked.contains("fix the parser") && asked.contains("Fixed the parser."));

    press(&mut app, agent, "Send suggestion 2");
    app.world_mut().flush();
    assert_eq!(suggested(app.world(), agent), None);
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 2
    });
    assert_eq!(
        typed_by_user(app.world(), agent),
        ["fix the parser", "update CHANGELOG"]
    );
    let messages = app
        .world()
        .get::<Conversation>(agent)
        .map_or(0, |conversation| conversation.messages().len());
    assert_eq!(
        messages, 4,
        "the suggestions are no part of the conversation"
    );
}

#[test]
fn a_message_sent_first_cancels_the_suggesting_call() {
    let agent_model =
        MockCompletionModel::from_stream_turns([reply("Fixed the parser."), reply("Ok.")]);
    let suggester = MockCompletionModel::from_stream_turns([reply("run the lexer tests")]);
    let (mut app, wakes, agent) = start("fh-suggest-cancel", &agent_model, &suggester);
    app.add_systems(
        AgentPass,
        hold_first::<With<SuggestCall>>.in_set(Stage::Gate),
    );
    let typed = Deliver::new(agent, "fix the parser", DeliveryMode::Steer);
    app.world_mut().trigger(typed);
    let held = |world: &World| {
        world
            .try_query_filtered::<(), With<Slow>>()
            .is_some_and(|mut holds| holds.iter(world).next().is_some())
    };
    run_until(&mut app, &wakes, held);

    let typed = Deliver::new(agent, "now check the lexer", DeliveryMode::Steer);
    app.world_mut().trigger(typed);
    let cancelled = |world: &World| {
        world
            .resource::<Collected<EffectRecorded>>()
            .0
            .iter()
            .any(|EffectRecorded(record)| {
                record.purpose.as_deref() == Some(NAME)
                    && record.disposition == Disposition::Cancelled
            })
    };
    run_until(&mut app, &wakes, cancelled);
    assert!(cancelled(app.world()));

    // The next turn's answer gets suggestions of its own; the cancelled
    // call never reached the suggester. (The next turn may end in the
    // frame the call was cancelled in.)
    run_until(&mut app, &wakes, |world| suggested(world, agent).is_some());
    assert_eq!(suggester.request_count(), 1);
    assert!(format!("{:?}", suggester.requests()).contains("now check the lexer"));
}

#[test]
fn an_interrupted_turn_gets_no_suggestions() {
    let agent_model = MockCompletionModel::from_stream_turns([reply("Fixed the parser.")]);
    let suggester = MockCompletionModel::from_stream_turns([reply("run the tests")]);
    let (mut app, wakes, agent) = start("fh-suggest-interrupt", &agent_model, &suggester);
    app.add_systems(AgentPass, hold_first::<With<TurnCall>>.in_set(Stage::Gate));
    let typed = Deliver::new(agent, "fix the parser", DeliveryMode::Steer);
    app.world_mut().trigger(typed);
    let held = |world: &World| {
        world
            .try_query_filtered::<(), With<Slow>>()
            .is_some_and(|mut holds| holds.iter(world).next().is_some())
    };
    run_until(&mut app, &wakes, held);

    app.world_mut().trigger(StopTurn::new(agent));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    for _ in 0..3 {
        app.update();
    }
    let ended = &app.world().resource::<Ended>().0;
    assert!(
        matches!(ended.first(), Some(TurnOutcome::Stopped(_))),
        "{ended:?}"
    );
    assert_eq!(app.world().resource::<Asked>().0, 0);
    assert_eq!(suggester.request_count(), 0);
}
