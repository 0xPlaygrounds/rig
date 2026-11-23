use std::sync::mpsc::Receiver;

use rig_basics::subagents::{SubagentsPlugin, TASK};
use rig_core::completion::{CompletionRequest, Message};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{
    Ended, app_on, connect, first_agent, log_ended, run_until, scripted_models,
};
use serde_json::json;

use super::*;

/// A plugin that makes a model call of its own on `/probe`, marked as its
/// own, on its own scripted model.
#[derive(Default)]
struct ProbePlugin;

/// The probe's model.
#[derive(Resource)]
struct ProbeModel(Connection);

impl Plugin for ProbePlugin {
    fn build(&self, app: &mut App) {
        app.add_command("probe", "Name the session", probe);
    }
}

fn probe(
    _: In<CommandArgs>,
    model: Res<ProbeModel>,
    me: Res<PluginEntity<ProbePlugin>>,
    mut commands: Commands,
) {
    let request = CompletionRequest::new(Message::user("Name this session."));
    commands.spawn((
        ModelRequest::new(request).on(model.0.clone()),
        Effect::default().with_purpose("namer"),
        ProvidedBy(me.entity()),
    ));
}

/// A reply of `events` that reports `input` tokens, `cached` of them from
/// the cache, and `output` tokens.
fn used(
    mut events: Vec<MockStreamEvent>,
    (input, cached, output): (u64, u64, u64),
) -> Vec<MockStreamEvent> {
    let mut usage = Usage::default();
    usage.input_tokens = Some(input);
    usage.cached_input_tokens = Some(cached);
    usage.output_tokens = Some(output);
    events.push(MockStreamEvent::final_response(usage));
    events
}

/// A session on `store` with the subagents, the probe and the spending by
/// origin, after its first frame.
fn start(
    store: &MemoryStore,
    main: &MockCompletionModel,
    sub: &MockCompletionModel,
    probe: &MockCompletionModel,
) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    app.insert_resource(scripted_models([("test/main", main), ("test/sub", sub)]))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            SubagentsPlugin,
            SpendByOriginPlugin,
        ))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    if let Some(probe) = connect(probe) {
        app.insert_resource(ProbeModel(probe));
    }
    rig_harness::load::<ProbePlugin>(&mut app, "probe", "path tests");
    app.finish();
    app.update();
    (app, wakes)
}

/// The text of every notice written since the last drain.
fn notices(world: &mut World) -> Vec<String> {
    let mut notices = world.resource_mut::<Messages<Notice>>();
    notices.drain().map(|notice| notice.text).collect()
}

/// What `origin` spent in the session: its calls and its input, cached and
/// output tokens.
fn spent(world: &World, origin: &str) -> (u64, u64, u64, u64) {
    let spend = world.resource::<SpendByOrigin>();
    spend.session.get(origin).map_or((0, 0, 0, 0), |spent| {
        let tokens = spent.tokens;
        (
            spent.calls,
            tokens.input_tokens.unwrap_or(0),
            tokens.cached_input_tokens.unwrap_or(0),
            tokens.output_tokens.unwrap_or(0),
        )
    })
}

#[test]
fn every_call_is_spent_by_its_agent_subagent_or_plugin_and_kept() {
    let task = json!({ "description": "scout", "prompt": "Look around.", "model": "test/sub" });
    let main = MockCompletionModel::from_stream_turns([
        used(
            vec![MockStreamEvent::tool_call("t1", TASK, task)],
            (100, 0, 10),
        ),
        used(
            vec![MockStreamEvent::text("Started the scout.")],
            (200, 100, 20),
        ),
        used(
            vec![MockStreamEvent::text("The scout is done.")],
            (300, 200, 30),
        ),
    ]);
    let sub = MockCompletionModel::from_stream_turns([used(
        vec![MockStreamEvent::text("Found it.")],
        (50, 0, 5),
    )]);
    let probe = MockCompletionModel::from_stream_turns([used(
        vec![MockStreamEvent::text("Spend audit")],
        (7, 0, 3),
    )]);
    let store = MemoryStore::default();
    let (mut app, wakes) = start(&store, &main, &sub, &probe);
    let world = app.world_mut();
    let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
    world.entity_mut(agent).insert(Name::new("main"));
    world.trigger(SetModel {
        entity: agent,
        model: "test/main".to_owned(),
        kind: ChoiceKind::ForThisRun,
    });
    world.trigger(Deliver::new(agent, "Send a scout.", DeliveryMode::Steer));
    // The parent reads its subagent's report in a call of its own.
    run_until(&mut app, &wakes, |world| {
        spent(world, "main").0 == 3
            && spent(world, "main › scout").0 == 1
            && world.get::<ActiveTurn>(agent).is_none()
    });
    app.world_mut()
        .trigger(RunCommand::new(agent, "probe".to_owned()));
    run_until(&mut app, &wakes, |world| {
        spent(world, "ProbePlugin (namer)").0 == 1
    });

    let world = app.world_mut();
    assert_eq!(spent(world, "main"), (3, 600, 300, 60));
    assert_eq!(spent(world, "main › scout"), (1, 50, 0, 5));
    assert_eq!(spent(world, "ProbePlugin (namer)"), (1, 7, 0, 3));
    let spend = world.resource::<SpendByOrigin>().clone();
    assert_eq!(spend.session.len(), 3, "{:?}", spend.session.keys());

    notices(world);
    world.trigger(RunCommand::new(agent, "spend".to_owned()));
    world.flush();
    let shown = notices(world).join("\n");
    assert!(!shown.contains(UNKNOWN), "{shown}");
    for row in ["main ", "main › scout ", "ProbePlugin (namer) ", "total "] {
        assert!(
            shown.lines().any(|line| line.starts_with(row)),
            "{row}: {shown}"
        );
    }

    // A new app on the same session has the same totals.
    drop(app);
    let (restarted, _wakes) = start(&store, &main, &sub, &probe);
    assert_eq!(
        restarted.world().resource::<SpendByOrigin>().session,
        spend.session
    );
}
