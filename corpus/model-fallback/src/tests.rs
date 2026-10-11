use std::sync::mpsc::Receiver;

use bevy_time::TimeUpdateStrategy;
use rig_cassette::journal::MemoryStore;
use rig_core::ProviderResponseError;
use rig_core::catalog::Catalog;
use rig_core::operation::Completion;
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ModelAdapter;
use rig_core::test_utils::{MockCompletionModel, MockError, MockStreamEvent};
use rig_harness::rig_ecs::effects::Handler;
use rig_harness_test_support::{app_on, reply, run_until};

use super::*;

/// How far the clocks move each frame: the retries' waits pass quickly,
/// and a turn's frames stay far within the cooldown.
const FRAME: Duration = Duration::from_secs(4);

/// How each turn ended, in order, and every notice.
#[derive(Resource, Default)]
struct Seen {
    ended: Vec<TurnOutcome>,
    notices: Vec<String>,
}

/// `model` as the catalog model `reference`.
fn connection(reference: &str, model: &MockCompletionModel) -> Option<Connection> {
    let spec = Catalog::builtin().resolve(reference).ok()?.shared();
    let handler = ModelAdapter::<Completion>::new(spec.reference(), model.clone());
    Some(Connection {
        spec,
        handler: Handler(ErasedHandler::new(handler)),
    })
}

/// An overloaded provider's reply.
fn overloaded() -> Vec<MockStreamEvent> {
    let status = http::StatusCode::from_u16(529).unwrap_or(http::StatusCode::SERVICE_UNAVAILABLE);
    let refusal = ProviderResponseError::new(status, "Overloaded");
    vec![MockStreamEvent::Error(MockError::ProviderResponse(refusal))]
}

/// Sends `text` and runs frames until one more turn answered.
fn ask(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str) -> Option<String> {
    let answered = |world: &World| {
        let seen = world.resource::<Seen>();
        seen.ended
            .iter()
            .filter(|outcome| matches!(outcome, TurnOutcome::Answered(_)))
            .count()
    };
    let before = answered(app.world());
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Queue));
    run_until(app, wakes, |world| answered(world) > before);
    match app.world().resource::<Seen>().ended.last() {
        Some(TurnOutcome::Answered(message)) => final_answer(message),
        _ => None,
    }
}

#[test]
fn an_overloaded_model_falls_back_and_comes_back_after_the_cooldown() {
    let a = MockCompletionModel::from_stream_turns((0..10).map(|_| overloaded()));
    let b = MockCompletionModel::from_stream_turns((0..3).map(|_| reply("ok")));
    let models = (
        connection("anthropic/claude-opus-4-5", &a),
        connection("openai/gpt-5", &b),
    );
    assert!(
        models.0.is_some() && models.1.is_some(),
        "the catalog lists both models"
    );
    let (Some(on_a), Some(on_b)) = models else {
        return;
    };
    let (ref_a, ref_b) = (on_a.spec.reference(), on_b.spec.reference());

    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    app.set_error_handler(bevy_ecs::error::warn)
        .add_plugins((AgentPlugin, JournalPlugin, ModelFallbackPlugin))
        .insert_resource(FallbackChain {
            chain: vec![ref_a.clone(), ref_b.clone()],
            cooldown: COOLDOWN,
        })
        .insert_resource(Connected(HashMap::from([
            (ref_a.clone(), on_a.clone()),
            (ref_b, on_b),
        ])))
        .insert_resource(TimeUpdateStrategy::ManualDuration(FRAME))
        .init_resource::<Seen>()
        .add_observer(
            |settled: On<Add<Settled>>, turns: Query<&TurnOutcome>, mut seen: ResMut<Seen>| {
                seen.ended.extend(turns.get(settled.entity).cloned());
            },
        )
        .add_systems(
            Last,
            |mut notices: MessageReader<Notice>, mut seen: ResMut<Seen>| {
                seen.notices
                    .extend(notices.read().map(|notice| notice.text.clone()));
            },
        );
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = world.query_filtered::<Entity, With<Agent>>().single(world);
    assert!(agent.is_ok(), "the app has an agent");
    let Ok(agent) = agent else {
        return;
    };
    world.entity_mut(agent).insert(on_a);

    // A fails each of its retries; the turn goes on on B.
    assert_eq!(ask(&mut app, &wakes, agent, "hello").as_deref(), Some("ok"));
    assert_eq!(a.request_count(), 5, "A is retried as usual first");
    assert_eq!(b.request_count(), 1);
    let notices = &app.world().resource::<Seen>().notices;
    assert!(
        notices
            .iter()
            .any(|text| text == "claude-opus-4-5 overloaded; continuing on gpt-5"),
        "{notices:?}"
    );

    // Within the cooldown, the next turn goes straight to B.
    assert_eq!(ask(&mut app, &wakes, agent, "again").as_deref(), Some("ok"));
    assert_eq!(a.request_count(), 5, "A is skipped while it cools down");
    assert_eq!(b.request_count(), 2);

    // Past the cooldown, the next turn tries A first.
    app.insert_resource(TimeUpdateStrategy::ManualDuration(COOLDOWN));
    app.update();
    app.insert_resource(TimeUpdateStrategy::ManualDuration(FRAME));
    assert_eq!(ask(&mut app, &wakes, agent, "later").as_deref(), Some("ok"));
    assert_eq!(a.request_count(), 10, "A is back once it cooled down");
    assert_eq!(b.request_count(), 3);
}
