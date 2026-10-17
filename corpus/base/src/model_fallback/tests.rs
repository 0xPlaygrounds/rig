use std::sync::mpsc::Receiver;

use bevy_time::{TimeUpdateStrategy, Virtual};
use rig_basics::compaction::CompactionPlugin;
use rig_core::ProviderResponseError;
use rig_core::catalog::{Catalog, ModelSpec, Transport};
use rig_core::operation::Completion;
use rig_core::providers::registry::{ConnectError, ProviderId};
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ModelAdapter;
use rig_core::test_utils::{MockCompletionModel, MockError, MockStreamEvent};
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{Ended, app_on, log_ended, reply, run_until};

use super::*;

/// How far the clocks move each frame: the retries' waits pass quickly,
/// and a turn's frames stay far within the cooldown.
const FRAME: Duration = Duration::from_secs(4);

/// Every notice.
#[derive(Resource, Default)]
struct Notices(Vec<String>);

/// The vendor's models answer as `model` scripts.
fn scripted(models: &mut Models, vendor: &str, model: &MockCompletionModel) {
    let model = model.clone();
    let provider = move |spec: &ModelSpec,
                         _: &Catalog,
                         _: &Transport|
          -> Result<ErasedHandler, ConnectError> {
        let handler = ModelAdapter::<Completion>::new(spec.reference(), model.clone());
        Ok(ErasedHandler::new(handler))
    };
    if let Some(vendor) = ProviderId::catalog(vendor) {
        models.0.provide(vendor, provider);
    }
}

/// An overloaded provider's reply.
fn overloaded() -> Vec<MockStreamEvent> {
    let status = http::StatusCode::from_u16(529).unwrap_or(http::StatusCode::SERVICE_UNAVAILABLE);
    let refusal = ProviderResponseError::new(status, "Overloaded");
    vec![MockStreamEvent::Error(MockError::ProviderResponse(refusal))]
}

/// Sends `text` and runs frames until one more turn ended; its answer.
fn ask(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str) -> Option<String> {
    let ended = |world: &World| world.resource::<Ended>().0.len();
    let before = ended(app.world());
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Queue));
    run_until(app, wakes, |world| ended(world) > before);
    match app.world().resource::<Ended>().0.last() {
        Some(TurnOutcome::Answered(message)) => message.final_answer(),
        _ => None,
    }
}

#[test]
fn an_overloaded_model_falls_back_and_comes_back_after_the_cooldown() {
    let a = MockCompletionModel::from_stream_turns((0..10).map(|_| overloaded()));
    let b = MockCompletionModel::from_stream_turns((0..3).map(|_| reply("ok")));
    let mut models = Models::default();
    scripted(&mut models, "anthropic", &a);
    scripted(&mut models, "openai", &b);

    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    app.set_error_handler(bevy_ecs::error::warn)
        .insert_resource(models)
        .insert_resource(Settings::from_toml(
            "[model-fallback]\n\
             chain = [\"anthropic/claude-opus-4-5\", \"openai/gpt-5\"]\n\
             cooldown = \"5m\"",
        ))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            CompactionPlugin,
            ModelFallbackPlugin,
        ))
        .insert_resource(TimeUpdateStrategy::ManualDuration(FRAME))
        .init_resource::<Ended>()
        .init_resource::<Notices>()
        .add_observer(log_ended)
        .add_systems(
            Last,
            |mut notices: MessageReader<Notice>, mut seen: ResMut<Notices>| {
                seen.0
                    .extend(notices.read().map(|notice| notice.text.clone()));
            },
        );
    let cooldown = app.world().resource::<FallbackChain>().cooldown;
    assert_eq!(cooldown, Duration::from_secs(300), "read from the settings");
    app.finish();
    // Bevy's clock moves a whole frame at a time, or the retries' waits
    // take hundreds of frames.
    app.world_mut()
        .resource_mut::<Time<Virtual>>()
        .set_max_delta(FRAME);
    app.update();
    let world = app.world_mut();
    let agent = world.query_filtered::<Entity, With<Agent>>().single(world);
    assert!(agent.is_ok(), "the app has an agent");
    let Ok(agent) = agent else {
        return;
    };
    let choice = ModelChoice("anthropic/claude-opus-4-5".to_owned());
    world.entity_mut(agent).insert(choice);

    // A fails each of its retries; the turn goes on on B.
    assert_eq!(ask(&mut app, &wakes, agent, "hello").as_deref(), Some("ok"));
    assert_eq!(a.request_count(), 5, "A is retried as usual first");
    assert_eq!(b.request_count(), 1);
    let notices = &app.world().resource::<Notices>().0;
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
    app.insert_resource(TimeUpdateStrategy::ManualDuration(cooldown));
    app.update();
    app.insert_resource(TimeUpdateStrategy::ManualDuration(FRAME));
    assert_eq!(ask(&mut app, &wakes, agent, "later").as_deref(), Some("ok"));
    assert_eq!(a.request_count(), 10, "A is back once it cooled down");
    assert_eq!(b.request_count(), 3);
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(ModelFallbackPlugin);
}

/// The model plugins, loaded together.
#[test]
fn stages_are_ordered_with_the_other_model_plugins() {
    rig_defaults_test_support::assert_stages_ordered((
        ModelFallbackPlugin,
        crate::model_router::ModelRouterPlugin,
        crate::local_models::LocalModelsPlugin,
        corpus_ho::api_key_pool::KeyPoolPlugin,
    ));
}
