use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::{MemoryStore, app_on, hold_turn_calls, run_until, scripted_models};

use super::*;
use rig_ecs::spawn::spawn;

#[test]
fn the_users_stop_cancels_the_build_only_on_an_idle_agent() {
    let model = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let models = scripted_models([("reload-main/m", &model)]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(models)
        .add_plugins((AgentPlugin, SessionLogPlugin, ReloadPlugin))
        .add_systems(AgentPass, hold_turn_calls.in_set(Stage::Gate));
    let choice = || ModelChoice("reload-main/m".to_owned());
    let agent = spawn_root(app.world_mut()).insert(choice()).id();
    // Idle below it, so the stop reaches an idle agent either way.
    spawn(
        app.world_mut(),
        Origin::ChildOf(agent),
        AgentSpec::default(),
    )
    .ok();
    app.finish();
    app.update();
    let building = ReloadStatus::Building { latest: None };
    app.world_mut().insert_resource(building.clone());
    app.world_mut()
        .trigger(Deliver::new(agent, "go", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        world.get::<ActiveTurn>(agent).is_some()
    });

    app.world_mut().trigger(StopTurn::new(agent));
    app.update();
    assert_eq!(*app.world().resource::<ReloadStatus>(), building);

    app.world_mut().trigger(StopTurn::new(agent));
    app.update();
    assert_eq!(*app.world().resource::<ReloadStatus>(), ReloadStatus::Idle);
}

/// The text of every notice written since the last drain.
fn notices(world: &mut World) -> Vec<String> {
    let mut notices = world.resource_mut::<Messages<Notice>>();
    notices.drain().map(|notice| notice.text).collect()
}

/// A queued `/update` is refused and cancelled in its own words: they name
/// the update and `/update cancel`, never `/reload`.
#[test]
fn a_queued_update_is_refused_and_cancelled_as_an_update() {
    let model = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(scripted_models([("reload-main/m", &model)]))
        .add_plugins((AgentPlugin, SessionLogPlugin, ReloadPlugin))
        .add_systems(AgentPass, hold_turn_calls.in_set(Stage::Gate));
    let choice = ModelChoice("reload-main/m".to_owned());
    let agent = spawn_root(app.world_mut()).insert(choice).id();
    app.finish();
    app.update();
    // A turn runs, so the queued update waits.
    app.world_mut()
        .trigger(Deliver::new(agent, "go", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        world.get::<ActiveTurn>(agent).is_some()
    });
    let queued = ReloadStatus::Queued {
        agent,
        rebuild: Rebuild::Update,
    };
    app.world_mut().insert_resource(queued.clone());
    notices(app.world_mut());

    app.world_mut().trigger(RunCommand::new(agent, "update"));
    app.update();
    let refused = notices(app.world_mut());
    assert_eq!(*app.world().resource::<ReloadStatus>(), queued);

    app.world_mut()
        .trigger(RunCommand::new(agent, "update cancel"));
    app.update();
    let cancelled = notices(app.world_mut());
    assert_eq!(*app.world().resource::<ReloadStatus>(), ReloadStatus::Idle);

    let about_the_update = |said: &[String]| {
        !said.is_empty()
            && said
                .iter()
                .all(|text| text.contains("update") && !text.contains("reload"))
    };
    assert!(about_the_update(&refused), "{refused:?}");
    assert!(refused.iter().all(|text| text.contains("/update cancel")));
    assert!(about_the_update(&cancelled), "{cancelled:?}");
}
