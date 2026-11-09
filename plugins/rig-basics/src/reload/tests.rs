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
