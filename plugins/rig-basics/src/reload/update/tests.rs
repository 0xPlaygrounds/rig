use rig_harness_test_support::{MemoryStore, app_on};

use super::*;
use crate::reload::tests::slash_commands;
use crate::reload::{Rebuild, ReloadStatus};

/// With the reload plugin left out, `/update` runs and `/reload` is no
/// command, so `/reload cancel` leaves a queued update alone.
#[test]
fn without_the_reload_plugin_update_runs_and_reload_is_no_command() {
    let (mut app, _) = app_on(&MemoryStore::default());
    app.add_plugins((AgentPlugin, SessionLogPlugin, UpdatePlugin));
    let agent = spawn_root(app.world_mut(), ()).id();
    app.finish();
    app.update();
    let commands = slash_commands(app.world_mut());
    assert!(commands.iter().any(|name| name == "/update"), "{commands:?}");
    assert!(!commands.iter().any(|name| name == "/reload"), "{commands:?}");

    let queued = ReloadStatus::Queued {
        agent,
        rebuild: Rebuild::Update,
    };
    app.world_mut().insert_resource(queued.clone());
    app.world_mut()
        .trigger(RunCommand::new(agent, "reload cancel"));
    app.update();
    assert_eq!(*app.world().resource::<ReloadStatus>(), queued);

    app.world_mut()
        .trigger(RunCommand::new(agent, "update cancel"));
    app.update();
    assert_eq!(*app.world().resource::<ReloadStatus>(), ReloadStatus::Idle);
}
