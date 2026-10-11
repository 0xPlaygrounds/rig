use std::path::PathBuf;

use bevy_app::prelude::*;
use rig::harness_protocol::{Home, env};

use super::{Processes, ProjectRoot, RigHome, ServicesPlugin, TokioRuntime};

#[test]
fn what_the_app_inserted_stays_and_the_rest_comes_from_the_process() {
    let mut app = App::new();
    app.insert_resource(RigHome(Home::at("/srv/rig-home")))
        .insert_resource(ProjectRoot(PathBuf::from("/srv/project")))
        .add_plugins(ServicesPlugin);
    let world = app.world();
    assert_eq!(
        world.resource::<RigHome>().0.root(),
        PathBuf::from("/srv/rig-home")
    );
    assert_eq!(
        world.resource::<ProjectRoot>().0,
        PathBuf::from("/srv/project")
    );
    // No child process acts as this agent.
    let processes = world.resource::<Processes>();
    let command = processes.launcher().command("true");
    let unset: Vec<_> = command
        .get_envs()
        .filter(|(_, value)| value.is_none())
        .map(|(name, _)| name.to_string_lossy().into_owned())
        .collect();
    assert!(
        env::AGENT_ONLY
            .iter()
            .all(|name| unset.contains(&(*name).to_owned()))
    );
    // A change reaches every holder of the resource, a tool's too.
    let held = processes.clone();
    processes.set(processes.launcher().wrapped(["nice"]));
    assert_eq!(held.launcher().prefix, ["nice"]);
    // Plugins run futures on rig's own runtime.
    let answer = world
        .get_resource::<TokioRuntime>()
        .map(|runtime| runtime.0.block_on(async { 6 * 7 }));
    assert_eq!(answer, Some(42));
}

#[test]
fn without_one_the_project_is_the_working_directory() {
    let mut app = App::new();
    app.add_plugins(ServicesPlugin);
    let here = std::env::current_dir().ok();
    assert_eq!(
        app.world()
            .get_resource::<ProjectRoot>()
            .map(|root| root.0.clone()),
        here
    );
}
