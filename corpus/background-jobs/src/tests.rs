use std::time::{Duration, Instant};

use rig_cassette::journal::MemoryStore;
use rig_core::catalog::{Catalog, Connector};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::harness_protocol::SessionId;
use serde_json::json;

use super::*;
use rig_harness_test_support::{Ended, app_on, connect, log_ended, reply, run_until};

/// The long job's command, unique enough to find among the processes.
const LONG: &str = "sleep 37.25";

/// Whether a process runs whose command line contains `text`.
fn running(text: &str) -> bool {
    let Ok(entries) = fs::read_dir("/proc") else {
        return false;
    };
    entries.flatten().any(|entry| {
        let cmdline = fs::read(entry.path().join("cmdline")).unwrap_or_default();
        String::from_utf8_lossy(&cmdline)
            .replace('\0', " ")
            .contains(text)
    })
}

/// A job that exits after the turn that started it ends starts a turn of
/// its idle agent (`notify_idle`) whose request says how it ended; jobs
/// start as every child process does (here with a variable the launcher
/// sets, as a sandbox's wrapper would apply); a job still running when the
/// app goes is killed with it.
#[test]
fn a_finished_job_tells_its_idle_agent_and_none_outlives_the_app() {
    let start = |id: &str, command: &str| {
        MockStreamEvent::tool_call(id, "job_start", json!({ "command": command }))
    };
    let model = MockCompletionModel::from_stream_turns([
        vec![
            start("1", "sleep 0.5; echo finished $VIA"),
            start("2", LONG),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        reply("started"),
        reply("seen"),
    ]);
    let target = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target");
    let home = std::env::var_os("CARGO_TARGET_DIR")
        .map_or(target, PathBuf::from)
        .join("corpus-background-jobs-test");
    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    let processes = Processes::default();
    processes.set(processes.launcher().with_env("VIA", "the-launcher"));
    app.insert_resource(Models(Connector::new(Catalog::default())))
        .insert_resource(Settings::from_toml("[jobs]\nnotify_idle = true"))
        .insert_resource(SessionPaths(Home::at(home).session(&SessionId::generate())))
        .insert_resource(processes)
        .add_plugins((AgentPlugin, JournalPlugin, BackgroundJobsPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next();
    let connection = connect(&model);
    assert!(
        agent.is_some() && connection.is_some(),
        "an agent on the model"
    );
    let (Some(agent), Some(connection)) = (agent, connection) else {
        return;
    };
    world.entity_mut(agent).insert(connection);
    world.trigger(Deliver::new(agent, "Run the tests.", DeliveryMode::Steer));

    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() >= 2
    });

    assert_eq!(model.request_count(), 3, "the exit started a turn");
    let third = format!("{:?}", model.requests().get(2));
    assert!(third.contains("exited with 0"), "{third}");
    assert!(third.contains("finished the-launcher"), "{third}");
    assert!(running(LONG), "the long job runs past the turns");

    drop(app);
    let gone = Instant::now();
    while running(LONG) && gone.elapsed() < Duration::from_secs(2) {
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(!running(LONG), "the long job died with the app");
}
