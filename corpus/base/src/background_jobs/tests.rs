use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::harness_protocol::{Home, SessionId};
use rig_harness::host::exit::ExitPlugin;
use serde_json::json;

use super::*;
use rig_harness_test_support::{
    Ended, app_in, calls, first_agent, log_ended, reply, run_until, scratch_home, scripted_models,
};

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

/// Waits up to two seconds for no process with `text` in its command line.
fn gone(text: &str) -> bool {
    let since = Instant::now();
    while running(text) && since.elapsed() < Duration::from_secs(2) {
        std::thread::sleep(Duration::from_millis(20));
    }
    !running(text)
}

/// The session `id` of `home` with the plugin, the `[jobs]` settings
/// `jobs`, and its agent answered by `model`.
fn session(
    home: &Home,
    id: &SessionId,
    jobs: &str,
    model: &MockCompletionModel,
    processes: Processes,
) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_in(home, id);
    app.insert_resource(scripted_models([("test/model", model)]))
        .insert_resource(Settings::from_toml(&format!("[jobs]\n{jobs}")))
        .insert_resource(processes)
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            ExitPlugin,
            BackgroundJobsPlugin,
        ))
        .init_resource::<Ended>()
        .add_observer(log_ended);
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

/// A job that exits after the turn that started it ends starts a turn of
/// its idle agent (`notify_idle`) whose request says how it ended; jobs
/// start as every child process does (here with a variable the launcher
/// sets, as a sandbox's wrapper would apply); a job still running when the
/// app goes is killed with it.
#[test]
fn a_finished_job_tells_its_idle_agent_and_none_outlives_the_app() {
    const LONG: &str = "sleep 37.25";
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
    let processes = Processes::default();
    processes.set(processes.launcher().with_env("VIA", "the-launcher"));
    let home = scratch_home("background-jobs-idle");
    let id = SessionId::generate();
    let (mut app, wakes, agent) = session(&home, &id, "notify_idle = true", &model, processes);
    let ask = Deliver::new(agent, "Run the tests.", DeliveryMode::Steer);
    app.world_mut().trigger(ask);

    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() >= 2
    });

    assert_eq!(model.request_count(), 3, "the exit started a turn");
    let third = format!("{:?}", model.requests().get(2));
    assert!(third.contains("exited with 0"), "{third}");
    assert!(third.contains("finished the-launcher"), "{third}");
    assert!(running(LONG), "the long job runs past the turns");

    drop(app);
    assert!(gone(LONG), "the long job died with the app");
}

/// With `survive_reload`, a `/reload` leaves a job running, and the next
/// process re-attaches to it from the saved session and reports its exit.
#[test]
fn a_job_survives_a_reload_and_the_next_process_reports_its_exit() {
    const SLOW: &str = "sleep 1.25; echo survived";
    let jobs = "notify_idle = true\nsurvive_reload = true";
    let home = scratch_home("background-jobs-reload");
    let id = SessionId::generate();
    let before = MockCompletionModel::from_stream_turns([
        calls("1", "job_start", json!({ "command": SLOW, "name": "slow" })),
        reply("started"),
    ]);
    let (mut app, wakes, agent) = session(&home, &id, jobs, &before, Processes::default());
    let ask = Deliver::new(agent, "Start it.", DeliveryMode::Steer);
    app.world_mut().trigger(ask);
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    app.world_mut().trigger(Exit(ExitReason::Reload));
    app.update();
    drop(app);
    assert!(running(SLOW), "the job outlives the reloaded process");

    let after = MockCompletionModel::from_stream_turns([reply("seen")]);
    let (mut app, wakes, _) = session(&home, &id, jobs, &after, Processes::default());
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });

    let request = format!("{:?}", after.requests().first());
    assert!(
        request.contains("job `slow` (id 1) exited with 0"),
        "{request}"
    );
    assert!(request.contains("survived"), "{request}");
    drop(app);
    assert!(gone(SLOW), "nothing is left");
}
