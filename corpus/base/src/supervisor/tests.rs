use std::sync::mpsc::Receiver;

use rig_core::test_utils::MockCompletionModel;
use rig_harness::clock::ClockPlugin;
use rig_harness_test_support::MemoryStore;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use super::*;
use rig_harness_test_support::{
    Collected, Ended, app_on, calls, collect, first_agent, log_ended, pass_wall_clock, reply,
    run_until, scripted_models,
};

/// The open `read` calls, with their paths, until the test answers them.
#[derive(Resource, Default)]
struct Reads(Vec<(Entity, String)>);

#[derive(Deserialize, JsonSchema)]
struct ReadArgs {
    path: String,
}

struct Session {
    app: App,
    wakes: Receiver<()>,
    worker: Entity,
}

impl Session {
    /// A worker on `test/worker`, given a supervisor by `typed` (a
    /// `/supervise` naming `model=test/supervisor`), then asked to fix the
    /// parser.
    fn new(worker: &MockCompletionModel, supervisor: &MockCompletionModel, typed: &str) -> Self {
        let (mut app, wakes) = app_on(&MemoryStore::default());
        let read = |called: On<ToolCalled<ReadArgs>>, mut reads: ResMut<Reads>| {
            reads.0.push((called.call, called.args.path.clone()));
        };
        let models = [("test/worker", worker), ("test/supervisor", supervisor)];
        app.insert_resource(scripted_models(models))
            .add_plugins((AgentPlugin, SessionLogPlugin, ClockPlugin, SupervisorPlugin))
            .init_resource::<Reads>()
            .init_resource::<Ended>()
            .add_open_tool("read", "Read a file.", ToolOptions::default(), read)
            .add_observer(log_ended);
        collect::<Notice>(&mut app);
        app.finish();
        app.update();
        let worker = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
        let model = "test/worker".to_owned();
        let world = app.world_mut();
        world.trigger(SetModel {
            entity: worker,
            model,
            kind: ChoiceKind::ForThisRun,
        });
        world.trigger(Deliver::new(worker, typed, DeliveryMode::Steer));
        run_until(&mut app, &wakes, |world| {
            let mut supervisors = world.try_query_filtered::<&SpawnedBy, With<Supervisor>>();
            let mut supervisors = supervisors.iter_mut().flat_map(|query| query.iter(world));
            supervisors.any(|&SpawnedBy(of)| of == worker)
        });
        let asked = "The parser drops trailing commas; fix it.";
        app.world_mut()
            .trigger(Deliver::new(worker, asked, DeliveryMode::Queue));
        Self { app, wakes, worker }
    }

    /// Runs until `done`.
    fn run_until(&mut self, done: impl Fn(&World) -> bool) {
        run_until(&mut self.app, &self.wakes, done);
    }

    /// Waits for the worker's next `read` call and answers it.
    fn answer_read(&mut self) {
        self.run_until(|world| !world.resource::<Reads>().0.is_empty());
        let world = self.app.world_mut();
        let reads = std::mem::take(&mut world.resource_mut::<Reads>().0);
        for (call, path) in reads {
            let output = ToolResult::success(format!("contents of {path}").into());
            if let Ok(mut call) = world.get_entity_mut(call) {
                call.insert(Outcome::from(output));
            }
        }
    }

    /// Whether a notice said `text`.
    fn told(&self, text: &str) -> bool {
        let notices = &self.app.world().resource::<Collected<Notice>>().0;
        notices.iter().any(|notice| notice.text.contains(text))
    }
}

/// A worker that reads three files, then answers.
fn reader() -> MockCompletionModel {
    MockCompletionModel::from_stream_turns([
        calls("r1", "read", json!({ "path": "src/parser.rs" })),
        calls("r2", "read", json!({ "path": "src/lexer.rs" })),
        calls("r3", "read", json!({ "path": "tests/parser.rs" })),
        reply("Fixed."),
    ])
}

/// A supervisor that gives one verdict, then closes its turn.
fn judge(verdict: serde_json::Value) -> MockCompletionModel {
    MockCompletionModel::from_stream_turns([calls("v1", VERDICT, verdict), reply("Done.")])
}

/// Whether `worker` accepted a steering message saying `text`.
fn steered(world: &World, worker: Entity, text: &str) -> bool {
    let mut messages = world.try_query_filtered::<(&Incoming, &ForAgent), With<Started>>();
    let mut messages = messages.iter_mut().flat_map(|query| query.iter(world));
    messages.any(|(message, &ForAgent(to))| {
        to == worker && message.mode == DeliveryMode::Steer && message.text() == text
    })
}

#[test]
fn the_supervisor_checks_the_worker_and_steers_or_stops_it() {
    let every_two = "/supervise calls=2 model=test/supervisor Fix the parser bug";

    // Every 2 calls: the check after the 2nd call steers the worker while
    // it reads on.
    let supervisor = judge(json!({ "decision": "steer", "message": "check tests first" }));
    let worker = reader();
    let mut session = Session::new(&worker, &supervisor, every_two);
    session.answer_read();
    session.answer_read();
    let at = session.worker;
    session.run_until(|world| steered(world, at, "check tests first"));
    session.answer_read();
    session.run_until(|_| worker.request_count() >= 4 && supervisor.request_count() >= 2);

    assert_eq!(supervisor.request_count(), 2);
    let checked = format!("{:?}", supervisor.requests().first());
    for part in [
        "Fix the parser bug",
        "src/parser.rs",
        "src/lexer.rs",
        "contents of src/lexer.rs",
    ] {
        assert!(checked.contains(part), "{part}: {checked}");
    }
    assert!(!checked.contains("tests/parser.rs"), "{checked}");
    let sent = format!("{:?}", worker.requests().get(2));
    assert!(!sent.contains("check tests first"), "{sent}");
    let steered = format!("{:?}", worker.requests().get(3));
    assert!(steered.contains("check tests first"), "{steered}");

    // A supervisor that answers `stop` ends the worker's turn as stopped,
    // and the user is told why.
    let supervisor = judge(json!({ "decision": "stop", "message": "it reads files in circles" }));
    let worker = reader();
    let mut session = Session::new(&worker, &supervisor, every_two);
    session.answer_read();
    session.answer_read();
    let stopped = |world: &World| {
        let ended = &world.resource::<Ended>().0;
        ended.iter().any(|outcome| {
            matches!(outcome, TurnOutcome::Stopped(stop)
                if stop.cause().to_string().contains("it reads files in circles"))
        })
    };
    session.run_until(stopped);
    assert!(stopped(session.app.world()));
    assert!(session.told("it reads files in circles"));
    assert_eq!(worker.request_count(), 3);

    // By time: once the minutes passed, the supervisor gets what the worker
    // did meanwhile.
    let supervisor = judge(json!({ "decision": "continue" }));
    let worker = reader();
    let by_time = "/supervise calls=0 minutes=5 model=test/supervisor Fix the parser bug";
    let mut session = Session::new(&worker, &supervisor, by_time);
    session.answer_read();
    session.run_until(|_| worker.request_count() >= 2);
    pass_wall_clock(session.app.world_mut(), Duration::from_secs(6 * 60));
    session.run_until(|_| supervisor.request_count() >= 1);
    let checked = format!("{:?}", supervisor.requests().first());
    assert!(checked.contains("contents of src/parser.rs"), "{checked}");
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(SupervisorPlugin);
}
