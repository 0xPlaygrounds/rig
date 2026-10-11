use std::sync::mpsc::Receiver;

use rig_cassette::journal::MemoryStore;
use rig_core::catalog::{Catalog, Connector};
use rig_core::test_utils::MockCompletionModel;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use super::*;
use rig_harness_test_support::{app_on, calls, connect, reply, run_until};

/// The open `read` calls, with their paths, until the test answers them.
#[derive(Resource, Default)]
struct Reads(Vec<(Entity, String)>);

/// The notices written and the turns ended, with whether they stopped.
#[derive(Resource, Default)]
struct Seen {
    notices: Vec<String>,
    ended: Vec<(Entity, bool)>,
}

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
    /// A worker answered by `worker`, supervised every 2 tool calls by a
    /// supervisor answered by `supervisor`, and asked to fix the parser.
    fn new(worker: &MockCompletionModel, supervisor: &MockCompletionModel) -> Self {
        let (mut app, wakes) = app_on(&MemoryStore::default());
        let read = |called: On<ToolCalled<ReadArgs>>, mut reads: ResMut<Reads>| {
            reads.0.push((called.call, called.args.path.clone()));
        };
        app.insert_resource(Models(Connector::new(Catalog::default())))
            .add_plugins((AgentPlugin, JournalPlugin, SupervisorPlugin))
            .init_resource::<Reads>()
            .init_resource::<Seen>()
            .add_open_tool("read", "Read a file.", ToolOptions::default(), read)
            .add_observer(|end: On<TurnEnded>, mut seen: ResMut<Seen>| {
                let stopped = matches!(end.outcome, TurnOutcome::Stopped);
                seen.ended.push((end.entity, stopped));
            })
            .add_systems(
                Last,
                |mut notices: MessageReader<Notice>, mut seen: ResMut<Seen>| {
                    let texts = notices.read().map(|notice| notice.text.clone());
                    seen.notices.extend(texts);
                },
            );
        let worker = connect(worker).map(|connection| {
            let choice = ModelChoice(connection.spec.reference());
            app.world_mut().spawn((Agent, connection, choice)).id()
        });
        let worker = worker.unwrap_or(Entity::PLACEHOLDER);
        app.finish();
        app.update();
        let world = app.world_mut();
        let typed = "/supervise calls=2 Fix the parser bug";
        world.trigger(Deliver::new(worker, typed, DeliveryMode::Steer));
        world.flush();
        let mut supervisors = world.query_filtered::<Entity, With<Supervisor>>();
        let hired = supervisors.iter(world).next();
        if let (Some(hired), Some(connection)) = (hired, connect(supervisor)) {
            world.entity_mut(hired).insert(connection);
        }
        let asked = "The parser drops trailing commas; fix it.";
        world.trigger(Deliver::new(worker, asked, DeliveryMode::Queue));
        Self { app, wakes, worker }
    }

    /// Runs until `done`.
    fn run_until(&mut self, done: impl Fn(&World, Entity) -> bool) {
        let worker = self.worker;
        run_until(&mut self.app, &self.wakes, |world| done(world, worker));
    }

    /// Waits for the worker's next `read` call and answers it.
    fn answer_read(&mut self) {
        self.run_until(|world, _| !world.resource::<Reads>().0.is_empty());
        self.answer_reads();
    }

    fn answer_reads(&mut self) {
        let world = self.app.world_mut();
        let reads = std::mem::take(&mut world.resource_mut::<Reads>().0);
        for (call, path) in reads {
            let output = ToolResult::success(format!("contents of {path}").into());
            if let Ok(mut call) = world.get_entity_mut(call) {
                call.insert(ToolOutput(output));
            }
        }
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

#[test]
fn the_supervisor_checks_every_two_calls_and_steers_or_stops_the_worker() {
    let steer = json!({ "decision": "steer", "message": "check tests first" });
    let supervisor =
        MockCompletionModel::from_stream_turns([calls("v1", VERDICT, steer), reply("Steered.")]);
    let worker = reader();
    let mut session = Session::new(&worker, &supervisor);
    session.answer_read();
    session.answer_read();
    // The check after the 2nd call steers the worker while it reads on.
    session.run_until(|world, worker| {
        let steered = world
            .get::<Inbox>(worker)
            .is_some_and(|inbox| !inbox.steering.is_empty());
        steered && !world.resource::<Reads>().0.is_empty()
    });
    session.answer_reads();
    session.run_until(|_, _| worker.request_count() >= 4 && supervisor.request_count() >= 2);

    let checked = format!("{:?}", supervisor.requests().first());
    for part in [
        "Fix the parser bug",
        "src/parser.rs",
        "src/lexer.rs",
        "contents of src/lexer.rs",
    ] {
        assert!(checked.contains(part), "{part}: {checked}");
    }
    let world = session.app.world_mut();
    let mut checks = world.query::<&Supervisor>();
    let checks: Vec<u32> = checks
        .iter(world)
        .map(|supervisor| supervisor.checks)
        .collect();
    assert_eq!(checks, [1]);
    let steered = format!("{:?}", worker.requests().get(3));
    assert!(steered.contains("check tests first"), "{steered}");
    assert!(steered.contains("supervisor"), "{steered}");
    let sent = format!("{:?}", worker.requests().get(2));
    assert!(!sent.contains("check tests first"), "{sent}");

    // A supervisor that answers `stop` interrupts the worker's turn, and
    // the user is told why.
    let stop = json!({ "decision": "stop", "message": "it reads files in circles" });
    let supervisor =
        MockCompletionModel::from_stream_turns([calls("v1", VERDICT, stop), reply("Stopped.")]);
    let worker = reader();
    let mut session = Session::new(&worker, &supervisor);
    session.answer_read();
    session.answer_read();
    session.run_until(|world, worker| {
        world
            .resource::<Seen>()
            .ended
            .iter()
            .any(|(agent, _)| *agent == worker)
    });
    let seen = session.app.world().resource::<Seen>();
    assert!(
        seen.ended.contains(&(session.worker, true)),
        "{:?}",
        seen.ended
    );
    let told = seen
        .notices
        .iter()
        .any(|notice| notice.contains("it reads files in circles"));
    assert!(told, "{:?}", seen.notices);
    assert_eq!(worker.request_count(), 3);
}
