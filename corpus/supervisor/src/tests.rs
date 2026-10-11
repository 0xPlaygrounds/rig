use std::sync::mpsc::Receiver;

use rig_cassette::journal::MemoryStore;
use rig_core::catalog::{Catalog, Connector};
use rig_core::test_utils::MockCompletionModel;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use super::*;
use rig_harness_test_support::{Ended, app_on, calls, connect, log_ended, reply, run_until};

/// The open `read` calls, with their paths, until the test answers them.
#[derive(Resource, Default)]
struct Reads(Vec<(Entity, String)>);

/// The notices written.
#[derive(Resource, Default)]
struct Notices(Vec<String>);

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
            .init_resource::<Notices>()
            .init_resource::<Ended>()
            .add_open_tool("read", "Read a file.", ToolOptions::default(), read)
            .add_observer(log_ended)
            .add_systems(
                Last,
                |mut notices: MessageReader<Notice>, mut seen: ResMut<Notices>| {
                    seen.0
                        .extend(notices.read().map(|notice| notice.text.clone()));
                },
            );
        let worker = connect(worker).map(|connection| {
            let choice = ModelChoice(connection.spec.reference());
            app.world_mut().spawn((Agent, connection, choice)).id()
        });
        let worker = worker.unwrap_or(Entity::PLACEHOLDER);
        app.finish();
        app.update();
        let typed = "/supervise calls=2 Fix the parser bug";
        app.world_mut()
            .trigger(Deliver::new(worker, typed, DeliveryMode::Steer));
        let hired = |world: &World| {
            let spawned = world.get::<Spawned>(worker);
            let supervisors = spawned.into_iter().flat_map(Spawned::iter);
            supervisors
                .filter(|&agent| world.get::<Supervisor>(agent).is_some())
                .last()
        };
        run_until(&mut app, &wakes, |world| hired(world).is_some());
        if let (Some(hired), Some(connection)) = (hired(app.world()), connect(supervisor)) {
            app.world_mut().entity_mut(hired).insert(connection);
        }
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
        self.answer_reads();
    }

    fn answer_reads(&mut self) {
        let world = self.app.world_mut();
        let reads = std::mem::take(&mut world.resource_mut::<Reads>().0);
        for (call, path) in reads {
            let output = ToolResult::success(format!("contents of {path}").into());
            if let Ok(mut call) = world.get_entity_mut(call) {
                call.insert(Outcome::from(output));
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

/// Whether `worker` accepted a steering message saying `text`.
fn steered(world: &mut World, worker: Entity, text: &str) -> bool {
    let mut messages = world.query_filtered::<(&Incoming, &ForAgent), With<Started>>();
    messages.iter(world).any(|(message, &ForAgent(to))| {
        to == worker && message.mode == DeliveryMode::Steer && message.text == text
    })
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
    let at = session.worker;
    let mut found = false;
    for _ in 0..100 {
        session.app.update();
        session.wakes.recv_timeout(Duration::from_millis(100)).ok();
        if steered(session.app.world_mut(), at, "check tests first") {
            found = true;
            break;
        }
    }
    assert!(found, "the worker was never steered");
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

    // A supervisor that answers `stop` interrupts the worker's turn, and
    // the user is told why.
    let stop = json!({ "decision": "stop", "message": "it reads files in circles" });
    let supervisor =
        MockCompletionModel::from_stream_turns([calls("v1", VERDICT, stop), reply("Stopped.")]);
    let worker = reader();
    let mut session = Session::new(&worker, &supervisor);
    session.answer_read();
    session.answer_read();
    session.run_until(|world| {
        let ended = &world.resource::<Ended>().0;
        ended
            .iter()
            .any(|outcome| matches!(outcome, TurnOutcome::Stopped(_)))
    });
    let ended = &session.app.world().resource::<Ended>().0;
    let stopped = ended.iter().find_map(|outcome| match outcome {
        TurnOutcome::Stopped(why) => why.clone(),
        _ => None,
    });
    let stopped = stopped.unwrap_or_default();
    assert!(stopped.contains("it reads files in circles"), "{ended:?}");
    let notices = &session.app.world().resource::<Notices>().0;
    let told = notices
        .iter()
        .any(|notice| notice.contains("it reads files in circles"));
    assert!(told, "{notices:?}");
    assert_eq!(worker.request_count(), 3);
}
