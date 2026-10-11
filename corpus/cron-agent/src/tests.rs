use std::sync::mpsc::Receiver;

use bevy_time::TimeUpdateStrategy;
use rig_cassette::journal::MemoryStore;
use rig_core::catalog::{Catalog, Connector};
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{app_on, connect, reply, run_until};

use super::*;

/// What the test watched: each message delivered (agent, mode, text) and
/// the name of each job run started.
#[derive(Resource, Default)]
struct Seen {
    delivered: Vec<(Entity, DeliveryMode, String)>,
    started: Vec<String>,
}

/// An app on `store` with the cron plugin on a clock that moves only when
/// the test moves it, after its first frame restored the session, and its
/// agent nothing spawned; a new one answered by `model`, when given.
fn session(
    store: &MemoryStore,
    model: Option<&MockCompletionModel>,
) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(store);
    app.insert_resource(Models(Connector::new(Catalog::default())))
        .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::ZERO))
        .add_plugins((AgentPlugin, JournalPlugin, CronPlugin))
        .init_resource::<Seen>()
        .add_observer(|sent: On<Deliver>, mut seen: ResMut<Seen>| {
            let entry = (sent.entity, sent.mode, sent.text.clone());
            seen.delivered.push(entry);
        })
        .add_observer(
            |run: On<Add<CronRun>>, names: Query<&Name>, mut seen: ResMut<Seen>| {
                let name = names.get(run.entity).map(|name| name.as_str().to_owned());
                seen.started.push(name.unwrap_or_default());
            },
        );
    if let Some(connection) = model.and_then(connect) {
        let choice = ModelChoice(connection.spec.reference());
        app.world_mut().spawn((Agent, connection, choice));
    }
    app.finish();
    app.update();
    let world = app.world_mut();
    let mut roots = world.query_filtered::<Entity, (With<Agent>, Without<SpawnedBy>)>();
    let agent = roots.iter(world).next().unwrap_or(Entity::PLACEHOLDER);
    (app, wakes, agent)
}

/// Moves the clock on by `minutes`, a minute a frame.
fn advance(app: &mut App, minutes: u32) {
    for _ in 0..minutes {
        app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs(60)));
        app.update();
    }
    app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::ZERO));
    app.update();
}

fn seen(app: &App) -> &Seen {
    app.world().resource::<Seen>()
}

#[test]
fn a_job_runs_in_its_own_subagent_on_its_tick_and_survives_a_restart() {
    let store = MemoryStore::default();
    let model = MockCompletionModel::from_stream_turns([reply("CI green")]);
    let (mut app, wakes, agent) = session(&store, Some(&model));
    let typed = Deliver::new(
        agent,
        "/cron add \"every 10m\" check CI",
        DeliveryMode::Queue,
    );
    app.world_mut().trigger(typed);
    app.update();

    // Nothing runs before the tick; on it, one subagent `cron:1` gets the
    // prompt, and its answer reaches the agent as a note.
    advance(&mut app, 9);
    assert!(seen(&app).started.is_empty());
    advance(&mut app, 1);
    let noted = |world: &World| {
        let seen = world.resource::<Seen>();
        seen.delivered.iter().any(|(to, mode, text)| {
            *to == agent && *mode == DeliveryMode::Note && text == "CI green"
        })
    };
    run_until(&mut app, &wakes, noted);
    assert!(noted(app.world()), "{:?}", seen(&app).delivered);
    assert_eq!(seen(&app).started, ["cron:1"]);
    let asked = format!("{:?}", model.requests());
    assert!(asked.contains("check CI"), "{asked}");
    // The subagent is gone once it answered.
    let world = app.world_mut();
    assert_eq!(world.query::<&CronRun>().iter(world).count(), 0);

    // Five more minutes: no second run.
    advance(&mut app, 5);
    assert_eq!(seen(&app).started, ["cron:1"]);
    assert_eq!(model.request_count(), 1);
    drop(app);

    // The restored session has the job, and not the finished run's agent.
    let (mut app, _wakes, agent) = session(&store, None);
    app.world_mut().trigger(RunCommand {
        entity: agent,
        line: "cron list".to_owned(),
    });
    app.update();
    let notices = app.world().resource::<Messages<Notice>>();
    let mut cursor = notices.get_cursor();
    let texts: Vec<&str> = cursor
        .read(notices)
        .map(|notice| notice.text.as_str())
        .collect();
    let listed = texts
        .iter()
        .any(|text| text.contains("cron:1 \"every 10m\"") && text.contains("check CI"));
    assert!(listed, "{texts:?}");
    // The session log brought the run's agent back, and it went again.
    assert_eq!(seen(&app).started.len(), 1);
    let world = app.world_mut();
    assert_eq!(world.query::<&CronRun>().iter(world).count(), 0);
}
