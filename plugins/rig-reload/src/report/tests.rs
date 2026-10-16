use std::sync::mpsc::Receiver;

use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::{
    MemoryStore, app_on, first_agent, reply, run_until, scripted_models,
};

use super::*;
use crate::ReloadPlugin;

/// The model the agent uses.
pub(crate) const MODEL: &str = "test/model";

/// The messages sent to agents: who wrote each, its mode and its text.
#[derive(Resource, Default)]
pub(crate) struct Sent(pub(crate) Vec<(AuthorKind, DeliveryMode, String)>);

/// An app with the reload plugin on `store`, started as `start`, its
/// agents answered by `model`, after its first frame.
pub(crate) fn session(
    store: &MemoryStore,
    start: StartReason,
    model: &MockCompletionModel,
) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    app.insert_resource(AppStart(start))
        .insert_resource(scripted_models([(MODEL, model)]))
        .add_plugins((AgentPlugin, SessionLogPlugin, ReloadPlugin))
        .init_resource::<Sent>()
        .add_observer(
            |sent: On<Add<Incoming>>, messages: Query<&Incoming>, mut log: ResMut<Sent>| {
                if let Ok(message) = messages.get(sent.entity) {
                    let entry = (
                        message.author.kind.clone(),
                        message.mode,
                        message.text().into(),
                    );
                    log.0.push(entry);
                }
            },
        );
    app.finish();
    app.update();
    (app, wakes)
}

/// The first agent of `app`, on [`MODEL`].
pub(crate) fn agent_on_model(app: &mut App) -> Entity {
    let world = app.world_mut();
    let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
    let model = MODEL.to_owned();
    world.trigger(SetModel {
        entity: agent,
        model,
    });
    agent
}

/// The texts of the reports rig-reload sent, each starting a turn.
pub(crate) fn reports(app: &App) -> Vec<String> {
    let sent = app.world().resource::<Sent>().0.iter();
    let reports = sent.filter(|(author, mode, _)| {
        *author == AuthorKind::Plugin(ORIGIN.to_owned()) && *mode == DeliveryMode::Queue
    });
    reports.map(|(.., text)| text.clone()).collect()
}

/// A session whose model asked for a reload that would do `then`, as the
/// `reload` tool leaves it, saved in a new store.
fn asked_for_a_reload(then: &str) -> MemoryStore {
    let store = MemoryStore::default();
    let model = MockCompletionModel::from_stream_turns([reply("Reloading.")]);
    let (mut app, wakes) = session(&store, StartReason::New, &model);
    let agent = agent_on_model(&mut app);
    app.world_mut().trigger(Deliver::new(
        agent,
        "Add a word_count tool.",
        DeliveryMode::Steer,
    ));
    run_until(&mut app, &wakes, |world| {
        model.request_count() == 1 && world.get::<ActiveTurn>(agent).is_none()
    });
    let asked = ReloadAsked {
        then: then.to_owned(),
        attempts: 1,
        owed: true,
    };
    app.world_mut().entity_mut(agent).insert(asked);
    app.update();
    store
}

/// After the restart on the new build, and after a rollback, the agent
/// that asked gets one message that starts a turn, and no later start
/// sends it again.
#[test]
fn a_model_asked_reload_reports_once_after_the_restart_and_after_a_rollback() {
    let said = [
        (StartReason::Reloaded, "the new build started"),
        (
            StartReason::Restarted,
            "the new build stopped during startup",
        ),
    ];
    for (start, meant) in said {
        let store = asked_for_a_reload("count the words of README.md");
        let model = MockCompletionModel::from_stream_turns([reply("12 words.")]);
        let (mut app, wakes) = session(&store, start, &model);
        run_until(&mut app, &wakes, |_| model.request_count() == 1);
        for _ in 0..3 {
            app.update();
        }
        let sent = reports(&app);
        let once = matches!(sent.as_slice(), [report] if report.contains(meant));
        assert!(once, "{start:?}: {sent:?}");
        let read = format!("{:?}", model.requests());
        assert!(read.contains(meant), "the turn it started read it: {read}");
        let reloaded = start == StartReason::Reloaded;
        assert_eq!(read.contains("count the words of README.md"), reloaded);
        drop(app);

        // A later start of the same session: nothing more.
        let quiet = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
        let (mut app, _) = session(&store, start, &quiet);
        for _ in 0..5 {
            app.update();
        }
        assert_eq!(reports(&app), Vec::<String>::new(), "{start:?}");
        assert_eq!(quiet.request_count(), 0);
    }
}

/// The user stopping the agent clears what it is owed, and a resume after
/// a reload nobody knows the end of sends nothing.
#[test]
fn a_stop_or_a_resume_drops_the_report() {
    let store = asked_for_a_reload("");
    let model = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let (mut app, _) = session(&store, StartReason::Resumed, &model);
    for _ in 0..3 {
        app.update();
    }
    assert_eq!(reports(&app), Vec::<String>::new());

    let store = asked_for_a_reload("");
    let (mut app, _) = session(&store, StartReason::New, &model);
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    app.world_mut().trigger(StopTurn::new(agent));
    app.update();
    let owed = app
        .world()
        .get::<ReloadAsked>(agent)
        .map(|asked| asked.owed);
    assert_eq!(owed, Some(false));
}
