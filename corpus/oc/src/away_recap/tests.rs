use std::sync::mpsc::Receiver;

use rig_core::catalog::{Catalog, ModelSpec, Transport};
use rig_core::effect::{EffectKind, HandlerDescriptor, family};
use rig_core::operation::Completion;
use rig_core::providers::registry::ProviderId;
use rig_core::serve::adapters::ModelAdapter;
use rig_core::serve::{Dispatch, ErasedHandler, Reply, Serve};
use rig_core::test_utils::MockCompletionModel;
use rig_harness::prelude::*;
use rig_harness_test_support::{
    Ended, MemoryStore, app_on, calls, first_agent, log_ended, pass_wall_clock, press, reply,
    run_until, scripted_models, tui_view,
};
use rig_tui::TuiTestScreen;

use super::{AwayRecapPlugin, RecapCall, RecapCard, TITLE};

const RECAP: &str = "Goal: X\n- Edited src/auth.rs and src/client.rs\nPending: run tests";

/// A model that never answers.
struct Never(ErasedHandler);

impl Serve for Never {
    type Family = family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        self.0.descriptor()
    }

    async fn serve(&self, _: EffectKind, _: Dispatch) -> Reply {
        std::future::pending().await
    }
}

/// How many recap calls were started.
#[derive(Resource, Default)]
struct RecapsAsked(usize);

/// The scripted `test/main` and `test/cheap`, and `stall/never`, which
/// never answers.
fn served(main: &MockCompletionModel, cheap: &MockCompletionModel) -> Models {
    let mut models = scripted_models([("test/main", main), ("test/cheap", cheap)]);
    if let Ok(id) = ProviderId::declare("stall") {
        models.0.catalog_mut().insert(ModelSpec::new(id, "never"));
        let quiet = cheap.clone();
        models
            .0
            .provide(id, move |spec: &ModelSpec, _: &Catalog, _: &Transport| {
                let inner = ModelAdapter::<Completion>::new(spec.reference(), quiet.clone());
                Ok(ErasedHandler::new(Never(ErasedHandler::new(inner))))
            });
    }
    models
}

/// An app on `store` started for `reason`, with the plugin, an `edit`
/// that only says it edited and the terminal view on a test screen,
/// recapping on `recap_model`; one frame run.
fn app(
    store: &MemoryStore,
    reason: StartReason,
    models: Models,
    recap_model: &str,
) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    let settings = format!("[recap]\nmodel = \"{recap_model}\"\n");
    app.insert_resource(models)
        .insert_resource(AppStart(reason))
        .insert_resource(TuiTestScreen::new(80, 30))
        .insert_resource(Settings::from_toml(&settings))
        .add_plugins((AgentPlugin, SessionLogPlugin, AwayRecapPlugin))
        .add_open_tool("edit", "Edit a file.", ToolOptions::default(), edited)
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .init_resource::<RecapsAsked>()
        .add_observer(|_: On<Add<RecapCall>>, mut started: ResMut<RecapsAsked>| started.0 += 1);
    tui_view(&mut app);
    app.finish();
    app.cleanup();
    app.update();
    (app, wakes)
}

fn edited(called: On<ToolCalled<serde_json::Value>>, mut calls: Query<&mut Decision, ToolCalls>) {
    if let Ok(mut call) = calls.get_mut(called.call) {
        call.decide(
            Decider::plugin("edit"),
            ToolResult::success("edited".into()),
        );
    }
}

/// Sends `text` as the user and runs until the turn ended.
fn turn(app: &mut App, wakes: &Receiver<()>, text: &str) {
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let ended = app.world().resource::<Ended>().0.len();
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Steer));
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() > ended
    });
}

/// The recap's card shown, if any.
fn card(world: &World) -> Option<String> {
    let mut cards = world.try_query::<&RecapCard>()?;
    cards.iter(world).next().map(|card| card.0.clone())
}

/// What the test screen shows.
fn screen(app: &App) -> String {
    app.world().resource::<TuiTestScreen>().rows().join("\n")
}

/// How many messages the session saved for its only agent.
fn saved_messages(store: &MemoryStore) -> usize {
    let session = Session::read(store).ok();
    let agent = session.as_ref().and_then(|session| session.agents.first());
    agent.map_or(0, |agent| agent.conversation.messages().len())
}

#[test]
fn a_resumed_session_is_recapped_beside_the_work() {
    let edit = |id: &str, path: &str| calls(id, "edit", serde_json::json!({ "path": path }));
    let main = MockCompletionModel::from_stream_turns([
        edit("e1", "src/auth.rs"),
        reply("Edited auth."),
        edit("e2", "src/client.rs"),
        reply("Edited the client."),
        edit("e3", "src/auth.rs"),
        reply("Auth uses the new client."),
        edit("e4", "src/client.rs"),
        reply("Done; run the tests next."),
        reply("Continuing."),
        reply("On it."),
    ]);
    let cheap = MockCompletionModel::from_stream_turns([reply(RECAP)]);
    let store = MemoryStore::default();

    // Four turns that edit two files, saved and closed.
    let (mut first, wakes) = app(
        &store,
        StartReason::New,
        served(&main, &cheap),
        "test/cheap",
    );
    let agent = first_agent(first.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    first.world_mut().trigger(SetModel {
        entity: agent,
        model: "test/main".to_owned(),
        kind: ChoiceKind::ForThisRun,
    });
    for ask in ["Migrate auth.", "Now the client.", "Wire them.", "Finish."] {
        turn(&mut first, &wakes, ask);
    }
    assert_eq!(
        first.world().resource::<RecapsAsked>().0,
        0,
        "no recap while working"
    );
    drop(first);
    let saved = saved_messages(&store);
    assert!(saved >= 16, "{saved}");

    // Resumed: the recap shows above the input, never reaching the
    // conversation, the model or the log.
    let (mut resumed, wakes) = app(
        &store,
        StartReason::Resumed,
        served(&main, &cheap),
        "test/cheap",
    );
    run_until(&mut resumed, &wakes, |world| card(world).is_some());
    assert_eq!(card(resumed.world()).as_deref(), Some(RECAP));
    resumed.update();
    let shown = screen(&resumed);
    assert!(
        shown.contains(TITLE) && shown.contains("Pending: run tests"),
        "{shown}"
    );
    assert_eq!(saved_messages(&store), saved);
    let state = Session::read(&store).map(|session| format!("{:?}", session.saved));
    assert!(state.is_ok_and(|state| state.contains("Pending: run tests")));
    let agent = first_agent(resumed.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let live = resumed.world().get::<Conversation>(agent);
    let live = live.map(|conversation| format!("{conversation:?}"));
    assert!(live.is_some_and(|live| !live.contains("Pending: run tests")));
    press(&mut resumed, "esc");
    assert!(card(resumed.world()).is_none(), "Esc dismisses it");
    turn(&mut resumed, &wakes, "continue");
    let requests = main.requests();
    let sent = requests.last().map(|request| format!("{request:?}"));
    let sent = sent.unwrap_or_default();
    assert!(
        sent.contains("continue") && !sent.contains("Pending: run tests"),
        "{sent}"
    );
    assert_eq!(cheap.request_count(), 1);
    drop(resumed);

    // After `/reload`: no recap.
    let (mut reloaded, wakes) = app(
        &store,
        StartReason::Reloaded,
        served(&main, &cheap),
        "test/cheap",
    );
    reloaded.update();
    drop(wakes);
    assert_eq!(reloaded.world().resource::<RecapsAsked>().0, 0);
    drop(reloaded);

    // Resumed again on a model that never answers: the user's message
    // starts its turn at once, and no recap comes.
    let (mut again, wakes) = app(
        &store,
        StartReason::Resumed,
        served(&main, &cheap),
        "stall/never",
    );
    assert_eq!(
        again.world().resource::<RecapsAsked>().0,
        1,
        "a recap was asked for"
    );
    turn(&mut again, &wakes, "keep going");
    assert_eq!(
        again.world().resource::<Ended>().0.len(),
        1,
        "the turn ran without waiting"
    );
    for _ in 0..5 {
        again.update();
    }
    assert!(card(again.world()).is_none());
    let mut pending = again
        .world_mut()
        .query_filtered::<(), (With<RecapCall>, Without<Settled>)>();
    assert_eq!(pending.iter(again.world()).count(), 0, "discarded");

    // Live, idle three hours: the next keystroke asks for one recap.
    pass_wall_clock(again.world_mut(), Duration::from_secs(3 * 60 * 60));
    again.update();
    press(&mut again, "a");
    press(&mut again, "b");
    assert_eq!(again.world().resource::<RecapsAsked>().0, 2);
    assert_eq!(pending.iter(again.world()).count(), 1);

    // It never answers: given up after the timeout, with no card.
    pass_wall_clock(again.world_mut(), Duration::from_secs(16));
    again.update();
    again.update();
    assert_eq!(pending.iter(again.world()).count(), 0, "timed out");
    assert!(card(again.world()).is_none());
}
