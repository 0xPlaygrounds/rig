use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::StreamExt;
use rig_core::effect::{EffectKind, HandlerDescriptor, family};
use rig_core::serve::{Dispatch, ErasedHandler, Reply, Serve};
use rig_core::streaming::{Item, Relayed};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::ask::AskPlugin;
use rig_harness::harness_protocol::{Home, SessionId};
use rig_harness::prelude::*;
use rig_harness::rig_ecs::effects::Handler;
use rig_harness::rig_ecs::journal::SessionStore;
use rig_harness_test_support::{
    Collected, answer_open, app_in, collect, connect, first_agent, reply, run_until, scratch_home,
};
use rig_tools::Shell;
use rig_tui::TuiInput;

use super::{CrashResumePlugin, Interrupted, Progress, TITLE, read_progress};

/// An app on `session` with the plugin, `shell`, someone who answers
/// questions and an input box, one frame run.
fn app(home: &Home, session: &SessionId) -> (App, std::sync::mpsc::Receiver<()>) {
    let (mut app, wakes) = app_in(home, session);
    app.add_plugins((AgentPlugin, JournalPlugin, AskPlugin, CrashResumePlugin))
        .add_tool(Shell::default())
        .init_resource::<TuiInput>();
    app.world_mut().spawn(Answerer::default());
    collect::<Notice>(&mut app);
    app.finish();
    app.cleanup();
    app.update();
    (app, wakes)
}

/// The agent, the session's store and the agent's saved progress.
fn progress(world: &mut World) -> Option<Progress> {
    let agent = first_agent(world)?;
    let id = world.get::<AgentId>(agent)?.clone();
    read_progress(&*world.get_resource::<SessionStore>()?.0, &id)
}

/// The agent's model, `model`, on its first call stalling after its text
/// when `stall` is set.
fn use_model(world: &mut World, model: &MockCompletionModel, stall: bool) {
    let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
    let Some(mut connection) = connect(model) else {
        return;
    };
    if stall {
        let inner = connection.handler.0.clone();
        let served = AtomicUsize::new(0);
        connection.handler = Handler(ErasedHandler::new(StallsAfterText { inner, served }));
    }
    world.entity_mut(agent).insert(connection);
}

/// A model whose first reply sends its text, then nothing more.
struct StallsAfterText {
    inner: ErasedHandler,
    served: AtomicUsize,
}

impl Serve for StallsAfterText {
    type Family = family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        self.inner.descriptor()
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        let reply = self.inner.serve(kind, dispatch).await;
        match reply {
            Reply::Stream(events) if self.served.fetch_add(1, Ordering::SeqCst) == 0 => {
                let opening = |item: &Result<Relayed, _>| {
                    let event = |event: &StreamEvent| {
                        matches!(event, StreamEvent::Start { .. } | StreamEvent::Text { .. })
                    };
                    std::future::ready(match item {
                        Ok(Relayed::Origin(_)) => true,
                        Ok(Relayed::Item(Item::Event(sent))) => event(sent),
                        _ => false,
                    })
                };
                let stalled = events.filter(opening).chain(futures::stream::pending());
                Reply::Stream(Box::pin(stalled))
            }
            other => other,
        }
    }
}

/// The entry the plugin put in the transcript.
fn entry(world: &mut World) -> Option<TranscriptEntry> {
    let mut entries = world.query::<&TranscriptEntry>();
    entries
        .iter(world)
        .find(|entry| entry.title.starts_with(TITLE))
        .cloned()
}

/// Every message of the `n`th model request, as JSON text.
fn request(model: &MockCompletionModel, n: usize) -> String {
    let requests = model.requests();
    let history = requests.get(n).map(|request| &request.chat_history);
    serde_json::to_string(&history).unwrap_or_default()
}

/// Two `shell` calls: the first finishes, the second runs when the app is
/// killed. On resume the entry shows one finished call and one of unknown
/// outcome; Continue sends the real first result and an error for the
/// second.
#[test]
fn a_killed_turn_continues_with_real_and_unknown_results() {
    let home = scratch_home("crash-resume-calls");
    let session = SessionId::generate();
    let file = home.root().join("a.txt");
    let echo = format!("echo a > {}", file.display());
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("c1", "shell", serde_json::json!({ "command": echo })),
            MockStreamEvent::tool_call("c2", "shell", serde_json::json!({ "command": "sleep 30" })),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        reply("Carrying on."),
    ]);

    let (mut first, wakes) = app(&home, &session);
    use_model(first.world_mut(), &model, false);
    let agent = first_agent(first.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    first
        .world_mut()
        .trigger(Deliver::new(agent, "Do the work.", DeliveryMode::Steer));
    let second_running = |world: &World| {
        let progress = world
            .get::<AgentId>(agent)
            .and_then(|id| read_progress(&*world.get_resource::<SessionStore>()?.0, id));
        Path::new(&file).exists()
            && progress.is_some_and(|progress| {
                let started: Vec<_> = progress
                    .calls
                    .iter()
                    .map(|call| (call.started, call.output.is_some()))
                    .collect();
                started == [(true, true), (true, false)]
            })
    };
    run_until(&mut first, &wakes, second_running);
    let saved = progress(first.world_mut()).unwrap_or_default();
    let output = saved
        .calls
        .first()
        .and_then(|call| call.output.clone())
        .unwrap_or_default();
    assert!(output.contains("[exit code 0]"), "{saved:?}");
    // Killed: no exit runs, nothing more is written.
    drop(first);

    let (mut second, wakes) = app(&home, &session);
    let shown = entry(second.world_mut())
        .map(|entry| entry.text)
        .unwrap_or_default();
    assert!(
        shown.contains("finished: shell") && shown.contains("[exit code 0]"),
        "{shown}"
    );
    assert!(
        shown.contains("unknown outcome: may have run: shell"),
        "{shown}"
    );
    assert_eq!(
        model.requests().len(),
        1,
        "nothing is sent before the user chose"
    );
    use_model(second.world_mut(), &model, false);
    assert!(answer_open(
        second.world_mut(),
        TITLE,
        &Answer::Choice("continue".into())
    ));
    run_until(&mut second, &wakes, |_| model.requests().len() >= 2);

    let sent = request(&model, 1);
    let real = serde_json::to_string(&output).unwrap_or_default();
    assert!(sent.contains(real.trim_matches('"')), "{sent}");
    assert!(sent.contains("may or may not have completed"), "{sent}");
    assert!(!sent.contains("interrupted by a restart"), "{sent}");
}

/// A reply killed while it streams: on resume, after a torn record, its
/// text is shown; Discard takes the turn back, puts the message back in
/// the input, and the next turn sends no trace of it.
#[test]
fn a_streamed_turn_is_shown_and_discarded_after_a_torn_record() {
    let home = scratch_home("crash-resume-stream");
    let session = SessionId::generate();
    let model = MockCompletionModel::from_stream_turns([
        reply("Step one is done. Now"),
        reply("Fresh start."),
    ]);

    let (mut first, wakes) = app(&home, &session);
    use_model(first.world_mut(), &model, true);
    let agent = first_agent(first.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    first.world_mut().trigger(Deliver::new(
        agent,
        "Explain the plan.",
        DeliveryMode::Steer,
    ));
    let streamed = |world: &World| {
        world.get::<AgentId>(agent).is_some_and(|id| {
            let store = world.get_resource::<SessionStore>();
            let progress = store.and_then(|store| read_progress(&*store.0, id));
            progress.is_some_and(|progress| progress.text == "Step one is done. Now")
        })
    };
    run_until(&mut first, &wakes, streamed);
    assert!(streamed(first.world()));
    let store = first
        .world()
        .get_resource::<SessionStore>()
        .map(|store| Arc::clone(&store.0));
    let id = first.world().get::<AgentId>(agent).cloned();
    drop(first);
    if let (Some(store), Some(id)) = (store, id) {
        assert!(store.append(&id.0, b"{\"half\":").is_ok());
    }

    let (mut second, wakes) = app(&home, &session);
    let notices = &second.world().resource::<Collected<Notice>>().0;
    assert!(
        notices
            .iter()
            .any(|notice| notice.text.contains("torn line")),
        "{notices:?}"
    );
    let shown = entry(second.world_mut())
        .map(|entry| entry.text)
        .unwrap_or_default();
    assert!(
        shown.contains("Partial reply (incomplete): Step one is done. Now"),
        "{shown}"
    );
    use_model(second.world_mut(), &model, false);
    assert!(answer_open(
        second.world_mut(),
        TITLE,
        &Answer::Choice("discard".into())
    ));
    let agent = first_agent(second.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    run_until(&mut second, &wakes, |world| {
        world.get::<Interrupted>(agent).is_none()
    });

    let world = second.world();
    let left = world
        .get::<Conversation>(agent)
        .map(|conversation| conversation.messages().len());
    assert_eq!(left, Some(0));
    assert_eq!(world.resource::<TuiInput>().text(), "Explain the plan.");
    second
        .world_mut()
        .trigger(Deliver::new(agent, "Start over.", DeliveryMode::Steer));
    run_until(&mut second, &wakes, |_| model.requests().len() >= 2);
    let sent = request(&model, 1);
    assert!(sent.contains("Start over."), "{sent}");
    assert!(
        !sent.contains("Explain the plan") && !sent.contains("Step one"),
        "{sent}"
    );
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(CrashResumePlugin);
}
