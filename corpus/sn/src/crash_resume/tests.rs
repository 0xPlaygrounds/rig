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
use rig_harness::rig_core::conversation::Note;
use rig_harness::rig_ecs::effects::Handler;
use rig_harness::rig_ecs::session_log::SessionStore;
use rig_harness_test_support::{
    Collected, answer_open, app_in, collect, connect, first_agent, reply, run_until, scratch_home,
};
use rig_tools::{Read, Shell};
use rig_tui::{Editor, MainEditor};

use super::{CrashResumePlugin, Progress, Recovering, TITLE, read_progress};

/// An app on `session` with the plugin, `shell`, someone who answers
/// questions and an input box, one frame run.
fn app(home: &Home, session: &SessionId) -> (App, std::sync::mpsc::Receiver<()>) {
    let (mut app, wakes) = app_in(home, session);
    app.add_plugins((AgentPlugin, SessionLogPlugin, AskPlugin, CrashResumePlugin))
        .add_tool(Shell::default())
        .add_tool(Read::default());
    app.world_mut().spawn((Editor::default(), MainEditor));
    app.world_mut().spawn(Answerer::default());
    collect::<Notice>(&mut app);
    app.finish();
    app.cleanup();
    app.update();
    (app, wakes)
}

/// The saved progress of `agent`.
fn progress(world: &World, agent: Entity) -> Option<Progress> {
    let id = world.get::<AgentId>(agent)?;
    read_progress(&*world.get_resource::<PluginData>()?.session, id)
}

/// Whether `agent` saved `text` as its streamed text.
fn streamed(world: &World, agent: Entity, text: &str) -> bool {
    progress(world, agent).is_some_and(|progress| progress.text == text)
}

/// The agent's model, `model`, its call number `stall` (from 0) stalling
/// after its text.
fn use_model(world: &mut World, model: &MockCompletionModel, stall: Option<usize>) {
    let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
    let Some(mut connection) = connect(model) else {
        return;
    };
    if let Some(stall) = stall {
        let inner = connection.handler.0.clone();
        let served = AtomicUsize::new(0);
        let stalls = StallsAfterText {
            inner,
            served,
            stall,
        };
        connection.handler = Handler(ErasedHandler::new(stalls));
    }
    world.entity_mut(agent).insert(connection);
}

/// A model whose reply number `stall` sends its text, then nothing more.
struct StallsAfterText {
    inner: ErasedHandler,
    served: AtomicUsize,
    stall: usize,
}

impl Serve for StallsAfterText {
    type Family = family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        self.inner.descriptor()
    }

    async fn serve(&self, kind: EffectKind, dispatch: Dispatch) -> Reply {
        let reply = self.inner.serve(kind, dispatch).await;
        match reply {
            Reply::Stream(events) if self.served.fetch_add(1, Ordering::SeqCst) == self.stall => {
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

/// The note the plugin put in the agent's transcript.
fn entry(world: &mut World) -> Option<Note> {
    let agent = first_agent(world)?;
    let conversation = world.get::<Conversation>(agent)?;
    let mut notes = conversation
        .asides()
        .filter_map(|(_, aside)| Note::of(aside));
    notes.find(|note| note.title.starts_with(TITLE))
}

/// Every message of the `n`th model request, as JSON text.
fn request(model: &MockCompletionModel, n: usize) -> String {
    let requests = model.requests();
    let history = requests.get(n).map(|request| &request.chat_history);
    serde_json::to_string(&history).unwrap_or_default()
}

/// A `read` and two `shell` calls: the `read` and the first `shell`
/// finish, the second `shell` runs when the app is killed. On resume the
/// entry shows the two finished calls with their results and one of
/// unknown outcome; Continue sends the real results and an error for the
/// second `shell`.
#[test]
fn a_killed_turn_continues_with_real_and_unknown_results() {
    let home = scratch_home("crash-resume-calls");
    let session = SessionId::generate();
    let file = home.root().join("a.txt");
    let echo = format!("echo a > {}", file.display());
    let notes = home.root().join("notes.txt");
    let read = serde_json::json!({ "path": notes.display().to_string() });
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("c1", "read", read),
            MockStreamEvent::tool_call("c2", "shell", serde_json::json!({ "command": echo })),
            MockStreamEvent::tool_call("c3", "shell", serde_json::json!({ "command": "sleep 30" })),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        reply("Carrying on."),
    ]);

    let (mut first, wakes) = app(&home, &session);
    assert!(std::fs::write(&notes, "read before the crash\n").is_ok());
    use_model(first.world_mut(), &model, None);
    let agent = first_agent(first.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    first
        .world_mut()
        .trigger(Deliver::new(agent, "Do the work.", DeliveryMode::Steer));
    // The first call settled, its result marked for a restart, and the
    // second runs.
    let second_running = |world: &World| {
        let calls = world.try_query_filtered::<(Has<Started>, Has<Settled>), With<ToolRequest>>();
        let mut calls: Vec<(bool, bool)> =
            calls.map_or_else(Vec::new, |mut calls| calls.iter(world).collect());
        calls.sort_unstable();
        Path::new(&file).exists() && calls == [(true, false), (true, true), (true, true)]
    };
    run_until(&mut first, &wakes, second_running);
    assert!(second_running(first.world()));
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
    assert!(
        shown.contains("finished: read") && shown.contains("read before the crash"),
        "{shown}"
    );
    assert_eq!(
        model.requests().len(),
        1,
        "nothing is sent before the user chose"
    );
    use_model(second.world_mut(), &model, None);
    assert!(answer_open(
        second.world_mut(),
        TITLE,
        &Answer::choice("continue")
    ));
    run_until(&mut second, &wakes, |_| model.requests().len() >= 2);

    let sent = request(&model, 1);
    assert!(sent.contains("[exit code 0]"), "{sent}");
    assert!(sent.contains("may or may not have completed"), "{sent}");
    assert!(sent.contains("read before the crash"), "{sent}");
    assert!(!sent.contains(super::NOT_RUN), "{sent}");
    assert!(!sent.contains("Stopped: the session restarted"), "{sent}");
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
    use_model(first.world_mut(), &model, Some(0));
    let agent = first_agent(first.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    first.world_mut().trigger(Deliver::new(
        agent,
        "Explain the plan.",
        DeliveryMode::Steer,
    ));
    let saved = |world: &World| streamed(world, agent, "Step one is done. Now");
    run_until(&mut first, &wakes, saved);
    assert!(saved(first.world()));
    let store = first
        .world()
        .get_resource::<SessionStore>()
        .map(|store| Arc::clone(&store.0));
    let id = first.world().get::<AgentId>(agent).cloned();
    drop(first);
    if let (Some(store), Some(id)) = (store, id) {
        assert!(store.append_log(&id.0, b"{\"half\":").is_ok());
    }

    let (mut second, wakes) = app(&home, &session);
    let notices = &second.world().resource::<Collected<Notice>>().0;
    assert!(
        notices
            .iter()
            .any(|notice| notice.text.contains("ended in a line cut short")),
        "{notices:?}"
    );
    let shown = entry(second.world_mut())
        .map(|entry| entry.text)
        .unwrap_or_default();
    assert!(
        shown.contains("Partial reply (incomplete): Step one is done. Now"),
        "{shown}"
    );
    use_model(second.world_mut(), &model, None);
    assert!(answer_open(
        second.world_mut(),
        TITLE,
        &Answer::choice("discard")
    ));
    let agent = first_agent(second.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    run_until(&mut second, &wakes, |world| {
        world.get::<Recovering>(agent).is_none()
    });

    let world = second.world_mut();
    let left = world
        .get::<Conversation>(agent)
        .map(|conversation| conversation.messages().len());
    assert_eq!(left, Some(0));
    let mut main = world.query_filtered::<&Editor, With<MainEditor>>();
    let typed = main.single(world).map(|input| input.text().to_owned());
    assert_eq!(typed.ok().as_deref(), Some("Explain the plan."));
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

/// Killed while the turn's second model call streams, after the first
/// reply's call finished and its result was committed: Continue keeps that
/// real result and the partial text, marked incomplete.
#[test]
fn a_turn_killed_in_a_later_model_call_keeps_its_results_and_text() {
    let home = scratch_home("crash-resume-later-call");
    let session = SessionId::generate();
    let file = home.root().join("b.txt");
    let echo = format!("echo b > {}", file.display());
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("c1", "shell", serde_json::json!({ "command": echo })),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        reply("Halfway through"),
        reply("Carrying on."),
    ]);

    let (mut first, wakes) = app(&home, &session);
    use_model(first.world_mut(), &model, Some(1));
    let agent = first_agent(first.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    first
        .world_mut()
        .trigger(Deliver::new(agent, "Do the work.", DeliveryMode::Steer));
    let saved = |world: &World| streamed(world, agent, "Halfway through");
    run_until(&mut first, &wakes, saved);
    assert!(saved(first.world()));
    drop(first);

    let (mut second, wakes) = app(&home, &session);
    use_model(second.world_mut(), &model, None);
    assert!(answer_open(
        second.world_mut(),
        TITLE,
        &Answer::choice("continue")
    ));
    run_until(&mut second, &wakes, |_| model.requests().len() >= 3);

    let sent = request(&model, 2);
    assert!(sent.contains("exit code 0"), "{sent}");
    assert!(!sent.contains(super::NOT_RUN), "{sent}");
    assert!(
        sent.contains("Halfway through") && sent.contains("[incomplete"),
        "{sent}"
    );
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(CrashResumePlugin);
}
