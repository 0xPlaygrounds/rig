use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;
use std::sync::{Mutex, PoisonError};

use rig_core::ProviderResponseError;
use rig_core::completion::{FinishReason, Message, Usage};
use rig_core::conversation::Entry;
use rig_core::message::{AssistantContent, AssistantMessage, CallId, ToolName};
use rig_core::test_utils::{MockCompletionModel, MockError, MockStreamEvent};
use rig_harness::services::ProjectRoot;
use rig_harness::workspace::ProjectWorkspace;
use rig_memory::CompactionSettings;
use rig_tools::workspace::{Contents, Stat, Workspace};

use super::*;
use rig_harness_test_support::{
    Collected, Ended, MemoryStore, advance, app_on, collect, commit_message, connect, log_ended,
    reply, run_until,
};

mod pending;

/// The kernel and the compaction plugin on `store`, keeping `keep` tokens, after `before` set the
/// app up and the first frame restored the session, with its loop's wakes.
fn start_with(
    store: &MemoryStore,
    keep: u64,
    before: impl FnOnce(&mut App),
) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    before(&mut app);
    app.set_error_handler(bevy_ecs::error::warn)
        .add_plugins((AgentPlugin, SessionLogPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    rig_harness::load::<CompactionPlugin>(&mut app, "rig-basics", "path plugins/rig-basics");
    app.finish();
    app.update();
    app.insert_resource(CompactionDefault(CompactionSettings {
        keep_recent_tokens: keep,
        ..CompactionSettings::default()
    }));
    collect::<Notice>(&mut app);
    (app, wakes)
}

/// Where the files of a [`Files`] project are.
const PROJECT: &str = "/project";

/// A project held in memory, which the built-in `read` tool reads.
#[derive(Debug, Default)]
struct Files(Mutex<BTreeMap<PathBuf, Vec<u8>>>);

impl Files {
    /// Puts `text` in the file `name`.
    fn put(&self, name: &str, text: &str) {
        let mut files = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        files.insert(Path::new(PROJECT).join(name), text.as_bytes().to_vec());
    }
}

impl Workspace for Files {
    fn stat(&self, path: &Path) -> io::Result<Stat> {
        let files = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        Ok(match files.get(path) {
            Some(bytes) => Stat::File {
                size: bytes.len() as u64,
                read_only: false,
                modified: None,
            },
            None => Stat::Missing,
        })
    }

    fn read(&self, path: &Path, _: u64) -> io::Result<Contents> {
        let files = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        let bytes = files.get(path).ok_or(io::ErrorKind::NotFound)?.clone();
        let size = bytes.len() as u64;
        Ok(Contents { bytes, size })
    }

    fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()> {
        let mut files = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        files.insert(path.to_path_buf(), bytes.to_vec());
        Ok(())
    }

    crate::tests::workspace_stubs!();
}

/// The app's first agent, answered by `model` with a `window`-token context window, with
/// `messages` in its conversation.
fn agent(
    app: &mut App,
    model: &MockCompletionModel,
    window: u32,
    messages: Vec<Message>,
) -> Option<Entity> {
    let mut connection = connect(model)?;
    let spec = connection.spec.as_ref().clone().with_context_window(window);
    connection.spec = Arc::new(spec.with_max_output_tokens(8_000));
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next()?;
    world.entity_mut(agent).insert(connection);
    for message in messages {
        commit_message(world, agent, message).ok()?;
    }
    Some(agent)
}

/// Sends `agent` `text`, then runs frames until `turns` turns ended.
fn ask(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str, turns: usize) {
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Steer));
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() >= turns
    });
}

/// Compacts `agent` with `focus`, then runs frames until `turns` turns ended.
fn compact(app: &mut App, wakes: &Receiver<()>, agent: Entity, focus: &str, turns: usize) {
    let focus = focus.to_owned();
    app.world_mut().trigger(Compact {
        entity: agent,
        focus,
    });
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() >= turns
    });
}

/// The answer the last turn of `app` ended with.
fn answer(app: &App) -> Option<String> {
    match app.world().resource::<Ended>().0.last() {
        Some(TurnOutcome::Answered(message)) => message.final_answer(),
        _ => None,
    }
}

/// Every entry of `agent`'s conversation.
fn entries(world: &World, agent: Entity) -> Vec<Entry> {
    let Some(conversation) = world.get::<Conversation>(agent) else {
        return Vec::new();
    };
    let at = 0..conversation.messages().len();
    at.filter_map(|at| conversation.entry(at)).collect()
}

/// The text of each notice `app` wrote.
fn notices(app: &App) -> Vec<String> {
    let notices = app.world().resource::<Collected<Notice>>();
    notices.0.iter().map(|notice| notice.text.clone()).collect()
}

/// Each request `model` got, as JSON.
fn sent(model: &MockCompletionModel) -> Vec<String> {
    model
        .requests()
        .iter()
        .map(|request| serde_json::to_string(request).unwrap_or_default())
        .collect()
}

/// A call of the `read` tool for `path` and its `output`.
fn read(id: &str, path: &str, output: &str) -> Vec<Message> {
    let Ok(name) = ToolName::new("read") else {
        return Vec::new();
    };
    let arguments = serde_json::json!({ "path": path });
    let call = AssistantContent::tool_call(id, name.clone(), arguments);
    vec![
        Message::Assistant(AssistantMessage::new(vec![call])),
        Message::tool_result(CallId::from_wire(id), name, output),
    ]
}

/// A history of `count` turns, each a long question and a short answer, so a cut keeping a few
/// hundred tokens lands on a question.
fn turns(from: usize, count: usize) -> Vec<Message> {
    (from..from + count)
        .flat_map(|turn| {
            [
                Message::user(format!("question {turn}: {}", "word ".repeat(200))),
                Message::assistant(format!("answer {turn}")),
            ]
        })
        .collect()
}

/// A second compaction updates the first summary: its request carries that summary, exact code
/// and decision included, in `<previous-summary>`, under the update prompt, and the summary that
/// replaces it is the model's answer to that.
#[test]
fn a_second_compaction_updates_the_first_summary() {
    let first = "## Goal\nShip the parser.\n\n## Key Decisions\n- **Use sqlite**: one file.\n\n\
                 ## Critical Context\n- Release code: ZX-4471";
    let second = "## Goal\nShip the parser.\n\n## Key Decisions\n- **Use sqlite**: one file.\n\n\
                  ## Critical Context\n- Release code: ZX-4471\n- Tests pass.";
    let model = MockCompletionModel::from_stream_turns([reply(first), reply(second)]);
    let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |_| {});
    let mut messages = vec![Message::user("Read src/lib.rs")];
    messages.extend(read("1", "src/lib.rs", "pub fn parse() {}"));
    messages.extend(turns(0, 4));
    let agent = agent(&mut app, &model, 200_000, messages);
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };
    compact(&mut app, &wakes, agent, "", 1);
    for message in turns(4, 4) {
        let _ = commit_message(app.world_mut(), agent, message);
    }
    compact(&mut app, &wakes, agent, "keep the release code", 2);

    let requests = sent(&model);
    assert_eq!(requests.len(), 2, "{:?}", notices(&app));
    let (Some(initial), Some(update)) = (requests.first(), requests.get(1)) else {
        return;
    };
    assert!(
        initial.contains("context summarization assistant"),
        "{initial}"
    );
    assert!(
        initial.contains("Create a structured context checkpoint"),
        "{initial}"
    );
    let payload = update
        .split_once("<previous-summary>")
        .and_then(|(_, rest)| rest.split_once("</previous-summary>"))
        .map(|(payload, _)| payload);
    assert!(
        payload.is_some_and(|payload| payload.contains("ZX-4471")
            && payload.contains("Use sqlite")
            && payload.contains("<read-files>\\nsrc/lib.rs")),
        "{update}"
    );
    assert!(update.contains("NEW conversation messages"), "{update}");
    assert!(
        update.contains("Additional focus: keep the release code"),
        "{update}"
    );
    assert!(!update.contains("question 0"), "summarized once: {update}");
    let live = entries(app.world(), agent);
    let summary = live
        .first()
        .map(|entry| entry.message.rag_text().unwrap_or_default());
    assert!(
        summary.is_some_and(|text| text.contains("Tests pass.")
            && text.contains("ZX-4471")
            && text.contains("Use sqlite")),
        "{live:?}"
    );
    let conversation = app.world().get::<Conversation>(agent);
    assert_eq!(conversation.map(|it| summarized_prefix(it)), Some(1));
}

/// A tool-call chain that nears the window compacts between its calls, summarizing the turn's
/// earlier part apart, and the turn finishes.
#[test]
fn a_tool_chain_near_the_window_compacts_mid_chain_and_finishes() {
    let files = Arc::new(Files::default());
    let line = |n: usize| format!("// line {n:04}: {}\n", "x".repeat(40));
    let text: String = (0..300).map(line).collect();
    for name in ["0.rs", "1.rs", "2.rs"] {
        files.put(name, &text);
    }
    let tokens = (text.len() / 4) as u64;
    // Room for two outputs, not three.
    let window = u32::try_from(tokens * 3 + 16_384 - tokens / 2).unwrap_or(u32::MAX);
    let call = |n: usize| {
        let arguments = serde_json::json!({ "path": format!("{n}.rs") });
        vec![
            MockStreamEvent::tool_call(format!("call-{n}"), "read", arguments),
            MockStreamEvent::final_response_with_default_usage(),
        ]
    };
    let model = MockCompletionModel::from_stream_turns([
        call(0),
        call(1),
        call(2),
        reply("## Original Request\nRead the three files."),
        reply("done"),
    ]);
    let (mut app, wakes) = start_with(&MemoryStore::default(), tokens + tokens / 2, |app| {
        app.insert_resource(ProjectWorkspace(files.clone()))
            .insert_resource(ProjectRoot(PROJECT.into()))
            .add_plugins(crate::coding_tools::ReadTool);
    });
    let agent = agent(&mut app, &model, window, Vec::new());
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };
    ask(&mut app, &wakes, agent, "Read the three files.", 1);

    assert_eq!(answer(&app).as_deref(), Some("done"), "{:?}", notices(&app));
    let requests = sent(&model);
    assert_eq!(requests.len(), 5, "{:?}", notices(&app));
    let (Some(prefix), Some(resent)) = (requests.get(3), requests.get(4)) else {
        return;
    };
    assert!(prefix.contains("# Instructions"), "{prefix}");
    assert!(prefix.contains("Read the three files."), "{prefix}");
    assert!(resent.contains("No prior history."), "{resent}");
    assert!(resent.contains("Turn Context (split turn)"), "{resent}");
    assert!(resent.contains("call-2"), "the newest call stays: {resent}");
    assert!(!resent.contains("call-0"), "{resent}");
    let live = entries(app.world(), agent);
    assert!(
        live.first()
            .is_some_and(|entry| entry.author == Some(summary_author())),
        "{live:?}"
    );
}

/// A refusal of the request as longer than the model's window.
fn too_long() -> Vec<MockStreamEvent> {
    let refusal = ProviderResponseError::without_status("prompt is too long: 210000 tokens");
    vec![MockStreamEvent::Error(MockError::ProviderResponse(refusal))]
}

/// After the provider refuses a request as too long, the conversation is compacted and the
/// request sent again, and the turn finishes. The question over the budget alone is summarized as
/// the turn's earlier part rather than sent again as it was.
#[test]
fn an_overflow_compacts_and_sends_the_request_again() {
    let model = MockCompletionModel::from_stream_turns([
        too_long(),
        reply("## Goal\nAnswer the questions."),
        reply("## Original Request\nThe release of the parser."),
        reply("done"),
    ]);
    let (mut app, wakes) = start_with(&MemoryStore::default(), 10, |_| {});
    let agent = agent(&mut app, &model, 200_000, turns(0, 4));
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };
    let question = "And the next question, about the release of the parser?";
    ask(&mut app, &wakes, agent, question, 1);

    assert_eq!(answer(&app).as_deref(), Some("done"), "{:?}", notices(&app));
    let requests = sent(&model);
    assert_eq!(requests.len(), 4, "{:?}", notices(&app));
    let (Some(history), Some(prefix), Some(resent)) =
        (requests.get(1), requests.get(2), requests.get(3))
    else {
        return;
    };
    assert!(history.contains("question 3"), "{history}");
    assert!(prefix.contains(question), "{prefix}");
    assert!(
        resent.contains("Answer the questions.") && resent.contains("Turn Context"),
        "{resent}"
    );
    assert!(
        !resent.contains("question 3") && !resent.contains(question),
        "{resent}"
    );
}

/// A request that overflows again after its compaction fails the turn: one recovery, never a
/// loop.
#[test]
fn an_overflow_recovers_once() {
    let model = MockCompletionModel::from_stream_turns([
        too_long(),
        reply("## Goal\nAnswer the questions."),
        too_long(),
        reply("never asked"),
    ]);
    let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |_| {});
    let agent = agent(&mut app, &model, 200_000, turns(0, 4));
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };
    ask(&mut app, &wakes, agent, "And the next question?", 1);
    assert_eq!(sent(&model).len(), 3, "{:?}", notices(&app));
    assert!(
        matches!(
            app.world().resource::<Ended>().0.as_slice(),
            [TurnOutcome::Failed(_)]
        ),
        "{:?}",
        app.world().resource::<Ended>().0
    );
}

/// A reply the window cut short, below the model's output limit, is dropped, the conversation
/// compacted and the request sent again; with compaction disabled, nothing is compacted.
#[test]
fn a_reply_cut_short_by_the_window_compacts_and_is_asked_again() {
    let cut = || {
        vec![
            MockStreamEvent::text("A half"),
            MockStreamEvent::FinalResponse(rig_core::operation::Finish {
                reason: Some(FinishReason::Length),
                ..rig_core::test_utils::mock_final(Usage::new().output_tokens(10))
            }),
        ]
    };
    for enabled in [true, false] {
        let model = MockCompletionModel::from_stream_turns([
            cut(),
            reply("## Goal\nAnswer the questions."),
            reply("done"),
        ]);
        let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |_| {});
        app.world_mut()
            .resource_mut::<CompactionDefault>()
            .0
            .enabled = enabled;
        let agent = agent(&mut app, &model, 200_000, turns(0, 4));
        assert!(agent.is_some(), "the catalog lists the model");
        let Some(agent) = agent else {
            return;
        };
        ask(&mut app, &wakes, agent, "And the next question?", 1);
        let requests = sent(&model);
        match enabled {
            true => {
                assert_eq!(answer(&app).as_deref(), Some("done"), "{:?}", notices(&app));
                assert_eq!(requests.len(), 3, "{:?}", notices(&app));
                let resent = requests.get(2).cloned().unwrap_or_default();
                assert!(!resent.contains("A half"), "{resent}");
            }
            false => assert_eq!(requests.len(), 1, "{:?}", notices(&app)),
        }
    }
}

/// The usage the provider reported for the last reply decides, however small the local
/// estimate: 24k reported in a 40k window compacts before the next request goes; with compaction
/// disabled it does not.
#[test]
fn reported_usage_triggers_a_compaction_before_the_next_request() {
    let measured = vec![
        MockStreamEvent::text("first answer"),
        MockStreamEvent::final_response_with_total_tokens(24_000),
    ];
    for enabled in [true, false] {
        let model = MockCompletionModel::from_stream_turns([
            measured.clone(),
            reply("## Goal\nAnswer the questions."),
            reply("second answer"),
        ]);
        // Keeping 8 tokens keeps the last question and its answer: no turn is split.
        let (mut app, wakes) = start_with(&MemoryStore::default(), 8, |_| {});
        app.world_mut()
            .resource_mut::<CompactionDefault>()
            .0
            .enabled = enabled;
        let agent = agent(&mut app, &model, 40_000, turns(0, 2));
        assert!(agent.is_some(), "the catalog lists the model");
        let Some(agent) = agent else {
            return;
        };
        ask(&mut app, &wakes, agent, "The first question?", 1);
        if enabled {
            run_until(&mut app, &wakes, |world| {
                let compacted = world
                    .get::<Conversation>(agent)
                    .is_some_and(|it| summarized_prefix(it) > 0);
                compacted && world.get::<ActiveTurn>(agent).is_none()
            });
        }
        let second = "The second question?";
        app.world_mut()
            .trigger(Deliver::new(agent, second, DeliveryMode::Steer));
        run_until(&mut app, &wakes, |world| {
            let ended = &world.resource::<Ended>().0;
            let answered = ended
                .iter()
                .filter(|it| matches!(it, TurnOutcome::Answered(_)));
            answered.count() == 2
        });
        let requests = sent(&model);
        match enabled {
            true => {
                assert_eq!(requests.len(), 3, "{:?}", notices(&app));
                let summary = requests.get(1).cloned().unwrap_or_default();
                assert!(
                    summary.contains("context summarization assistant"),
                    "{summary}"
                );
                let next = requests.get(2).cloned().unwrap_or_default();
                assert!(next.contains("Answer the questions."), "{next}");
            }
            false => assert_eq!(requests.len(), 2, "{:?}", notices(&app)),
        }
    }
}

/// A summary reply that failed or was cut at the output limit replaces nothing, and the notice
/// says why; the session reopened holds the messages as they were, and the next request sends
/// them.
#[test]
fn a_failed_or_cut_summary_keeps_the_messages_and_says_why() {
    let cut = vec![
        MockStreamEvent::text("## Goal\nAnswer the qu"),
        MockStreamEvent::FinalResponse(rig_core::operation::Finish {
            reason: Some(FinishReason::Length),
            ..rig_core::test_utils::mock_final(Usage::default())
        }),
    ];
    let refused = ProviderResponseError::without_status("invalid request: bad summary");
    let failed = vec![MockStreamEvent::Error(MockError::ProviderResponse(refused))];
    let truncated =
        "Summarization failed: generation hit the token cap and the summary is incomplete";
    for (failure, why) in [(cut, truncated), (failed, "bad summary")] {
        let store = MemoryStore::default();
        let model = MockCompletionModel::from_stream_turns([failure]);
        let (mut app, wakes) = start_with(&store, 10, |_| {});
        let agent = agent(&mut app, &model, 200_000, turns(0, 4));
        assert!(agent.is_some(), "the catalog lists the model");
        let Some(agent) = agent else {
            return;
        };
        let before = entries(app.world(), agent);
        compact(&mut app, &wakes, agent, "", 1);
        assert_eq!(entries(app.world(), agent), before, "{why}");
        let said = notices(&app);
        assert!(
            said.iter()
                .any(|text| text.contains("Summarization failed") && text.contains(why)),
            "{said:?}"
        );
        drop(app);

        let model = MockCompletionModel::from_stream_turns([reply("Ready.")]);
        let (mut resumed, wakes) = start_with(&store, 10, |_| {});
        let reopened = self::agent(&mut resumed, &model, 200_000, Vec::new());
        let Some(reopened) = reopened else {
            return;
        };
        assert_eq!(
            entries(resumed.world(), reopened),
            before,
            "{why}: reopened"
        );
        ask(&mut resumed, &wakes, reopened, "Go on.", 1);
        let next = sent(&model);
        assert!(
            next.first().is_some_and(|it| it.contains("question 0")),
            "{why}: {next:?}"
        );
    }
}

/// A failure the provider says is passing.
fn unavailable() -> Vec<MockStreamEvent> {
    let busy = ProviderResponseError::without_status("overloaded").with_transient(Some(true));
    vec![MockStreamEvent::Error(MockError::ProviderResponse(busy))]
}

/// Whether `entry` holds a compaction's summary.
fn holds_summary(entry: &Entry) -> bool {
    entry.author == Some(summary_author())
}

/// On the hold this test puts on an answered summary call, for that call.
#[derive(Component)]
struct Reviewing(Entity);

/// On a summary call the test decided.
#[derive(Component)]
struct Reviewed;

/// Holds each answered summary call for review, as a plugin asking the user would.
fn hold_for_review(
    calls: Query<Entity, (With<Summarizing>, ReadyToSettle, Without<Reviewed>)>,
    asked: Query<&Reviewing>,
    mut commands: Commands,
) {
    for call in &calls {
        if asked.iter().all(|Reviewing(it)| *it != call) {
            commands.spawn((Reviewing(call), HoldOf(call)));
        }
    }
}

/// Runs frames until a summary call is held for review, and returns it.
fn held(app: &mut App, wakes: &Receiver<()>) -> Option<Entity> {
    run_until(app, wakes, |world| {
        world.iter_entities().any(|e| e.contains::<Reviewing>())
    });
    let world = app.world_mut();
    world.query::<&Reviewing>().iter(world).next().map(|r| r.0)
}

/// Decides the summary held for review: sets it to `summary` and lets it go.
fn decide(app: &mut App, wakes: &Receiver<()>, summary: Option<&str>) {
    let Some(call) = held(app, wakes) else {
        return;
    };
    let world = app.world_mut();
    if let Some(mut summarizing) = world.get_mut::<Summarizing>(call) {
        summarizing.summary = summary.map(str::to_owned);
    }
    world.entity_mut(call).insert(Reviewed);
    let review: Vec<Entity> = world
        .query_filtered::<Entity, With<Reviewing>>()
        .iter(world)
        .collect();
    for entity in review {
        world.despawn(entity);
    }
}

/// Whether the one turn of `app` was stopped by the user.
fn stopped_by_user(app: &App) -> bool {
    matches!(
        app.world().resource::<Ended>().0.as_slice(),
        [TurnOutcome::Stopped(stop)] if *stop.cause() == StopCause::User
    )
}

/// Esc while a summary near the window is written cancels it: nothing is summarized, nothing is
/// left running, and the turn does not come back.
#[test]
fn stopping_the_turn_during_a_summary_cancels_the_compaction() {
    let model = MockCompletionModel::from_stream_turns([
        reply("## Goal\nAnswer the questions."),
        reply("never asked"),
    ]);
    let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |app| {
        app.add_systems(AgentPass, hold_for_review.in_set(ReviewSystems::Act));
    });
    // About 1k tokens of conversation in a window that leaves less than the reserve.
    let agent = agent(&mut app, &model, 17_000, turns(0, 4));
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };
    let before = entries(app.world(), agent);
    app.world_mut().trigger(Deliver::new(
        agent,
        "And the next question?",
        DeliveryMode::Steer,
    ));
    assert!(held(&mut app, &wakes).is_some(), "{:?}", notices(&app));
    app.world_mut().trigger(StopTurn::new(agent));
    for _ in 0..5 {
        app.update();
    }
    assert!(
        stopped_by_user(&app),
        "{:?}",
        app.world().resource::<Ended>().0
    );
    assert_eq!(sent(&model).len(), 1, "the summary alone");
    let after = entries(app.world(), agent);
    assert!(!after.iter().any(holds_summary), "{after:?}");
    assert_eq!(after.get(..before.len()), Some(before.as_slice()));
    let world = app.world_mut();
    assert_eq!(world.query::<&Summarizing>().iter(world).count(), 0);
    assert_eq!(world.query::<&TurnCall>().iter(world).count(), 0);
}

/// A conversation whose `/compact` cut splits its last turn: older turns, then a question and a
/// long answer over the budget alone.
fn split_turn() -> Vec<Message> {
    let mut messages = turns(0, 3);
    messages.push(Message::user("The last question?"));
    messages.push(Message::assistant("x".repeat(2_000)));
    messages
}

/// Esc between the history summary and the summary of the split turn's earlier part cancels the
/// compaction: nothing is summarized.
#[test]
fn stopping_between_the_two_summaries_of_a_split_turn_cancels_the_compaction() {
    let model = MockCompletionModel::from_stream_turns([
        reply("## Goal\nAnswer the questions."),
        reply("## Original Request\nThe last question."),
    ]);
    let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |app| {
        let hold_prefix = |calls: Query<(Entity, &Summarizing), Without<Reviewed>>,
                           asked: Query<&Reviewing>,
                           mut commands: Commands| {
            for (call, summarizing) in &calls {
                let free = asked.iter().all(|Reviewing(it)| *it != call);
                if summarizing.history.is_some() && free {
                    commands.spawn((Reviewing(call), HoldOf(call)));
                }
            }
        };
        app.add_systems(AgentPass, hold_prefix.in_set(Stage::Prepare));
    });
    let agent = agent(&mut app, &model, 1_000_000, split_turn());
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };
    let before = entries(app.world(), agent);
    app.world_mut().trigger(Compact {
        entity: agent,
        focus: String::new(),
    });
    assert!(held(&mut app, &wakes).is_some(), "{:?}", notices(&app));
    app.world_mut().trigger(StopTurn::new(agent));
    for _ in 0..5 {
        app.update();
    }
    assert!(
        stopped_by_user(&app),
        "{:?}",
        app.world().resource::<Ended>().0
    );
    assert_eq!(entries(app.world(), agent), before);
    assert!(sent(&model).len() <= 2, "{:?}", sent(&model));
}

/// Each summary call of a split turn gets its own retries: with one retry allowed, the history
/// summary and the turn-prefix summary each fail once and are sent again, and the summary goes in.
#[test]
fn each_summary_call_is_retried_on_its_own() {
    let model = MockCompletionModel::from_stream_turns([
        unavailable(),
        reply("## Goal\nAnswer the questions."),
        unavailable(),
        reply("## Original Request\nThe last question."),
    ]);
    let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |app| {
        app.insert_resource(RetryPolicy {
            max_retries: 1,
            ..RetryPolicy::DEFAULT
        });
    });
    let agent = agent(&mut app, &model, 1_000_000, split_turn());
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };
    app.world_mut().trigger(Compact {
        entity: agent,
        focus: String::new(),
    });
    for _ in 0..2 {
        run_until(&mut app, &wakes, |world| {
            world
                .iter_entities()
                .any(|entity| entity.contains::<Backoff>())
        });
        advance(&mut app, std::time::Duration::from_secs(3));
        run_until(&mut app, &wakes, |world| {
            !world
                .iter_entities()
                .any(|entity| entity.contains::<Backoff>())
        });
    }
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    assert_eq!(sent(&model).len(), 4, "{:?}", notices(&app));
    let live = entries(app.world(), agent);
    let summary = live.first().filter(|entry| holds_summary(entry));
    let text = summary.map(|entry| entry.message.rag_text().unwrap_or_default());
    assert!(
        text.is_some_and(|text| text.contains("Answer the questions.")
            && text.contains("Turn Context (split turn)")
            && text.contains("The last question.")),
        "{live:?}"
    );
}

/// Esc during the wait before a summary call is sent again cancels the compaction: nothing more
/// is sent and nothing is summarized.
#[test]
fn a_stop_during_the_retry_wait_cancels_the_compaction() {
    let model =
        MockCompletionModel::from_stream_turns([unavailable(), reply("## Goal\nNever asked.")]);
    let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |_| {});
    let agent = agent(&mut app, &model, 1_000_000, turns(0, 4));
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };
    let before = entries(app.world(), agent);
    app.world_mut().trigger(Compact {
        entity: agent,
        focus: String::new(),
    });
    run_until(&mut app, &wakes, |world| {
        world
            .iter_entities()
            .any(|entity| entity.contains::<Backoff>())
    });
    app.world_mut().trigger(StopTurn::new(agent));
    app.world_mut().flush();
    app.update();
    advance(&mut app, std::time::Duration::from_secs(3));
    for _ in 0..3 {
        app.update();
    }
    assert_eq!(sent(&model).len(), 1, "one summary call");
    assert!(
        stopped_by_user(&app),
        "{:?}",
        app.world().resource::<Ended>().0
    );
    assert_eq!(entries(app.world(), agent), before);
    let said = notices(&app);
    assert!(
        !said.iter().any(|it| it.contains("not compacted")),
        "{said:?}"
    );
}

/// The summaries sent again, and whether each was the plugin's.
#[derive(Resource, Default)]
struct Resent(Vec<bool>);

/// Sends the first summary again once it is answered, before it settles.
fn resend_first_summary(
    answered: Query<(Entity, Has<ProvidedBy>), (With<Summarizing>, ReadyToSettle)>,
    mut resent: ResMut<Resent>,
    mut commands: Commands,
) {
    for (summary, provided) in &answered {
        if resent.0.is_empty() {
            resent.0.push(provided);
            commands.queue(rig_harness::successor::resend(summary));
        }
    }
}

/// A summary call sent again before it settles is taken from its successor, split-turn fields
/// and all.
#[test]
fn a_summary_sent_again_mid_flight_is_taken_from_its_successor() {
    let model = MockCompletionModel::from_stream_turns([
        reply("## Goal\nThe stale summary."),
        reply("## Goal\nThe fresh summary."),
        reply("## Original Request\nThe last question."),
    ]);
    let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |app| {
        app.init_resource::<Resent>()
            .add_systems(AgentPass, resend_first_summary.in_set(Stage::Review));
    });
    let agent = agent(&mut app, &model, 1_000_000, split_turn());
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };
    compact(&mut app, &wakes, agent, "", 1);
    assert_eq!(app.world().resource::<Resent>().0, [true], "the plugin's");
    let live = entries(app.world(), agent);
    let text = live
        .first()
        .map(|entry| entry.message.rag_text().unwrap_or_default());
    assert!(
        text.is_some_and(|text| text.contains("fresh")
            && !text.contains("stale")
            && text.contains("The last question.")),
        "{live:?}"
    );
}

/// A reviewer cancels one summary and edits the next, which goes in as edited; undoing it puts
/// the messages back with their ids.
#[test]
fn a_held_summary_is_applied_as_reviewed_or_cancelled_and_undone_with_its_ids() {
    let model = MockCompletionModel::from_stream_turns([
        reply("## Goal\nThe first draft."),
        reply("## Goal\nThe second draft."),
    ]);
    let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |app| {
        app.add_systems(AgentPass, hold_for_review.in_set(ReviewSystems::Act));
    });
    let agent = agent(&mut app, &model, 1_000_000, turns(0, 4));
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };
    app.update();
    let original = entries(app.world(), agent);
    let ended = |count: usize| move |world: &World| world.resource::<Ended>().0.len() == count;
    let start_compaction = |app: &mut App| {
        app.world_mut().trigger(Compact {
            entity: agent,
            focus: String::new(),
        });
    };

    start_compaction(&mut app);
    decide(&mut app, &wakes, None);
    run_until(&mut app, &wakes, ended(1));
    assert_eq!(entries(app.world(), agent), original);

    start_compaction(&mut app);
    decide(&mut app, &wakes, Some("## Goal\nThe reviewed summary."));
    run_until(&mut app, &wakes, ended(2));
    let live = entries(app.world(), agent);
    let text = live
        .first()
        .map(|entry| entry.message.rag_text().unwrap_or_default());
    assert!(
        text.is_some_and(|it| it.contains("The reviewed summary.")),
        "{live:?}"
    );

    app.update();
    app.world_mut().trigger(Uncompact { entity: agent });
    app.update();
    assert_eq!(entries(app.world(), agent), original);
}
