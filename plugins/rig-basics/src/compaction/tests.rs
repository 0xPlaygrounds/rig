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

/// A reply saying `text` whose usage the provider reported as `tokens`.
fn measured(text: &str, tokens: u64) -> Vec<MockStreamEvent> {
    vec![
        MockStreamEvent::text(text),
        MockStreamEvent::final_response_with_total_tokens(tokens),
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

/// The usage the provider reported for the last reply decides before the next prompt, however
/// small the local estimate: 24k reported in a 40k window compacts before the next request goes,
/// the pending prompt neither counted nor cut; with compaction disabled it does not.
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
        // Keeping 8 tokens keeps the first question and its answer: no turn is split.
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
        let second = format!("The second question? {}", "word ".repeat(100));
        ask(&mut app, &wakes, agent, &second, 2);
        let requests = sent(&model);
        match enabled {
            true => {
                assert_eq!(requests.len(), 3, "{:?}", notices(&app));
                let summary = requests.get(1).cloned().unwrap_or_default();
                assert!(
                    summary.contains("context summarization assistant"),
                    "{summary}"
                );
                assert!(!summary.contains("The second question"), "{summary}");
                let next = requests.get(2).cloned().unwrap_or_default();
                assert!(
                    next.contains("Answer the questions.") && next.contains("The first question"),
                    "{next}"
                );
            }
            false => assert_eq!(requests.len(), 2, "{:?}", notices(&app)),
        }
    }
}

/// After a reply that reports no usage, the last reply that did plus an estimate of what came after
/// it is the context, as pi counts it: 20k reported, then a 4k exchange measured at nothing,
/// crosses a 40k window's threshold before the next prompt.
#[test]
fn growth_after_a_reply_that_reports_nothing_is_counted() {
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::text("first answer"),
            MockStreamEvent::final_response_with_total_tokens(20_000),
        ],
        reply(&format!("second answer {}", "word ".repeat(3_400))),
        reply("## Goal\nAnswer the questions."),
        // The long second answer alone passes the keep: the cut splits its turn.
        reply("## Original Request\nThe second question."),
        reply("third answer"),
    ]);
    let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |_| {});
    let agent = agent(&mut app, &model, 40_000, turns(0, 2));
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };
    ask(&mut app, &wakes, agent, "The first question?", 1);
    ask(&mut app, &wakes, agent, "The second question?", 2);
    ask(&mut app, &wakes, agent, "The third question?", 3);
    let requests = sent(&model);
    assert_eq!(requests.len(), 5, "{:?}", notices(&app));
    let summary = requests.get(2).cloned().unwrap_or_default();
    assert!(
        summary.contains("context summarization assistant"),
        "{summary}"
    );
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
/// and all, and the call sent again is retried like the first: it fails once and still goes in.
#[test]
fn a_summary_sent_again_mid_flight_is_taken_from_its_successor() {
    let model = MockCompletionModel::from_stream_turns([
        reply("## Goal\nThe stale summary."),
        unavailable(),
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
    app.world_mut().trigger(Compact {
        entity: agent,
        focus: String::new(),
    });
    until_ended_with_retries(&mut app, &wakes, 1);
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

/// A reviewer cancels one summary and edits the next, which goes in as edited; one a reviewer
/// blanks replaces nothing and says why.
#[test]
fn a_held_summary_is_applied_as_reviewed_or_cancelled_and_a_blank_one_fails() {
    let model = MockCompletionModel::from_stream_turns([
        reply("## Goal\nThe first draft."),
        reply("## Goal\nThe blanked draft."),
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
    decide(&mut app, &wakes, Some(" \n"));
    run_until(&mut app, &wakes, ended(2));
    assert_eq!(entries(app.world(), agent), original);
    let said = notices(&app);
    assert!(
        said.iter().any(|it| it.contains("the summary is empty")),
        "{said:?}"
    );

    start_compaction(&mut app);
    decide(&mut app, &wakes, Some("## Goal\nThe reviewed summary."));
    run_until(&mut app, &wakes, ended(3));
    let live = entries(app.world(), agent);
    let text = live
        .first()
        .map(|entry| entry.message.rag_text().unwrap_or_default());
    assert!(
        text.is_some_and(|it| it.contains("The reviewed summary.")),
        "{live:?}"
    );
}

/// A request refused as too long whose recovery does not go in, because its summary fails or
/// there is nothing to compact, is dropped as pi omits it: the turn stops there with a notice of
/// why, the provider's own reason for a failed summary included, and the user's prompt and its
/// image stay in the conversation, not given back.
#[test]
fn a_refused_request_whose_recovery_does_not_go_in_keeps_the_prompt() {
    let refused = ProviderResponseError::without_status("invalid request: bad summary");
    for nothing in [false, true] {
        let failed = vec![MockStreamEvent::Error(MockError::ProviderResponse(
            refused.clone(),
        ))];
        let model = MockCompletionModel::from_stream_turns([too_long(), failed]);
        let keep = if nothing { 1_000_000 } else { 100 };
        let (mut app, wakes) = start_with(&MemoryStore::default(), keep, |app| {
            collect::<rig_ecs::inbox::Recalled>(app);
        });
        let agent = agent(&mut app, &model, 200_000, turns(0, 4));
        assert!(agent.is_some(), "the catalog lists the model");
        let Some(agent) = agent else {
            return;
        };
        app.world_mut()
            .trigger(with_image(agent, "What is in the picture?"));
        run_until(&mut app, &wakes, |world| {
            !world.resource::<Ended>().0.is_empty()
        });
        let ended = &app.world().resource::<Ended>().0;
        assert!(
            matches!(ended.as_slice(), [TurnOutcome::Stopped(_)]),
            "{nothing}: {ended:?}"
        );
        let said = notices(&app);
        let why: &[&str] = match nothing {
            true => &["nothing to compact"],
            false => &["Context overflow recovery failed", "bad summary"],
        };
        assert!(
            said.iter().any(|it| why.iter().all(|why| it.contains(why))),
            "{nothing}: {said:?}"
        );
        let live = serde_json::to_string(&entries(app.world(), agent)).unwrap_or_default();
        assert!(
            live.contains("What is in the picture?") && live.contains("aGk="),
            "{nothing}: the prompt is kept"
        );
        let recalled = &app
            .world()
            .resource::<Collected<rig_ecs::inbox::Recalled>>()
            .0;
        assert!(recalled.is_empty(), "{nothing}: {recalled:?}");
    }
}

/// A threshold summary a reviewer cancels, or one that fails, replaces nothing, and the request it
/// held goes out as it was and the turn answers.
#[test]
fn a_cancelled_or_failed_threshold_summary_sends_the_request_as_it_was() {
    let refused = ProviderResponseError::without_status("invalid request: bad summary");
    let failed = vec![MockStreamEvent::Error(MockError::ProviderResponse(refused))];
    for (summary, cancel) in [(reply("## Goal\nAnswer."), true), (failed, false)] {
        // The answer is measured well under the window, so no compaction follows the run.
        let model = MockCompletionModel::from_stream_turns([summary, measured("done", 100)]);
        let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |app| {
            if cancel {
                app.add_systems(AgentPass, hold_for_review.in_set(ReviewSystems::Act));
            }
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
        if cancel {
            decide(&mut app, &wakes, None);
        }
        run_until(&mut app, &wakes, |world| {
            !world.resource::<Ended>().0.is_empty()
        });
        assert_eq!(answer(&app).as_deref(), Some("done"), "{:?}", notices(&app));
        let requests = sent(&model);
        assert_eq!(requests.len(), 2, "{:?}", notices(&app));
        let request = requests.get(1).cloned().unwrap_or_default();
        assert!(
            request.contains("question 0") && request.contains("next question"),
            "{request}"
        );
        let after = entries(app.world(), agent);
        assert!(!after.iter().any(holds_summary), "{after:?}");
        assert_eq!(after.get(..before.len()), Some(before.as_slice()));
    }
}

/// A request that overflows right after a threshold compaction still gets its one recovery; with
/// compaction disabled, an overflow is not recovered from.
#[test]
fn overflow_recovery_follows_a_threshold_compaction_and_stops_when_disabled() {
    for enabled in [true, false] {
        // The question over the keep alone is summarized as the turn's earlier part after the
        // overflow, as pi does.
        let mut script = vec![
            reply("## Goal\nAnswer the questions."),
            too_long(),
            reply("## Goal\nAnswer them all."),
            reply("## Original Request\nThe next question."),
            measured("done", 100),
        ];
        if !enabled {
            script.remove(0);
        }
        let model = MockCompletionModel::from_stream_turns(script);
        let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |_| {});
        app.world_mut()
            .resource_mut::<CompactionDefault>()
            .0
            .enabled = enabled;
        let agent = agent(&mut app, &model, 17_000, turns(0, 4));
        assert!(agent.is_some(), "the catalog lists the model");
        let Some(agent) = agent else {
            return;
        };
        let question = format!("And the next question? {}", "word ".repeat(100));
        ask(&mut app, &wakes, agent, &question, 1);
        match enabled {
            true => {
                assert_eq!(answer(&app).as_deref(), Some("done"), "{:?}", notices(&app));
                assert_eq!(sent(&model).len(), 5, "{:?}", notices(&app));
            }
            false => {
                assert_eq!(sent(&model).len(), 1, "{:?}", notices(&app));
                let ended = &app.world().resource::<Ended>().0;
                assert!(
                    matches!(ended.as_slice(), [TurnOutcome::Failed(_)]),
                    "{ended:?}"
                );
            }
        }
    }
}

/// A turn's request asks for no more output than the room its input leaves in the window, as pi
/// asks.
#[test]
fn a_turn_asks_for_the_output_room_its_input_leaves() {
    let model = MockCompletionModel::from_stream_turns([reply("done")]);
    let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |_| {});
    app.world_mut()
        .resource_mut::<CompactionDefault>()
        .0
        .enabled = false;
    let agent = agent(&mut app, &model, 12_000, turns(0, 4));
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };
    ask(&mut app, &wakes, agent, "And the next question?", 1);
    let asked = model
        .requests()
        .first()
        .and_then(|request| request.max_tokens);
    // About 1k tokens of input in a 12k window, less pi's 4096-token margin, under the 8k output
    // limit.
    assert!(
        asked.is_some_and(|it| (6_000..8_000).contains(&it)),
        "{asked:?}"
    );
}

/// A conversation replaced live, which its log records as a reset, drops what its compactions
/// replaced, as a restart from that log does: whether it is the same value saved again or another
/// whose entries reuse the summary's id, the summary never shows the messages from before.
#[cfg(feature = "tui")]
#[test]
fn a_reset_drops_what_compactions_replaced_live() {
    for unrelated in [false, true] {
        let model = MockCompletionModel::from_stream_turns([reply("## Goal\nThe summary.")]);
        let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |app| {
            let seen = |changed: Query<(), Changed<rig_tui::Replaced>>,
                        mut redrawn: ResMut<Redrawn>| {
                redrawn.0 |= !changed.is_empty();
            };
            app.init_resource::<Redrawn>().add_systems(PostUpdate, seen);
        });
        let agent = agent(&mut app, &model, 1_000_000, turns(0, 4));
        assert!(agent.is_some(), "the catalog lists the model");
        let Some(agent) = agent else {
            return;
        };
        app.update();
        app.world_mut().trigger(Compact {
            entity: agent,
            focus: String::new(),
        });
        run_until(&mut app, &wakes, |world| {
            world.resource::<Ended>().0.len() == 1
        });
        let archived = |app: &App| {
            app.world()
                .get::<rig_tui::Replaced>(agent)
                .map_or(0, |replaced| replaced.0.len())
        };
        assert_eq!(archived(&app), 1, "{unrelated}: {:?}", notices(&app));

        let world = app.world_mut();
        world.resource_mut::<Redrawn>().0 = false;
        let Some(mut conversation) = world.get_mut::<Conversation>(agent) else {
            return;
        };
        let json = serde_json::to_string(&**conversation).unwrap_or_default();
        let value = serde_json::from_str::<rig_core::conversation::Conversation>(&json);
        assert!(value.is_ok(), "{json}");
        let Ok(mut value) = value else {
            return;
        };
        if unrelated {
            value.edit_each(Author::plugin("test"), |entry| {
                Some(Message::user(format!("other {}", entry.index)))
            });
        }
        **conversation = value;
        app.update();
        app.update();
        assert_eq!(archived(&app), 0, "{unrelated}");
        assert!(app.world().resource::<Redrawn>().0, "{unrelated}");
    }
}

/// Whether what an agent's compactions replaced changed since it was last set to false, which
/// is what lays the terminal view's pane out again.
#[cfg(feature = "tui")]
#[derive(Resource, Default)]
struct Redrawn(bool);

/// A conversation replaced whole by another of the same messages, then compacted by a summary
/// already on its way before anything reads the reset: the summary keeps what it replaced, as
/// a restart from the log shows it, and what a summary from before the reset replaced is gone.
#[cfg(feature = "tui")]
#[test]
fn a_summary_right_after_a_reset_keeps_what_it_replaced_live_as_a_restart_does() {
    fn replace_under_the_summary(
        answered: On<Add<Outcome>>,
        calls: Query<&ForAgent, With<Summarizing>>,
        mut agents: Query<&mut Conversation>,
    ) {
        let agent = calls.get(answered.entity).map(|&ForAgent(agent)| agent);
        if let Some(mut conversation) = agent.ok().and_then(|agent| agents.get_mut(agent).ok()) {
            let messages = conversation.messages().to_vec();
            **conversation = rig_core::conversation::Conversation::from_messages(messages);
        }
    }

    let store = MemoryStore::default();
    let model = MockCompletionModel::from_stream_turns([
        reply("## Goal\nThe first summary."),
        reply("## Goal\nThe second summary."),
    ]);
    let (mut app, wakes) = start_with(&store, 100, |_| {});
    let agent = agent(&mut app, &model, 1_000_000, turns(0, 4));
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };
    app.update();
    compact(&mut app, &wakes, agent, "", 1);
    for message in turns(4, 4) {
        assert!(commit_message(app.world_mut(), agent, message).is_ok());
    }
    app.update();
    app.add_observer(replace_under_the_summary);
    compact(&mut app, &wakes, agent, "", 2);
    app.update();
    app.update();

    let ids = |sets: &BTreeMap<MessageId, Vec<Entry>>| -> Vec<(MessageId, Vec<Option<MessageId>>)> {
        let ids = |entries: &Vec<Entry>| entries.iter().map(|entry| entry.id).collect();
        sets.iter()
            .map(|(id, entries)| (*id, ids(entries)))
            .collect()
    };
    let live = app.world().get::<rig_tui::Replaced>(agent);
    let live = live.map(|replaced| ids(&replaced.0)).unwrap_or_default();
    let id = app.world().get::<AgentId>(agent).map(|id| id.0.clone());
    let replayed = rig_memory::summarized_in_log(&store, &id.unwrap_or_default());
    assert!(replayed.is_ok(), "{replayed:?}");
    let replayed = replayed.map(|sets| ids(&sets)).unwrap_or_default();
    assert_eq!(live.len(), 1, "{live:?} {:?}", notices(&app));
    assert_eq!(live, replayed);
}

/// What each compaction replaced, the messages themselves, stays drawn above its summary, a
/// cancelled one drawing nothing and a second one nesting the first: after a restart, for the
/// agent and a subagent, and in a fork; a fork made before a compaction draws nothing.
#[cfg(feature = "tui")]
#[test]
fn what_compactions_replaced_stays_drawn_live_after_a_restart_and_in_a_fork() {
    use rig_ecs::fork::ForkAgent;
    use rig_ecs::spawn::{AgentSpec, Origin, spawn};

    type Sets = Vec<(MessageId, Vec<(Option<MessageId>, Message)>)>;
    let replaced = |world: &World, agent: Entity| -> Sets {
        let Some(replaced) = world.get::<rig_tui::Replaced>(agent) else {
            return Vec::new();
        };
        let entry = |entry: &Entry| (entry.id, (*entry.message).clone());
        let sets = replaced.0.iter();
        sets.map(|(summary, entries)| (*summary, entries.iter().map(entry).collect()))
            .collect()
    };
    let store = MemoryStore::default();
    let model = MockCompletionModel::from_stream_turns([
        reply("## Goal\nCancelled."),
        reply("## Goal\nThe first summary."),
        reply("## Goal\nThe second summary."),
    ]);
    let (mut app, wakes) = start_with(&store, 100, |app| {
        app.add_systems(AgentPass, hold_for_review.in_set(ReviewSystems::Act));
    });
    let mut messages = vec![Message::user("What does the crate do?")];
    messages.extend(read("1", "src/lib.rs", "pub fn answer() -> u32 { 42 }"));
    messages.extend(turns(0, 4));
    let agent = agent(&mut app, &model, 1_000_000, messages);
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };
    app.update();
    let original = entries(app.world(), agent);
    let compact = |app: &mut App, review: Option<&str>, count: usize| {
        app.world_mut().trigger(Compact {
            entity: agent,
            focus: String::new(),
        });
        decide(app, &wakes, review);
        run_until(app, &wakes, |world| {
            world.resource::<Ended>().0.len() == count
        });
    };
    compact(&mut app, None, 1);
    assert!(replaced(app.world(), agent).is_empty());

    compact(&mut app, Some("## Goal\nThe first summary."), 2);
    let live = entries(app.world(), agent);
    let first = live.first().and_then(|entry| entry.id);
    let sets = replaced(app.world(), agent);
    assert_eq!(sets.len(), 1, "{sets:?}");
    let Some((summary, taken)) = sets.first() else {
        return;
    };
    assert_eq!(Some(*summary), first);
    let gone: Vec<_> = original
        .iter()
        .filter(|entry| !live.contains(entry))
        .map(|entry| (entry.id, (*entry.message).clone()))
        .collect();
    assert_eq!(taken, &gone);
    let asks = |message: &Message| format!("{message:?}").contains("What does the crate do?");
    let calls_read = |message: &Message| format!("{message:?}").contains("src/lib.rs");
    assert!(taken.iter().any(|(_, message)| asks(message)), "{taken:?}");
    assert!(
        taken.iter().any(|(_, message)| calls_read(message)),
        "{taken:?}"
    );
    // The model's view is the summary alone.
    assert!(live.first().is_some_and(holds_summary));
    assert!(!live.iter().any(|entry| asks(&entry.message)));

    for message in [
        Message::user(format!("question 9 {}", "word ".repeat(200))),
        Message::assistant("answer 9"),
    ] {
        assert!(commit_message(app.world_mut(), agent, message).is_ok());
    }
    compact(&mut app, Some("## Goal\nThe second summary."), 3);
    let sets = replaced(app.world(), agent);
    assert_eq!(sets.len(), 2, "{sets:?}");
    let nested = sets.iter().find(|(summary, _)| Some(*summary) != first);
    let nested = nested.is_some_and(|(_, taken)| taken.iter().any(|(id, _)| *id == first));
    assert!(nested, "{sets:?}");

    let world = app.world_mut();
    let child = spawn(world, Origin::ChildOf(agent), AgentSpec::default());
    let child = child.unwrap_or(Entity::PLACEHOLDER);
    for text in ["one", "two", "three"] {
        assert!(commit_message(world, child, Message::user(text)).is_ok());
    }
    let state = SummaryState {
        summary: "The subagent's summary.".to_owned(),
        ..SummaryState::default()
    };
    if let Some(mut conversation) = world.get_mut::<Conversation>(child) {
        let read = conversation
            .messages()
            .get(..2)
            .unwrap_or_default()
            .to_vec();
        assert!(rig_memory::apply_summary(&mut conversation, &read, &state).is_some());
    }
    app.update();
    let ids = |app: &App, entity| app.world().get::<AgentId>(entity).cloned();
    let (Some(main_id), Some(child_id)) = (ids(&app, agent), ids(&app, child)) else {
        return;
    };
    drop(app);

    let (mut restarted, _wakes) = start_with(&store, 100, |_| {});
    let find = |app: &mut App, id: &AgentId| {
        let world = app.world_mut();
        let mut agents = world.query::<(Entity, &AgentId)>();
        id.find_in(agents.iter(world))
    };
    let restored = (
        find(&mut restarted, &main_id),
        find(&mut restarted, &child_id),
    );
    assert!(
        matches!(restored, (Some(_), Some(_))),
        "both agents were restored"
    );
    let (Some(agent), Some(child)) = restored else {
        return;
    };
    assert_eq!(replaced(restarted.world(), agent), sets);
    let child_sets = replaced(restarted.world(), child);
    let child_taken = child_sets.first().map(|(_, taken)| taken.len());
    assert_eq!(child_taken, Some(2), "{child_sets:?}");

    restarted.world_mut().spawn(ForkAgent::new(agent, None));
    restarted.update();
    let world = restarted.world_mut();
    let mut agents = world.query_filtered::<Entity, With<Agent>>();
    let fork = agents.iter(world).find(|it| ![agent, child].contains(it));
    assert!(fork.is_some(), "the fork was made");
    let Some(fork) = fork else { return };
    assert_eq!(replaced(restarted.world(), fork), sets);

    restarted.world_mut().spawn(ForkAgent::new(agent, first));
    restarted.update();
    let world = restarted.world_mut();
    let mut agents = world.query_filtered::<Entity, With<Agent>>();
    let earlier = agents
        .iter(world)
        .find(|it| ![agent, child, fork].contains(it));
    assert!(earlier.is_some(), "the fork was made");
    let Some(earlier) = earlier else { return };
    assert_eq!(replaced(restarted.world(), earlier), Vec::new());
}

/// A reply cut short by the window is dropped before compaction is tried, as pi omits it: with
/// nothing to summarize, or with the recovery summary cancelled by a reviewer, the turn stops
/// there without the partial reply, the user's prompt and its image kept, as pi's run ends.
#[test]
fn a_reply_cut_short_is_dropped_when_recovery_does_not_go_through() {
    let cut = || {
        vec![
            MockStreamEvent::text("A half"),
            MockStreamEvent::FinalResponse(rig_core::operation::Finish {
                reason: Some(FinishReason::Length),
                ..rig_core::test_utils::mock_final(Usage::new().output_tokens(10))
            }),
        ]
    };
    for cancel in [false, true] {
        let model = MockCompletionModel::from_stream_turns([cut(), reply("## Goal\nAnswer.")]);
        let keep = if cancel { 100 } else { 1_000_000 };
        let (mut app, wakes) = start_with(&MemoryStore::default(), keep, |app| {
            if cancel {
                app.add_systems(AgentPass, hold_for_review.in_set(ReviewSystems::Act));
            }
        });
        let agent = agent(&mut app, &model, 200_000, turns(0, 4));
        assert!(agent.is_some(), "the catalog lists the model");
        let Some(agent) = agent else {
            return;
        };
        app.world_mut()
            .trigger(with_image(agent, "What is in the picture?"));
        if cancel {
            decide(&mut app, &wakes, None);
        }
        run_until(&mut app, &wakes, |world| {
            !world.resource::<Ended>().0.is_empty()
        });
        let ended = &app.world().resource::<Ended>().0;
        assert!(
            matches!(ended.as_slice(), [TurnOutcome::Stopped(_)]),
            "{cancel}: {ended:?}"
        );
        let live = serde_json::to_string(&entries(app.world(), agent)).unwrap_or_default();
        assert!(!live.contains("A half"), "{cancel}: {live}");
        assert!(
            live.contains("What is in the picture?") && live.contains("aGk="),
            "{cancel}: {live}"
        );
    }
}

/// A prompt that takes the request over the threshold compacts before its first request goes,
/// though the history before it is short, also when that ends in a tool result kept from a turn
/// that stopped: the prompt is counted, and never summarized. A 30k prompt in a 40k window.
#[test]
fn a_big_prompt_over_the_threshold_compacts_before_the_first_send() {
    let model = MockCompletionModel::from_stream_turns([
        reply("## Goal\nAnswer the questions."),
        measured("done", 100),
    ]);
    let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |_| {});
    let mut messages = turns(0, 2);
    messages.extend(read("1", "src/lib.rs", "pub fn answer() -> u32 { 42 }"));
    let agent = agent(&mut app, &model, 40_000, messages);
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };
    let prompt = format!("Read this: {}", "word ".repeat(24_000));
    ask(&mut app, &wakes, agent, &prompt, 1);
    assert_eq!(answer(&app).as_deref(), Some("done"), "{:?}", notices(&app));
    let requests = sent(&model);
    assert_eq!(requests.len(), 2, "{:?}", notices(&app));
    let summary = requests.first().cloned().unwrap_or_default();
    assert!(
        summary.contains("context summarization assistant") && !summary.contains("Read this:"),
        "{summary}"
    );
    let asked = requests.last().cloned().unwrap_or_default();
    assert!(
        asked.contains("Read this:") && asked.contains("Answer the questions."),
        "{asked}"
    );
}

/// The before-prompt check alone, as on a resumed session whose last reply was measured near the
/// window: the summary goes before the prompt's request.
#[test]
fn a_measured_conversation_compacts_before_the_next_prompt() {
    let model = MockCompletionModel::from_stream_turns([
        reply("## Goal\nAnswer the questions."),
        reply("done"),
    ]);
    let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |_| {});
    let agent = agent(&mut app, &model, 40_000, turns(0, 4));
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };
    let through = app
        .world()
        .get::<Conversation>(agent)
        .and_then(|conversation| conversation.entries().last().map(|entry| entry.id));
    let measured = through.map(|through| rig_memory::Anchor {
        usage: Usage::new().total_tokens(30_000),
        through,
    });
    app.world_mut()
        .entity_mut(agent)
        .insert(LastUsage(measured));
    ask(&mut app, &wakes, agent, "And the next question?", 1);
    assert_eq!(answer(&app).as_deref(), Some("done"), "{:?}", notices(&app));
    let requests = sent(&model);
    assert_eq!(requests.len(), 2, "{:?}", notices(&app));
    let summary = requests.first().cloned().unwrap_or_default();
    assert!(
        summary.contains("context summarization assistant"),
        "{summary}"
    );
}

/// The threshold counts the request as it goes, the same count the output limit is fitted by:
/// with no usage reported, a long system prompt takes a short conversation over, and its older
/// messages are summarized before the request is sent.
#[test]
fn the_threshold_counts_the_request_as_it_is_sent() {
    let model = MockCompletionModel::from_stream_turns([
        reply("## Goal\nAnswer the questions."),
        reply("done"),
    ]);
    let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |app| {
        let long = |mut calls: Query<&mut ModelRequest, (Added<Admitted>, With<TurnCall>)>| {
            for mut call in &mut calls {
                if let Some(prompt) = &mut call.prompt {
                    prompt.sections.push(rig_core::completion::Section::new(
                        "long",
                        "word ".repeat(20_000),
                    ));
                }
            }
        };
        app.add_systems(AgentPass, long.in_set(PrepareSystems::Edit));
    });
    let agent = agent(&mut app, &model, 40_000, turns(0, 2));
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };
    ask(&mut app, &wakes, agent, "And the next question?", 1);
    assert_eq!(answer(&app).as_deref(), Some("done"), "{:?}", notices(&app));
    let requests = sent(&model);
    assert_eq!(requests.len(), 2, "{:?}", notices(&app));
    let summary = requests.first().cloned().unwrap_or_default();
    assert!(
        summary.contains("context summarization assistant"),
        "{summary}"
    );
}

/// A request a plugin brought whole, with a history of its own, is measured as it goes: a
/// conversation last measured near the window neither compacts for it nor takes room from its
/// answer, and a long history of its own takes that room but compacts nothing, since a summary of
/// the conversation would not shorten it.
#[test]
fn a_request_brought_whole_is_measured_by_what_it_sends() {
    for (words, clamped) in [(2, false), (21_000, true)] {
        let model = MockCompletionModel::from_stream_turns([reply("done")]);
        let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |app| {
            let history = "word ".repeat(words);
            let whole =
                move |mut calls: Query<&mut ModelRequest, (Added<Admitted>, With<TurnCall>)>| {
                    for mut call in &mut calls {
                        call.history = None;
                        call.request_mut().chat_history = vec![Arc::new(Message::user(&history))];
                    }
                };
            app.add_systems(AgentPass, whole.in_set(PrepareSystems::Edit));
        });
        let agent = agent(&mut app, &model, 40_000, turns(0, 4));
        assert!(agent.is_some(), "the catalog lists the model");
        let Some(agent) = agent else {
            return;
        };
        let through = app
            .world()
            .get::<Conversation>(agent)
            .and_then(|conversation| conversation.entries().last().map(|entry| entry.id));
        let measured = through.map(|through| rig_memory::Anchor {
            usage: Usage::new().total_tokens(30_000),
            through,
        });
        app.world_mut()
            .entity_mut(agent)
            .insert(LastUsage(measured));
        let before = entries(app.world(), agent);
        ask(&mut app, &wakes, agent, "And the next question?", 1);
        let requests = model.requests();
        assert_eq!(requests.len(), 1, "{words}: {:?}", notices(&app));
        let asked = requests.first().and_then(|request| request.max_tokens);
        match clamped {
            true => assert!(
                asked.is_some_and(|asked| asked < 8_000),
                "{words}: {asked:?}"
            ),
            false => assert_eq!(asked, Some(8_000), "{words}"),
        }
        let after = entries(app.world(), agent);
        assert_eq!(
            after.get(..before.len()),
            Some(before.as_slice()),
            "{words}"
        );
    }
}

/// Runs `app` until its turns ended `count` times, moving the clock on for each retry wait.
fn until_ended_with_retries(app: &mut App, wakes: &Receiver<()>, count: usize) {
    let mut ended = false;
    for _ in 0..400 {
        ended = app.world().resource::<Ended>().0.len() >= count
            && app
                .world_mut()
                .query_filtered::<(), With<ActiveTurn>>()
                .iter(app.world())
                .next()
                .is_none();
        if ended {
            break;
        }
        advance(app, std::time::Duration::from_secs(3));
        app.update();
        wakes
            .recv_timeout(std::time::Duration::from_millis(20))
            .ok();
    }
    assert!(
        ended,
        "the turns never ended: {:?}",
        app.world().resource::<Ended>().0
    );
}

/// Before a prompt, a tool result kept from an earlier turn counts and may be cut; only the prompt
/// itself is neither: a large kept result and a short prompt in a 40k window compact first.
#[test]
fn a_kept_tool_result_before_the_prompt_counts() {
    let model =
        MockCompletionModel::from_stream_turns([reply("## Goal\nRead the file."), reply("done")]);
    let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |_| {});
    let mut messages = turns(0, 2);
    messages.extend(read("1", "src/big.rs", &"x".repeat(100_000)));
    let agent = agent(&mut app, &model, 40_000, messages);
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };
    ask(&mut app, &wakes, agent, "And now?", 1);
    let requests = sent(&model);
    let summary = requests.first().cloned().unwrap_or_default();
    assert!(
        summary.contains("context summarization assistant"),
        "{requests:?}"
    );
    assert!(!summary.contains("And now?"), "{summary}");
    // As in pi, the context is unknown after the compaction until a reply reports it.
    assert!(app.world().get::<ContextStale>(agent).is_some());
}

/// The user's `text` for `agent`, with a small PNG attached.
fn with_image(agent: Entity, text: &str) -> Deliver {
    let mut prompt = Deliver::new(agent, text, DeliveryMode::Steer);
    let image = rig_core::message::UserContent::image_base64(
        "aGk=",
        Some(rig_core::message::ImageMediaType::PNG),
        None,
    );
    prompt
        .message
        .parts
        .push(rig_ecs::inbox::MessagePart::Attachment(
            rig_ecs::inbox::Attachment {
                label: "picture.png".to_owned(),
                content: image,
            },
        ));
    prompt
}

/// Esc while the summary recovering from a reply cut short is written stops the turn without the
/// partial reply, with or without a tool call in it: it was left out before the summary started,
/// as pi omits it, its call never ran, and it is kept as an aside.
#[test]
fn stopping_a_cut_short_recovery_saves_none_of_the_reply() {
    for with_call in [false, true] {
        let mut cut = vec![MockStreamEvent::text("A half")];
        if with_call {
            let arguments = serde_json::json!({ "path": "a.rs" });
            cut.push(MockStreamEvent::tool_call("call-cut", "read", arguments));
        }
        cut.push(MockStreamEvent::FinalResponse(
            rig_core::operation::Finish {
                reason: Some(FinishReason::Length),
                ..rig_core::test_utils::mock_final(Usage::new().output_tokens(10))
            },
        ));
        let model = MockCompletionModel::from_stream_turns([
            cut,
            reply("## Goal\nAnswer."),
            measured("done", 100),
        ]);
        let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |app| {
            app.add_systems(AgentPass, hold_for_review.in_set(ReviewSystems::Act));
        });
        let agent = agent(&mut app, &model, 200_000, turns(0, 4));
        let Some(agent) = agent else {
            return;
        };
        app.world_mut()
            .trigger(Deliver::new(agent, "Write it.", DeliveryMode::Steer));
        assert!(held(&mut app, &wakes).is_some(), "{:?}", notices(&app));
        app.world_mut().trigger(StopTurn::new(agent));
        run_until(&mut app, &wakes, |world| {
            world.get::<ActiveTurn>(agent).is_none()
        });
        let live = serde_json::to_string(&entries(app.world(), agent)).unwrap_or_default();
        assert!(!live.contains("A half"), "the partial reply was saved");
        assert!(!live.contains("call-cut"), "{with_call}: {live}");
        assert!(live.contains("Write it."), "the prompt is kept");
        assert_eq!(sent(&model).len(), 2, "nothing after the summary");
        let conversation = app.world().get::<Conversation>(agent);
        let asides = conversation.into_iter().flat_map(|it| it.asides());
        let kept: Vec<_> = asides.filter(|(_, it)| it.kind == OMITTED).collect();
        assert!(
            matches!(kept.as_slice(), [(_, it)] if it.value.to_string().contains("A half")),
            "{with_call}: {kept:?}"
        );
    }
}

/// Before a prompt, only the message it delivered is pending: a prompt an earlier stopped turn
/// kept, and a note delivered while idle, are context, counted and summarized.
#[test]
fn messages_before_the_prompt_it_delivered_are_counted() {
    for inform in [false, true] {
        let model = MockCompletionModel::from_stream_turns([
            reply("## Goal\nRead it."),
            measured("done", 100),
        ]);
        let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |_| {});
        let mut messages = turns(0, 2);
        if !inform {
            // What a stopped turn left: the user's prompt with no answer.
            messages.push(Message::user(format!("Kept: {}", "word ".repeat(20_000))));
        }
        let agent = agent(&mut app, &model, 40_000, messages);
        let Some(agent) = agent else {
            return;
        };
        if inform {
            let note = format!("Noted: {}", "word ".repeat(20_000));
            app.world_mut()
                .trigger(Deliver::new(agent, note, DeliveryMode::Inform));
            for _ in 0..3 {
                app.update();
            }
        }
        ask(&mut app, &wakes, agent, "And now?", 1);
        let requests = sent(&model);
        let summary = requests.first().cloned().unwrap_or_default();
        assert!(
            summary.contains("context summarization assistant"),
            "{inform}: {requests:?}"
        );
        assert!(
            !summary.contains("And now?"),
            "{inform}: the prompt is pending"
        );
    }
}

/// `/compact` with nothing to summarize says pi's reason: "Already compacted" right after a
/// compaction, and "Nothing to compact (session too small)" for a conversation within the keep.
#[test]
fn compact_with_nothing_to_summarize_says_why_as_pi() {
    let model = MockCompletionModel::from_stream_turns([reply("## Goal\nAnswer.")]);
    let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |_| {});
    let compacted = agent(&mut app, &model, 1_000_000, turns(0, 4));
    let Some(compacted) = compacted else {
        return;
    };
    compact(&mut app, &wakes, compacted, "", 1);
    assert!(
        entries(app.world(), compacted)
            .first()
            .is_some_and(holds_summary)
    );
    compact(&mut app, &wakes, compacted, "", 2);
    let said = notices(&app);
    assert!(
        said.iter()
            .any(|it| it == "Compaction failed: Already compacted"),
        "{said:?}"
    );

    let model = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let (mut app, wakes) = start_with(&MemoryStore::default(), 1_000_000, |_| {});
    let agent = agent(&mut app, &model, 1_000_000, turns(0, 2));
    let Some(agent) = agent else {
        return;
    };
    compact(&mut app, &wakes, agent, "", 1);
    let said = notices(&app);
    assert!(
        said.iter()
            .any(|it| it == "Compaction failed: Nothing to compact (session too small)"),
        "{said:?}"
    );
    assert!(sent(&model).is_empty(), "{:?}", sent(&model));
}

/// What [`mark_wrapped`] saw: each connection it wrapped that a wrapper had wrapped already.
#[derive(Resource, Default)]
struct Rewrapped(Vec<String>);

/// An adapter that marks the connection it wraps, and records one it finds marked already.
fn mark_wrapped(
    mut calls: Query<(&mut Adapted, Has<Summarizing>), Added<Admitted>>,
    mut seen: ResMut<Rewrapped>,
) {
    for (mut adapted, summary) in &mut calls {
        let name = adapted.0.spec.display_name.clone();
        if name.ends_with(" (wrapped)") {
            seen.0.push(format!("summary={summary}: {name}"));
        }
        let mut spec = adapted.0.spec.as_ref().clone();
        spec.display_name = format!("{name} (wrapped)");
        adapted.0.spec = Arc::new(spec);
    }
}

/// A summary goes through the agent's adapters once, as the turn's request does: the summary
/// before a request near the window and the one recovering from an overflow are sent on the
/// route the request named, not on the connection the adapters already wrapped.
#[test]
fn a_summary_is_wrapped_by_the_adapters_once() {
    let scripts = [
        (
            "near the window",
            vec![
                measured("first answer", 30_000),
                reply("## Goal\nAnswer."),
                measured("done", 100),
            ],
        ),
        (
            "recovery",
            vec![too_long(), reply("## Goal\nAnswer."), measured("done", 100)],
        ),
    ];
    for (case, script) in scripts {
        let model = MockCompletionModel::from_stream_turns(script);
        let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |app| {
            app.init_resource::<Rewrapped>();
            app.add_systems(
                AgentPass,
                mark_wrapped.in_set(PrepareSystems::AdaptOutermost),
            );
        });
        let agent = agent(&mut app, &model, 40_000, turns(0, 4));
        let Some(agent) = agent else {
            return;
        };
        ask(&mut app, &wakes, agent, "The first question?", 1);
        if case == "near the window" {
            ask(&mut app, &wakes, agent, "The second question?", 2);
        }
        for _ in 0..40 {
            app.update();
            wakes
                .recv_timeout(std::time::Duration::from_millis(20))
                .ok();
        }
        let requests = sent(&model);
        let summarized = requests
            .iter()
            .any(|it| it.contains("context summarization assistant"));
        assert!(summarized, "{case}: {:?}", notices(&app));
        let seen = &app.world().resource::<Rewrapped>().0;
        assert!(seen.is_empty(), "{case}: {seen:?}");
    }
}

/// A resend refused again after its recovery fails the turn, and no compaction follows it, as pi's
/// check returns once a recovery was tried.
#[test]
fn a_second_refusal_after_recovery_is_not_compacted_again() {
    let model = MockCompletionModel::from_stream_turns([
        too_long(),
        reply("## Goal\nAnswer."),
        too_long(),
        reply("## Goal\nNot asked."),
    ]);
    let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |_| {});
    let agent = agent(&mut app, &model, 40_000, turns(0, 4));
    let Some(agent) = agent else {
        return;
    };
    ask(&mut app, &wakes, agent, "And the next question?", 1);
    for _ in 0..40 {
        app.update();
        wakes
            .recv_timeout(std::time::Duration::from_millis(20))
            .ok();
    }
    let ended = &app.world().resource::<Ended>().0;
    assert!(
        matches!(ended.as_slice(), [TurnOutcome::Failed(_)]),
        "{ended:?}"
    );
    assert_eq!(sent(&model).len(), 3, "{:?}", notices(&app));
}

/// A reply the window cut short that recovery leaves out, here with nothing to compact, stays in
/// the saved conversation as an aside, as pi keeps the attempt in its session file: the next
/// request does not send it, and the session reopened still holds it.
#[test]
fn a_reply_cut_short_and_left_out_stays_in_the_saved_session() {
    let cut = vec![
        MockStreamEvent::text("A half"),
        MockStreamEvent::FinalResponse(rig_core::operation::Finish {
            reason: Some(FinishReason::Length),
            ..rig_core::test_utils::mock_final(Usage::new().output_tokens(10))
        }),
    ];
    let model = MockCompletionModel::from_stream_turns([cut, reply("The whole answer.")]);
    let store = MemoryStore::default();
    let (mut app, wakes) = start_with(&store, 1_000_000, |_| {});
    let agent = agent(&mut app, &model, 200_000, turns(0, 4));
    let Some(agent) = agent else {
        return;
    };
    ask(&mut app, &wakes, agent, "Write it.", 1);
    ask(&mut app, &wakes, agent, "Write it again.", 2);
    assert_eq!(answer(&app).as_deref(), Some("The whole answer."));
    let requests = sent(&model);
    assert_eq!(requests.len(), 2, "{:?}", notices(&app));
    assert!(
        !requests.iter().any(|request| request.contains("A half")),
        "{requests:?}"
    );
    let omitted = |world: &World, agent: Entity| -> Vec<String> {
        let conversation = world.get::<Conversation>(agent);
        let asides = conversation.into_iter().flat_map(|it| it.asides());
        asides
            .filter(|(_, aside)| aside.kind == OMITTED && aside.saved)
            .map(|(_, aside)| aside.value.to_string())
            .collect()
    };
    let live = omitted(app.world(), agent);
    assert!(
        matches!(live.as_slice(), [one] if one.contains("A half")),
        "{live:?}"
    );
    app.update();
    let id = app.world().get::<AgentId>(agent).cloned();
    drop(app);

    let (mut reopened, _wakes) = start_with(&store, 1_000_000, |_| {});
    let world = reopened.world_mut();
    let mut agents = world.query::<(Entity, &AgentId)>();
    let restored = id.and_then(|id| id.find_in(agents.iter(world)));
    assert!(restored.is_some(), "the agent was restored");
    let Some(restored) = restored else {
        return;
    };
    assert_eq!(omitted(reopened.world(), restored), live);
    let messages = serde_json::to_string(&entries(reopened.world(), restored)).unwrap_or_default();
    assert!(!messages.contains("A half"), "{messages}");
}

/// A follow-up naming a file, queued while `/compact` writes its summary, goes on as the inbox
/// prepared it: the next request holds the file once.
#[test]
fn a_follow_up_queued_during_compact_keeps_its_file_attached_once() {
    let files = Arc::new(Files::default());
    files.put("note.txt", "NOTE_BODY");
    let model = MockCompletionModel::from_stream_turns([
        reply("## Goal\nAnswer."),
        measured("It is a note.", 100),
    ]);
    let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |app| {
        collect::<rig_ecs::inbox::Recalled>(app);
        app.insert_resource(ProjectWorkspace(files.clone()))
            .insert_resource(ProjectRoot(PROJECT.into()))
            .add_plugins(crate::coding_tools::AttachPlugin)
            .add_systems(AgentPass, hold_for_review.in_set(ReviewSystems::Act));
    });
    let agent = agent(&mut app, &model, 1_000_000, turns(0, 4));
    let Some(agent) = agent else {
        return;
    };
    app.world_mut().trigger(Compact {
        entity: agent,
        focus: String::new(),
    });
    assert!(held(&mut app, &wakes).is_some(), "{:?}", notices(&app));
    app.world_mut().trigger(Deliver::new(
        agent,
        "Explain @note.txt",
        DeliveryMode::Queue,
    ));
    run_until(&mut app, &wakes, |world| {
        world.iter_entities().any(|it| {
            it.contains::<Incoming>() && it.contains::<Started>() && !it.contains::<Outcome>()
        })
    });
    decide(&mut app, &wakes, Some("## Goal\nAnswer."));
    // The turn started for the summary goes on with the follow-up.
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    assert_eq!(answer(&app).as_deref(), Some("It is a note."));
    let last = sent(&model).get(1).cloned().unwrap_or_default();
    assert_eq!(last.matches("NOTE_BODY").count(), 1, "{last}");
    assert!(last.contains("Explain @note.txt"), "{last}");
    let recalled = &app
        .world()
        .resource::<Collected<rig_ecs::inbox::Recalled>>()
        .0;
    assert!(recalled.is_empty(), "{recalled:?}");
}

/// Whether [`send_as_compaction_ends`] sent its prompt.
#[derive(Resource, Default)]
struct SentAtEnd(bool);

/// Sends the agent a newer prompt the moment its first turn ends.
fn send_as_compaction_ends(
    end: On<Remove<TurnOf>>,
    turns: Query<&TurnOf>,
    mut sent: ResMut<SentAtEnd>,
    mut commands: Commands,
) {
    if let Ok(&TurnOf(agent)) = turns.get(end.entity)
        && !std::mem::replace(&mut sent.0, true)
    {
        commands.trigger(Deliver::new(agent, "NEWER_PROMPT", DeliveryMode::Steer));
    }
}

/// A prompt queued during `/compact` is answered before a newer one that arrives as the
/// compaction ends: the next request holds both, the queued one first.
#[test]
fn a_follow_up_queued_during_compact_goes_before_a_newer_prompt() {
    let model = MockCompletionModel::from_stream_turns([
        reply("## Goal\nAnswer."),
        measured("Both answered.", 100),
        measured("Again.", 100),
    ]);
    let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |app| {
        collect::<rig_ecs::inbox::Recalled>(app);
        app.init_resource::<SentAtEnd>()
            .add_observer(send_as_compaction_ends)
            .add_systems(AgentPass, hold_for_review.in_set(ReviewSystems::Act));
    });
    let agent = agent(&mut app, &model, 1_000_000, turns(0, 4));
    let Some(agent) = agent else {
        return;
    };
    app.world_mut().trigger(Compact {
        entity: agent,
        focus: String::new(),
    });
    assert!(held(&mut app, &wakes).is_some(), "{:?}", notices(&app));
    app.world_mut()
        .trigger(Deliver::new(agent, "QUEUED_PROMPT", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| {
        world.iter_entities().any(|it| {
            it.contains::<Incoming>() && it.contains::<Started>() && !it.contains::<Outcome>()
        })
    });
    decide(&mut app, &wakes, Some("## Goal\nAnswer."));
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() >= 2 && world.get::<ActiveTurn>(agent).is_none()
    });
    let requests = sent(&model);
    let next = requests.get(1).cloned().unwrap_or_default();
    let (queued, newer) = (next.find("QUEUED_PROMPT"), next.find("NEWER_PROMPT"));
    assert!(queued.is_some(), "{:?}: {next}", notices(&app));
    assert!(newer.is_none_or(|newer| queued < Some(newer)), "{next}");
    let recalled = &app
        .world()
        .resource::<Collected<rig_ecs::inbox::Recalled>>()
        .0;
    assert!(recalled.is_empty(), "{recalled:?}");
}

/// A model whose input limit is below its context window compacts against the input limit: 30k
/// reported of a 40k input limit compacts before the next request, though the window is 200k.
#[test]
fn a_model_whose_input_limit_is_below_its_window_compacts_before_it_overflows() {
    let model = MockCompletionModel::from_stream_turns([
        measured("first answer", 30_000),
        reply("## Goal\nAnswer the questions."),
        measured("second answer", 100),
    ]);
    let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |_| {});
    let agent = agent(&mut app, &model, 200_000, turns(0, 4));
    let Some(agent) = agent else {
        return;
    };
    if let Some(mut connection) = app.world_mut().get_mut::<Connection>(agent) {
        let spec = connection.spec.as_ref().clone();
        connection.spec = Arc::new(spec.with_max_input_tokens(40_000));
    }
    ask(&mut app, &wakes, agent, "The first question?", 1);
    ask(&mut app, &wakes, agent, "The second question?", 2);
    let requests = sent(&model);
    assert_eq!(requests.len(), 3, "{:?}", notices(&app));
    let summary = requests.get(1).cloned().unwrap_or_default();
    assert!(
        summary.contains("context summarization assistant"),
        "{summary}"
    );
    assert_eq!(answer(&app).as_deref(), Some("second answer"));
}

/// A turn's call a plugin routed stays on its route when it is built again, after the summary
/// that recovers it and after a retry, and the summary goes there too; the next turn's call is
/// routed afresh. Nothing goes to the agent's own model.
#[test]
fn a_rebuilt_call_keeps_its_route_and_a_new_call_is_still_routed() {
    let own = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let routed = MockCompletionModel::from_stream_turns([
        too_long(),
        reply("## Goal\nAnswer the questions."),
        unavailable(),
        measured("done", 100),
        measured("again", 100),
    ]);
    let target = connect(&routed).map(|mut connection| {
        let spec = connection
            .spec
            .as_ref()
            .clone()
            .with_context_window(200_000);
        connection.spec = Arc::new(spec.with_max_output_tokens(8_000));
        connection
    });
    let Some(target) = target else {
        return;
    };
    let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |app| {
        let route = move |mut calls: Query<
            &mut ModelRequest,
            (Added<Admitted>, With<TurnCall>, Without<Supersedes>),
        >| {
            for mut call in &mut calls {
                call.connection = Some(target.clone());
            }
        };
        app.add_systems(AgentPass, route.in_set(PrepareSystems::Edit));
    });
    let agent = agent(&mut app, &own, 200_000, turns(0, 4));
    let Some(agent) = agent else {
        return;
    };
    app.world_mut().trigger(Deliver::new(
        agent,
        "And the next question?",
        DeliveryMode::Steer,
    ));
    until_ended_with_retries(&mut app, &wakes, 1);
    assert_eq!(answer(&app).as_deref(), Some("done"), "{:?}", notices(&app));
    app.world_mut()
        .trigger(Deliver::new(agent, "And another?", DeliveryMode::Steer));
    until_ended_with_retries(&mut app, &wakes, 2);
    assert_eq!(
        answer(&app).as_deref(),
        Some("again"),
        "{:?}",
        notices(&app)
    );
    assert_eq!(sent(&routed).len(), 5, "{:?}", notices(&app));
    assert!(sent(&own).is_empty(), "{:?}", sent(&own));
}

/// Messages sent while `/compact` writes its summary are delivered by the kernel as after any
/// reply: the steering one first, then the queued ones together in the order they were sent,
/// still queued, the image of one attached; nothing is handed back as text.
#[test]
fn messages_sent_during_compact_keep_their_priority_order_and_attachments() {
    let model = MockCompletionModel::from_stream_turns([
        reply("## Goal\nAnswer."),
        measured("Steered.", 100),
        measured("Queued.", 100),
    ]);
    let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |app| {
        collect::<rig_ecs::inbox::Recalled>(app);
        app.add_systems(AgentPass, hold_for_review.in_set(ReviewSystems::Act));
    });
    let agent = agent(&mut app, &model, 1_000_000, turns(0, 4));
    let Some(agent) = agent else {
        return;
    };
    app.world_mut().trigger(Compact {
        entity: agent,
        focus: String::new(),
    });
    assert!(held(&mut app, &wakes).is_some(), "{:?}", notices(&app));
    let mut first = with_image(agent, "QUEUED_ONE");
    first.message.mode = DeliveryMode::Queue;
    app.world_mut().trigger(first);
    app.update();
    app.world_mut()
        .trigger(Deliver::new(agent, "QUEUED_TWO", DeliveryMode::Queue));
    app.update();
    app.world_mut()
        .trigger(Deliver::new(agent, "STEERED", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        let waiting = |it: &EntityRef| {
            it.contains::<Incoming>() && it.contains::<Started>() && !it.contains::<Outcome>()
        };
        world.iter_entities().filter(waiting).count() == 3
    });
    let modes = |world: &mut World| -> Vec<DeliveryMode> {
        let mut messages = world.query::<&Incoming>();
        let mut modes: Vec<_> = messages.iter(world).map(|it| it.mode).collect();
        modes.sort_by_key(|mode| format!("{mode:?}"));
        modes
    };
    let before = modes(app.world_mut());
    decide(&mut app, &wakes, Some("## Goal\nAnswer."));
    app.update();
    assert_eq!(
        modes(app.world_mut()),
        before,
        "a queued message stays queued"
    );
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty() && world.get::<ActiveTurn>(agent).is_none()
    });
    assert_eq!(answer(&app).as_deref(), Some("Queued."));
    let requests = sent(&model);
    assert_eq!(requests.len(), 3, "{:?}", notices(&app));
    let (steered, queued) = (
        requests.get(1).cloned().unwrap_or_default(),
        requests.get(2).cloned().unwrap_or_default(),
    );
    assert!(
        steered.contains("STEERED") && !steered.contains("QUEUED"),
        "{steered}"
    );
    let (one, two) = (queued.find("QUEUED_ONE"), queued.find("QUEUED_TWO"));
    assert!(one.is_some() && one < two, "{queued}");
    assert_eq!(queued.matches("aGk=").count(), 1, "{queued}");
    let recalled = &app
        .world()
        .resource::<Collected<rig_ecs::inbox::Recalled>>()
        .0;
    assert!(recalled.is_empty(), "{recalled:?}");
}

/// A rate limit whose `Retry-After` header asks for seven seconds.
fn rate_limited() -> Vec<MockStreamEvent> {
    let mut headers = http::HeaderMap::new();
    headers.insert("retry-after", http::HeaderValue::from_static("7"));
    let limited = ProviderResponseError::new(http::StatusCode::TOO_MANY_REQUESTS, "rate limited")
        .with_headers(Some(headers));
    vec![MockStreamEvent::Error(MockError::ProviderResponse(limited))]
}

/// A summary call is retried by the kernel as `[retry]` says, after the wait the provider asked
/// for, each summary call on a count of its own that leaves the turn's alone: with one retry
/// allowed, a summary that failed once goes in and the turn's own call is still retried once;
/// a summary that fails twice is given up, and the notice names the setting.
#[test]
fn summary_retries_follow_the_retry_settings_and_leave_the_turns_budget() {
    let policy = rig_ecs::retry::RetryPolicy {
        max_retries: 1,
        ..rig_ecs::retry::RetryPolicy::DEFAULT
    };
    let model = MockCompletionModel::from_stream_turns([
        rate_limited(),
        reply("## Goal\nAnswer the questions."),
        unavailable(),
        measured("done", 100),
    ]);
    let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |app| {
        app.insert_resource(policy);
    });
    // About 1k tokens of conversation in a window that leaves less than the reserve.
    let agent = agent(&mut app, &model, 17_000, turns(0, 4));
    let Some(agent) = agent else {
        return;
    };
    app.world_mut().trigger(Deliver::new(
        agent,
        "And the next question?",
        DeliveryMode::Steer,
    ));
    run_until(&mut app, &wakes, |world| {
        world.iter_entities().any(|it| it.contains::<Backoff>())
    });
    let world = app.world_mut();
    let wait = world.query::<&Backoff>().iter(world).next();
    let wait = wait.map(Backoff::seconds_left);
    assert!(wait.is_some_and(|left| (6..=7).contains(&left)), "{wait:?}");
    until_ended_with_retries(&mut app, &wakes, 1);
    assert_eq!(answer(&app).as_deref(), Some("done"), "{:?}", notices(&app));
    assert_eq!(sent(&model).len(), 4, "{:?}", notices(&app));
    let live = entries(app.world(), agent);
    assert!(live.first().is_some_and(holds_summary), "{live:?}");

    let model = MockCompletionModel::from_stream_turns([
        unavailable(),
        unavailable(),
        reply("## Goal\nToo late."),
    ]);
    let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |app| {
        app.insert_resource(policy);
    });
    let agent = self::agent(&mut app, &model, 1_000_000, turns(0, 4));
    let Some(agent) = agent else {
        return;
    };
    app.world_mut().trigger(Compact {
        entity: agent,
        focus: String::new(),
    });
    until_ended_with_retries(&mut app, &wakes, 1);
    assert_eq!(sent(&model).len(), 2, "{:?}", notices(&app));
    let said = notices(&app);
    assert!(
        said.iter()
            .any(|it| it.contains("Not retrying") && it.contains("[retry] max_retries")),
        "{said:?}"
    );
    assert!(!entries(app.world(), agent).iter().any(holds_summary));
}

/// A reply the window cut short that holds a tool call is recovered from like any other: the
/// call never runs, the reply is kept as an aside the moment it is dropped, and the request is
/// sent again.
#[test]
fn a_cut_short_reply_with_a_tool_call_runs_no_tool_and_is_kept() {
    let files = Arc::new(Files::default());
    files.put("a.rs", "FILE_BODY");
    let cut = vec![
        MockStreamEvent::text("Reading"),
        MockStreamEvent::tool_call("call-cut", "read", serde_json::json!({ "path": "a.rs" })),
        MockStreamEvent::FinalResponse(rig_core::operation::Finish {
            reason: Some(FinishReason::Length),
            ..rig_core::test_utils::mock_final(Usage::new().output_tokens(10))
        }),
    ];
    let model = MockCompletionModel::from_stream_turns([
        cut,
        reply("## Goal\nAnswer the questions."),
        measured("done", 100),
    ]);
    let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |app| {
        app.insert_resource(ProjectWorkspace(files.clone()))
            .insert_resource(ProjectRoot(PROJECT.into()))
            .add_plugins(crate::coding_tools::ReadTool)
            .add_systems(AgentPass, hold_for_review.in_set(ReviewSystems::Act));
    });
    let agent = agent(&mut app, &model, 200_000, turns(0, 4));
    let Some(agent) = agent else {
        return;
    };
    app.world_mut()
        .trigger(Deliver::new(agent, "Read a.rs.", DeliveryMode::Steer));
    assert!(held(&mut app, &wakes).is_some(), "{:?}", notices(&app));
    let omitted = |world: &World| -> Vec<String> {
        let conversation = world.get::<Conversation>(agent);
        let asides = conversation.into_iter().flat_map(|it| it.asides());
        asides
            .filter(|(_, aside)| aside.kind == OMITTED)
            .map(|(_, aside)| aside.value.to_string())
            .collect()
    };
    let kept = omitted(app.world());
    assert!(
        matches!(kept.as_slice(), [it] if it.contains("call-cut")),
        "kept before the summary is in: {kept:?}"
    );
    decide(&mut app, &wakes, Some("## Goal\nAnswer the questions."));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    assert_eq!(answer(&app).as_deref(), Some("done"), "{:?}", notices(&app));
    let requests = sent(&model);
    assert_eq!(requests.len(), 3, "{:?}", notices(&app));
    let all = requests.join("\n");
    assert!(!all.contains("FILE_BODY"), "the call ran: {all}");
    let live = serde_json::to_string(&entries(app.world(), agent)).unwrap_or_default();
    assert!(!live.contains("call-cut"), "{live}");
    assert_eq!(omitted(app.world()).len(), 1);
}

/// One recovery for each request, as pi's flag: the summary going in does not allow a second,
/// so a resend refused again fails the turn; a message of the user's delivered into the resend
/// does, so that request is recovered once more.
#[test]
fn new_user_input_allows_another_recovery_and_a_summary_does_not() {
    for steered in [false, true] {
        let model = MockCompletionModel::from_stream_turns([
            too_long(),
            reply("## Goal\nAnswer the questions."),
            too_long(),
            // The second recovery splits the turn: its history, then its earlier part.
            reply("## Goal\nAnswer them all."),
            reply("## Original Request\nThe next question."),
            measured("done", 100),
        ]);
        let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |app| {
            app.add_systems(AgentPass, hold_for_review.in_set(ReviewSystems::Act));
        });
        let agent = agent(&mut app, &model, 200_000, turns(0, 4));
        let Some(agent) = agent else {
            return;
        };
        app.world_mut().trigger(Deliver::new(
            agent,
            "And the next question?",
            DeliveryMode::Steer,
        ));
        assert!(held(&mut app, &wakes).is_some(), "{:?}", notices(&app));
        if steered {
            let more = format!("One more thing: {}", "word ".repeat(300));
            app.world_mut()
                .trigger(Deliver::new(agent, more, DeliveryMode::Steer));
            run_until(&mut app, &wakes, |world| {
                world.iter_entities().any(|it| {
                    it.contains::<Incoming>()
                        && it.contains::<Started>()
                        && !it.contains::<Outcome>()
                })
            });
        }
        decide(&mut app, &wakes, Some("## Goal\nAnswer the questions."));
        if steered {
            decide(&mut app, &wakes, Some("## Goal\nAnswer them all."));
            decide(
                &mut app,
                &wakes,
                Some("## Original Request\nThe next question."),
            );
        }
        run_until(&mut app, &wakes, |world| {
            !world.resource::<Ended>().0.is_empty()
        });
        let ended = &app.world().resource::<Ended>().0;
        match steered {
            true => {
                assert_eq!(answer(&app).as_deref(), Some("done"), "{:?}", notices(&app));
                assert_eq!(sent(&model).len(), 6, "{:?}", notices(&app));
            }
            false => {
                assert!(
                    matches!(ended.as_slice(), [TurnOutcome::Failed(_)]),
                    "{ended:?}"
                );
                assert_eq!(sent(&model).len(), 3, "{:?}", notices(&app));
            }
        }
    }
}

/// The output clamp trusts the reported usage only while no newer message was put before its
/// reply, as pi-ai's estimate: a few-shot prefix added to the request makes it estimate instead.
#[test]
fn the_output_clamp_drops_the_usage_when_a_prefix_is_added() {
    for prefix in [false, true] {
        let model = MockCompletionModel::from_stream_turns([reply("done")]);
        let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |app| {
            if prefix {
                let few_shot =
                    |mut calls: Query<&mut ModelRequest, (Added<Admitted>, With<TurnCall>)>| {
                        for mut call in &mut calls {
                            if let Some(history) = &mut call.history {
                                let text = format!("An example. {}", "word ".repeat(2_000));
                                let example = Entry::new(Message::user(text));
                                history.splice(0..0, [example], Author::plugin("few-shot"));
                            }
                        }
                    };
                app.add_systems(AgentPass, few_shot.in_set(PrepareSystems::Edit));
            }
        });
        app.world_mut()
            .resource_mut::<CompactionDefault>()
            .0
            .enabled = false;
        let agent = agent(&mut app, &model, 40_000, turns(0, 2));
        let Some(agent) = agent else {
            return;
        };
        let through = app
            .world()
            .get::<Conversation>(agent)
            .and_then(|conversation| conversation.entries().last().map(|entry| entry.id));
        let anchor = through.map(|through| rig_memory::Anchor {
            usage: Usage::new().total_tokens(39_000),
            through,
        });
        app.world_mut().entity_mut(agent).insert(LastUsage(anchor));
        ask(&mut app, &wakes, agent, "And now?", 1);
        let asked = model
            .requests()
            .first()
            .and_then(|request| request.max_tokens);
        match prefix {
            false => assert!(asked.is_some_and(|it| it < 1_000), "{asked:?}"),
            true => assert_eq!(asked, Some(8_000)),
        }
    }
}
