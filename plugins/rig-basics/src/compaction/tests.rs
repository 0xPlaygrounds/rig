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

/// Each summary call of a split turn gets its own retries: the history summary spends all three
/// and goes in on the last, and the turn-prefix summary still fails once and is sent again, which
/// a count the two shared would refuse.
#[test]
fn each_summary_call_is_retried_on_its_own() {
    let model = MockCompletionModel::from_stream_turns([
        unavailable(),
        unavailable(),
        unavailable(),
        reply("## Goal\nAnswer the questions."),
        unavailable(),
        reply("## Original Request\nThe last question."),
    ]);
    let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |_| {});
    let agent = agent(&mut app, &model, 1_000_000, split_turn());
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };
    app.world_mut().trigger(Compact {
        entity: agent,
        focus: String::new(),
    });
    // The calls fail on their own frames, so the clock moves on until every wait passed.
    until_ended_with_retries(&mut app, &wakes, 1);
    assert_eq!(sent(&model).len(), 6, "{:?}", notices(&app));
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

/// A summary call's passing failures are retried by pi's patterns, a bare `terminated` among them,
/// and a spent quota never is, whatever its status.
#[test]
fn summary_retries_follow_pis_patterns() {
    let report = |message: &str| ErrorReport::new(ErrorKind::Provider, message);
    assert!(retryable(&report("terminated")));
    assert!(retryable(&report("upstream connect error or disconnect")));
    let mut quota = report("429 Too Many Requests: insufficient_quota");
    quota.retryable = true;
    assert!(!retryable(&quota));
    assert!(!retryable(&report("invalid request: bad summary")));
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
        let upto = conversation.id(2);
        assert!(rig_memory::apply_summary(&mut conversation, upto, &state).is_some());
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

/// Before a prompt, the prompt itself is neither counted nor cut, also when the conversation ends
/// in a tool result kept from a turn that stopped: a 30k prompt after a short history in a 40k
/// window goes without a summary.
#[test]
fn the_pending_prompt_is_not_counted_after_a_kept_tool_result() {
    let model = MockCompletionModel::from_stream_turns([measured("done", 100)]);
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
    assert_eq!(sent(&model).len(), 1, "{:?}", notices(&app));
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

/// With no usage reported, the threshold counts the messages alone, as pi: a long system prompt
/// does not push a short conversation over.
#[test]
fn without_usage_the_threshold_counts_the_messages_alone() {
    let model = MockCompletionModel::from_stream_turns([reply("done")]);
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
    assert_eq!(sent(&model).len(), 1, "{:?}", notices(&app));
}

/// After a run whose last reply reported usage over the reserve, the answer is held and the
/// conversation compacted before the turn ends, as pi checks at the end of each run, on that
/// turn's model; a prompt sent while that summary is written is answered, not handed back.
#[test]
fn a_run_ending_over_the_reserve_compacts_and_a_prompt_sent_meanwhile_is_answered() {
    let measured = vec![
        MockStreamEvent::text("first answer"),
        MockStreamEvent::final_response_with_total_tokens(30_000),
    ];
    let model = MockCompletionModel::from_stream_turns([
        measured,
        reply("## Goal\nAnswer the questions."),
        reply("## Goal\nAnswer them all."),
        reply("second answer"),
    ]);
    let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |_| {});
    let agent = agent(&mut app, &model, 40_000, turns(0, 4));
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };
    app.world_mut().trigger(Deliver::new(
        agent,
        "The first question?",
        DeliveryMode::Steer,
    ));
    run_until(&mut app, &wakes, |world| {
        world
            .iter_entities()
            .any(|entity| entity.contains::<AfterRun>())
    });
    let second = "The second question, sent while the summary is written?";
    app.world_mut()
        .trigger(Deliver::new(agent, second, DeliveryMode::Queue));
    // The summary may go in before or after the prompt is taken; either way it is answered.
    run_until(&mut app, &wakes, |world| {
        let answered = sent(&model).iter().skip(2).any(|it| it.contains(second));
        answered && world.get::<ActiveTurn>(agent).is_none()
    });
    let requests = sent(&model);
    let summary = requests.get(1).cloned().unwrap_or_default();
    assert!(
        summary.contains("context summarization assistant"),
        "{summary}"
    );
    assert!(
        requests.iter().skip(2).any(|it| it.contains(second)),
        "the prompt reached the model: {requests:?}"
    );
    let ended = &app.world().resource::<Ended>().0;
    assert!(
        matches!(ended.last(), Some(TurnOutcome::Answered(_))),
        "{ended:?}"
    );
}

/// A completed summary reply whose stop is a passing error is retried as pi retries its
/// assistant's error, and the waits double from the first with no jitter, capped.
#[test]
fn a_summary_that_completes_with_a_passing_error_is_retried_with_pis_waits() {
    let overloaded = vec![
        MockStreamEvent::text("## Goal\nAnsw"),
        MockStreamEvent::FinalResponse(rig_core::operation::Finish {
            reason: Some(FinishReason::Other("overloaded".to_owned())),
            ..rig_core::test_utils::mock_final(Usage::default())
        }),
    ];
    let model = MockCompletionModel::from_stream_turns([overloaded, reply("## Goal\nAnswer.")]);
    let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |_| {});
    let agent = agent(&mut app, &model, 1_000_000, turns(0, 4));
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };
    app.world_mut().trigger(Compact {
        entity: agent,
        focus: String::new(),
    });
    for _ in 0..200 {
        if !app.world().resource::<Ended>().0.is_empty() {
            break;
        }
        advance(&mut app, std::time::Duration::from_secs(3));
        app.update();
        wakes
            .recv_timeout(std::time::Duration::from_millis(20))
            .ok();
    }
    assert_eq!(sent(&model).len(), 2, "{:?}", notices(&app));
    let live = entries(app.world(), agent);
    assert!(live.first().is_some_and(holds_summary), "{live:?}");
    // pi's retryDelayMs with pi's defaults: a 2 s first wait and a 60 s cap.
    let waits: Vec<u64> = (0..7)
        .map(|n| retry_delay(&SUMMARY_RETRY, n).as_secs())
        .collect();
    assert_eq!(waits, [2, 4, 8, 16, 32, 60, 60]);
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

/// A summary after the run that fails once and is sent again keeps holding the turn's answer: the
/// turn ends answered, the summary goes in, and a prompt sent meanwhile is answered; when every
/// retry fails, the turn still ends answered and the failure is shown, as pi.
#[test]
fn a_retried_summary_after_the_run_keeps_the_turn_answered() {
    let busy = || {
        let busy = ProviderResponseError::without_status("overloaded").with_transient(Some(true));
        vec![MockStreamEvent::Error(MockError::ProviderResponse(busy))]
    };
    for succeeds in [true, false] {
        let mut script = vec![measured("first answer", 30_000), busy()];
        match succeeds {
            true => script.extend([
                reply("## Goal\nAnswer the questions."),
                reply("second answer"),
            ]),
            false => script.extend([busy(), busy(), busy(), busy()]),
        }
        let model = MockCompletionModel::from_stream_turns(script);
        let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |_| {});
        let agent = agent(&mut app, &model, 40_000, turns(0, 4));
        assert!(agent.is_some(), "the catalog lists the model");
        let Some(agent) = agent else {
            return;
        };
        app.world_mut().trigger(Deliver::new(
            agent,
            "The first question?",
            DeliveryMode::Steer,
        ));
        run_until(&mut app, &wakes, |world| {
            world
                .iter_entities()
                .any(|entity| entity.contains::<Backoff>())
        });
        let second = "The second question, sent during the retry wait?";
        if succeeds {
            app.world_mut()
                .trigger(Deliver::new(agent, second, DeliveryMode::Queue));
        }
        until_ended_with_retries(&mut app, &wakes, 1);
        let ended = &app.world().resource::<Ended>().0;
        assert!(
            matches!(ended.last(), Some(TurnOutcome::Answered(_))),
            "{succeeds}: {ended:?}"
        );
        let requests = sent(&model);
        match succeeds {
            true => {
                let live = entries(app.world(), agent);
                assert!(live.first().is_some_and(holds_summary), "{live:?}");
                assert!(
                    requests.iter().any(|it| it.contains(second)),
                    "{requests:?}"
                );
            }
            false => {
                let said = notices(&app);
                assert!(
                    said.iter().any(|it| it.contains("Auto-compaction failed")),
                    "{said:?}"
                );
            }
        }
    }
}

/// Esc while the summary after a run is written stops it and the turn; a prompt the turn accepted
/// meanwhile is not lost: it stays in the conversation for the next request, as every message a
/// stopped turn accepted does.
#[test]
fn stopping_during_the_summary_after_the_run_keeps_the_prompt_sent_meanwhile() {
    let model = MockCompletionModel::from_stream_turns([
        measured("first answer", 30_000),
        reply("## Goal\nAnswer."),
    ]);
    let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |app| {
        app.add_systems(AgentPass, hold_for_review.in_set(ReviewSystems::Act));
    });
    let agent = agent(&mut app, &model, 40_000, turns(0, 4));
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };
    app.world_mut().trigger(Deliver::new(
        agent,
        "The first question?",
        DeliveryMode::Steer,
    ));
    assert!(held(&mut app, &wakes).is_some(), "{:?}", notices(&app));
    let second = "The second question, sent while the summary is written?";
    app.world_mut()
        .trigger(Deliver::new(agent, second, DeliveryMode::Queue));
    for _ in 0..5 {
        app.update();
    }
    app.world_mut().trigger(StopTurn::new(agent));
    run_until(&mut app, &wakes, |world| {
        world.get::<ActiveTurn>(agent).is_none()
    });
    let live = serde_json::to_string(&entries(app.world(), agent)).unwrap_or_default();
    assert!(live.contains(second), "{live}");
    assert!(
        !entries(app.world(), agent)
            .first()
            .is_some_and(holds_summary)
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
    let stale = app.world().get::<ContextStale>(agent);
    assert!(
        stale.is_some_and(|stale| stale.estimate.is_none()),
        "{stale:?}"
    );
}

/// The summary after a run measures against the window of the model the run's last request went
/// to, a smaller one it was routed to, as pi measures against the model that wrote the reply.
#[test]
fn the_check_after_a_run_uses_the_window_of_the_model_it_ran_on() {
    // Routed to a 40k model from a 1M agent: 30k reported compacts after the run.
    let model = MockCompletionModel::from_stream_turns([
        measured("first answer", 30_000),
        reply("## Goal\nAnswer."),
    ]);
    let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |_| {});
    let routed = agent(&mut app, &model, 1_000_000, turns(0, 4));
    assert!(routed.is_some(), "the catalog lists the model");
    let Some(routed) = routed else {
        return;
    };
    let small = app.world().get::<Connection>(routed).map(|connection| {
        let spec = connection.spec.as_ref().clone().with_context_window(40_000);
        Connection {
            spec: Arc::new(spec),
            handler: connection.handler.clone(),
        }
    });
    let Some(small) = small else {
        return;
    };
    let route = move |mut calls: Query<&mut ModelRequest, (Added<Admitted>, With<TurnCall>)>| {
        for mut call in &mut calls {
            call.connection = Some(small.clone());
        }
    };
    app.add_systems(AgentPass, route.in_set(PrepareSystems::Edit));
    ask(&mut app, &wakes, routed, "The first question?", 1);
    run_until(&mut app, &wakes, |world| {
        world.get::<ActiveTurn>(routed).is_none()
    });
    let requests = sent(&model);
    assert_eq!(requests.len(), 2, "{:?}", notices(&app));
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
                                let example = Entry::new(Message::user("An example."));
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

/// The summary after a run goes to the connection the run's last request was routed to, with that
/// model's output limit, not the agent's own.
#[test]
fn the_summary_after_a_run_goes_to_the_model_the_run_was_routed_to() {
    let own = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let routed = MockCompletionModel::from_stream_turns([
        measured("first answer", 30_000),
        reply("## Goal\nAnswer."),
    ]);
    let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |_| {});
    let agent = agent(&mut app, &own, 1_000_000, turns(0, 4));
    let small = connect(&routed).map(|connection| Connection {
        spec: Arc::new(
            connection
                .spec
                .as_ref()
                .clone()
                .with_context_window(40_000)
                .with_max_output_tokens(2_000),
        ),
        handler: connection.handler,
    });
    let (Some(agent), Some(small)) = (agent, small) else {
        return;
    };
    let route = move |mut calls: Query<&mut ModelRequest, (Added<Admitted>, With<TurnCall>)>| {
        for mut call in &mut calls {
            call.connection = Some(small.clone());
        }
    };
    app.add_systems(AgentPass, route.in_set(PrepareSystems::Edit));
    ask(&mut app, &wakes, agent, "The first question?", 1);
    run_until(&mut app, &wakes, |world| {
        world.get::<ActiveTurn>(agent).is_none()
    });
    assert!(own.requests().is_empty(), "{:?}", sent(&own));
    let requests = routed.requests();
    assert_eq!(requests.len(), 2, "{:?}", notices(&app));
    let summary = requests.get(1);
    assert!(
        summary.is_some_and(|it| format!("{it:?}").contains("context summarization assistant")),
        "{:?}",
        sent(&routed)
    );
    let asked = summary.and_then(|it| it.max_tokens);
    assert!(asked.is_some_and(|it| it <= 2_000), "{asked:?}");
}

/// Esc while the summary recovering from a reply cut short is written stops the turn without the
/// partial reply: it was left out before the summary started, as pi omits it.
#[test]
fn stopping_a_cut_short_recovery_saves_none_of_the_reply() {
    let cut = vec![
        MockStreamEvent::text("A half"),
        MockStreamEvent::FinalResponse(rig_core::operation::Finish {
            reason: Some(FinishReason::Length),
            ..rig_core::test_utils::mock_final(Usage::new().output_tokens(10))
        }),
    ];
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
    assert!(live.contains("Write it."), "the prompt is kept");
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

/// After a run that ended in a final error, the context it left is checked, as pi checks after an
/// error: a message another author sent stays in the conversation, and the usage reported before
/// it plus that message leave less than the reserve, so it compacts in a turn of its own.
#[test]
fn a_run_ending_in_an_error_is_checked_with_what_it_left() {
    let refused = ProviderResponseError::without_status("invalid request: refused");
    let model = MockCompletionModel::from_stream_turns([
        measured("first answer", 20_000),
        vec![MockStreamEvent::Error(MockError::ProviderResponse(refused))],
        reply("## Goal\nAnswer."),
    ]);
    let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |_| {});
    let agent = agent(&mut app, &model, 40_000, turns(0, 4));
    let Some(agent) = agent else {
        return;
    };
    ask(&mut app, &wakes, agent, "The first question?", 1);
    let note = format!("From the watcher: {}", "word ".repeat(4_000));
    app.world_mut().trigger(
        Deliver::new(agent, note, DeliveryMode::Steer).with_author(Author::plugin("watcher")),
    );
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() >= 2
    });
    for _ in 0..20 {
        app.update();
        wakes
            .recv_timeout(std::time::Duration::from_millis(20))
            .ok();
    }
    let requests = sent(&model);
    assert_eq!(requests.len(), 3, "{:?}", notices(&app));
    let summary = requests.get(2).cloned().unwrap_or_default();
    assert!(
        summary.contains("context summarization assistant"),
        "{summary}"
    );
    // The message the failed turn left is the newest, so it is kept.
    let live = entries(app.world(), agent);
    assert!(live.first().is_some_and(holds_summary), "{live:?}");
    let kept = serde_json::to_string(&live).unwrap_or_default();
    assert!(kept.contains("From the watcher"), "{kept}");
}

/// A turn whose resend was cut short again is not checked at its end, but a follow-up the turn
/// then answers starts its recovery and check afresh, as pi resets them on each user message: its
/// answer at 30k of a 40k window compacts after the run.
#[test]
fn a_follow_up_after_a_failed_recovery_is_checked_again() {
    let cut = || {
        vec![
            MockStreamEvent::text("A half"),
            MockStreamEvent::FinalResponse(rig_core::operation::Finish {
                reason: Some(FinishReason::Length),
                ..rig_core::test_utils::mock_final(Usage::new().output_tokens(10))
            }),
        ]
    };
    let model = MockCompletionModel::from_stream_turns([
        cut(),
        reply("## Goal\nAnswer."),
        cut(),
        measured("the follow-up's answer", 30_000),
        reply("## Goal\nAnswer it all."),
    ]);
    let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |_| {});
    let agent = agent(&mut app, &model, 40_000, turns(0, 4));
    let Some(agent) = agent else {
        return;
    };
    app.world_mut()
        .trigger(Deliver::new(agent, "Write it.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |_| !sent(&model).is_empty());
    // Long enough that the turns before it lie outside the keep.
    let follow_up = format!("And then? {}", "word ".repeat(1_000));
    app.world_mut()
        .trigger(Deliver::new(agent, follow_up, DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| {
        world.get::<ActiveTurn>(agent).is_none()
    });
    for _ in 0..20 {
        app.update();
        wakes
            .recv_timeout(std::time::Duration::from_millis(20))
            .ok();
    }
    let requests = sent(&model);
    assert_eq!(requests.len(), 5, "{:?}", notices(&app));
    let last = requests.get(4).cloned().unwrap_or_default();
    assert!(last.contains("context summarization assistant"), "{last}");
}

/// The output clamp falls back to an earlier usage report still valid when the newest no longer
/// describes the request, as pi-ai's estimate takes the newest valid one.
#[test]
fn the_output_clamp_falls_back_to_an_earlier_valid_usage() {
    let model = MockCompletionModel::from_stream_turns([reply("done")]);
    let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |app| {
        // A note put before the last message: the newest report no longer describes the prefix.
        let note = |mut calls: Query<&mut ModelRequest, (Added<Admitted>, With<TurnCall>)>| {
            for mut call in &mut calls {
                if let Some(history) = &mut call.history {
                    let at = history.messages().len().saturating_sub(2);
                    let example = Entry::new(Message::user("A note."));
                    history.splice(at..at, [example], Author::plugin("notes"));
                }
            }
        };
        app.add_systems(AgentPass, note.in_set(PrepareSystems::Edit));
    });
    app.world_mut()
        .resource_mut::<CompactionDefault>()
        .0
        .enabled = false;
    let agent = agent(&mut app, &model, 40_000, turns(0, 3));
    let Some(agent) = agent else {
        return;
    };
    let ids: Vec<_> = app
        .world()
        .get::<Conversation>(agent)
        .map(|conversation| conversation.entries().map(|entry| entry.id).collect())
        .unwrap_or_default();
    // An earlier report through the third message, then the newest through the last.
    let at = |index: usize| ids.get(index).copied();
    for (through, tokens) in [(at(3), 39_000), (ids.last().copied(), 100)] {
        let anchor = through.map(|through| rig_memory::Anchor {
            usage: Usage::new().total_tokens(tokens),
            through,
        });
        app.world_mut().entity_mut(agent).insert(LastUsage(anchor));
        app.update();
    }
    ask(&mut app, &wakes, agent, "And now?", 1);
    let asked = model
        .requests()
        .first()
        .and_then(|request| request.max_tokens);
    assert!(asked.is_some_and(|it| it < 1_000), "{asked:?}");
}

/// With the default settings, a summary that keeps failing is sent four times, pi's three retries,
/// and not a fifth.
#[test]
fn a_failing_summary_is_retried_three_times() {
    let model = MockCompletionModel::from_stream_turns([
        unavailable(),
        unavailable(),
        unavailable(),
        unavailable(),
        reply("## Goal\nToo late."),
    ]);
    let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |_| {});
    let agent = agent(&mut app, &model, 1_000_000, turns(0, 4));
    let Some(agent) = agent else {
        return;
    };
    app.world_mut().trigger(Compact {
        entity: agent,
        focus: String::new(),
    });
    until_ended_with_retries(&mut app, &wakes, 1);
    assert_eq!(sent(&model).len(), 4, "{:?}", notices(&app));
    let said = notices(&app);
    assert!(
        said.iter().any(|it| it.contains("Not retrying")),
        "{said:?}"
    );
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
/// after a run, the one recovering from an overflow and the one after a failed run are sent on the
/// route the request named, not on the connection the adapters already wrapped.
#[test]
fn a_summary_is_wrapped_by_the_adapters_once() {
    let refused = ProviderResponseError::without_status("invalid request: refused");
    let scripts = [
        (
            "after the run",
            vec![measured("first answer", 30_000), reply("## Goal\nAnswer.")],
        ),
        (
            "recovery",
            vec![too_long(), reply("## Goal\nAnswer."), measured("done", 100)],
        ),
        (
            "after an error",
            vec![
                measured("first answer", 20_000),
                vec![MockStreamEvent::Error(MockError::ProviderResponse(refused))],
                reply("## Goal\nAnswer."),
            ],
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
        if case == "after an error" {
            let note = format!("From the watcher: {}", "word ".repeat(4_000));
            app.world_mut().trigger(
                Deliver::new(agent, note, DeliveryMode::Steer)
                    .with_author(Author::plugin("watcher")),
            );
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

/// A follow-up with an image sent while the compaction turn after an error writes its summary is
/// answered once that turn ends, image and all, as pi delivers what was queued during a
/// compaction: nothing is handed back as text.
#[test]
fn a_follow_up_sent_during_the_compaction_after_an_error_keeps_its_image() {
    let refused = ProviderResponseError::without_status("invalid request: refused");
    let model = MockCompletionModel::from_stream_turns([
        measured("first answer", 20_000),
        vec![MockStreamEvent::Error(MockError::ProviderResponse(refused))],
        reply("## Goal\nAnswer."),
        measured("A greeting.", 100),
    ]);
    let (mut app, wakes) = start_with(&MemoryStore::default(), 100, |app| {
        collect::<rig_ecs::inbox::Recalled>(app);
        app.add_systems(AgentPass, hold_for_review.in_set(ReviewSystems::Act));
    });
    let agent = agent(&mut app, &model, 40_000, turns(0, 4));
    let Some(agent) = agent else {
        return;
    };
    ask(&mut app, &wakes, agent, "The first question?", 1);
    let note = format!("From the watcher: {}", "word ".repeat(4_000));
    app.world_mut().trigger(
        Deliver::new(agent, note, DeliveryMode::Steer).with_author(Author::plugin("watcher")),
    );
    assert!(held(&mut app, &wakes).is_some(), "{:?}", notices(&app));
    let mut follow_up = with_image(agent, "What is in the picture?");
    follow_up.message.mode = DeliveryMode::Queue;
    app.world_mut().trigger(follow_up);
    run_until(&mut app, &wakes, |world| {
        world.iter_entities().any(|it| {
            it.contains::<Incoming>() && it.contains::<Started>() && !it.contains::<Outcome>()
        })
    });
    decide(&mut app, &wakes, Some("## Goal\nAnswer."));
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() >= 4
    });
    assert_eq!(answer(&app).as_deref(), Some("A greeting."));
    let requests = sent(&model);
    let last = requests.get(3).cloned().unwrap_or_default();
    assert!(
        last.contains("What is in the picture?") && last.contains("aGk="),
        "{:?}: {last}",
        notices(&app)
    );
    let recalled = &app
        .world()
        .resource::<Collected<rig_ecs::inbox::Recalled>>()
        .0;
    assert!(recalled.is_empty(), "{recalled:?}");
}
