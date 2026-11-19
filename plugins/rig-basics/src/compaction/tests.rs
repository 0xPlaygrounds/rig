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
    Collected, Ended, MemoryStore, app_on, collect, commit_message, connect, log_ended, reply,
    run_until,
};

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
    connection.spec = Arc::new(connection.spec.as_ref().clone().with_context_window(window));
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
    assert!(update.contains("<previous-summary>"), "{update}");
    assert!(
        update.contains("ZX-4471") && update.contains("Use sqlite"),
        "{update}"
    );
    assert!(update.contains("<read-files>\\nsrc/lib.rs"), "{update}");
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
        summary.is_some_and(|text| text.contains("Tests pass.") && text.contains("ZX-4471")),
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
/// request sent again, and the turn finishes.
#[test]
fn an_overflow_compacts_and_sends_the_request_again() {
    let model = MockCompletionModel::from_stream_turns([
        too_long(),
        reply("## Goal\nAnswer the questions."),
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
    assert_eq!(requests.len(), 3, "{:?}", notices(&app));
    let (Some(summary), Some(resent)) = (requests.get(1), requests.get(2)) else {
        return;
    };
    assert!(summary.contains("question 3"), "{summary}");
    assert!(
        resent.contains("Answer the questions.") && resent.contains(question),
        "{resent}"
    );
    assert!(!resent.contains("question 3"), "{resent}");
}

/// A summary reply that failed or was cut at the output limit replaces nothing, and the notice
/// says why.
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
    let model = MockCompletionModel::from_stream_turns([cut, failed]);
    let (mut app, wakes) = start_with(&MemoryStore::default(), 10, |_| {});
    let agent = agent(&mut app, &model, 200_000, turns(0, 4));
    assert!(agent.is_some(), "the catalog lists the model");
    let Some(agent) = agent else {
        return;
    };
    let before = entries(app.world(), agent);
    compact(&mut app, &wakes, agent, "", 1);
    compact(&mut app, &wakes, agent, "", 2);

    assert_eq!(entries(app.world(), agent), before);
    let said = notices(&app);
    let truncated =
        "Summarization failed: generation hit the token cap and the summary is incomplete";
    assert!(said.iter().any(|text| text.contains(truncated)), "{said:?}");
    assert!(
        said.iter()
            .any(|text| text.contains("Summarization failed:") && text.contains("bad summary")),
        "{said:?}"
    );
}
