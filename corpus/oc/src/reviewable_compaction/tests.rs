use std::sync::mpsc::Receiver;

use rig_core::completion::{CompletionRequest, Message};
use rig_core::test_utils::MockCompletionModel;
use rig_harness::pick::PickPlugin;
use rig_harness::rig_memory::LogRecord;
use rig_harness_test_support::{
    AnswerAsks, Answers, Ended, MemoryStore, app_on, commit_message, connect, first_agent,
    log_ended, reply, run_until,
};

use super::*;

/// The kernel, the plugin and a front answering its questions with
/// `answers`, on `store`, after the first frame restored the session; its
/// agent answered by `model`.
fn start(
    store: &MemoryStore,
    model: &MockCompletionModel,
    answers: Vec<Answer>,
) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(store);
    app.add_plugins((
        AgentPlugin,
        SessionLogPlugin,
        PickPlugin,
        ReviewableCompactionPlugin,
        AnswerAsks,
    ))
    .insert_resource(Answers::new(answers))
    .init_resource::<Ended>()
    .add_observer(log_ended);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    if let Some(connection) = connect(model) {
        app.world_mut().entity_mut(agent).insert(connection);
    }
    (app, wakes, agent)
}

/// Runs frames until `count` turns ended.
fn until_ended(app: &mut App, wakes: &Receiver<()>, count: usize) {
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() >= count
    });
}

/// Sends `text` to `agent` and runs frames until its turn ended.
fn ask(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str) {
    let ended = app.world().resource::<Ended>().0.len();
    let message = Deliver::new(agent, text, DeliveryMode::Steer);
    app.world_mut().trigger(message);
    until_ended(app, wakes, ended + 1);
}

/// The text of the request's messages, and of its first user message.
fn shape(request: Option<&CompletionRequest>) -> (String, String) {
    let Some(request) = request else {
        return (String::new(), String::new());
    };
    let text = |message: &Message| serde_json::to_string(message).unwrap_or_default();
    let all = request
        .chat_history
        .iter()
        .map(|message| text(message))
        .collect();
    let user = |message: &&std::sync::Arc<Message>| matches!(&***message, Message::User { .. });
    let first = request
        .chat_history
        .iter()
        .find(user)
        .map(|message| text(message));
    (all, first.unwrap_or_default())
}

/// The messages of `agent`.
fn messages(app: &App, agent: Entity) -> Vec<Message> {
    let conversation = app.world().get::<Conversation>(agent);
    conversation
        .map(|it| it.messages().iter().map(|m| (**m).clone()).collect())
        .unwrap_or_default()
}

/// The ids of the entries of `agent`.
fn ids(app: &App, agent: Entity) -> Vec<MessageId> {
    let conversation = app.world().get::<Conversation>(agent);
    let ids = conversation.map(|it| it.entries().map(|entry| entry.id).collect());
    ids.unwrap_or_default()
}

#[test]
fn a_reviewed_compaction_is_edited_saved_and_undone() {
    let store = MemoryStore::default();
    let model = MockCompletionModel::from_stream_turns([
        reply("SUMMARY v1"),
        reply("SUMMARY v2"),
        reply("Answer after compaction."),
    ]);
    let answers = vec![
        Answer::choice("regenerate"),
        Answer::choice("edit"),
        Answer::text("SUMMARY v2 and file x.rs"),
        Answer::choice("accept"),
    ];
    let (mut app, wakes, agent) = start(&store, &model, answers);
    for turn in 1..=12 {
        let world = app.world_mut();
        commit_message(world, agent, Message::user(format!("Q{turn:02}: question"))).ok();
        commit_message(
            world,
            agent,
            Message::assistant(format!("A{turn:02}: answer")),
        )
        .ok();
    }
    app.update();
    let original = ids(&app, agent);

    // Regenerate, then Edit, then Accept.
    app.world_mut()
        .trigger(RunCommand::new(agent, "compact keep test names"));
    until_ended(&mut app, &wakes, 1);
    let requests = model.requests();
    assert_eq!(requests.len(), 2, "a summary, then another");
    let (summarizer, _) = shape(requests.first());
    assert!(summarizer.contains("keep test names"), "{summarizer}");
    let asked = &app.world().resource::<Answers>().asked;
    assert_eq!(asked.len(), 4, "{asked:?}");

    // The summary, then the last 2 turns as they were.
    ask(&mut app, &wakes, agent, "Q13: next");
    let requests = model.requests();
    let (all, first) = shape(requests.get(2));
    assert!(first.contains("SUMMARY v2 and file x.rs"), "{first}");
    for kept in ["Q11:", "A11:", "Q12:", "A12:", "Q13:"] {
        assert!(all.contains(kept), "{kept} in {all}");
    }
    for gone in ["Q01:", "A01:", "Q10:", "A10:", "SUMMARY v1"] {
        assert!(!all.contains(gone), "{gone} in {all}");
    }
    let id = app.world().get::<AgentId>(agent).map(|id| id.0.clone());
    drop(app);

    // Reloaded: the same request, and the log keeps the first message.
    let model = MockCompletionModel::from_stream_turns([
        reply("Answer after reload."),
        reply("Answer after uncompact."),
        reply("SUMMARY v3"),
    ]);
    let (mut app, wakes, agent) = start(&store, &model, vec![Answer::choice("cancel")]);
    ask(&mut app, &wakes, agent, "Q14: after reload");
    let (all, first) = shape(model.requests().first());
    assert!(first.contains("SUMMARY v2 and file x.rs"), "{first}");
    assert!(all.contains("Q12:") && !all.contains("Q01:"), "{all}");
    let records = history(&store, &id.unwrap_or_default()).unwrap_or_default();
    let kept_first = records.iter().any(|(_, record)| match record {
        LogRecord::Change(ChangeRecord {
            change: Change::Push { entries },
            ..
        }) => entries
            .iter()
            .any(|entry| entry.message == Message::user("Q01: question").into()),
        _ => false,
    });
    assert!(kept_first, "the log keeps the original first message");

    // `/uncompact`, choosing the compaction: the 12 turns with their ids,
    // then those after.
    app.world_mut().trigger(RunCommand::new(agent, "uncompact"));
    app.update();
    let world = app.world_mut();
    let picker = world
        .query_filtered::<Entity, With<Picker>>()
        .iter(world)
        .next();
    assert!(picker.is_some(), "the compactions to choose from");
    if let Some(picker) = picker {
        world.trigger(Pick::person(picker, Some(Chosen::item(0))));
    }
    app.update();
    let restored = ids(&app, agent);
    assert_eq!(restored.get(..original.len()), Some(original.as_slice()));
    ask(&mut app, &wakes, agent, "Q15: after uncompact");
    let (all, _) = shape(model.requests().get(1));
    assert!(!all.contains("SUMMARY"), "{all}");
    let at = |text: &str| all.find(text).unwrap_or(usize::MAX);
    assert!(at("Q01:") < at("A01:") && at("A01:") < at("Q12:"), "{all}");
    assert!(at("Q12:") < at("Q13:") && at("Q13:") < at("Q14:"), "{all}");
    assert!(at("Q14:") < usize::MAX, "{all}");

    // `/compact`, then Cancel: nothing changes.
    let before = messages(&app, agent);
    let ended = app.world().resource::<Ended>().0.len();
    app.world_mut().trigger(RunCommand::new(agent, "compact"));
    until_ended(&mut app, &wakes, ended + 1);
    assert_eq!(model.requests().len(), 3, "the summary was asked for");
    assert_eq!(messages(&app, agent), before);
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(ReviewableCompactionPlugin);
}

/// A failed editor leaves the summary as the model wrote it, says why,
/// and the review goes on.
#[cfg(all(unix, feature = "tui"))]
#[test]
fn a_failed_editor_keeps_the_summary_and_says_why() {
    use std::os::unix::process::ExitStatusExt;
    use std::process::ExitStatus;

    use rig_harness_test_support::{Collected, collect, scratch_home};
    use rig_tui::TerminalExit;

    let store = MemoryStore::default();
    let model = MockCompletionModel::from_stream_turns([
        reply("SUMMARY v1"),
        reply("Answer after compaction."),
    ]);
    let answers = vec![Answer::choice("editor"), Answer::choice("accept")];
    let (mut app, wakes, agent) = start(&store, &model, answers);
    let home = scratch_home("oc-reviewable-editor");
    let dir = home.root().to_path_buf();
    assert!(std::fs::create_dir_all(&dir).is_ok());
    app.insert_resource(Scratch(Some(dir.clone())));
    collect::<Notice>(&mut app);
    for turn in 1..=12 {
        let world = app.world_mut();
        commit_message(world, agent, Message::user(format!("Q{turn:02}: question"))).ok();
        commit_message(world, agent, Message::assistant(format!("A{turn:02}"))).ok();
    }
    app.update();

    app.world_mut().trigger(RunCommand::new(agent, "compact"));
    let editing = |world: &World| {
        world
            .iter_entities()
            .find_map(|entity| entity.get::<editor::EditorFor>().map(|_| entity.id()))
    };
    run_until(&mut app, &wakes, |world| editing(world).is_some());
    // The editor wrote the file, then exited with 3.
    std::fs::write(dir.join("reviewable-compaction-summary.md"), "EDITED").ok();
    if let Some(run) = editing(app.world()) {
        let failed = TerminalExit(Ok(ExitStatus::from_raw(3 << 8)));
        app.world_mut().entity_mut(run).insert(failed);
    }
    until_ended(&mut app, &wakes, 1);
    std::fs::remove_dir_all(&dir).ok();

    let notices = &app.world().resource::<Collected<Notice>>().0;
    let said: Vec<String> = notices.iter().map(|notice| notice.text.clone()).collect();
    assert!(
        said.contains(&"The editor failed (exit status 3); the summary is unchanged.".to_owned()),
        "{said:?}"
    );
    ask(&mut app, &wakes, agent, "Q13: next");
    let requests = model.requests();
    let (_, first) = shape(requests.get(1));
    assert!(
        first.contains("SUMMARY v1") && !first.contains("EDITED"),
        "{first}"
    );
}
