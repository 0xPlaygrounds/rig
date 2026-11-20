use std::sync::Arc;
use std::sync::mpsc::Receiver;

use rig_basics::telemetry::effect_log::{EffectLog, EffectLogPlugin};
use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::SessionId;
use rig_harness::rig_core::tool::ToolOutput;
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{
    Collected, Ended, app_in, app_on, calls, collect, commit_message, connect, first_agent,
    log_ended, reply, run_until, say, scratch_home, scripted_models,
};

use super::*;

/// A summary in the template, told apart by `marker`.
fn template(marker: &str) -> String {
    format!(
        "## Goal\nShip the API ({marker}).\n\n## Constraints the user gave\n- Keep the tests.\n\n\
         ## Decisions and why\n- **Dropped the cache**: it served stale data.\n\n\
         ## Files touched\n- `src/f1.rs`: read for the API.\n\n## Current state\nWorks.\n\n\
         ## Open questions / next steps\n1. Release."
    )
}

/// What the user says in turn `turn`.
fn says(turn: usize) -> String {
    format!("user says {turn}: do step {turn}")
}

/// What `read` is called with.
#[derive(Deserialize, schemars::JsonSchema)]
struct ReadArgs {
    path: String,
}

/// `read` answers with the path it was given, and 8,000 words more for a path in `big/`.
fn read_file(called: On<ToolCalled<ReadArgs>>, mut calls: Query<&mut Decision, ToolCalls>) {
    let path = &called.args.path;
    let more = if path.starts_with("big/") { 8_000 } else { 0 };
    let text = format!("contents of {path}{}", " word".repeat(more));
    if let Ok(mut call) = calls.get_mut(called.call) {
        call.decide(
            Decider::plugin("fh-structured-compaction"),
            ToolResult::success(ToolOutput::text(text)),
        );
    }
}

/// Runs frames until one more turn of `app` ended than `before`.
fn until_ended(app: &mut App, wakes: &Receiver<()>, before: usize) {
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() > before
    });
}

/// Each request `model` got, as JSON.
fn sent(model: &MockCompletionModel) -> Vec<String> {
    let requests = model.requests();
    let json = requests.iter().map(serde_json::to_string);
    json.map(Result::unwrap_or_default).collect()
}

/// Where `part` is in `text`, past the end when it is not.
fn at(text: &str, part: &str) -> usize {
    text.find(part).unwrap_or(usize::MAX)
}

#[test]
fn compact_writes_the_template_keeps_the_last_exchanges_and_builds_on_the_last_summary() {
    let home = scratch_home("fh-structured-compaction");
    let (mut app, wakes) = app_in(&home, &SessionId::generate());
    let mut script = Vec::new();
    for turn in 1..=6 {
        let path = serde_json::json!({ "path": format!("src/f{turn}.rs") });
        script.push(calls(&format!("c{turn}"), "read", path));
        script.push(reply(&format!("answer {turn}")));
    }
    script.push(reply("answer 7"));
    let model = MockCompletionModel::from_stream_turns(script);
    let summarizer = MockCompletionModel::from_stream_turns([
        reply(&template("FIRST-SUMMARY")),
        reply(&template("SECOND-SUMMARY")),
    ]);
    let summarizer_name = "test/summarizer";
    app.insert_resource(scripted_models([
        ("test/agent", &model),
        (summarizer_name, &summarizer),
    ]))
    .add_plugins((AgentPlugin, SessionLogPlugin, EffectLogPlugin))
    .add_plugins(StructuredCompactionPlugin)
    .insert_resource(StructuredCompaction {
        model: Some(summarizer_name.to_owned()),
        keep_user_messages: 3,
    })
    .add_open_tool("read", "Read a file.", ToolOptions::default(), read_file)
    .init_resource::<Ended>()
    .add_observer(log_ended);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut());
    assert!(agent.is_some(), "an agent");
    let Some(agent) = agent else {
        return;
    };
    let model_name = "test/agent".to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model: model_name,
        kind: ChoiceKind::ForThisRun,
    });
    let deliver = |app: &mut App, text: String| say(app, &wakes, agent, &text);
    let compact = |app: &mut App, line: &str| {
        let before = app.world().resource::<Ended>().0.len();
        app.world_mut().trigger(RunCommand::new(agent, line));
        until_ended(app, &wakes, before);
    };
    for turn in 1..=6 {
        deliver(&mut app, says(turn));
    }

    compact(&mut app, "compact keep the API details");
    deliver(&mut app, says(7));
    let asked = sent(&summarizer);
    let first = asked.first().map(String::as_str).unwrap_or_default();
    assert!(first.contains("keep the API details"), "{first}");
    assert!(first.contains("## Decisions and why"), "{first}");
    let requests = sent(&model);
    let next = requests.last().map(String::as_str).unwrap_or_default();
    // The summary, then the user's last three messages, each where it was with its whole
    // exchange, and the new message.
    let order = [
        at(next, "FIRST-SUMMARY"),
        at(next, &says(4)),
        at(next, "contents of src/f4.rs"),
        at(next, &says(5)),
        at(next, "contents of src/f5.rs"),
        at(next, &says(6)),
        at(next, "contents of src/f6.rs"),
        at(next, &says(7)),
    ];
    assert!(
        order.is_sorted() && order.last() < Some(&usize::MAX),
        "{order:?} {next}"
    );
    for turn in 1..=3 {
        assert!(!next.contains(&says(turn)), "{next}");
        assert!(
            !next.contains(&format!("contents of src/f{turn}.rs")),
            "{next}"
        );
    }

    compact(&mut app, "compact");
    let asked = sent(&summarizer);
    let second = asked.get(1).map(String::as_str).unwrap_or_default();
    assert!(second.contains("<previous-summary>"), "{second}");
    assert!(second.contains("FIRST-SUMMARY"), "{second}");

    // The saved session still has every message the user sent.
    app.update();
    let log = app.world().get_resource::<EffectLog>();
    let records = log.and_then(|log| log.read().ok()).unwrap_or_default();
    let texts: Vec<String> = records
        .iter()
        .filter_map(|record| MessageRecord::read(&record.kind))
        .map(|message| message.text)
        .collect();
    for turn in 1..=6 {
        assert!(texts.contains(&says(turn)), "{texts:?}");
    }
}

/// A turn of an agent whose conversation is near a 40k-token window, an image with the user's
/// last message before the turn, answered by `model`, with `summarizer` writing the template's
/// summaries and `keep` of the user's messages kept: the app once the turn ended, the agent and
/// its entries before the turn.
fn turn_near_the_window(
    (model, summarizer): (&MockCompletionModel, &MockCompletionModel),
    keep: usize,
) -> Option<(App, Entity, Vec<Entry>)> {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(scripted_models([("test/summarizer", summarizer)]))
        .add_plugins((AgentPlugin, SessionLogPlugin, StructuredCompactionPlugin))
        .insert_resource(StructuredCompaction {
            model: Some("test/summarizer".to_owned()),
            keep_user_messages: keep,
        })
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<Notice>(&mut app);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut())?;
    let mut connection = connect(model)?;
    connection.spec = Arc::new(connection.spec.as_ref().clone().with_context_window(40_000));
    app.world_mut().entity_mut(agent).insert(connection);
    let long = "word ".repeat(4_000);
    let mut messages = vec![Message::user("What does the crate do?")];
    for turn in 0..6 {
        messages.push(Message::assistant(format!("{turn}: {long}")));
        messages.push(pictured(format!("question {turn}")));
    }
    messages.push(Message::assistant("The last answer."));
    for message in messages {
        commit_message(app.world_mut(), agent, message).ok()?;
    }
    let before = entries(app.world(), agent);
    let next = Deliver::new(agent, "And the next question?", DeliveryMode::Steer);
    app.world_mut().trigger(next);
    until_ended(&mut app, &wakes, 0);
    Some((app, agent, before))
}

/// A message of the user's of `text` with an image.
fn pictured(text: String) -> Message {
    let mut message = Message::user(text);
    if let Message::User { content } = &mut message {
        content.push(rig_core::message::UserContent::image_base64(
            "aGk=",
            Some(rig_core::message::ImageMediaType::PNG),
            None,
        ));
    }
    message
}

/// Whether `entries`, after a summary, end with `before` from the one holding `text` on: the
/// kept messages stay the entries they were, attachments included.
fn kept_from(entries: &[Entry], before: &[Entry], text: &str) -> bool {
    let holds = |entry: &Entry| serde_json::to_string(entry).is_ok_and(|json| json.contains(text));
    let kept = before
        .iter()
        .position(holds)
        .and_then(|at| before.get(at..));
    let summarized = entries.first().is_some_and(|entry| entry.author.is_some());
    summarized && kept.is_some_and(|kept| entries.get(1..=kept.len()) == Some(kept))
}

/// A summary that twice lacks its headings falls back to the default summary, by the agent's
/// model, of the same messages: the user's last messages stay the entries they were, their
/// images included, and the turn goes on.
#[test]
fn a_summary_without_its_headings_twice_falls_back_to_the_default_summary_of_the_same_messages() {
    let summarizer =
        MockCompletionModel::from_stream_turns([reply("Just some prose."), reply("More prose.")]);
    let model = MockCompletionModel::from_stream_turns([
        reply("## Goal\nThe default summary (DEFAULT-SUMMARY)."),
        reply("done"),
    ]);
    let app = turn_near_the_window((&model, &summarizer), 3);
    assert!(app.is_some(), "an agent on a listed model");
    let Some((app, agent, before)) = app else {
        return;
    };

    assert_eq!(sent(&summarizer).len(), 2, "the template, asked twice");
    let requests = sent(&model);
    assert_eq!(requests.len(), 2, "one default summary, then the turn");
    let default = requests.first().map(String::as_str).unwrap_or_default();
    assert!(
        default.contains("You are a context summarization assistant"),
        "{default}"
    );
    assert!(
        default.contains("question 3") && !default.contains("question 4"),
        "{default}"
    );
    assert!(
        matches!(
            app.world().resource::<Ended>().0.as_slice(),
            [TurnOutcome::Answered(_)]
        ),
        "{:?}",
        app.world().resource::<Ended>().0
    );
    let last = requests.last().map(String::as_str).unwrap_or_default();
    assert!(last.contains("DEFAULT-SUMMARY"), "{last}");
    let after = entries(app.world(), agent);
    assert!(kept_from(&after, &before, "question 4"), "{after:?}");
    let notices = &app.world().resource::<Collected<Notice>>().0;
    assert!(
        notices
            .iter()
            .any(|notice| notice.text.contains("default way")),
        "{notices:?}"
    );
}

/// Near the window with fewer messages of the user's than are kept, nothing may be summarized:
/// the compaction is refused before any summary is asked for, with a notice, and the request
/// goes as it is.
#[test]
fn near_the_window_with_nothing_before_the_kept_messages_no_summary_is_asked_for() {
    let summarizer = MockCompletionModel::from_stream_turns([reply("unused")]);
    let model = MockCompletionModel::from_stream_turns([reply("done")]);
    let app = turn_near_the_window((&model, &summarizer), 20);
    assert!(app.is_some(), "an agent on a listed model");
    let Some((app, agent, before)) = app else {
        return;
    };
    assert_eq!(sent(&summarizer).len(), 0);
    let requests = sent(&model);
    assert_eq!(requests.len(), 1, "{requests:?}");
    let request = requests.first().map(String::as_str).unwrap_or_default();
    assert!(
        request.contains("What does the crate do?") && request.contains("question 5"),
        "{request}"
    );
    let after = entries(app.world(), agent);
    assert_eq!(after.get(..before.len()), Some(before.as_slice()));
    let ended = &app.world().resource::<Ended>().0;
    assert!(
        matches!(ended.as_slice(), [TurnOutcome::Answered(_)]),
        "{ended:?}"
    );
    let notices = &app.world().resource::<Collected<Notice>>().0;
    let refused = "nothing comes before the user's last 20 messages";
    assert!(
        notices.iter().any(|notice| notice.text.contains(refused)),
        "{notices:?}"
    );
}

/// A refusal of the request as longer than the model's window.
fn too_long() -> Vec<rig_core::test_utils::MockStreamEvent> {
    let refusal =
        rig_core::ProviderResponseError::without_status("prompt is too long: 210000 tokens");
    let refused = rig_core::test_utils::MockError::ProviderResponse(refusal);
    vec![rig_core::test_utils::MockStreamEvent::Error(refused)]
}

/// An agent of `app` on `model`, with `summarizer` writing the template's summaries, the
/// defaults otherwise.
fn agent_with_defaults(
    app: &mut App,
    (model, summarizer): (&MockCompletionModel, &MockCompletionModel),
) -> Option<Entity> {
    app.insert_resource(scripted_models([
        ("test/agent", model),
        ("test/summarizer", summarizer),
    ]))
    .add_plugins((AgentPlugin, SessionLogPlugin, StructuredCompactionPlugin))
    .insert_resource(StructuredCompaction {
        model: Some("test/summarizer".to_owned()),
        ..StructuredCompaction::default()
    })
    .add_open_tool("read", "Read a file.", ToolOptions::default(), read_file)
    .init_resource::<Ended>()
    .add_observer(log_ended);
    collect::<Notice>(app);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut())?;
    app.world_mut().trigger(SetModel {
        entity: agent,
        model: "test/agent".to_owned(),
        kind: ChoiceKind::ForThisRun,
    });
    Some(agent)
}

/// A session of two prompts and many long tool results whose next request is refused as too long
/// recovers: the first prompt's exchange is summarized though fewer than the three kept by
/// default came before, and the request is answered.
#[test]
fn an_overflow_with_few_prompts_and_long_tool_output_is_answered() {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    let mut script = Vec::new();
    for call in 1..=4 {
        let path = serde_json::json!({ "path": format!("big/f{call}.rs") });
        script.push(calls(&format!("c{call}"), "read", path));
    }
    script.extend([reply("built"), too_long(), reply("done")]);
    let model = MockCompletionModel::from_stream_turns(script);
    let summarizer = MockCompletionModel::from_stream_turns([reply(&template("RECOVERED"))]);
    let agent = agent_with_defaults(&mut app, (&model, &summarizer));
    assert!(agent.is_some(), "an agent");
    let Some(agent) = agent else {
        return;
    };
    say(&mut app, &wakes, agent, "Fix the build.");
    say(&mut app, &wakes, agent, "Now the tests.");

    assert_eq!(
        sent(&summarizer).len(),
        1,
        "{:?} {:?}",
        app.world().resource::<Collected<Notice>>().0,
        sent(&model)
    );
    let ended = &app.world().resource::<Ended>().0;
    assert!(
        matches!(ended.as_slice(), [_, TurnOutcome::Answered(_)]),
        "{ended:?} {:?}",
        app.world().resource::<Collected<Notice>>().0
    );
    let requests = sent(&model);
    let resent = requests.last().map(String::as_str).unwrap_or_default();
    assert!(
        resent.contains("RECOVERED")
            && resent.contains("Now the tests.")
            && !resent.contains("Fix the build."),
        "{resent}"
    );
}

/// A session of one prompt and many long tool results whose next request is refused as too long
/// fails the turn with a notice saying why: nothing comes before the user's only message, which
/// is never summarized, so no summary is asked for and the conversation stays as it was.
#[test]
fn an_overflow_with_one_prompt_fails_the_turn_and_changes_nothing() {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    let mut script = Vec::new();
    for call in 1..=4 {
        let path = serde_json::json!({ "path": format!("big/f{call}.rs") });
        script.push(calls(&format!("c{call}"), "read", path));
    }
    script.push(too_long());
    let model = MockCompletionModel::from_stream_turns(script);
    let summarizer = MockCompletionModel::from_stream_turns([reply(&template("UNUSED"))]);
    let agent = agent_with_defaults(&mut app, (&model, &summarizer));
    assert!(agent.is_some(), "an agent");
    let Some(agent) = agent else {
        return;
    };
    say(&mut app, &wakes, agent, "Fix the build.");

    let notices = &app.world().resource::<Collected<Notice>>().0;
    let ended = &app.world().resource::<Ended>().0;
    assert!(
        matches!(ended.as_slice(), [outcome] if !matches!(outcome, TurnOutcome::Answered(_))),
        "{ended:?} {notices:?}"
    );
    let failed = |text: &str| {
        text.starts_with("Context overflow recovery failed")
            && text.contains("nothing comes before")
    };
    assert!(
        notices.iter().any(|notice| failed(&notice.text)),
        "{notices:?}"
    );
    assert!(sent(&summarizer).is_empty() && sent(&model).len() == 5);
    let entries = entries(app.world(), agent);
    let first = entries.first().and_then(|entry| entry.message.rag_text());
    assert_eq!(first.as_deref(), Some("Fix the build."));
    assert_eq!(entries.len(), 9, "the prompt, four calls and their results");
    let summary = Some(rig_memory::summary_author());
    assert!(entries.iter().all(|entry| entry.author != summary));
}

/// A `/compact` of a conversation of one message of the user's, with an image, and a long reply
/// is refused: the user's only message is never summarized, and nothing changes.
#[test]
fn a_compact_with_only_the_users_first_message_to_keep_is_refused() {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    let model = MockCompletionModel::from_stream_turns([reply("## Goal\nDEFAULT-SUMMARY.")]);
    let summarizer = MockCompletionModel::from_stream_turns([reply(&template("UNUSED"))]);
    let agent = agent_with_defaults(&mut app, (&model, &summarizer));
    assert!(agent.is_some(), "an agent");
    let Some(agent) = agent else {
        return;
    };
    let long = Message::assistant("word ".repeat(16_001));
    for message in [pictured("Read this.".to_owned()), long] {
        assert!(commit_message(app.world_mut(), agent, message).is_ok());
    }
    let before = entries(app.world(), agent);
    app.world_mut().trigger(RunCommand::new(agent, "compact"));
    until_ended(&mut app, &wakes, 0);

    assert_eq!(entries(app.world(), agent), before);
    assert_eq!(sent(&summarizer).len() + sent(&model).len(), 0);
    let notices = &app.world().resource::<Collected<Notice>>().0;
    assert!(
        notices
            .iter()
            .any(|notice| notice.text.contains("nothing comes before")),
        "{notices:?}"
    );
}

/// A `/compact` of a conversation of exactly as many messages of the user's as are kept, an
/// image with the oldest, is refused: none of them is summarized, and nothing changes.
#[test]
fn a_compact_with_exactly_the_kept_messages_is_refused_and_changes_nothing() {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    let model = MockCompletionModel::from_stream_turns([reply("## Goal\nDEFAULT-SUMMARY.")]);
    let summarizer = MockCompletionModel::from_stream_turns([reply(&template("UNUSED"))]);
    let agent = agent_with_defaults(&mut app, (&model, &summarizer));
    assert!(agent.is_some(), "an agent");
    let Some(agent) = agent else {
        return;
    };
    let long = || Message::assistant("word ".repeat(16_001));
    let asked = [
        pictured(says(1)),
        Message::user(says(2)),
        Message::user(says(3)),
    ];
    for message in asked.into_iter().flat_map(|asked| [asked, long()]) {
        assert!(commit_message(app.world_mut(), agent, message).is_ok());
    }
    let before = entries(app.world(), agent);
    app.world_mut().trigger(RunCommand::new(agent, "compact"));
    until_ended(&mut app, &wakes, 0);

    assert_eq!(entries(app.world(), agent), before);
    assert_eq!(sent(&summarizer).len() + sent(&model).len(), 0);
    let notices = &app.world().resource::<Collected<Notice>>().0;
    let refused = "nothing comes before the user's last 3 messages";
    assert!(
        notices.iter().any(|notice| notice.text.contains(refused)),
        "{notices:?}"
    );
}

/// The agent's entries, oldest first.
fn entries(world: &World, agent: Entity) -> Vec<Entry> {
    let conversation = world.get::<Conversation>(agent);
    let all = conversation.map(|it| (0..it.messages().len()).filter_map(|at| it.entry(at)));
    all.into_iter().flatten().collect()
}

/// A compaction only ever replaces the messages before the ones kept by a summary: the kept
/// messages stay the entries they were, an image among them, and a compaction that is refused,
/// or whose summary comes back after the conversation changed, changes nothing.
#[test]
fn kept_messages_stay_the_entries_they_were_and_a_failed_compaction_changes_nothing() {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    let answers = [1, 3, 4].map(|turn| reply(&format!("answer {turn}")));
    let model = MockCompletionModel::from_stream_turns(answers);
    let summarizer = MockCompletionModel::from_stream_turns([
        reply(&template("FIRST-SUMMARY")),
        reply(&template("SECOND-SUMMARY")),
    ]);
    /// Rewrites the first entry while a summary is reviewed, once switched on.
    #[derive(Resource, Default)]
    struct EditMeanwhile(bool);
    let edit = |calls: Query<&ForAgent, (ReadyToSettle, With<Summarizing>)>,
                mut agents: Query<&mut Conversation>,
                mut on: ResMut<EditMeanwhile>| {
        for &ForAgent(agent) in &calls {
            if let (true, Ok(mut conversation)) = (on.0, agents.get_mut(agent)) {
                on.0 = false;
                conversation.edit_each(Author::plugin("test"), |entry| {
                    (entry.index == 0).then(|| Message::user("edited meanwhile"))
                });
            }
        }
    };
    app.insert_resource(scripted_models([
        ("test/agent", &model),
        ("test/summarizer", &summarizer),
    ]))
    .add_plugins((AgentPlugin, SessionLogPlugin, StructuredCompactionPlugin))
    .insert_resource(StructuredCompaction {
        model: Some("test/summarizer".to_owned()),
        keep_user_messages: 2,
    })
    .init_resource::<EditMeanwhile>()
    .add_systems(AgentPass, edit.in_set(ReviewSystems::Edit))
    .init_resource::<Ended>()
    .add_observer(log_ended);
    collect::<Notice>(&mut app);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut());
    assert!(agent.is_some(), "an agent");
    let Some(agent) = agent else {
        return;
    };
    app.world_mut().trigger(SetModel {
        entity: agent,
        model: "test/agent".to_owned(),
        kind: ChoiceKind::ForThisRun,
    });
    let compact = |app: &mut App| {
        let before = app.world().resource::<Ended>().0.len();
        app.world_mut().trigger(RunCommand::new(agent, "compact"));
        until_ended(app, &wakes, before);
    };
    say(&mut app, &wakes, agent, &says(1));
    for message in [pictured(says(2)), Message::assistant("answer 2")] {
        assert!(commit_message(app.world_mut(), agent, message).is_ok());
    }
    say(&mut app, &wakes, agent, &says(3));

    // The last two of the user's messages stay, the image with its message.
    let before = entries(app.world(), agent);
    compact(&mut app);
    let after = entries(app.world(), agent);
    let kept = before
        .iter()
        .position(|entry| serde_json::to_string(entry).is_ok_and(|json| json.contains(&says(2))));
    assert_eq!(after.get(1..), kept.and_then(|kept| before.get(kept..)));
    let json = serde_json::to_string(&after).unwrap_or_default();
    assert!(
        json.contains("FIRST-SUMMARY") && json.contains("aGk="),
        "{json}"
    );
    assert!(!json.contains(&says(1)), "{json}");

    // Nothing comes before the two kept messages now: refused, nothing changes.
    compact(&mut app);
    assert_eq!(entries(app.world(), agent), after);
    assert_eq!(sent(&summarizer).len(), 1);

    // A summary whose messages changed while it was written goes in nowhere.
    say(&mut app, &wakes, agent, &says(4));
    let before = entries(app.world(), agent);
    app.world_mut().resource_mut::<EditMeanwhile>().0 = true;
    compact(&mut app);
    let after = entries(app.world(), agent);
    assert_eq!(sent(&summarizer).len(), 2);
    let ids = |entries: &[Entry]| -> Vec<_> { entries.iter().map(|entry| entry.id).collect() };
    assert_eq!(ids(&after), ids(&before));
    assert_eq!(after.get(1..), before.get(1..));
    let json = serde_json::to_string(&after).unwrap_or_default();
    assert!(
        json.contains("edited meanwhile") && !json.contains("SECOND-SUMMARY"),
        "{json}"
    );
    let notices = &app.world().resource::<Collected<Notice>>().0;
    let said = |text: &str| notices.iter().any(|notice| notice.text.contains(text));
    assert!(said("changed meanwhile"), "{notices:?}");
}

rig_defaults_test_support::stages_ordered_test!(StructuredCompactionPlugin);
