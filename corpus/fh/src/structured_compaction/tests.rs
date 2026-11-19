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

/// `read` answers with the path it was given.
fn read_file(called: On<ToolCalled<ReadArgs>>, mut calls: Query<&mut Decision, ToolCalls>) {
    let text = format!("contents of {}", called.args.path);
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

/// A turn of an agent whose conversation is near a 40k-token window, answered by `model`, with
/// `summarizer` writing the template's summaries and `keep` of the user's messages kept: the app
/// once the turn ended.
fn turn_near_the_window(
    (model, summarizer): (&MockCompletionModel, &MockCompletionModel),
    keep: usize,
) -> Option<App> {
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
        messages.push(Message::user(format!("question {turn}")));
    }
    messages.push(Message::assistant("The last answer."));
    for message in messages {
        commit_message(app.world_mut(), agent, message).ok()?;
    }
    let next = Deliver::new(agent, "And the next question?", DeliveryMode::Steer);
    app.world_mut().trigger(next);
    until_ended(&mut app, &wakes, 0);
    Some(app)
}

#[test]
fn a_summary_without_its_headings_twice_falls_back_and_the_turn_goes_on() {
    let summarizer =
        MockCompletionModel::from_stream_turns([reply("Just some prose."), reply("More prose.")]);
    // The fallback is the default compaction, whose cut splits the last turn: a history summary,
    // then one of the turn's earlier part.
    let model = MockCompletionModel::from_stream_turns([
        reply("## Goal\nThe default summary (DEFAULT-SUMMARY)."),
        reply("## Original Request\nThe next question."),
        reply("done"),
    ]);
    let app = turn_near_the_window((&model, &summarizer), 3);
    assert!(app.is_some(), "an agent on a listed model");
    let Some(app) = app else {
        return;
    };

    assert_eq!(sent(&summarizer).len(), 2, "the template, asked twice");
    let requests = sent(&model);
    let default = requests.first().map(String::as_str).unwrap_or_default();
    assert!(
        default.contains("You are a context summarization assistant"),
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
    let notices = &app.world().resource::<Collected<Notice>>().0;
    assert!(
        notices
            .iter()
            .any(|notice| notice.text.contains("default way")),
        "{notices:?}"
    );
}

/// With fewer messages of the user's than are kept, nothing may be summarized: the compaction
/// is refused before any summary is asked for, with a notice, and the request goes as it is.
#[test]
fn a_compaction_with_nothing_before_the_kept_messages_is_refused() {
    let summarizer = MockCompletionModel::from_stream_turns([reply("unused")]);
    let model = MockCompletionModel::from_stream_turns([reply("done")]);
    let app = turn_near_the_window((&model, &summarizer), 20);
    assert!(app.is_some(), "an agent on a listed model");
    let Some(app) = app else {
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
    let mut pictured = Message::user(says(2));
    if let Message::User { content } = &mut pictured {
        content.push(rig_core::message::UserContent::image_base64(
            "aGk=",
            Some(rig_core::message::ImageMediaType::PNG),
            None,
        ));
    }
    for message in [pictured, Message::assistant("answer 2")] {
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
